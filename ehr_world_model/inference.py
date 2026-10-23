"""
Inference utilities: model construction, checkpoint I/O, patient-level
embedding trajectories and per-timestep risk probes.
"""
from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import TRANSFORMER_SIZES, TransformerWorldModel, VelocityWorldModel

HOURS_PER_STEP = 6

_LOSS_KW = dict(lambda_mse=1.0, lambda_path=0.5, lambda_sigreg=0.05, lambda_recon=0.20)


def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Model construction / checkpoints ──────────────────────────────────────────

def build_model(arch: str, state_dim: int, n_actions: int, n_continuous: int,
                action_conditioned: bool = True, size: str = "M") -> nn.Module:
    """arch: 'velocity' (v6) or 'transformer' (v7, size S/M/L)."""
    if arch == "velocity":
        return VelocityWorldModel(state_dim=state_dim, n_actions=n_actions, latent_dim=128,
                                  action_conditioned=action_conditioned,
                                  n_continuous=n_continuous, **_LOSS_KW)
    if arch == "transformer":
        return TransformerWorldModel(state_dim=state_dim, n_actions=n_actions,
                                     action_conditioned=action_conditioned,
                                     n_continuous=n_continuous, dropout=0.1,
                                     **TRANSFORMER_SIZES[size], **_LOSS_KW)
    raise ValueError(f"unknown arch '{arch}'")


def save_checkpoint(model: nn.Module, path, **config) -> None:
    """Saves weights together with the kwargs needed by ``build_model``."""
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def load_checkpoint(path, device=None) -> nn.Module:
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ── Patient trajectories ──────────────────────────────────────────────────────

def ema_smooth(traj: torch.Tensor, alpha: float = 0.25) -> torch.Tensor:
    if traj.shape[0] == 0:
        return traj
    out = [traj[0]]
    for i in range(1, traj.shape[0]):
        out.append(alpha * traj[i] + (1 - alpha) * out[-1])
    return torch.stack(out)


@torch.no_grad()
def build_patient_trajectories(model, aug_loader, dataset, max_steps: int = 128) -> Dict[str, Dict]:
    """Group windowed embeddings by patient.

    Returns {pid: {'z_true': (T, D) encoder embeddings,
                   'z_pred': (T, D) one-step world-model predictions,
                   'length': T}}  (both EMA-smoothed, α=0.25).
    ``aug_loader`` must be unshuffled so batch order matches dataset.patient_ids.
    """
    dev = next(model.parameters()).device
    model.eval()
    pid_zt, pid_zp = defaultdict(list), defaultdict(list)
    global_i = 0
    for batch in aug_loader:
        s = batch["s"].to(dev); a = batch["actions"].to(dev)
        out = model(s, a, tf_ratio=0.0)
        B = s.shape[0]
        for bi in range(B):
            pid = dataset.patient_ids[global_i + bi]
            if pid is None:
                continue
            pid_zt[str(pid)].append(ema_smooth(out["emb"][bi].cpu(), 0.25))
            pid_zp[str(pid)].append(ema_smooth(out["next_emb"][bi].cpu(), 0.25))
        global_i += B
    trajs = {}
    for pid_s in pid_zt:
        zt = torch.cat(pid_zt[pid_s], dim=0)[:max_steps]
        zp = torch.cat(pid_zp[pid_s], dim=0)[:max_steps]
        trajs[pid_s] = {"z_true": zt, "z_pred": zp, "length": zt.shape[0]}
    return trajs


# ── Linear trajectory risk probe ──────────────────────────────────────────────
# nn.Linear(D→1) applied at every timestep; patient risk = mean sigmoid over T.

def get_label(pid_s: str, col: str, label_map: Dict) -> float:
    key = int(pid_s) if pid_s.lstrip("-").isdigit() else pid_s
    v = label_map.get(key, {}).get(col, float("nan"))
    return float(v) if v is not None else float("nan")


def train_linear_traj_probe(traj_embs: List[torch.Tensor], labels: List[float],
                            n_epochs: int = 200, lr: float = 3e-3, patience: int = 20) -> nn.Linear:
    D = traj_embs[0].shape[1]
    probe = nn.Linear(D, 1)
    nn.init.xavier_uniform_(probe.weight, gain=0.5)
    nn.init.zeros_(probe.bias)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    y_t = torch.tensor(labels, dtype=torch.float32)
    best_loss, pc, best_w = float("inf"), 0, copy.deepcopy(probe.state_dict())
    for _ in range(n_epochs):
        probe.train(); opt.zero_grad()
        probs = torch.stack([torch.sigmoid(probe(e)).mean() for e in traj_embs])
        loss = F.binary_cross_entropy(probs, y_t)
        loss.backward(); opt.step()
        v = loss.item()
        if v < best_loss - 1e-5:
            best_loss, pc, best_w = v, 0, copy.deepcopy(probe.state_dict())
        else:
            pc += 1
            if pc >= patience:
                break
    probe.load_state_dict(best_w)
    return probe


def train_risk_probe(col: str, trajs: Dict[str, Dict], label_map: Dict) -> Optional[nn.Linear]:
    """Fit a trajectory probe for task ``col`` on encoder trajectories (z_true)."""
    embs, lbls = [], []
    for pid_s, tr in trajs.items():
        lbl = get_label(pid_s, col, label_map)
        if not np.isfinite(lbl):
            continue
        embs.append(tr["z_true"][:tr["length"]])
        lbls.append(lbl)
    if len(lbls) < 10 or sum(lbls) < 2:
        return None
    return train_linear_traj_probe(embs, lbls, n_epochs=300, patience=25)


@torch.no_grad()
def get_traj_risk_scores(probe, traj_embs) -> np.ndarray:
    """Mean per-timestep probability per patient → (N,)."""
    probe.eval()
    return np.array([torch.sigmoid(probe(e)).mean().item() for e in traj_embs])


@torch.no_grad()
def get_traj_risk_series(probe, traj_emb) -> np.ndarray:
    """Per-timestep risk for a single patient trajectory → (T,)."""
    probe.eval()
    return torch.sigmoid(probe(traj_emb)).squeeze(1).numpy()
