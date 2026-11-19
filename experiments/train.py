"""
Train / load the world models (wm5, wm5plus, wm78, wm0).

    wm5plus  5 drug groups + additive action forcing B(z,a)     (EHR-WM, experiments.models)
    wm5      EHRWorldModel, actions = 5 drug groups
    wm78     EHRWorldModel, actions = 78 medication codes
    wm0      EHRWorldModel without action conditioning     (ablation)

Checkpoints are written to ``<run-dir>/<name>/model.pt`` (weights + constructor
config); existing checkpoints are re-used rather than retrained.

    python -m experiments.train --proc-dir data/processed/mimic_wm --run-dir runs/mimic --models wm5plus --stages onestep
"""
from __future__ import annotations

import copy
import logging
import time
from pathlib import Path

import pandas as pd
import torch

from ehr_world_model.inference import _LOSS_KW, get_device, save_checkpoint
from ehr_world_model.inference import build_model as _build_base

from .data import Split, make_loader
from .models import AdditiveActionEHRWorldModel

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)

class _Specs(dict):
    """Base specs + variants named <base>[_s<seed>][_f<forcing scale>], e.g. wm5plus_s1, wm5plus_f0.5."""

    def __missing__(self, name):
        import re
        m = re.fullmatch(r"(wm5plus|wm5|wm78|wm0)((?:_s\d+)?)((?:_f[0-9.]+)?)", name)
        if not m or m.group(1) not in self:
            raise KeyError(name)
        spec = dict(dict.__getitem__(self, m.group(1)))
        if m.group(2):
            spec["seed"] = int(m.group(2)[2:]); spec["label"] += f", seed {spec['seed']}"
        if m.group(3):
            spec["forcing_scale"] = float(m.group(3)[2:]); spec["label"] += f", forcing x{spec['forcing_scale']:g}"
        return spec


MODEL_SPECS = _Specs({
    "wm5":  dict(action_key="act5",  action_conditioned=True,  label="EHR-WM (5 actions)"),
    "wm78": dict(action_key="act78", action_conditioned=True,  label="EHR-WM (78 med codes)"),
    "wm0":  dict(action_key="act5",  action_conditioned=False, label="EHR-WM (no actions)"),
    "wm5plus": dict(action_key="act5", action_conditioned=True, arch="velocity_additive",
                    label="EHR-WM+ (5 actions, additive forcing)"),
})


def build_model(arch, state_dim, n_actions, n_continuous, action_conditioned=True, forcing_scale=1.0, **kw):
    """'velocity' (ehr_world_model) or 'velocity_additive' (experiments.models)."""
    if arch == "velocity_additive":
        return AdditiveActionEHRWorldModel(state_dim=state_dim, n_actions=n_actions, latent_dim=128,
                                        action_conditioned=True, n_continuous=n_continuous,
                                        forcing_scale=forcing_scale, **_LOSS_KW)
    return _build_base(arch, state_dim, n_actions, n_continuous, action_conditioned)


def load_checkpoint(path, device=None):
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
SEED = 42


# 3-stage scheduled-sampling curriculum (tf=1.0/k=1 -> tf=0.5/k=2 -> tf=0.5/k=4).
FAST_STAGES = [
    dict(tf=1.0, rollout_k=1, epochs=8, patience=3),
    dict(tf=0.5, rollout_k=2, epochs=5, patience=3),
    dict(tf=0.5, rollout_k=4, epochs=5, patience=3),
]


# One-step training: stage 1 only (1-step prediction objective).
ONE_STEP_STAGES = [dict(tf=1.0, rollout_k=1, epochs=8, patience=3)]
STAGE_SETS = {"fast": FAST_STAGES, "onestep": ONE_STEP_STAGES}


def ckpt_path(run_dir, name):
    """Final checkpoint (written when the whole curriculum has finished)."""
    return Path(run_dir) / name / "model.pt"


def resolve_ckpt(run_dir, name):
    """Most advanced checkpoint available: final > best of the latest stage > latest epoch."""
    d = Path(run_dir) / name
    if (d / "model.pt").exists():
        return d / "model.pt"
    stages = sorted(d.glob("stage*_best.pt"))
    if stages:
        return stages[-1]
    return d / "latest.pt" if (d / "latest.pt").exists() else None


def _atomic_save(obj, path):
    """Write to a temp file and rename, so concurrent readers never see a partial checkpoint."""
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def train_curriculum_ckpt(model, train_loader, val_loader, out_dir, cfg, label="model", stages=None, lr=3e-4):
    """Scheduled-sampling curriculum. Every epoch writes latest.pt + train_history.csv, and each stage's
    best weights go to stage<k>_best.pt."""
    stages = stages or FAST_STAGES
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dev = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    hist = []
    for si, st in enumerate(stages):
        best, pat, best_w = float("inf"), 0, copy.deepcopy(model.state_dict())
        log.info("[%s] stage %d/%d tf=%s k=%s", label, si + 1, len(stages), st["tf"], st["rollout_k"])
        for ep in range(st["epochs"]):
            model.train(); tr = 0.0
            for b in train_loader:
                opt.zero_grad(set_to_none=True)
                loss = model(b["s"].to(dev), b["actions"].to(dev), tf_ratio=st["tf"], rollout_k=st["rollout_k"])["loss"]
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                tr += loss.item()
            model.eval(); vl = 0.0
            with torch.no_grad():
                for b in val_loader:
                    vl += model(b["s"].to(dev), b["actions"].to(dev), tf_ratio=0.0, rollout_k=st["rollout_k"])["loss"].item()
            tr /= max(1, len(train_loader)); vl /= max(1, len(val_loader))
            hist.append(dict(stage=si, epoch=ep, train_loss=tr, val_loss=vl))
            pd.DataFrame(hist).to_csv(out_dir / "train_history.csv", index=False)
            _atomic_save({"state_dict": model.state_dict(), "config": cfg}, out_dir / "latest.pt")
            log.info("[%s] stage %d ep %d  train=%.4f  val=%.4f", label, si + 1, ep + 1, tr, vl)
            if vl < best - 1e-5:
                best, pat, best_w = vl, 0, copy.deepcopy(model.state_dict())
                _atomic_save({"state_dict": best_w, "config": cfg}, out_dir / f"stage{si + 1}_best.pt")
            else:
                pat += 1
                if pat >= st["patience"]:
                    log.info("[%s] early stop", label); break
        model.load_state_dict(best_w)
    return pd.DataFrame(hist)


def train_one(name, splits, run_dir, device=None, stages=None, batch_size=64, max_train_windows=None,
              max_val_windows=20000):
    spec = MODEL_SPECS[name]
    device = device or get_device()
    seed = spec.get("seed", SEED)
    torch.manual_seed(seed)
    tr, va = splits["train"], splits["val"]
    n_actions = tr.act[spec["action_key"]].shape[1]
    n_cont = tr.state.shape[1] // 4
    cfg = dict(arch=spec.get("arch", "velocity"), state_dim=tr.state_dim, n_actions=n_actions, n_continuous=n_cont,
               action_conditioned=spec["action_conditioned"])
    if "forcing_scale" in spec:
        cfg["forcing_scale"] = spec["forcing_scale"]
    model = build_model(**cfg).to(device)
    tl = make_loader(tr, spec["action_key"], batch_size, shuffle=True, max_windows=max_train_windows, seed=seed)
    vl = make_loader(va, spec["action_key"], batch_size, shuffle=False, max_windows=max_val_windows, seed=SEED)
    log.info("[%s] params=%d  train windows=%d  val windows=%d", name,
             sum(p.numel() for p in model.parameters()), len(tl.dataset), len(vl.dataset))
    t0 = time.time()
    hist = train_curriculum_ckpt(model, tl, vl, Path(run_dir) / name, cfg, label=name, stages=stages)
    save_checkpoint(model, ckpt_path(run_dir, name), **cfg)
    log.info("[%s] done in %.1f min -> %s", name, (time.time() - t0) / 60, ckpt_path(run_dir, name))
    return model, hist


def load_or_train(name, splits, run_dir, device=None, **kw):
    """Load the most advanced checkpoint in runs/<name>/ (final, stage-best or latest); train only if none."""
    p = resolve_ckpt(run_dir, name)
    if p is not None:
        log.info("[%s] loading weights from %s", name, p)
        return load_checkpoint(p, device or get_device())
    log.info("[%s] no checkpoint in %s - training", name, Path(run_dir) / name)
    model, _ = train_one(name, splits, run_dir, device, **kw)
    model.eval()
    return model


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-dir", default=str(ROOT / "data/processed/mimic_wm"))
    ap.add_argument("--run-dir", default=str(ROOT / "runs"))
    ap.add_argument("--models", nargs="+", default=["wm5", "wm5plus", "wm78", "wm0"])
    ap.add_argument("--max-train-windows", type=int, default=None)
    ap.add_argument("--stages", default="fast", choices=list(STAGE_SETS))
    a = ap.parse_args()
    sp = {s: Split(a.proc_dir, s) for s in ["train", "val"]}
    for m in a.models:
        if ckpt_path(a.run_dir, m).exists():
            log.info("[%s] checkpoint exists, skipping", m)
            continue
        train_one(m, sp, a.run_dir, stages=STAGE_SETS[a.stages], max_train_windows=a.max_train_windows)
