"""
Model construction and checkpoint I/O.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .models import EHRWorldModel

_LOSS_KW = dict(lambda_mse=1.0, lambda_path=0.5, lambda_sigreg=0.05, lambda_recon=0.20)


def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(arch: str, state_dim: int, n_actions: int, n_continuous: int,
                action_conditioned: bool = True) -> nn.Module:
    """arch: 'velocity' (the additive-forcing variant is built by experiments.train)."""
    if arch == "velocity":
        return EHRWorldModel(state_dim=state_dim, n_actions=n_actions, latent_dim=128,
                                  action_conditioned=action_conditioned,
                                  n_continuous=n_continuous, **_LOSS_KW)
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
