"""
Fourier SIGReg — Epps-Pulley characteristic-function regulariser.

Computes  ∫ ‖φ_N(t) − φ_Gauss(t)‖² dt  over t ∈ [0.2, 4.0]
using M random 1-D projections of the latent codes. Pushes the embedding
distribution towards an isotropic Gaussian to prevent representation collapse.
"""
from __future__ import annotations

import torch


def epps_pulley_test(h: torch.Tensor, t_nodes: torch.Tensor) -> torch.Tensor:
    h_ext = h.unsqueeze(-1)
    t_ext = t_nodes.view(1, 1, -1)
    ht    = h_ext * t_ext
    phi_N_real = torch.mean(torch.cos(ht), dim=0)
    phi_N_imag = torch.mean(torch.sin(ht), dim=0)
    phi_0      = torch.exp(-0.5 * t_nodes ** 2).view(1, -1)
    diff_sq    = (phi_N_real - phi_0) ** 2 + phi_N_imag ** 2
    return torch.mean(torch.trapz(diff_sq, t_nodes, dim=-1))


def sigreg_loss_fourier(Z: torch.Tensor, M: int = 256, num_nodes: int = 17, max_samples: int = 1024) -> torch.Tensor:
    """Z: (N, D) latent codes."""
    B, D = Z.shape
    if B < 2:
        return torch.tensor(0.0, device=Z.device)
    if B > max_samples:
        Z = Z[torch.randperm(B, device=Z.device)[:max_samples]]
    U = torch.randn(D, M, device=Z.device)
    U = U / torch.norm(U, dim=0, keepdim=True)
    H = Z @ U
    t_nodes = torch.linspace(0.2, 4.0, steps=num_nodes, device=Z.device)
    return epps_pulley_test(H, t_nodes)
