"""
Velocity-Matrix World Model (v6) — discrete Neural-CDE-style latent dynamics.

    z_t      = projector(s_t)                         (MLP encoder)
    F_t      = predictor([z_t ; a_t])  ∈ R^{D×D}       (spectral-normed MLP)
    dz_{t+1} = F_t · dz_t
    z_{t+1}  = clamp(z_t + dz_{t+1}, ±50)

Loss = λ_mse·MSE(z_pred, z_tgt) + λ_path·MSE(dz_pred, dz_tgt)
     + λ_sigreg·SIGReg(emb, preds) + λ_recon·masked-MSE(recon_head(emb), s_cont)

Targets are stop-gradient encoder outputs; SIGReg and the reconstruction head
train the projector. ``action_conditioned=False`` gives the no-action ablation
(Model B). ``_step`` is the single-step transition used by rollout and the
counterfactual / MPC experiments.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..sigreg import sigreg_loss_fourier


class VelocityWorldModel(nn.Module):
    def __init__(self, state_dim, n_actions, latent_dim=128,
                 action_conditioned=True,
                 lambda_mse=1.0, lambda_path=0.5, lambda_sigreg=0.05,
                 n_continuous=0, lambda_recon=0.20):
        super().__init__()
        self.state_dim          = state_dim
        self.n_actions          = n_actions
        self.latent_dim         = D = latent_dim
        self.action_conditioned = action_conditioned
        self.lambda_mse         = lambda_mse
        self.lambda_path        = lambda_path
        self.lambda_sigreg      = lambda_sigreg
        self.n_continuous       = n_continuous
        self.lambda_recon       = lambda_recon

        pred_in = D + (n_actions if action_conditioned else 0)

        # Use LayerNorm instead of BatchNorm: works correctly regardless of
        # batch size, sequence length, or eval/train mode transitions
        self.projector = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256),
            nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, D),
        )
        self.predictor = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(pred_in, 256)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(256, D * D)),
        )

        # Reconstruction head: forces encoder to retain feature-level information
        if n_continuous > 0:
            self.recon_head = nn.Linear(D, n_continuous)
            nn.init.xavier_uniform_(self.recon_head.weight, gain=0.5)
            nn.init.zeros_(self.recon_head.bias)
        else:
            self.recon_head = None

        proj = self.projector

        class _Enc(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = proj
            def forward(self, s):
                if s.dim() == 2: return self.proj(s)
                B, T, _ = s.shape
                return self.proj(s.reshape(B * T, -1)).reshape(B, T, -1)

        self.encoder = _Enc()

    def _step(self, z, dz_prev, action):
        D   = self.latent_dim
        inp = torch.cat([z, action], -1) if self.action_conditioned else z
        F_m = self.predictor(inp).view(-1, D, D)
        dz  = torch.bmm(F_m, dz_prev.unsqueeze(-1)).squeeze(-1)
        return torch.clamp(z + dz, -50.0, 50.0), dz

    def forward(self, s, actions, tf_ratio=1.0, rollout_k=None):
        B, T, _ = s.shape
        D = self.latent_dim
        mse_fn = nn.MSELoss(reduction='none')

        # ── Encoder: single forward pass for consistent normalisation ──────────
        # emb has gradients — used for SIGReg and recon loss directly
        emb = self.projector(s.reshape(B * T, -1)).reshape(B, T, D)
        # stop-gradient target for dynamics loss (prevents trivial collapse via
        # the prediction path, but SIGReg+recon still train the projector)
        tgt = emb.detach()

        horizon = max(1, min(rollout_k if rollout_k is not None else T - 2, T - 2))
        total_mse = total_path = 0.0
        jumps = 0
        pred_buf = []
        one_step = {}

        for t in range(1, T - 1):
            z_c = emb[:, t,     :].detach()   # anchor: no grad through dynamics loop
            z_p = emb[:, t - 1, :].detach()
            dz  = z_c - z_p

            for h in range(horizon):
                fi = t + h + 1
                if fi >= T: break
                a_t          = actions[:, fi - 1, :]
                z_next, dz_n = self._step(z_c, dz, a_t)

                z_tgt  = tgt[:, fi,     :]
                dz_tgt = tgt[:, fi, :] - tgt[:, fi - 1, :]

                total_mse  = total_mse  + mse_fn(z_next, z_tgt).mean(1).mean()
                total_path = total_path + mse_fn(dz_n, dz_tgt).mean(1).mean()
                jumps += 1
                pred_buf.append(z_next)

                if h == 0:
                    one_step[t] = z_next.detach()

                if torch.rand(1).item() < tf_ratio:
                    z_c = tgt[:, fi, :];  dz = z_c - tgt[:, fi - 1, :]
                else:
                    z_c = z_next.detach(); dz = dz_n.detach()

        if jumps == 0:
            z = s.sum() * 0.0
            return dict(loss=z, pred_loss=z, sigreg_loss=z,
                        sigreg_pred_loss=z, recon_loss=z, emb=emb.detach(), next_emb=emb.detach())

        loss_mse  = total_mse  / jumps
        loss_path = total_path / jumps

        # ── SIGReg directly on emb (gradients flow through projector) ──────────
        emb_flat = emb.reshape(B * T, D)
        loss_sig = sigreg_loss_fourier(emb_flat)
        if pred_buf:
            ap = torch.cat(pred_buf, 0)
            if len(ap) > 1:
                loss_sig = loss_sig + sigreg_loss_fourier(ap)

        # ── Reconstruction loss (forces encoder to retain feature values) ───────
        recon_loss = torch.tensor(0.0, device=s.device)
        if self.recon_head is not None and self.n_continuous > 0:
            nc     = self.n_continuous
            values = s[:, :, :nc]          # z-scored values
            obs_m  = s[:, :, nc:2 * nc]    # observation mask
            s_hat  = self.recon_head(emb)  # (B, T, nc) — gradients through emb
            denom  = obs_m.sum().clamp(min=1)
            recon_loss = (((s_hat - values) ** 2) * obs_m).sum() / denom

        total = (self.lambda_mse    * loss_mse
               + self.lambda_path   * loss_path
               + self.lambda_sigreg * loss_sig
               + self.lambda_recon  * recon_loss)

        # Build next_emb (B, T, D)
        next_emb_cols = [one_step[t - 1] if (t - 1) in one_step else tgt[:, t, :] for t in range(T)]
        next_emb = torch.stack(next_emb_cols, dim=1)

        return dict(loss=total,
                    pred_loss=loss_mse.detach(),
                    sigreg_loss=loss_sig.detach(),
                    sigreg_pred_loss=loss_path.detach(),
                    recon_loss=recon_loss.detach(),
                    emb=emb.detach(), next_emb=next_emb)

    @torch.no_grad()
    def rollout(self, s0, actions, n_steps):
        self.eval()
        B = s0.shape[0]
        z  = self.projector(s0)
        dz = torch.zeros_like(z)
        out = []
        for t in range(n_steps):
            a = (actions[:, t, :] if t < actions.shape[1]
                 else torch.zeros(B, self.n_actions, device=s0.device))
            z, dz = self._step(z, dz, a)
            out.append(z)
        return torch.stack(out, dim=1)
