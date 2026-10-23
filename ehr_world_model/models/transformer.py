"""
Transformer World Model (v7).

Encoder   : Linear(state_dim→D) + learnable pos-emb +
            Pre-LN TransformerEncoder (bidirectional, n_enc_layers)
Predictor : concat(z_t, action_t) → Linear(→D) +
            Pre-LN TransformerEncoder with causal mask (n_pred_layers)
            pred[:, t, :] forecasts z[:, t+1, :]

Losses (same as the velocity model)
    MSE    : pred[:, :T-h] vs tgt[:, h:]  summed over h = 1..rollout_k
    Path   : Δpred vs Δtgt (velocity consistency)
    SIGReg : Epps-Pulley on encoder + predictor outputs
    Recon  : linear head z → continuous feature values

Size variants (see TRANSFORMER_SIZES):
    S : latent=64,  heads=2, enc=2, pred=1, ff=128   ~0.6M params
    M : latent=128, heads=4, enc=4, pred=2, ff=256   ~2.4M params
    L : latent=256, heads=8, enc=6, pred=3, ff=512   ~9.5M params

forward() output dict, ``model.encoder`` and ``model.rollout`` match
VelocityWorldModel, so training / inference / rollout code is shared.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..sigreg import sigreg_loss_fourier

TRANSFORMER_SIZES = {
    "S": dict(latent_dim=64,  n_heads=2, n_enc_layers=2, n_pred_layers=1, ff_dim=128),
    "M": dict(latent_dim=128, n_heads=4, n_enc_layers=4, n_pred_layers=2, ff_dim=256),
    "L": dict(latent_dim=256, n_heads=8, n_enc_layers=6, n_pred_layers=3, ff_dim=512),
}


class TransformerWorldModel(nn.Module):
    def __init__(self, state_dim, n_actions, latent_dim=128,
                 n_heads=4, n_enc_layers=4, n_pred_layers=2, ff_dim=256,
                 action_conditioned=True,
                 lambda_mse=1.0, lambda_path=0.5, lambda_sigreg=0.05,
                 n_continuous=0, lambda_recon=0.20,
                 dropout=0.1, max_seq_len=512):
        super().__init__()
        self.latent_dim         = D = latent_dim
        self.n_actions          = n_actions
        self.action_conditioned = action_conditioned
        self.lambda_mse         = lambda_mse
        self.lambda_path        = lambda_path
        self.lambda_sigreg      = lambda_sigreg
        self.n_continuous       = n_continuous
        self.lambda_recon       = lambda_recon
        self.max_seq_len        = max_seq_len

        # ── Encoder (bidirectional) ───────────────────────────────────────────
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, D), nn.LayerNorm(D)
        )
        self.pos_emb = nn.Embedding(max_seq_len, D)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self._enc_tf = nn.TransformerEncoder(
            enc_layer, num_layers=n_enc_layers, norm=nn.LayerNorm(D)
        )

        # ── Predictor (causal) ────────────────────────────────────────────────
        pred_in = D + (n_actions if action_conditioned else 0)
        self.pred_proj = nn.Sequential(
            nn.Linear(pred_in, D), nn.LayerNorm(D)
        )
        pred_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self._pred_tf = nn.TransformerEncoder(
            pred_layer, num_layers=n_pred_layers, norm=nn.LayerNorm(D)
        )

        # ── Reconstruction head ────────────────────────────────────────────────
        if n_continuous > 0:
            self.recon_head = nn.Linear(D, n_continuous)
        else:
            self.recon_head = None

        # ── Public encoder wrapper (matches v6 model.encoder API) ─────────────
        _m = self
        class _Enc(nn.Module):
            def forward(self, s):
                if s.dim() == 2:            # (T, state_dim) → (T, D)
                    return _m._encode(s.unsqueeze(0)).squeeze(0)
                return _m._encode(s)        # (B, T, state_dim) → (B, T, D)
        self.encoder = _Enc()

        self._init_weights()

    # ─────────────────────────────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _encode(self, s):
        """Bidirectional context encoder.  (B, T, state_dim) → (B, T, D)"""
        B, T, _ = s.shape
        pos = torch.arange(T, device=s.device).clamp(max=self.max_seq_len - 1)
        x = self.state_proj(s) + self.pos_emb(pos).unsqueeze(0)
        return self._enc_tf(x)

    def _predict(self, z, actions, tf_ratio=1.0):
        """
        Causal predictor.  pred[:, t] ≈ z[:, t+1].

        Scheduled sampling (tf_ratio < 1): randomly replace (1-tf_ratio) of
        input tokens with Gaussian noise to simulate free-running at train time.
        """
        B, T, D = z.shape
        if self.training and tf_ratio < 1.0:
            keep  = torch.bernoulli(torch.full((B, T, 1), tf_ratio, device=z.device))
            sigma = z.detach().std().clamp(min=0.05)
            z     = z * keep + torch.randn_like(z) * sigma * (1.0 - keep)

        x = (torch.cat([z, actions], dim=-1) if self.action_conditioned else z)
        x = self.pred_proj(x)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=z.device)
        return self._pred_tf(x, mask=mask, is_causal=True)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, s, actions, tf_ratio=1.0, rollout_k=None):
        B, T, _ = s.shape
        D       = self.latent_dim
        mse_fn  = nn.MSELoss(reduction="none")

        if T < 2:
            z = s.sum() * 0.0
            dummy = torch.zeros(B, T, D, device=s.device)
            return dict(loss=z, pred_loss=z, sigreg_loss=z,
                        sigreg_pred_loss=z, recon_loss=z,
                        emb=dummy, next_emb=dummy)

        # ── Encode (bidirectional — full sequence context) ────────────────────
        emb = self._encode(s)      # (B, T, D), gradients for SIGReg + recon
        tgt = emb.detach()

        # ── Causal prediction ─────────────────────────────────────────────────
        pred = self._predict(tgt, actions, tf_ratio)   # (B, T, D)

        # ── Multi-step MSE + path loss ────────────────────────────────────────
        horizon   = max(1, min(rollout_k or 1, T - 1))
        tot_mse = tot_path = 0.0
        n_terms = 0
        for h in range(1, horizon + 1):
            if T <= h: break
            mse_h = mse_fn(pred[:, :T - h], tgt[:, h:]).mean()
            tot_mse = tot_mse + mse_h
            if T > h + 1:
                dp = pred[:, 1:T - h]   - pred[:, :T - h - 1]
                dt = tgt[:,  h + 1:T]   - tgt[:,  h:T - 1]
                if dp.shape == dt.shape:
                    tot_path = tot_path + mse_fn(dp, dt).mean()
            n_terms += 1

        loss_mse  = tot_mse  / max(1, n_terms)
        loss_path = tot_path / max(1, n_terms)

        # ── SIGReg ────────────────────────────────────────────────────────────
        loss_sig = sigreg_loss_fourier(emb.reshape(B * T, D))
        pf = pred.detach().reshape(B * T, D)
        if pf.shape[0] > 1:
            loss_sig = loss_sig + sigreg_loss_fourier(pf)

        # ── Reconstruction ────────────────────────────────────────────────────
        recon_loss = torch.tensor(0.0, device=s.device)
        if self.recon_head is not None and self.n_continuous > 0:
            nc    = self.n_continuous
            s_hat = self.recon_head(emb)
            obs_m = s[:, :, nc:2 * nc]
            denom = obs_m.sum().clamp(min=1)
            recon_loss = (((s_hat - s[:, :, :nc]) ** 2) * obs_m).sum() / denom

        total = (self.lambda_mse    * loss_mse
               + self.lambda_path   * loss_path
               + self.lambda_sigreg * loss_sig
               + self.lambda_recon  * recon_loss)

        # next_emb[:, t] = model's prediction for z[:, t]
        #   = pred[:, t-1] for t ≥ 1,  tgt[:, 0] for t = 0
        next_emb = torch.cat([tgt[:, :1], pred[:, :-1]], dim=1)

        _d = lambda v: v.detach() if torch.is_tensor(v) else torch.tensor(float(v), device=s.device)
        return dict(loss=total,
                    pred_loss=_d(loss_mse),
                    sigreg_loss=loss_sig.detach(),
                    sigreg_pred_loss=_d(loss_path),
                    recon_loss=recon_loss.detach(),
                    emb=emb.detach(),
                    next_emb=next_emb.detach())

    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def rollout(self, s0, actions, n_steps):
        """
        Autoregressive rollout from a single initial state.
        s0      : (B, state_dim)
        actions : (B, n_steps, n_actions)
        Returns : (B, n_steps, D)
        """
        self.eval()
        B  = s0.shape[0]
        z  = self._encode(s0.unsqueeze(1))          # (B, 1, D)
        na = self.n_actions if self.action_conditioned else 0
        out = []
        for t in range(n_steps):
            T_ctx = z.shape[1]
            a_ctx = torch.zeros(B, T_ctx, max(na, 1), device=s0.device)
            if self.action_conditioned and t < actions.shape[1]:
                a_ctx[:, -1, :] = actions[:, t, :]
            z_next = self._predict(z, a_ctx, tf_ratio=1.0)[:, -1:, :]  # (B, 1, D)
            out.append(z_next)
            z = torch.cat([z, z_next], dim=1)
        return torch.cat(out, dim=1)   # (B, n_steps, D)
