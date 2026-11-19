"""
Neural forecasting baselines and the masked-MSE MLP decoder used by forecast_h.

    GRUD       GRU-D (input + hidden decay towards the mean) over the context, MLP head
    LatentODE  ODE-RNN encoder -> latent ODE (Euler) -> linear decoder per horizon
    Transf     transformer over context tokens + learned horizon queries
    _fit_mlp   masked-MSE MLP decoder with early stopping (EHR-WM decoder and Obs-MLP)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

CTX, H = 2, 8          # context bins, forecast horizons (6-h steps)


class GRUD(nn.Module):
    """GRU-D (Che et al. 2018): decays inputs toward the empirical mean and the hidden state with time gaps."""

    def __init__(self, nc, d_static, hid=128, out=H):
        super().__init__()
        self.nc, self.hid, self.out = nc, hid, out
        self.gx = nn.Linear(nc, nc)
        self.gh = nn.Linear(nc, hid)
        self.cell = nn.GRUCell(3 * nc + d_static, hid)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, out * nc))

    def forward(self, ctx, static):
        B, T, _ = ctx.shape
        nc = self.nc
        h = ctx.new_zeros(B, self.hid)
        x_last = ctx.new_zeros(B, nc)
        for t in range(T):
            v, m, ts = ctx[:, t, :nc], ctx[:, t, nc:2 * nc], ctx[:, t, 2 * nc:3 * nc]
            delta = ts * 168.0 / 6.0                      # gap in bins
            gx = torch.exp(-torch.relu(self.gx(delta)))
            gh = torch.exp(-torch.relu(self.gh(delta)))
            x_hat = m * v + (1 - m) * gx * x_last         # decay toward the (zero) mean
            x_last = m * v + (1 - m) * x_last
            h = self.cell(torch.cat([x_hat, m, delta, static], -1), gh * h)
        return self.head(h).view(B, self.out, nc)


class LatentODE(nn.Module):
    """ODE-RNN encoder -> z0; dz/dt = f(z) integrated with Euler (4 sub-steps per 6-h bin)."""

    def __init__(self, d_in, nc, latent=64, hid=128, out=H):
        super().__init__()
        self.enc = nn.GRU(d_in, hid, batch_first=True)
        self.to_z = nn.Linear(hid, latent)
        self.f = nn.Sequential(nn.Linear(latent, hid), nn.Tanh(), nn.Linear(hid, latent))
        self.dec = nn.Linear(latent, nc)
        self.out = out

    def forward(self, ctx, static):
        _, h = self.enc(torch.cat([ctx, static[:, None].expand(-1, ctx.shape[1], -1)], -1))
        z = self.to_z(h[-1])
        outs = []
        for _ in range(self.out):
            for _ in range(4):
                z = z + 0.25 * self.f(z)
            outs.append(self.dec(z))
        return torch.stack(outs, 1)


class Transf(nn.Module):
    def __init__(self, d_in, nc, d=128, out=H):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.randn(CTX + out, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 256, dropout=0.1, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, 2)
        self.q = nn.Parameter(torch.randn(out, d) * 0.02)
        self.head = nn.Linear(d, nc)
        self.out = out

    def forward(self, ctx, static):
        B = ctx.shape[0]
        tok = self.inp(torch.cat([ctx, static[:, None].expand(-1, ctx.shape[1], -1)], -1))
        seq = torch.cat([tok, self.q[None].expand(B, -1, -1)], 1) + self.pos[None]
        return self.head(self.tr(seq)[:, CTX:])


class _Dec(nn.Module):
    def __init__(self, d_in, nc):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 256), nn.GELU(),
                                 nn.Linear(256, nc))

    def forward(self, x):
        return self.net(x)


def _fit_mlp(F, Y, M, n_val, epochs=40, seed=0):
    """Masked-MSE MLP decoder (all variables jointly); early stopping on a held-out slice of TRAIN admissions."""
    torch.manual_seed(seed)
    N, Hh, D = F.shape
    nc = Y.shape[-1]
    Ff, Yf, Mf = F.reshape(-1, D), Y.reshape(-1, nc), M.reshape(-1, nc).astype(np.float32)
    nv = n_val * Hh
    mu, sd = Ff[nv:].mean(0), Ff[nv:].std(0) + 1e-6
    T = lambda a: torch.tensor(a, dtype=torch.float32)
    f, y, m = T((Ff[nv:] - mu) / sd), T(Yf[nv:]), T(Mf[nv:])
    fv, yv, mv = T((Ff[:nv] - mu) / sd), T(Yf[:nv]), T(Mf[:nv])
    net = _Dec(D, nc)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    best, bw, pat = np.inf, None, 0
    for _ in range(epochs):
        net.train()
        for b in torch.randperm(len(f)).split(1024):
            loss = (((net(f[b]) - y[b]) ** 2) * m[b]).sum() / m[b].sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = (((net(fv) - yv) ** 2) * mv).sum().item() / max(mv.sum().item(), 1)
        if vl < best - 1e-5:
            best, bw, pat = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 5:
                break
    net.load_state_dict(bw); net.eval()

    def predict(Fx):
        with torch.no_grad():
            return net(T((Fx.reshape(-1, D) - mu) / sd)).numpy().reshape(Fx.shape[0], Fx.shape[1], nc)
    return predict
