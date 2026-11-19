"""
Forecasting in the clinical feature space at fixed horizons.

Every horizon is scored on the same target bins, so the columns differ only in how far ahead the forecast is
(not in position within the stay: early bins are much harder, e.g. 37% of bin-2 targets have any prior
measurement vs 95% at bin 9).

    python -m experiments.forecast_h --site mimic        # also: eicu, nwicu

Protocol (identical for every method)
    forecast origin o : any bin; context = the 2 bins (o-1, o) plus everything summarised in their state channels
                        (time-since, decayed carry-forward) and, for LOCF-type inputs, the last value seen up to o
    horizon h = 1..8  : target bin t = o + h (6..48 h ahead)
    targets           : the SAME set of target bins t = 9..17 for every horizon (origin o = t - h), so the
                        columns differ only in how far ahead the forecast is
    training          : all methods trained on windows with random origins from TRAIN admissions
    windows           : 0-6h (h=1), 6-12h (h=2), 12-24h (h=3-4), 24-48h (h=5-8), all

Methods
    G-Mean, LOCF (last value up to the origin), Ridge, GRU-D, L-ODE, Transf.
    Obs-MLP            MLP decoder on [last observed values ; their age ; horizon] (no world model)
    EHR-WM (linear)    world-model free rollout from the origin, per-variable ridge probe
    EHR-WM             world-model free rollout from the origin + MLP decoder on [predicted state ; last observed
                       values ; age ; horizon]
    EHR-WM (oracle)    ridge probe on the TRUE encoder state at the target bin (reconstruction ceiling)
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from . import evaluate as ev
from .data import Split
from .features import LABEL
from .forecast_models import GRUD, LatentODE, Transf, _fit_mlp
from .sites import OUT, SITES
from .train import load_checkpoint, resolve_ckpt

log = logging.getLogger(__name__)
CTX, H = 2, 8
T0, T1 = CTX - 1 + H, 17              # common target bins 9..17
NT = T1 - T0 + 1
T_LOAD = T1 + 1 + H                   # actions needed up to o + H - 1
O_MAX = T1 - 1                        # origins 1..16
WINDOWS = {"0-6h": [1], "6-12h": [2], "12-24h": [3, 4], "24-48h": [5, 6, 7, 8]}
FOUT = OUT / "forecast_h"
RIDGE_ALPHAS = [(10.0, "Ridge"), (1000.0, "Ridge (alpha=1000)")]      # two regularisation strengths, both reported


def ridge_predictions(Ftr, Ytr, Mtr, Fte, gmean, alpha):
    """Per-(horizon, variable) ridge regression on the flattened 2-bin context + demographics."""
    nc = Ytr.shape[-1]
    R = np.zeros((len(Fte), H, nc), np.float32)
    for h in range(H):
        for f in range(nc):
            ok = Mtr[:, h, f] > 0
            R[:, h, f] = Ridge(alpha=alpha).fit(Ftr[ok], Ytr[ok, h, f]).predict(Fte) if ok.sum() > 50 else gmean[f]
    return R


def load(site, split, max_n, seed):
    sp = Split(SITES[site]["proc"], split)
    idx = np.flatnonzero(sp.adm.n_bins.values >= CTX + H)
    if len(idx) > max_n:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_n, replace=False))
    X, A, L = ev.padded(sp, idx, T_LOAD, "act5")
    return X, A, L, sp.state.shape[1] // 4


def locf_tables(X, nc):
    """V[:, b] = last observed value in bins <= b (0 if none); B[:, b] = bin of that observation (-99 if none)."""
    v, m = X[..., :nc], X[..., nc:2 * nc] > 0
    V = np.zeros_like(v); B = np.full(v.shape, -99, np.int16)
    cur = np.zeros_like(v[:, 0]); cb = np.full(cur.shape, -99, np.int16)
    for b in range(X.shape[1]):
        cur = np.where(m[:, b], v[:, b], cur); cb = np.where(m[:, b], b, cb)
        V[:, b], B[:, b] = cur, cb
    return V, B


def make_windows(X, L, origins_per_adm, rng=None):
    """Return (adm index, origin) pairs. origins_per_adm=None -> all origins 1..O_MAX."""
    pairs = []
    for i, n in enumerate(L):
        cand = np.arange(1, min(n - 2, O_MAX) + 1) if origins_per_adm else np.arange(1, O_MAX + 1)
        if origins_per_adm and len(cand) > origins_per_adm:
            cand = rng.choice(cand, origins_per_adm, replace=False)
        pairs += [(i, o) for o in cand]
    p = np.array(pairs)
    return p[:, 0], p[:, 1]


def window_arrays(X, ai, oi, nc):
    ctx = np.stack([X[ai, oi - 1, :4 * nc], X[ai, oi, :4 * nc]], 1)
    st = X[ai, 0, 4 * nc:]
    tg = oi[:, None] + np.arange(1, H + 1)[None]
    Y = X[ai[:, None], tg, :nc]; M = (X[ai[:, None], tg, nc:2 * nc] > 0).astype(np.float32)
    return ctx, st, Y, M


def obs_feats(V, B, ai, oi):
    """[last value up to origin ; age of it at target (bins / 20, 1.0 if never) ; horizon one-hot] per horizon."""
    lv = V[ai, oi]                                                      # (W, nc)
    lb = B[ai, oi].astype(np.float32)
    hz = np.arange(1, H + 1, dtype=np.float32)
    age = np.where(lb[:, None] < 0, 20.0, oi[:, None, None] + hz[None, :, None] - lb[:, None]) / 20.0
    age = np.minimum(age, 1.0).astype(np.float32)
    oh = np.broadcast_to(np.eye(H, dtype=np.float32)[None], (len(ai), H, H))
    return np.concatenate([np.repeat(lv[:, None], H, 1), age, oh], -1)


def train_nn(model, C, S, Y, M, epochs=8, bs=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    C, S, Y, M = map(torch.from_numpy, (C, S, Y, M))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()
    for _ in range(epochs):
        for b in torch.randperm(len(C)).split(bs):
            loss = (((model(C[b], S[b]) - Y[b]) ** 2) * M[b]).sum() / M[b].sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_nn(model, C, S):
    C, S = torch.from_numpy(C), torch.from_numpy(S)
    return torch.cat([model(C[s:s + 4096], S[s:s + 4096]) for s in range(0, len(C), 4096)]).numpy()


def wm_rollouts(model, Z, A, ai, oi, batch=20000):
    out = []
    for s in range(0, len(ai), batch):
        a, o = ai[s:s + batch], oi[s:s + batch]
        seed = np.stack([Z[a, o - 1], Z[a, o]], 1)
        act = A[a[:, None], o[:, None] + np.arange(H)[None]]           # a_o ... a_{o+H-1}
        out.append(ev.rollout(model, seed, act))
    return np.concatenate(out)


def gather(Pw, N):
    """Pw: (N*O_MAX, H, nc) predictions for all test origins 1..O_MAX (adm-major) ->
    (N, NT, H, nc) where [:, t-T0, h-1] is the forecast of target bin t made h steps ahead."""
    Pw = Pw.reshape(N, O_MAX, H, -1)
    G = np.empty((N, NT, H, Pw.shape[-1]), np.float32)
    for k, t in enumerate(range(T0, T1 + 1)):
        for h in range(1, H + 1):
            G[:, k, h - 1] = Pw[:, t - h - 1, h - 1]
    return G


def score(G, Yt, Mt, feats, site, method):
    rows = []
    for win, hs in list(WINDOWS.items()) + [("all", list(range(1, H + 1)))]:
        for f in range(len(feats)):
            mm = Mt[..., f]
            if mm.sum() < 20:
                continue
            y = np.concatenate([Yt[..., f][mm]] * len(hs))
            p = np.concatenate([G[:, :, h - 1, f][mm] for h in hs])
            rows.append(dict(site=site, model="wm5plus" if "EHR-WM" in method else "baseline", method=method,
                             window=win, feature=feats[f], n=int(mm.sum()) * len(hs),
                             R2=r2_score(y, p) if y.std() > 1e-8 else np.nan, MAE=float(np.abs(y - p).mean())))
    return rows


def run(site, max_train=15000, max_test=8000, per_adm=4, device=None, seed=0):
    t_start = time.time()
    cfg = SITES[site]
    FOUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((cfg["proc"] / "feature_manifest.json").read_text()) if (cfg["proc"] / "feature_manifest.json").exists() \
        else json.loads((cfg["proc"].parent / "xsite_manifest.json").read_text())
    feats = [LABEL.get(c, c) for c in man["vital_codes"] + man["lab_codes"]]
    rng = np.random.default_rng(seed)
    Xtr, Atr, Ltr, nc = load(site, "train", max_train, 0)
    Xte, Ate, Lte, _ = load(site, "test", max_test, 1)
    N = len(Xte)
    ai_tr, oi_tr = make_windows(Xtr, Ltr, per_adm, rng)
    ai_te, oi_te = make_windows(Xte, Lte, None)
    Ctr, Str, Ytr, Mtr = window_arrays(Xtr, ai_tr, oi_tr, nc)
    Cte, Ste, _, _ = window_arrays(Xte, ai_te, oi_te, nc)
    Yt = Xte[:, T0:T1 + 1, :nc]; Mt = Xte[:, T0:T1 + 1, nc:2 * nc] > 0
    log.info("[%s] train %d adm / %d windows; test %d adm, targets bins %d-%d (%d obs)", site, len(Xtr), len(ai_tr),
             N, T0, T1, Mt.sum())
    Vtr, Btr = locf_tables(Xtr, nc); Vte, Bte = locf_tables(Xte, nc)
    rows, keep = [], {}

    def score_g(name, G):
        rows.extend(score(G, Yt, Mt, feats, cfg["label"], name))
        d = pd.DataFrame([r for r in rows if r["method"] == name]).groupby("window").R2.median()
        log.info("  %-28s median R2 %s", name, " ".join(f"{w}={d[w]:.3f}" for w in list(WINDOWS) + ["all"]))

    def add(name, Pw):
        G = gather(Pw.astype(np.float32), N)
        rows.extend(score(G, Yt, Mt, feats, cfg["label"], name))
        if name in ("EHR-WM", "LOCF", "Obs-MLP"):
            keep[name] = G
        d = pd.DataFrame([r for r in rows if r["method"] == name]).groupby("window").R2.median()
        log.info("  %-16s median R2 %s  (%.1f min)", name, " ".join(f"{w}={d[w]:.3f}" for w in list(WINDOWS) + ["all"]),
                 (time.time() - t_start) / 60)

    # ── simple baselines
    v, m = Xtr[..., :nc], Xtr[..., nc:2 * nc] > 0
    gmean = np.array([v[..., f][m[..., f]].mean() for f in range(nc)], np.float32)
    add("G-Mean", np.broadcast_to(gmean, (len(ai_te), H, nc)))
    lv = Vte[ai_te, oi_te]; seen = Bte[ai_te, oi_te] >= 0
    add("LOCF", np.repeat(np.where(seen, lv, gmean)[:, None], H, 1))
    Ftr, Fte = np.concatenate([Ctr.reshape(len(Ctr), -1), Str], 1), np.concatenate([Cte.reshape(len(Cte), -1), Ste], 1)
    for alpha, rname in RIDGE_ALPHAS:
        add(rname, ridge_predictions(Ftr, Ytr, Mtr, Fte, gmean, alpha))
    d_static = Str.shape[1]
    for name, mdl in [("GRU-D", GRUD(nc, d_static)), ("L-ODE", LatentODE(4 * nc + d_static, nc)),
                      ("Transf.", Transf(4 * nc + d_static, nc))]:
        add(name, predict_nn(train_nn(mdl, Ctr, Str, Ytr, Mtr, seed=seed), Cte, Ste))
    n_val = len(ai_tr) // 8                       # MLP early-stopping slice (windows are shuffled by rng choice order)
    perm = rng.permutation(len(ai_tr))
    Otr, Ote = obs_feats(Vtr, Btr, ai_tr, oi_tr)[perm], obs_feats(Vte, Bte, ai_te, oi_te)
    Ytr_p, Mtr_p = Ytr[perm], Mtr[perm]
    add("Obs-MLP", _fit_mlp(Otr, Ytr_p, Mtr_p, n_val, seed=seed)(Ote))

    # ── world model
    model = load_checkpoint(resolve_ckpt(cfg["runs"], "wm5plus"), device)
    Ztr, Zte = ev.encode(model, Xtr), ev.encode(model, Xte)
    Rtr, Rte = wm_rollouts(model, Ztr, Atr, ai_tr, oi_tr), wm_rollouts(model, Zte, Ate, ai_te, oi_te)
    D = Rtr.shape[-1]
    P = np.zeros((len(Rte), H, nc), np.float32)
    for h in range(H):                                  # per-horizon probes (as for the Ridge baseline)
        for f in range(nc):
            ok = Mtr[:, h, f] > 0
            if ok.sum() >= 50:
                P[:, h, f] = Ridge(alpha=1.0).fit(Rtr[ok, h], Ytr[ok, h, f]).predict(Rte[:, h])
    add("EHR-WM (linear)", P)
    add("EHR-WM", _fit_mlp(np.concatenate([Rtr[perm], Otr], -1), Ytr_p, Mtr_p, n_val, seed=seed)(np.concatenate([Rte, Ote], -1)))
    # oracle: ridge probe on the true encoder state at the target bin (same for every horizon)
    tg_tr = oi_tr[:, None] + np.arange(1, H + 1)[None]
    Zt_tr = Ztr[ai_tr[:, None], tg_tr]
    Zt_te = Zte[ai_te[:, None], oi_te[:, None] + np.arange(1, H + 1)[None]]
    P = np.zeros((len(Zt_te), H, nc), np.float32)
    for f in range(nc):
        ok = Mtr[..., f] > 0
        if ok.sum() >= 50:
            P[..., f] = Ridge(alpha=1.0).fit(Zt_tr[ok], Ytr[..., f][ok]).predict(Zt_te.reshape(-1, D)).reshape(len(Zt_te), H)
    add("EHR-WM (oracle)", P)

    pd.DataFrame(rows).to_csv(FOUT / f"{site}.csv", index=False)
    # scatter-plot arrays in (row, horizon, var) layout: row = (admission, target bin)
    rep = lambda a: np.repeat(a.reshape(-1, 1, nc), H, 1)
    np.savez_compressed(FOUT / f"{site}_predictions.npz", y=rep(Yt).astype(np.float16), mask=rep(Mt),
                        feats=np.array(feats), **{k.replace(" ", "_"): G.reshape(-1, H, nc).astype(np.float16)
                                                  for k, G in keep.items()})
    log.info("[%s] done in %.1f min", site, (time.time() - t_start) / 60)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=list(SITES))
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    torch.set_num_threads(4)
    run(a.site, device=a.device)
