"""
Survival and gradient-boosting baselines for the inpatient tasks (and XGBoost on the MIMIC all-task benchmark).

Same cohort, labels, prediction time (48 h) and input as experiments.inpatient:
    x = [mean of the 171-dim state over the first 8 six-hour bins ; state at bin 7]
Baselines (each reports AUROC/AUPRC for the binary label and Harrell's C-index for time-to-event)
    CoxPH     L2-penalised linear Cox model (Breslow)                      [risk = linear predictor]
    DeepSurv  MLP Cox model (Katzman et al. 2018), early stopping on val     [risk = network output]
    DeepHit   discrete-time neural survival with ranking loss (Lee et al. 2018), 20 time bins
              [risk = cumulative incidence at the task horizon]
    XGBoost   -> experiments.xgb_baselines (separate process: xgboost and torch ship conflicting OpenMP
              runtimes on macOS). This module caches the exact feature matrices it needs.
For CoxPH/DeepSurv/DeepHit the AUROC uses the survival risk score against the binary label.

    python -m experiments.baselines_surv --site mimic_meds      # also: eicu, nwicu
    python -m experiments.baselines_surv --site mimic_meds --bench   # cache MIMIC benchmark features for XGBoost
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import evaluate as ev
from .data import Split
from .inpatient import OUT, SITES, TASKS, _boot, fit_cox, representations, task_table

log = logging.getLogger(__name__)
SEED = 0


def _cindex_ci(t, risk, e, n_boot=100):
    c = concordance_index(t, -risk, e)
    lo, hi = _boot(lambda b: concordance_index(t[b], -risk[b], e[b]), len(t), n_boot, SEED)
    return c, lo, hi


def _auc_ci(y, s, n_boot=200):
    lo, hi = _boot(lambda b: roc_auc_score(y[b], s[b]) if 0 < y[b].sum() < len(b) else np.nan, len(y), n_boot, SEED)
    return roc_auc_score(y, s), lo, hi


# ── DeepSurv ────────────────────────────────────────────────────────────────────

def _mlp(d_in, d_out, hid=256, p=0.2):
    return nn.Sequential(nn.Linear(d_in, hid), nn.ReLU(), nn.Dropout(p), nn.Linear(hid, hid // 2), nn.ReLU(),
                         nn.Dropout(p), nn.Linear(hid // 2, d_out))


def _cox_nll(risk, t, e):
    order = torch.argsort(t, descending=True)
    r, ev_ = risk[order], e[order]
    return -((r - torch.logcumsumexp(r, 0)) * ev_).sum() / ev_.sum().clamp(min=1)


def deepsurv(Xtr, ttr, etr, Xva, tva, eva, Xte, epochs=40, bs=1024, lr=1e-3):
    torch.manual_seed(SEED)
    net = _mlp(Xtr.shape[1], 1)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    T = lambda a: torch.tensor(a, dtype=torch.float32)
    Xt, tt, et = T(Xtr), T(ttr), T(etr)
    Xv, tv, evv = T(Xva), T(tva), T(eva)
    best, best_w, pat = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        for b in torch.randperm(len(Xt)).split(bs):
            loss = _cox_nll(net(Xt[b]).squeeze(-1), tt[b], et[b])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = _cox_nll(net(Xv).squeeze(-1), tv, evv).item()
        if vl < best - 1e-4:
            best, best_w, pat = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 5:
                break
    net.load_state_dict(best_w); net.eval()
    with torch.no_grad():
        return net(T(Xte)).squeeze(-1).numpy()


# ── DeepHit (single risk) ───────────────────────────────────────────────────────

def deephit(Xtr, ttr, etr, Xva, tva, eva, Xte, horizon, n_bins=20, epochs=40, bs=1024, lr=1e-3, alpha=0.2, sigma=0.1):
    torch.manual_seed(SEED)
    ev_times = ttr[etr > 0]
    cuts = np.unique(np.quantile(ev_times if len(ev_times) > 20 else ttr, np.linspace(0, 1, n_bins + 1)[1:-1]))
    K = len(cuts) + 1
    bin_of = lambda t: np.searchsorted(cuts, t, side="left")
    net = _mlp(Xtr.shape[1], K + 1)          # last unit = "beyond the last bin"
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt)
    Xt, kt, et = T(Xtr), T(bin_of(ttr), torch.long), T(etr)
    Xv, kv, evv = T(Xva), T(bin_of(tva), torch.long), T(eva)

    def loss_fn(logits, k, e):
        pmf = torch.softmax(logits, -1)
        cif = torch.cumsum(pmf, -1)
        idx = torch.arange(len(k))
        nll_ev = -torch.log(pmf[idx, k].clamp(min=1e-8)) * e
        nll_cen = -torch.log((1 - cif[idx, k]).clamp(min=1e-8)) * (1 - e)
        nll = (nll_ev + nll_cen).mean()
        # ranking loss: pairs (i event at k_i, j with k_j > k_i) should have CIF_i(k_i) > CIF_j(k_i)
        ei = e > 0
        if ei.sum() > 1:
            Fi = cif[ei, k[ei]]                          # (n_e,)
            Fj = cif[:, k[ei]].T                         # (n_e, n)
            mask = (k[None, :] > k[ei][:, None]).float()
            rank = (mask * torch.exp(-(Fi[:, None] - Fj) / sigma)).sum() / mask.sum().clamp(min=1)
        else:
            rank = torch.zeros(())
        return nll + alpha * rank

    best, best_w, pat = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        for b in torch.randperm(len(Xt)).split(bs):
            loss = loss_fn(net(Xt[b]), kt[b], et[b])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = loss_fn(net(Xv[:5000]), kv[:5000], evv[:5000]).item()
        if vl < best - 1e-4:
            best, best_w, pat = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 5:
                break
    net.load_state_dict(best_w); net.eval()
    with torch.no_grad():
        cif = torch.cumsum(torch.softmax(net(T(Xte)), -1), -1).numpy()
    return cif[:, min(bin_of(np.array([horizon]))[0], K - 1)]


HORIZON = {"30-day mortality": 30.0, "LOS": 7.0, "30-day readmission": 30.0, "Sepsis": 7.0}


def run_site(site, max_train=40000):
    cfg = SITES[site]
    sp = {s: Split(cfg["proc"], s) for s in ["train", "val", "test"]}
    T = {s: task_table(site, s, sp[s]) for s in sp}
    idx = {s: np.flatnonzero(T[s].eligible.values) for s in sp}
    T = {s: T[s].iloc[idx[s]].reset_index(drop=True) for s in sp}
    F = {s: representations(None, sp[s], sp[s].adm.index.values[idx[s]], "act5")["raw"] for s in sp}
    tasks = [t for t in TASKS if f"y_{t}" in T["train"]]
    rows = {b: [] for b in ["coxph", "deepsurv", "deephit"]}
    rng = np.random.default_rng(SEED)
    for task in tasks:
        sel = {}
        for s in sp:
            ok = np.flatnonzero(np.isfinite(T[s][f"y_{task}"].values))
            if s == "train" and len(ok) > max_train:
                ok = np.sort(rng.choice(ok, max_train, replace=False))
            sel[s] = ok
        if T["train"][f"y_{task}"].values[sel["train"]].sum() < 10:
            continue
        sc = StandardScaler().fit(F["train"][sel["train"]])
        X = {s: sc.transform(F[s][sel[s]]).astype(np.float32) for s in sp}
        y = {s: T[s][f"y_{task}"].values[sel[s]] for s in sp}
        t = {s: T[s][f"t_{task}"].values[sel[s]] for s in sp}
        e = {s: T[s][f"e_{task}"].values[sel[s]] for s in sp}
        base = dict(site=cfg["label"], task=task, traj="raw", n_test=len(y["test"]), events_test=int(y["test"].sum()))
        np.savez_compressed(OUT / f"cache_{site}_{task.replace(' ', '_')}.npz", label=cfg["label"], task=task,
                            **{f"{k}_{s}": v[s] for k, v in [("X", X), ("y", y), ("t", t), ("e", e)] for s in sp})

        def add(name, s_cls, risk):
            if task == "LOS" and name != "xgboost":
                s_cls = -s_cls      # LOS "event" = discharge: high hazard = short stay, label = long stay
            auc, lo, hi = _auc_ci(y["test"], s_cls)
            c, clo, chi = _cindex_ci(t["test"], risk, e["test"])
            rows[name].append(dict(model=name, **base, AUROC=auc, AUROC_lo=lo, AUROC_hi=hi,
                                   AUPRC=average_precision_score(y["test"], s_cls), C_index=c, C_lo=clo, C_hi=chi))
            log.info("[%s] %-9s %-19s AUROC %.3f  C %.3f", site, name, task, auc, c)

        w = fit_cox(X["train"], t["train"], e["train"])
        r = X["test"] @ w
        add("coxph", r, r)
        r = deepsurv(X["train"], t["train"], e["train"], X["val"], t["val"], e["val"], X["test"])
        add("deepsurv", r, r)
        r = deephit(X["train"], t["train"], e["train"], X["val"], t["val"], e["val"], X["test"], HORIZON[task])
        add("deephit", r, r)
    for name, rr in rows.items():
        pd.DataFrame(rr).to_csv(OUT / f"res_{site}_{name}.csv", index=False)


def cache_mimic_bench():
    """Patient-mean raw state (steps 2..23, as the raw LR baseline) + labels for XGBoost on the all-task benchmark."""
    sp = {s: Split(SITES["mimic_meds"]["proc"], s) for s in ["train", "val", "test"]}
    out = {}
    for s in sp:
        split = sp[s]
        idx = np.where(split.adm.n_bins.values >= ev.CONTEXT + 1)[0]
        X, _, L = ev.padded(split, idx, ev.MAX_STEPS, "act5")
        valid = (np.arange(X.shape[1])[None] >= ev.CONTEXT) & (np.arange(X.shape[1])[None] < L[:, None])
        pid = np.repeat(split.adm.patient_id.values[idx][:, None], X.shape[1], 1)[valid]
        F = pd.DataFrame(X[valid]).groupby(pid).mean()
        lm = split.patient_labels.set_index("patient_id")
        out[f"X_{s}"] = F.values.astype(np.float32)
        for task, col in ev.TASKS.items():
            out[f"y_{s}_{task.replace(' ', '_')}"] = F.index.map(lm[col]).values.astype(float)
    np.savez_compressed(OUT / "cache_mimic_bench.npz", tasks=np.array(list(ev.TASKS)), **out)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=list(SITES))
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    if a.bench:
        cache_mimic_bench()
    else:
        run_site(a.site)
