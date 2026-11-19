"""
Encoding, free rollout and the downstream probe benchmark.

Downstream prediction
   For every test patient we build two latent trajectories per admission
   (first ``MAX_STEPS`` 6-h steps):
       ORACLE    z_t      = encoder(x_t)                       (reference: encoder of the real data)
       ROLLOUT   z~_t     free-running from the first 2 encoder states under the observed actions
   All world-model predictions are free rollouts; the model never sees the true
   state after the 2-step context.
   A logistic-regression probe is fit per timestep on TRAIN oracle states
   (each state carries its patient's label, weighted 1/#steps), applied to both
   trajectories, and the patient score is the mean probability over steps
   t >= 2. AUROC / AUPRC with 95% bootstrap CIs.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .data import Split

MAX_STEPS = 24          # 6 days per admission
CONTEXT = 2             # the world model needs 2 states to define dz
EMA_ALPHA = 0.25        # EMA smoothing of generated trajectories; None = off


def ema(gen: np.ndarray, anchor: np.ndarray, alpha=None) -> np.ndarray:
    """EMA over generated steps: s_h = a*z^_h + (1-a)*s_{h-1}, with s_{-1} = last true encoder state.

    gen: (N, H, D) generated states; anchor: (N, D). Applied to generated trajectories only, to damp
    rollout drift; encoder (oracle) trajectories are never smoothed."""
    alpha = EMA_ALPHA if alpha is None else alpha
    if not alpha or alpha >= 1:
        return gen
    out = np.empty_like(gen)
    s = anchor.astype(gen.dtype)
    for h in range(gen.shape[1]):
        s = alpha * gen[:, h] + (1 - alpha) * s
        out[:, h] = s
    return out

# MIMIC-IV benchmark tasks -> label column in patient_labels.parquet
TASKS = {
    "Acute MI":           "has_acute_mi",
    "Lupus":              "has_lupus",
    "Hyperlipidemia":     "has_hyperlipidemia",
    "Hypertension":       "has_hypertension",
    "Celiac":             "has_celiac",
    "Pancreatic Cancer":  "has_pancreatic_cancer",
    "ICU Admission":      "icu_admission",
    "Long LOS":           "los_long",
    "30-day Readmission": "readmit_30d",
    "Sepsis":             "has_Sepsis",
}


# ── Tensor helpers ─────────────────────────────────────────────────────────────

def _t(x):
    """Contiguous float32 tensor from a numpy array (strided slices crash MPS copies)."""
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))

def padded(split: Split, idxs: Sequence[int], T: int, action_key: str):
    """X (N,T,Dx) float32, A (N,T,A) float32, L (N,) valid lengths."""
    n_a = split.act[action_key].shape[1]
    X = np.zeros((len(idxs), T, split.state_dim), np.float32)
    A = np.zeros((len(idxs), T, n_a), np.float32)
    L = np.zeros(len(idxs), np.int64)
    for k, i in enumerate(idxs):
        x, a = split.timeline(int(i), action_key, max_bins=T)
        X[k, :len(x)], A[k, :len(a)], L[k] = x, a, len(x)
    return X, A, L


@torch.no_grad()
def encode(model, X: np.ndarray, batch: int = 2048) -> np.ndarray:
    dev = next(model.parameters()).device
    out = []
    for s in range(0, len(X), batch):
        out.append(model.encoder(_t(X[s:s + batch]).to(dev)).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def step(model, z, dz, a):
    """Batched single transition (tensors on the model device)."""
    if model.n_actions == 0 or a.shape[-1] != model.n_actions:
        a = torch.zeros(z.shape[0], model.n_actions, device=z.device)
    return model._step(z, dz, a)


@torch.no_grad()
def rollout(model, z_seed: np.ndarray, actions: np.ndarray, batch: int = 1024) -> np.ndarray:
    """Free-running rollout.

    z_seed : (N, 2, D) last two encoder states (z_{c-2}, z_{c-1})
    actions: (N, H, A) actions a_{c-1} ... a_{c+H-2}
    returns: (N, H, D) predicted z_c ... z_{c+H-1}
    """
    dev = next(model.parameters()).device
    out = []
    for s in range(0, len(z_seed), batch):
        zs = _t(z_seed[s:s + batch]).to(dev)
        ac = _t(actions[s:s + batch]).to(dev)
        z, dz = zs[:, 1], zs[:, 1] - zs[:, 0]
        traj = []
        for h in range(ac.shape[1]):
            z, dz = step(model, z, dz, ac[:, h])
            traj.append(z)
        out.append(torch.stack(traj, 1).cpu().numpy())
    return ema(np.concatenate(out), z_seed[:, 1])


@torch.no_grad()
def decode(model, Z: np.ndarray) -> np.ndarray:
    dev = next(model.parameters()).device
    sh = Z.shape
    r = model.recon_head(_t(Z.reshape(-1, sh[-1])).to(dev)).cpu().numpy()
    return r.reshape(*sh[:-1], -1)


# ── Trajectories for every admission ──────────────────────────────────────────

def admission_trajectories(model, split: Split, action_key: str, T: int = MAX_STEPS, max_n: int = None,
                           seed: int = 0) -> Dict:
    """ORACLE / ROLLOUT trajectories for admissions with >= 3 bins (optionally subsampled)."""
    idxs = np.where(split.adm.n_bins.values >= CONTEXT + 1)[0]
    if max_n and len(idxs) > max_n:
        idxs = np.sort(np.random.default_rng(seed).choice(idxs, max_n, replace=False))
    X, A, L = padded(split, idxs, T, action_key)
    Z = encode(model, X)
    Zr = np.concatenate([Z[:, :CONTEXT], rollout(model, Z[:, :CONTEXT], A[:, CONTEXT - 1:T - 1])], 1)
    return dict(idx=idxs, X=X, A=A, L=L, oracle=Z, rollout=Zr,
                pid=split.adm.patient_id.values[idxs])


def _flatten(tr, key, t_min=CONTEXT):
    """Stack valid steps (t >= t_min) -> rows, with owning patient id."""
    Z, L, pid = tr[key], tr["L"], tr["pid"]
    T = Z.shape[1]
    valid = (np.arange(T)[None] >= t_min) & (np.arange(T)[None] < L[:, None])
    return Z[valid], np.repeat(pid[:, None], T, 1)[valid]


def bootstrap(y, s, fn, n=200, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        b = rng.integers(0, len(y), len(y))
        if y[b].min() == y[b].max():
            continue
        vals.append(fn(y[b], s[b]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def probe_benchmark(tr_train: Dict, tr_test: Dict, lab_train: pd.DataFrame, lab_test: pd.DataFrame,
                    model_name: str, tasks: Dict[str, str] = TASKS, n_boot: int = 200,
                    max_rows: int = 150_000, seed: int = 0) -> pd.DataFrame:
    Xtr, ptr = _flatten(tr_train, "oracle")
    if len(Xtr) > max_rows:                      # subsample training states for speed
        keep = np.random.default_rng(seed).choice(len(Xtr), max_rows, replace=False)
        Xtr, ptr = Xtr[keep], ptr[keep]
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr)
    w_tr = 1.0 / pd.Series(ptr).map(pd.Series(ptr).value_counts()).values
    test_rows = {k: _flatten(tr_test, k) for k in ["oracle", "rollout"]}
    rows = []
    lm_tr = lab_train.set_index("patient_id")
    lm_te = lab_test.set_index("patient_id")
    for task, col in tasks.items():
        y_tr = pd.Series(ptr).map(lm_tr[col]).values.astype(float)
        ok = np.isfinite(y_tr)
        if y_tr[ok].sum() < 5:
            continue
        clf = LogisticRegression(C=0.1, max_iter=300)
        clf.fit(Xtr[ok], y_tr[ok], sample_weight=w_tr[ok])
        for kind, (Zte, pte) in test_rows.items():
            p = clf.predict_proba(sc.transform(Zte))[:, 1]
            score = pd.Series(p).groupby(pte).mean()
            y = score.index.map(lm_te[col]).values.astype(float)
            m = np.isfinite(y)
            y, s = y[m], score.values[m]
            if y.sum() < 2:
                continue
            auc, ap = roc_auc_score(y, s), average_precision_score(y, s)
            lo, hi = bootstrap(y, s, roc_auc_score, n_boot)
            rows.append(dict(model=model_name, task=task, traj=kind, prevalence=y.mean(), n=len(y),
                             AUROC=auc, AUROC_lo=lo, AUROC_hi=hi, AUPRC=ap))
    return pd.DataFrame(rows)


def raw_feature_baseline(sp_train: Split, sp_test: Split, tr_train, tr_test, tasks=TASKS, n_boot=200):
    """Logistic regression on the patient-mean raw state x_t (same steps as the probes)."""
    def pmean(tr):
        X, L, pid = tr["X"], tr["L"], tr["pid"]
        T = X.shape[1]
        valid = (np.arange(T)[None] >= CONTEXT) & (np.arange(T)[None] < L[:, None])
        rows, p = X[valid], np.repeat(pid[:, None], T, 1)[valid]
        return pd.DataFrame(rows).groupby(p).mean()
    Ftr, Fte = pmean(tr_train), pmean(tr_test)
    sc = StandardScaler().fit(Ftr.values)
    lm_tr = sp_train.patient_labels.set_index("patient_id")
    lm_te = sp_test.patient_labels.set_index("patient_id")
    out = []
    for task, col in tasks.items():
        y = Ftr.index.map(lm_tr[col]).values.astype(float)
        ok = np.isfinite(y)
        if y[ok].sum() < 5:
            continue
        clf = LogisticRegression(C=0.1, max_iter=500).fit(sc.transform(Ftr.values[ok]), y[ok])
        s = clf.predict_proba(sc.transform(Fte.values))[:, 1]
        yt = Fte.index.map(lm_te[col]).values.astype(float)
        m = np.isfinite(yt)
        lo, hi = bootstrap(yt[m], s[m], roc_auc_score, n_boot)
        out.append(dict(model="Raw features (LR)", task=task, traj="raw", prevalence=yt[m].mean(), n=m.sum(),
                        AUROC=roc_auc_score(yt[m], s[m]), AUROC_lo=lo, AUROC_hi=hi,
                        AUPRC=average_precision_score(yt[m], s[m])))
    return pd.DataFrame(out)
