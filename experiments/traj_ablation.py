"""
Ablation: direct prediction vs trajectory-based prediction (MIMIC-IV inpatient tasks, EHR-WM = wm5plus).

Frozen encoder + transition model. Every admission is initialised from its first two observed 6-h bins (z_1, z_2);
outcomes are the inpatient tasks of experiments/inpatient.py, predicted at 48 h (end of bin H = 8), with the same
cohorts, eligibility and train / val / test patient splits.

Probe inputs
  Direct                     [z_1, z_2]
  Direct + actions           [z_1, z_2, a_2 ... a_7]          (same action information the trajectories receive)
  Trajectory (free rollout)  z^_H from z_1, z_2 and the observed actions a_2 ... a_7  (no observed state after 12 h)
Probe: L2 logistic regression on standardised inputs; C chosen on the validation split by AUROC from the same grid
for every input. Metrics: AUROC, AUPRC on held-out test admissions; 95% CIs and differences vs Direct from a paired
bootstrap that resamples test PATIENTS (all methods scored on the same resamples).

    python -m experiments.traj_ablation      ->  outputs/traj_ablation/mimic_wm5plus.csv (+ _boot.csv)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import evaluate as ev
from .data import Split
from .inpatient import PRED_BINS, SITES, TASKS, task_table
from .train import MODEL_SPECS, ROOT, load_checkpoint, resolve_ckpt

log = logging.getLogger(__name__)
OUT = ROOT / "outputs/traj_ablation"
C_GRID = [1e-3, 1e-2, 5e-2, 1e-1, 1.0]
N_BOOT = 1000
MAX_TRAIN = 40000
METHODS = ["Direct", "Direct + actions", "Trajectory (free rollout)"]
INPUTS = {"Direct": "[z_1, z_2]", "Direct + actions": "[z_1, z_2, a_2:7]",
          "Trajectory (free rollout)": "z^_H (free rollout)"}


def features(model, split: Split, idx: np.ndarray, action_key: str) -> dict:
    X, A, _ = ev.padded(split, idx, PRED_BINS, action_key)
    Z = ev.encode(model, X)                                              # (N, 8, D) encoder states of bins 1..8
    acts = A[:, ev.CONTEXT - 1:PRED_BINS - 1]                            # a_2 ... a_7 (drive z^_3 ... z^_8)
    free = ev.rollout(model, Z[:, :ev.CONTEXT], acts)                    # z^_3 ... z^_8 from z_1, z_2 only
    z12 = Z[:, :ev.CONTEXT].reshape(len(Z), -1)
    return {"Direct": z12,
            "Direct + actions": np.concatenate([z12, acts.reshape(len(acts), -1)], 1),
            "Trajectory (free rollout)": free[:, -1]}


def fit_probe(Ftr, ytr, Fva, yva, seed=0):
    """Same protocol for every input: subsample train, standardise, pick C on validation AUROC."""
    itr = np.flatnonzero(np.isfinite(ytr))
    if len(itr) > MAX_TRAIN:
        itr = np.random.default_rng(seed).choice(itr, MAX_TRAIN, replace=False)
    iva = np.flatnonzero(np.isfinite(yva))
    sc = StandardScaler().fit(Ftr[itr])
    best = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Ftr[itr]), ytr[itr])
        auc = roc_auc_score(yva[iva], clf.predict_proba(sc.transform(Fva[iva]))[:, 1])
        if best is None or auc > best[0]:
            best = (auc, C, clf)
    return sc, best[2], best[1], best[0]


def run(site="mimic_meds", model_name="wm5plus", seed=0):
    cfg = SITES[site]
    OUT.mkdir(parents=True, exist_ok=True)
    ck = resolve_ckpt(cfg["runs"], model_name)
    model = load_checkpoint(ck)
    key = MODEL_SPECS[model_name]["action_key"]
    log.info("[%s] %s <- %s", site, model_name, ck)
    sp = {s: Split(cfg["proc"], s) for s in ["train", "val", "test"]}
    T = {s: task_table(site, s, sp[s]) for s in sp}
    idx = {s: np.flatnonzero(T[s].eligible.values) for s in sp}
    T = {s: T[s].iloc[idx[s]].reset_index(drop=True) for s in sp}
    F = {s: features(model, sp[s], sp[s].adm.index.values[idx[s]], key) for s in sp}
    pid_te = sp["test"].adm.iloc[idx["test"]].patient_id.values

    rows, boots = [], []
    for task in [t for t in TASKS if f"y_{t}" in T["train"]]:
        ytr, yva, yte = (T[s][f"y_{task}"].values for s in ["train", "val", "test"])
        ite = np.flatnonzero(np.isfinite(yte))
        y = yte[ite]
        P = {}
        for m in METHODS:
            sc, clf, C, val_auc = fit_probe(F["train"][m], ytr, F["val"][m], yva, seed)
            P[m] = clf.predict_proba(sc.transform(F["test"][m][ite]))[:, 1]
            rows.append(dict(task=task, method=m, input=INPUTS[m], C=C, val_AUROC=val_auc, n_test=len(y),
                             events_test=int(y.sum()), AUROC=roc_auc_score(y, P[m]),
                             AUPRC=average_precision_score(y, P[m])))
            log.info("%-20s %-26s C=%-6g AUROC %.3f AUPRC %.3f", task, m, C, rows[-1]["AUROC"], rows[-1]["AUPRC"])
        # paired cluster bootstrap over test patients
        pids = pid_te[ite]
        upid, inv = np.unique(pids, return_inverse=True)
        members = [np.flatnonzero(inv == k) for k in range(len(upid))]
        rng = np.random.default_rng(seed)
        for b in range(N_BOOT):
            bi = np.concatenate([members[k] for k in rng.integers(0, len(upid), len(upid))])
            if not 0 < y[bi].sum() < len(bi):
                continue
            for m in METHODS:
                boots.append(dict(task=task, b=b, method=m, AUROC=roc_auc_score(y[bi], P[m][bi]),
                                  AUPRC=average_precision_score(y[bi], P[m][bi])))
    R, B = pd.DataFrame(rows), pd.DataFrame(boots)
    # CIs and paired differences vs Direct
    W = B.pivot_table(index=["task", "b"], columns="method", values=["AUROC", "AUPRC"])
    for i, r in R.iterrows():
        for met in ["AUROC", "AUPRC"]:
            v = W[(met, r.method)].xs(r.task, level="task")
            d = v - W[(met, "Direct")].xs(r.task, level="task")
            R.loc[i, f"{met}_lo"], R.loc[i, f"{met}_hi"] = np.nanpercentile(v, [2.5, 97.5])
            R.loc[i, f"d{met}_vs_direct"] = r[met] - R[(R.task == r.task) & (R.method == "Direct")][met].iloc[0]
            R.loc[i, f"d{met}_lo"], R.loc[i, f"d{met}_hi"] = np.nanpercentile(d, [2.5, 97.5])
    R.insert(0, "site", cfg["label"]); R.insert(1, "model", model_name)
    R.to_csv(OUT / f"mimic_{model_name}.csv", index=False)
    B.to_csv(OUT / f"mimic_{model_name}_boot.csv", index=False)
    log.info("\n%s", R[["task", "method", "AUROC", "AUROC_lo", "AUROC_hi", "AUPRC", "dAUROC_vs_direct",
                         "dAUROC_lo", "dAUROC_hi"]].round(3).to_string())
    return R


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
