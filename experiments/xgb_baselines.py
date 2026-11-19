"""
XGBoost baselines. Runs in its own process WITHOUT importing torch: xgboost and torch ship different
OpenMP runtimes on macOS and segfault when both are loaded.

Inputs are the exact feature matrices cached by experiments.baselines_surv:
    outputs/inpatient/cache_<site>_<task>.npz   (inpatient tasks, raw 48-h features)
    outputs/inpatient/cache_mimic_bench.npz     (MIMIC all-task benchmark, patient-mean raw state)
XGBoost: binary:logistic for classification (AUROC/AUPRC), survival:cox for time-to-event (C-index);
early stopping on the validation split.

    python -m experiments.xgb_baselines              # all cached sites/tasks + MIMIC benchmark
"""
from __future__ import annotations

import xgboost as xgb  # noqa: I001  (imported before the other numerical libraries)

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "outputs/inpatient"
OUT = ROOT / "outputs"
log = logging.getLogger(__name__)
SEED = 0
PARAMS = dict(max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
              tree_method="hist", nthread=4, seed=SEED)


def _boot(fn, n, k, seed=SEED):
    rng = np.random.default_rng(seed)
    v = [x for x in (fn(rng.integers(0, n, n)) for _ in range(k)) if np.isfinite(x)]
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)


def classify(Xtr, ytr, Xva, yva, Xte):
    bst = xgb.train({**PARAMS, "objective": "binary:logistic", "eval_metric": "logloss"},
                    xgb.DMatrix(Xtr, label=ytr), 1000, evals=[(xgb.DMatrix(Xva, label=yva), "val")],
                    early_stopping_rounds=50, verbose_eval=False)
    return bst.predict(xgb.DMatrix(Xte), iteration_range=(0, bst.best_iteration + 1))


def cox(Xtr, ttr, etr, Xva, tva, eva, Xte):
    lab = lambda t, e: np.where(e > 0, t, -t)          # negative label = right-censored
    bst = xgb.train({**PARAMS, "objective": "survival:cox", "eval_metric": "cox-nloglik"},
                    xgb.DMatrix(Xtr, label=lab(ttr, etr)), 1000,
                    evals=[(xgb.DMatrix(Xva, label=lab(tva, eva)), "val")], early_stopping_rounds=50, verbose_eval=False)
    return np.log(bst.predict(xgb.DMatrix(Xte), iteration_range=(0, bst.best_iteration + 1)))


def run_inpatient():
    by_site = {}
    for p in sorted(INP.glob("cache_*_*.npz")):
        if p.name == "cache_mimic_bench.npz":
            continue
        z = np.load(p, allow_pickle=True)
        site = p.name[len("cache_"):].split("_")[0] if not p.name.startswith("cache_mimic_meds") else "mimic_meds"
        y, t, e = z["y_test"], z["t_test"], z["e_test"]
        pcls = classify(z["X_train"], z["y_train"], z["X_val"], z["y_val"], z["X_test"])
        risk = cox(z["X_train"], z["t_train"], z["e_train"], z["X_val"], z["t_val"], z["e_val"], z["X_test"])
        auc = roc_auc_score(y, pcls)
        lo, hi = _boot(lambda b: roc_auc_score(y[b], pcls[b]) if 0 < y[b].sum() < len(b) else np.nan, len(y), 200)
        c = concordance_index(t, -risk, e)
        clo, chi = _boot(lambda b: concordance_index(t[b], -risk[b], e[b]), len(y), 100)
        by_site.setdefault(site, []).append(dict(model="xgboost", site=str(z["label"]), task=str(z["task"]), traj="raw",
                                                 n_test=len(y), events_test=int(y.sum()), AUROC=auc, AUROC_lo=lo,
                                                 AUROC_hi=hi, AUPRC=average_precision_score(y, pcls),
                                                 C_index=c, C_lo=clo, C_hi=chi))
        log.info("[%s] xgboost %-19s AUROC %.3f  C %.3f", site, z["task"], auc, c)
    for site, rows in by_site.items():
        pd.DataFrame(rows).to_csv(INP / f"res_{site}_xgboost.csv", index=False)


def run_bench():
    p = INP / "cache_mimic_bench.npz"
    if not p.exists():
        return
    z = np.load(p, allow_pickle=True)
    out, raw = [], []
    for task in z["tasks"]:
        k = str(task).replace(" ", "_")
        y = {s: z[f"y_{s}_{k}"] for s in ["train", "val", "test"]}
        ok = {s: np.isfinite(y[s]) for s in y}
        if y["train"][ok["train"]].sum() < 5 or y["test"][ok["test"]].sum() < 2:
            continue
        pr = classify(z["X_train"][ok["train"]], y["train"][ok["train"]], z["X_val"][ok["val"]], y["val"][ok["val"]],
                      z["X_test"][ok["test"]])
        yt = y["test"][ok["test"]]
        lo, hi = _boot(lambda b: roc_auc_score(yt[b], pr[b]) if 0 < yt[b].sum() < len(b) else np.nan, len(yt), 200)
        out.append(dict(model="XGBoost", task=str(task), traj="xgb", prevalence=yt.mean(), n=len(yt),
                        AUROC=roc_auc_score(yt, pr), AUROC_lo=lo, AUROC_hi=hi, AUPRC=average_precision_score(yt, pr)))
        log.info("[mimic bench] XGBoost %-20s AUROC %.3f", task, out[-1]["AUROC"])
        # logistic regression on the same patient-mean raw state (the "Raw features (LR)" baseline)
        sc = StandardScaler().fit(z["X_train"][ok["train"]])
        clf = LogisticRegression(C=0.1, max_iter=500).fit(sc.transform(z["X_train"][ok["train"]]), y["train"][ok["train"]])
        pl = clf.predict_proba(sc.transform(z["X_test"][ok["test"]]))[:, 1]
        lo, hi = _boot(lambda b: roc_auc_score(yt[b], pl[b]) if 0 < yt[b].sum() < len(b) else np.nan, len(yt), 200)
        raw.append(dict(model="Raw features (LR)", task=str(task), traj="raw", prevalence=yt.mean(), n=len(yt),
                        AUROC=roc_auc_score(yt, pl), AUROC_lo=lo, AUROC_hi=hi, AUPRC=average_precision_score(yt, pl)))
    (OUT / "mimic").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(OUT / "mimic" / "xgb_baseline.csv", index=False)
    pd.DataFrame(raw).to_csv(OUT / "mimic" / "raw_baseline.csv", index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run_inpatient()
    run_bench()
