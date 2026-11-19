"""
Evaluation driver: one (site, model) per process, results under outputs/<site>/<model>/.

    python -m experiments.run_site --site mimic --model wm5plus

Computes
    bench.csv          downstream probes, MIMIC-IV benchmark tasks (MIMIC only)
    disc_step.csv      per-timestep real-vs-generated discriminator AUROC
    disc_seq.csv       sequence discriminator, ordered vs time-permuted
    risktraj.parquet   per-step risk, oracle vs rollout, positives / negatives
    rct_*.csv          emulated RCT (5-action models)
Inpatient classification / time-to-event: experiments.inpatient (one process per site).
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import evaluate as ev
from . import rct
from .data import Split
from .sites import OUT, SITES
from .train import MODEL_SPECS, load_checkpoint, resolve_ckpt

log = logging.getLogger(__name__)
RISK_TASKS = {"mimic": {"Sepsis": ("patient", "has_Sepsis"), "ICU admission": ("patient", "icu_admission")},
              "eicu": {"In-hospital mortality": ("adm", "y_mortality"), "Long LOS": ("adm", "y_long_los")},
              "nwicu": {"In-hospital mortality": ("adm", "y_mortality"), "Long LOS": ("adm", "y_long_los")}}


def _labels(split, tr, kind, col):
    if kind == "patient":
        lm = split.patient_labels.set_index("patient_id")[col]
        return pd.Series(tr["pid"]).map(lm).values.astype(float)
    return split.adm[col].values[tr["idx"]].astype(float)


def discriminator(trtr, trte, max_t=ev.MAX_STEPS):
    rows = []
    for t in range(ev.CONTEXT, max_t):
        okr, oke = trtr["L"] > t, trte["L"] > t
        if okr.sum() < 200 or oke.sum() < 100:
            break
        r = dict(t=t, hours=6 * t, n_test=int(oke.sum()))
        for k in ["rollout"]:
            Xtr = np.r_[trtr["oracle"][okr, t], trtr[k][okr, t]]
            ytr = np.r_[np.zeros(okr.sum()), np.ones(okr.sum())]
            Xte = np.r_[trte["oracle"][oke, t], trte[k][oke, t]]
            yte = np.r_[np.zeros(oke.sum()), np.ones(oke.sum())]
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(C=1.0, max_iter=300).fit(sc.transform(Xtr), ytr)
            r[f"auroc_{k}"] = roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1])
        rows.append(r)
    return pd.DataFrame(rows)


def sequence_discriminator(trtr, trte, L=8, seed=0):
    rng = np.random.default_rng(seed)
    sl = slice(ev.CONTEXT, ev.CONTEXT + L)

    def perm(Z):
        return np.stack([z[rng.permutation(L)] for z in Z])

    def get(tr):
        ok = tr["L"] >= ev.CONTEXT + L
        return tr["oracle"][ok, sl], tr["rollout"][ok, sl]
    o_tr, r_tr = get(trtr)
    o_te, r_te = get(trte)
    conds = {"real vs rollout (ordered)": (lambda o, r: (o, r)),
             "real vs rollout (rollout permuted)": (lambda o, r: (o, perm(r))),
             "real vs rollout (real permuted)": (lambda o, r: (perm(o), r)),
             "real vs real (one permuted)": (lambda o, r: (o, perm(o)))}
    rows = []
    for name, f in conds.items():
        A0, A1 = f(o_tr, r_tr); B0, B1 = f(o_te, r_te)
        Xtr = np.r_[A0.reshape(len(A0), -1), A1.reshape(len(A1), -1)]
        Xte = np.r_[B0.reshape(len(B0), -1), B1.reshape(len(B1), -1)]
        ytr, yte = np.r_[np.zeros(len(A0)), np.ones(len(A1))], np.r_[np.zeros(len(B0)), np.ones(len(B1))]
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=0.01, max_iter=500).fit(sc.transform(Xtr), ytr)
        rows.append(dict(condition=name, auroc=roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1]),
                         n_test=len(B0)))
    return pd.DataFrame(rows)


def risk_trajectories(site, sp, trtr, trte, n_rep=5, max_t=16):
    out = []
    for task, (kind, col) in RISK_TASKS[site].items():
        if (kind == "patient" and col not in sp["train"].patient_labels) or (kind == "adm" and col not in sp["train"].adm):
            continue
        ytr, yte = _labels(sp["train"], trtr, kind, col), _labels(sp["test"], trte, kind, col)
        T = trtr["oracle"].shape[1]
        valid = (np.arange(T)[None] >= ev.CONTEXT) & (np.arange(T)[None] < trtr["L"][:, None]) & np.isfinite(ytr)[:, None]
        X = trtr["oracle"][valid]; y = np.repeat(ytr[:, None], T, 1)[valid]
        if len(X) > 200000:
            k = np.random.default_rng(0).choice(len(X), 200000, replace=False); X, y = X[k], y[k]
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=0.1, max_iter=300).fit(sc.transform(X), y)
        risk = {k: clf.predict_proba(sc.transform(trte[k].reshape(-1, trte[k].shape[-1])))[:, 1].reshape(trte[k].shape[:2])
                for k in ["oracle", "rollout"]}
        L = trte["L"]
        for cls in [1, 0]:
            m = (yte == cls) & (L >= max_t)
            ids = np.flatnonzero(m)
            if len(ids) < 5:
                continue
            for k, rk in risk.items():
                for t in range(ev.CONTEXT, max_t):
                    out.append(dict(task=task, cls=cls, traj=k, patient="mean", hour=6 * t,
                                    mean=float(rk[ids, t].mean()), sem=float(rk[ids, t].std() / np.sqrt(len(ids)))))
            # representative patients: median band of oracle mean risk
            mr = risk["oracle"][ids, ev.CONTEXT:max_t].mean(1)
            order = ids[np.argsort(mr)]
            reps = order[np.linspace(len(order) * 0.35, len(order) * 0.65, n_rep).astype(int)]
            for j, i in enumerate(reps):
                for k, rk in risk.items():
                    for t in range(ev.CONTEXT, max_t):
                        out.append(dict(task=task, cls=cls, traj=k, patient=f"P{j + 1}", hour=6 * t,
                                        mean=float(rk[i, t]), sem=0.0))
    return pd.DataFrame(out)


def run(site, model_name, parts=("bench", "disc", "risk", "rct"), n_boot=100, device=None):
    t0 = time.time()
    cfg = SITES[site]
    out = OUT / site / model_name
    out.mkdir(parents=True, exist_ok=True)
    ck = resolve_ckpt(cfg["runs"], model_name)
    model = load_checkpoint(ck, device)
    key = MODEL_SPECS[model_name]["action_key"]
    sp = {s: Split(cfg["proc"], s) for s in ["train", "test"]}
    meta = dict(site=site, model=model_name, checkpoint=str(ck))
    log.info("[%s/%s] %s", site, model_name, ck)

    if {"bench", "disc", "risk"} & set(parts):
        trtr = ev.admission_trajectories(model, sp["train"], key, max_n=20000)
        trte = ev.admission_trajectories(model, sp["test"], key, max_n=None if site == "mimic" else 20000, seed=1)
        if "bench" in parts and site == "mimic":
            ev.probe_benchmark(trtr, trte, sp["train"].patient_labels, sp["test"].patient_labels, model_name) \
              .to_csv(out / "bench.csv", index=False)
            if model_name == "wm5":
                ev.raw_feature_baseline(sp["train"], sp["test"], trtr, trte).to_csv(OUT / site / "raw_baseline.csv", index=False)
            log.info("[%s/%s] bench done", site, model_name)
        if "disc" in parts:
            discriminator(trtr, trte).to_csv(out / "disc_step.csv", index=False)
            sequence_discriminator(trtr, trte).to_csv(out / "disc_seq.csv", index=False)
            log.info("[%s/%s] discriminator done", site, model_name)
        if "risk" in parts:
            risk_trajectories(site, sp, trtr, trte).to_parquet(out / "risktraj.parquet")
        del trtr, trte
    if "rct" in parts and key == "act5":
        n_cont = sp["train"].state.shape[1] // 4
        man_p = cfg["proc"] / "feature_manifest.json"
        man = json.loads(man_p.read_text()) if man_p.exists() else json.loads((cfg["proc"].parent / "xsite_manifest.json").read_text())
        fidx = rct.feature_index(man["vital_codes"] + man["lab_codes"])
        co = rct.cohort_arrays(model, sp["test"])
        probes = rct.fit_outcome_probes(model, sp["train"], co, cfg["outcomes"])
        meta["rct_probe_auroc"] = {k: p["test_auroc"] for k, p in probes.items()}
        res = {d: rct.experiment_for_drug(model, co, d, n_cont, fidx, probes, n_boot=n_boot) for d in range(5)}
        pd.DataFrame([r["summary"] for r in res.values()]).to_csv(out / "rct_summary.csv", index=False)
        for k in ["probes", "traj", "features", "balance"]:
            parts_ = [r[k].assign(drug=r["summary"]["drug"]) if k == "balance" else r[k] for r in res.values() if len(r[k])]
            if parts_:
                pd.concat(parts_).to_csv(out / f"rct_{k}.csv", index=False)
        if model.action_conditioned:
            rct.specificity_matrix(model, co, res, n_cont, fidx).to_csv(out / "rct_spec.csv", index=False)
        log.info("[%s/%s] RCT done", site, model_name)
    meta["minutes"] = (time.time() - t0) / 60
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    import argparse
    import torch
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=list(SITES))
    ap.add_argument("--model", required=True)
    ap.add_argument("--parts", default="bench,disc,risk,rct")
    ap.add_argument("--device", default=None, help="cpu / mps (default: best available)")
    a = ap.parse_args()
    torch.set_num_threads(4)
    run(a.site, a.model, tuple(a.parts.split(",")), device=torch.device(a.device) if a.device else None)
