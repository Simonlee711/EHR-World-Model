"""
Inpatient tasks at 48 h: classification + time-to-event, for every site.

Tasks (prediction time t* = bin origin + 48 h; input = first 8 six-hour bins)
    mortality_30d   in-hospital death within 30 d of t*        TTE: time to death, censored at discharge / 30 d
    los             total LOS > 7 d                            TTE: time from t* to live discharge (deaths censored)
    readmission_30d next admission within 30 d of discharge    TTE: time from discharge, censored at 30 d
                    (not available in eICU: one stay per subject)
    sepsis          sepsis-coded admission (ICD-10 A40/A41/R65.2; ICD-9 038/995.91/995.92/785.52) with
                    suspected-infection onset (first systemic antibiotic start) after t*;
                    coded admissions with onset before t* (or no antibiotic) are excluded
                    TTE: time from t* to onset, censored at discharge
Eligibility: >= 8 bins, alive and in hospital at t*.

Representations: for every model and trajectory type (oracle = encoder of real data / rollout = free rollout)
    h = [mean_{t=2..7} z_t ; z_7]  (2 x 128);   raw baseline: [mean_t x_t ; x_7]
Classifier: L2 logistic regression (AUROC, AUPRC); TTE: L2-penalised Cox (Breslow), Harrell's C-index.

    python -m experiments.inpatient --site mimic_meds --models wm5plus
    python -m experiments.inpatient --site eicu --models wm5plus
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import evaluate as ev
from .data import Split
from .train import MODEL_SPECS, ROOT, load_checkpoint, resolve_ckpt

log = logging.getLogger(__name__)
OUT = ROOT / "outputs/inpatient"
PRED_BINS = 8
TASKS = ["30-day mortality", "LOS", "30-day readmission", "Sepsis"]
SEPSIS10 = ("A40", "A41", "R652")
SEPSIS9 = ("038", "99591", "99592", "78552")
SITES = {
    "mimic_meds": dict(label="MIMIC-IV", proc=ROOT / "data/processed/mimic_wm", runs=ROOT / "runs/mimic"),
    "eicu":       dict(label="eICU-CRD", proc=ROOT / "data/processed/xsite/eicu", runs=ROOT / "runs/eicu"),
    "nwicu":      dict(label="NWICU", proc=ROOT / "data/processed/xsite/nwicu", runs=ROOT / "runs/nwicu"),
}
HORIZONS = [1, 2, 4, 6]      # rollout steps before the 48-h prediction time (calibration analysis)


# ── Event times per admission ──────────────────────────────────────────────────

def _first_abx_bin(split: Split) -> np.ndarray:
    a = split.act["act5"][:, 0]
    out = np.full(len(split.adm), np.nan)
    for i, (o, n) in enumerate(zip(split.adm.offset.values, split.adm.n_bins.values)):
        hit = np.flatnonzero(a[o:o + n])
        if len(hit):
            out[i] = hit[0]
    return out


def _meds_sepsis_hadm(site_cfg, split_name: str, pids) -> set:
    d = ROOT / "data/meds/MIMIC_MEDS_Extract_v0.0.3_test/data" / {"train": "train", "val": "tuning", "test": "held_out"}[split_name]
    hadm = set()
    for f in sorted(d.glob("*.parquet")):
        t = pq.read_table(f, columns=["patient_id", "hadm_id", "code"], read_dictionary=["code"],
                          filters=[("patient_id", "in", list(pids))]).to_pandas()
        cats = pd.Series(t.code.cat.categories)
        sep = cats[cats.str.startswith(tuple(f"DIAGNOSIS//ICD//10//{p}" for p in SEPSIS10) +
                                       tuple(f"DIAGNOSIS//ICD//9//{p}" for p in SEPSIS9))]
        hadm |= set(t.loc[t.code.isin(sep), "hadm_id"].dropna().astype(int))
    return hadm


def _harm_sepsis_enc(site: str) -> set:
    enc = set()
    for f in sorted((ROOT / "data" / site / "data").glob("*.parquet")):
        t = pq.read_table(f, columns=["encounter_id", "code"], read_dictionary=["code"]).to_pandas()
        cats = pd.Series(t.code.cat.categories)
        sep = cats[cats.str.startswith(tuple(f"DX//ICD10//{p}" for p in SEPSIS10))]
        enc |= set(t.loc[t.code.isin(sep), "encounter_id"].dropna().astype(int))
    return enc


def task_table(site: str, split_name: str, split: Split) -> pd.DataFrame:
    """y_<task> (NaN = excluded), t_<task> (days), e_<task> per admission."""
    a = split.adm.copy()
    if site == "mimic_meds":
        t0 = a.t0
        died = a.death_time.notna() & (a.death_time <= a.discharge_time + pd.Timedelta(hours=24)) \
            & (a.death_time >= a.admit_time)
        death = a.death_time.where(died)
        nxt = a.sort_values(["patient_id", "admit_time"]).groupby("patient_id").admit_time.shift(-1).reindex(a.index)
        sep = a.hadm_id.astype(int).isin(_meds_sepsis_hadm(SITES[site], split_name, a.patient_id.unique()))
        has_readmit = True
    else:
        t0 = a.admit_time
        died = a.died_in_hospital.fillna(False).astype(bool)
        death = a.death_time.where(died, pd.NaT).fillna(a.discharge_time.where(died))
        allenc = pd.read_parquet(ROOT / "data" / site / "metadata/encounters.parquet")[["subject_id", "admit_time"]]
        allenc = allenc.sort_values("admit_time")
        m = pd.merge_asof(a[["patient_id", "discharge_time"]].rename(columns={"patient_id": "subject_id"})
                          .reset_index().sort_values("discharge_time"),
                          allenc.rename(columns={"admit_time": "next_admit"}), left_on="discharge_time",
                          right_on="next_admit", by="subject_id", direction="forward", allow_exact_matches=False)
        nxt = m.set_index("index").next_admit.reindex(a.index)
        sep = a.encounter_id.astype(int).isin(_harm_sepsis_enc(site))
        has_readmit = site != "eicu"
    pred = t0 + pd.Timedelta(hours=6 * PRED_BINS)
    day = lambda td: td.dt.total_seconds() / 86400.0
    elig = (a.n_bins >= PRED_BINS) & (a.discharge_time > pred) & ~(died & (death <= pred))
    out = pd.DataFrame(index=a.index)
    out["eligible"] = elig

    # mortality
    to_dis = day(a.discharge_time - pred)
    to_death = day(death - pred)
    y = (died & (to_death <= 30)).astype(float)
    out["y_30-day mortality"] = y.where(elig)
    out["t_30-day mortality"] = np.minimum(np.minimum(to_death.fillna(np.inf), to_dis), 30.0)
    out["e_30-day mortality"] = y
    # LOS
    out["y_LOS"] = (day(a.discharge_time - a.admit_time) > 7).astype(float).where(elig)
    out["t_LOS"] = to_dis
    out["e_LOS"] = (~died).astype(float)
    # readmission
    if has_readmit:
        gap = day(nxt - a.discharge_time)
        yr = ((gap > 0) & (gap <= 30)).astype(float)
        out["y_30-day readmission"] = yr.where(elig & ~died)
        out["t_30-day readmission"] = np.minimum(gap.fillna(30.0), 30.0).clip(lower=1e-3)
        out["e_30-day readmission"] = yr
    # sepsis
    onset = _first_abx_bin(split)
    ok_sep = elig & (~sep | (np.isfinite(onset) & (onset >= PRED_BINS)))
    ys = sep.astype(float)
    out["y_Sepsis"] = ys.where(ok_sep)
    out["t_Sepsis"] = np.where(sep, (np.nan_to_num(onset) * 6 + 3 - 6 * PRED_BINS) / 24.0, to_dis)
    out["e_Sepsis"] = ys
    for c in [c for c in out.columns if c.startswith("t_")]:
        out[c] = pd.Series(out[c]).clip(lower=1e-3)
    return out


# ── Representations ────────────────────────────────────────────────────────────

def representations(model, split: Split, idx: np.ndarray, action_key: str, horizons=True) -> dict:
    X, A, _ = ev.padded(split, idx, PRED_BINS, action_key)
    feats = {"raw": np.concatenate([X.mean(1), X[:, -1]], 1)}
    if model is None:
        return feats
    Z = ev.encode(model, X)
    traj = {"oracle": Z,
            "rollout": np.concatenate([Z[:, :ev.CONTEXT],
                                       ev.rollout(model, Z[:, :ev.CONTEXT], A[:, ev.CONTEXT - 1:PRED_BINS - 1])], 1)}
    if horizons:   # h rollout steps before 48 h: context = first PRED_BINS - h bins
        for h in HORIZONS:
            c = PRED_BINS - h
            traj[f"h{h}"] = np.concatenate([Z[:, :c], ev.rollout(model, Z[:, c - 2:c], A[:, c - 1:PRED_BINS - 1])], 1)
    for k, v in traj.items():
        feats[k] = np.concatenate([v[:, ev.CONTEXT:].mean(1), v[:, -1]], 1)
    return feats


# ── Cox model (L2-penalised, Breslow ties) ─────────────────────────────────────

def fit_cox(X, t, e, l2=1e-2, iters=200):
    Xt = torch.tensor(X, dtype=torch.float64)
    order = torch.argsort(torch.tensor(t), descending=True)
    Xs, es = Xt[order], torch.tensor(e, dtype=torch.float64)[order]
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1, max_iter=iters, line_search_fn="strong_wolfe")
    n_ev = max(es.sum().item(), 1.0)

    def closure():
        opt.zero_grad()
        r = Xs @ w
        ll = ((r - torch.logcumsumexp(r, 0)) * es).sum() / n_ev
        loss = -ll + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().numpy()


def _boot(fn, n, k=100, seed=0):
    rng = np.random.default_rng(seed)
    vals = [v for v in (fn(rng.integers(0, n, n)) for _ in range(k)) if np.isfinite(v)]
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


def evaluate_features(Ftr, Fte, Ttr, Tte, task, max_train=40000, seed=0, return_models=False):
    ytr, yte = Ttr[f"y_{task}"].values, Tte[f"y_{task}"].values
    mtr, mte = np.isfinite(ytr), np.isfinite(yte)
    if ytr[mtr].sum() < 10 or yte[mte].sum() < 5:
        return None
    itr = np.flatnonzero(mtr)
    if len(itr) > max_train:
        itr = np.random.default_rng(seed).choice(itr, max_train, replace=False)
    ite = np.flatnonzero(mte)
    sc = StandardScaler().fit(Ftr[itr])
    Xtr, Xte = sc.transform(Ftr[itr]), sc.transform(Fte[ite])
    clf = LogisticRegression(C=0.05, max_iter=1000).fit(Xtr, ytr[itr])
    p = clf.predict_proba(Xte)[:, 1]
    y = yte[ite]
    lo, hi = _boot(lambda b: roc_auc_score(y[b], p[b]) if 0 < y[b].sum() < len(b) else np.nan, len(y), 200, seed)
    w = fit_cox(Xtr, Ttr[f"t_{task}"].values[itr], Ttr[f"e_{task}"].values[itr])
    risk = Xte @ w
    tt, ee = Tte[f"t_{task}"].values[ite], Tte[f"e_{task}"].values[ite]
    c = concordance_index(tt, -risk, ee)
    clo, chi = _boot(lambda b: concordance_index(tt[b], -risk[b], ee[b]), len(y), 100, seed)
    out = dict(n_test=len(y), events_test=int(y.sum()), AUROC=roc_auc_score(y, p), AUROC_lo=lo, AUROC_hi=hi,
               AUPRC=average_precision_score(y, p), C_index=c, C_lo=clo, C_hi=chi)
    if return_models:
        return out, dict(scaler=sc, clf=clf, cox_w=w, test_index=ite, p=p, lp=risk)
    return out


def ece(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    rows, e = [], 0.0
    for k in range(n_bins):
        m = b == k
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
            rows.append((k, float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return e, rows


def run_site(site: str, models, include_raw=True):
    cfg = SITES[site]
    OUT.mkdir(parents=True, exist_ok=True)
    sp = {s: Split(cfg["proc"], s) for s in ["train", "val", "test"]}
    T = {s: task_table(site, s, sp[s]) for s in sp}
    idx = {s: np.flatnonzero(T[s].eligible.values) for s in sp}
    T = {s: T[s].iloc[idx[s]].reset_index(drop=True) for s in sp}
    tasks = [t for t in TASKS if f"y_{t}" in T["train"]]
    pd.DataFrame([dict(site=cfg["label"], task=t, n_train=int(np.isfinite(T["train"][f"y_{t}"]).sum()),
                       ev_train=int(np.nansum(T["train"][f"y_{t}"])),
                       n_test=int(np.isfinite(T["test"][f"y_{t}"]).sum()), ev_test=int(np.nansum(T["test"][f"y_{t}"])))
                  for t in tasks]).to_csv(OUT / f"cohort_{site}.csv", index=False)
    adm_te = sp["test"].adm.iloc[idx["test"]].reset_index(drop=True)
    runs = ([("raw", None)] if include_raw else []) + [(m, m) for m in models]
    for name, m in runs:
        model, key, ck = None, "act5", None
        if m is not None:
            ck = resolve_ckpt(cfg["runs"], m)
            if ck is None:
                log.warning("[%s] no checkpoint for %s", site, m)
                continue
            model = load_checkpoint(ck)
            key = MODEL_SPECS[m]["action_key"]
            log.info("[%s] %s <- %s", site, m, ck)
        F = {s: representations(model, sp[s], sp[s].adm.index.values[idx[s]], key) for s in sp}
        rows, score_rows, cal_rows, rel_rows = [], [], [], []
        for traj in (["raw"] if model is None else ["oracle", "rollout"]):
            for task in tasks:
                r = evaluate_features(F["train"][traj], F["test"][traj], T["train"], T["test"], task, return_models=True)
                if not r:
                    continue
                r, mdl = r
                rows.append(dict(site=cfg["label"], task=task, model=name, traj=traj,
                                 checkpoint=ck.name if ck else "", **r))
                log.info("[%s] %-8s %-7s %-19s AUROC %.3f  C %.3f", site, name, traj, task, r["AUROC"], r["C_index"])
                if traj in ("oracle", "rollout", "raw"):
                    ii = mdl["test_index"]
                    score_rows.append(pd.DataFrame(dict(site=cfg["label"], model=name, traj=traj, task=task,
                        y=T["test"][f"y_{task}"].values[ii], t=T["test"][f"t_{task}"].values[ii],
                        e=T["test"][f"e_{task}"].values[ii], p=mdl["p"], lp=mdl["lp"],
                        age=adm_te["age"].values[ii], sex=adm_te["sex"].values[ii])))
                # calibration across rollout horizons: probe fit on ORACLE train features, Platt on val
                if traj == "oracle" and model is not None:
                    yv, ytst = T["val"][f"y_{task}"].values, T["test"][f"y_{task}"].values
                    mv, mt = np.isfinite(yv), np.isfinite(ytst)
                    for rep in ["oracle"] + [f"h{h}" for h in HORIZONS]:
                        pv = mdl["clf"].decision_function(mdl["scaler"].transform(F["val"][rep][mv]))
                        pt = mdl["clf"].decision_function(mdl["scaler"].transform(F["test"][rep][mt]))
                        platt = LogisticRegression(C=1e4).fit(pv[:, None], yv[mv])
                        pc = platt.predict_proba(pt[:, None])[:, 1]
                        e_, rel = ece(ytst[mt], pc)
                        hz = 0 if rep == "oracle" else int(rep[1:])
                        cal_rows.append(dict(site=cfg["label"], model=name, task=task, rep=rep, horizon=hz,
                                             ece=e_, auroc=roc_auc_score(ytst[mt], pc)))
                        rel_rows += [dict(site=cfg["label"], model=name, task=task, rep=rep, bin=k, mean_pred=a_,
                                          frac_pos=b_, n=c_) for k, a_, b_, c_ in rel]
        pd.DataFrame(rows).to_csv(OUT / f"res_{site}_{name}.csv", index=False)
        if score_rows:
            pd.concat(score_rows).to_parquet(OUT / f"scores_{site}_{name}.parquet")
        if cal_rows:
            pd.DataFrame(cal_rows).to_csv(OUT / f"calib_{site}_{name}.csv", index=False)
            pd.DataFrame(rel_rows).to_csv(OUT / f"reliab_{site}_{name}.csv", index=False)
    aggregate()


def aggregate():
    order = {v["label"]: i for i, v in enumerate(SITES.values())}
    parts = [pd.read_csv(p) for p in sorted(OUT.glob("res_*.csv")) if p.stat().st_size > 5]
    if parts:
        df = pd.concat(parts, ignore_index=True)
        df["_s"] = df.site.map(order); df["_t"] = df.task.map({t: i for i, t in enumerate(TASKS)})
        df["_m"] = df.model.map({"wm5": 0, "wm5plus": 1, "wm78": 2, "wm0": 3, "raw": 4})
        df["_r"] = df.traj.map({"oracle": 0, "rollout": 1, "raw": 2})
        df.sort_values(["_s", "_t", "_m", "_r"]).drop(columns=["_s", "_t", "_m", "_r"]) \
          .to_csv(OUT / "inpatient_results.csv", index=False)
    cs = [pd.read_csv(p) for p in sorted(OUT.glob("cohort_*.csv"))]
    if cs:
        pd.concat(cs).to_csv(OUT / "inpatient_cohorts.csv", index=False)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=list(SITES))
    ap.add_argument("--models", nargs="*", default=["wm5", "wm5plus", "wm0"])
    ap.add_argument("--no-raw", action="store_true")
    a = ap.parse_args()
    run_site(a.site, a.models, include_raw=not a.no_raw)
