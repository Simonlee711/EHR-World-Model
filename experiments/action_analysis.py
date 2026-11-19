"""
Action-specificity and physiological-validation analyses for the emulated RCT.

Question: does the world model learn clinically meaningful, ACTION-SPECIFIC dynamics, or just a
generic "treated patient" direction?

Per (site, model):
 1. Matching      covariate balance of PSM (caliper 0.2) vs Mahalanobis matching on core covariates, both
                  with exact co-treatment strata; RCT read-outs with PSM  -> rct_* (as experiments.rct)
 2. Physiology    Delta_a(t) = h(z^do(a)) - h(z^do(0)) on a population of held-out admissions:
                  do(0) = observed actions with a removed in the 24-54 h window, do(a) = a given in every
                  window bin; 6-step free rollouts (EMA-smoothed) decoded with the reconstruction head (z-units and native units)
 3. Observed      matched-cohort difference-in-differences per variable:
                  (x_T[48-60h] - x_T[0-24h]) - (x_C[48-60h] - x_C[0-24h])      (removes residual baseline imbalance)
 4. Specificity   "diagonal" profile matrix: corr( simulated profile of action a , observed profile of cohort b )
                  raw and after removing the component shared by all actions; permutation test on the trace;
                  cohort-conditional version (a applied to cohort b's controls on b's schedule)
 5. Generic axis  cosine similarity between latent / decoded effects of different actions; projection of each
                  effect on the mortality-probe ("severity") direction; effect magnitudes vs action prevalence
 6. Clinical      sign agreement of simulated and observed effects with pharmacological expectations
 7. Outcomes      probe-predicted long-LOS / mortality under C_fact vs C_do with CIs vs observed rates

    python -m experiments.action_analysis --site mimic --model wm5plus
"""
from __future__ import annotations

import itertools
import json
import logging
import time

import numpy as np
import pandas as pd

from . import rct
from .actions import GROUP_LABELS, GROUPS
from .data import Split
from .evaluate import decode
from .features import BY_CODE, LABEL
from .sites import ROOT, SITES
from .train import load_checkpoint, resolve_ckpt

log = logging.getLogger(__name__)
OUT = ROOT / "outputs/action_analysis"
C, R = rct.C_CTX, rct.R
GL = [GROUP_LABELS[g] for g in GROUPS]
MATCHING = "psm"    # exact co-treatment strata + propensity NN (caliper 0.2)

# Pharmacological expectations (sign of the effect within ~1-2 days); +1 increase, -1 decrease.
# Heuristic priors from standard ICU pharmacology, used only as a plausibility check.
EXPECTED = {
    "Vasopressors": {"Systolic BP": +1, "Diastolic BP": +1},
    "IV Fluids": {"Hemoglobin": -1, "Hematocrit": -1, "Chloride": +1, "BUN": -1, "Creatinine": -1,
                  "Albumin": -1, "Bicarbonate": -1, "Lactate": -1, "Heart rate": -1},
    "Antibiotics": {"Temperature": -1, "WBC": -1, "Heart rate": -1, "Lactate": -1},
    "Pain Medications (Analgesics)": {"Temperature": -1, "Respiratory rate": -1, "Heart rate": -1},
    "Sedatives": {"GCS eye": -1, "GCS verbal": -1, "GCS motor": -1, "Respiratory rate": -1,
                  "Systolic BP": -1, "Heart rate": -1},
}


def _manifest(site):
    p = SITES[site]["proc"] / "feature_manifest.json"
    return json.loads(p.read_text()) if p.exists() else json.loads((SITES[site]["proc"].parent / "xsite_manifest.json").read_text())


def native(z, codes, man):
    """z-units (possibly log-scaled) -> native clinical units."""
    mu = np.array([man["state_mean"][c] for c in codes]); sd = np.array([man["state_std"][c] for c in codes])
    v = z * sd + mu
    lg = np.array([BY_CODE[c]["log"] for c in codes])
    return np.where(lg, np.exp(v), v)


def population_response(model, co, codes, man, n_max=6000, n_boot=200, seed=0):
    """Delta_a(t) for every action on a random population of eligible held-out admissions."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(co["Z"]), min(n_max, len(co["Z"])), replace=False)
    Z, A = co["Z"][idx], co["A"][idx]
    rows, lat = [], {}
    for d, g in enumerate(GL):
        A0 = A.copy(); A0[:, C:R, d] = 0
        A1 = A0.copy(); A1[:, C:R, d] = 1
        z0, z1 = rct.generate(model, Z, A0), rct.generate(model, Z, A1)                      # (n, 6, D), free rollout
        lat[g] = (z1 - z0)
        h0, h1 = decode(model, z0), decode(model, z1)                                         # (n, 6, nc)
        dz = h1 - h0
        dn = native(h1, codes, man) - native(h0, codes, man)
        for t in range(dz.shape[1]):
            boots = np.array([dz[rng.integers(0, len(dz), len(dz)), t].mean(0) for _ in range(n_boot)])
            lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
            for f, c in enumerate(codes):
                rows.append(dict(action=g, variable=LABEL[c], hour=6 * (C + t) + 3, delta_z=float(dz[:, t, f].mean()),
                                 lo_z=float(lo[f]), hi_z=float(hi[f]), delta_native=float(dn[:, t, f].mean()),
                                 unit=BY_CODE[c]["unit"]))
    return pd.DataFrame(rows), lat, idx


def observed_did(co, pairs, t_idx, c_idx, n_cont, codes):
    """Matched difference-in-differences per variable, and per hour, in z-units."""
    X = co["X"]
    ti, ci = t_idx[pairs[:, 0]], c_idx[pairs[:, 1]]
    base = rct.last_observed(X, n_cont, 0, C - 1)
    post = rct.last_observed(X, n_cont, R - 1, R)
    dT, dC = post[ti] - base[ti], post[ci] - base[ci]
    prof = {LABEL[c]: float(np.nanmean(dT[:, f]) - np.nanmean(dC[:, f]))
            if np.isfinite(dT[:, f]).sum() >= 20 and np.isfinite(dC[:, f]).sum() >= 20 else np.nan
            for f, c in enumerate(codes)}
    traj = []
    for t in range(C, R + 1):
        cur = rct.last_observed(X, n_cont, t, t)
        for f, c in enumerate(codes):
            a, b = cur[ti, f] - base[ti, f], cur[ci, f] - base[ci, f]
            if np.isfinite(a).sum() >= 20 and np.isfinite(b).sum() >= 20:
                traj.append(dict(variable=LABEL[c], hour=6 * t + 3, did=float(np.nanmean(a) - np.nanmean(b)),
                                 n_T=int(np.isfinite(a).sum()), n_C=int(np.isfinite(b).sum())))
    return prof, pd.DataFrame(traj)


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def profile_matrix(sim: pd.DataFrame, obs: pd.DataFrame):
    """sim/obs: (actions x variables). Returns raw and centered correlation matrices + permutation p-values."""
    acts = [a for a in GL if a in sim.index and a in obs.index]
    S, O = sim.loc[acts], obs.loc[acts].reindex(columns=sim.columns)
    out = {}
    for kind, (Sx, Ox) in {"raw": (S, O), "centered": (S - S.mean(0), O - O.mean(0))}.items():
        M = np.array([[_corr(Sx.loc[a].values, Ox.loc[b].values) for b in acts] for a in acts])
        tr = np.nanmean(np.diag(M))
        perms = [np.nanmean([M[i, p[i]] for i in range(len(acts))]) for p in itertools.permutations(range(len(acts)))]
        out[kind] = dict(M=pd.DataFrame(M, index=acts, columns=acts), diag_mean=float(tr),
                         offdiag_mean=float(np.nanmean(M[~np.eye(len(acts), dtype=bool)])),
                         perm_p=float(np.mean(np.array(perms) >= tr - 1e-12)),
                         n_diag_rowmax=int(sum(np.nanargmax(M[i]) == i for i in range(len(acts)) if np.isfinite(M[i]).any())))
    return out


def run(site, model_name, out_root=OUT, n_boot=100):
    t0 = time.time()
    cfg = SITES[site]
    out = out_root / site / model_name
    out.mkdir(parents=True, exist_ok=True)
    ck = resolve_ckpt(cfg["runs"], model_name)
    model = load_checkpoint(ck)
    man = _manifest(site)
    codes = man["vital_codes"] + man["lab_codes"]
    n_cont = len(codes)
    fidx = rct.feature_index(codes)
    sp = {s: Split(cfg["proc"], s) for s in ["train", "test"]}
    co = rct.cohort_arrays(model, sp["test"])
    probes = rct.fit_outcome_probes(model, sp["train"], co, cfg["outcomes"])
    log.info("[%s/%s] %s, %d eligible admissions", site, model_name, ck.name, len(co["Z"]))

    # 1. matching balance comparison + RCT read-outs
    res, bal_rows = {}, []
    for d in range(5):
        r_main = rct.experiment_for_drug(model, co, d, n_cont, fidx, probes, n_boot=n_boot, matching=MATCHING)
        r_core = rct.experiment_for_drug(model, co, d, n_cont, fidx, probes, n_boot=2, matching="core")
        res[d] = r_main
        for tag, r in [("mahalanobis (core)", r_core), (MATCHING, r_main)]:
            b = r["balance"].assign(drug=GL[d], method=tag, n_pairs=len(r["pairs"]))
            bal_rows.append(b)
    pd.concat(bal_rows).to_csv(out / "matching_balance.csv", index=False)
    pd.DataFrame([r["summary"] for r in res.values()]).to_csv(out / "rct_summary.csv", index=False)
    for k in ["probes", "traj", "features"]:
        parts = [r[k] for r in res.values() if len(r[k])]
        if parts:
            pd.concat(parts).to_csv(out / f"rct_{k}.csv", index=False)
    if model.action_conditioned:
        rct.specificity_matrix(model, co, res, n_cont, fidx).to_csv(out / "rct_spec.csv", index=False)
    log.info("[%s/%s] RCT done %.1f min", site, model_name, (time.time() - t0) / 60)

    # 2. population physiological response
    sens, lat, pop_idx = population_response(model, co, codes, man)
    sens.to_csv(out / "sensitivity_traj.csv", index=False)
    read = sens[sens.hour == sens.hour.max()]
    sim_pop = read.pivot(index="action", columns="variable", values="delta_z").reindex(GL)

    # 3. observed DiD per cohort, and cohort-conditional simulated profiles (a applied to cohort b's controls)
    obs_prof, obs_traj, cond = {}, [], []
    for d, r in res.items():
        if len(r["pairs"]) < 20:
            continue
        prof, tr = observed_did(co, r["pairs"], r["t_idx"], r["c_idx"], n_cont, codes)
        obs_prof[GL[d]] = prof
        obs_traj.append(tr.assign(cohort=GL[d]))
        for d2 in range(5):
            arms, _, _ = rct.run_arms(model, co, r["pairs"], r["t_idx"], r["c_idx"], d, d_apply=d2)
            dd = decode(model, arms["C_do"][:, -1]) - decode(model, arms["C_fact"][:, -1])
            cond.append(dict(cohort=GL[d], applied=GL[d2], **{LABEL[c]: float(dd[:, f].mean()) for f, c in enumerate(codes)}))
    obs = pd.DataFrame(obs_prof).T.reindex(columns=sim_pop.columns)
    obs.to_csv(out / "observed_did_profile.csv")
    sim_pop.to_csv(out / "simulated_profile_population.csv")
    if obs_traj:
        pd.concat(obs_traj).to_csv(out / "observed_did_traj.csv", index=False)
    cond = pd.DataFrame(cond)
    cond.to_csv(out / "simulated_profile_conditional.csv", index=False)

    # 4. diagonal profile matrices
    pm = profile_matrix(sim_pop, obs)
    summ = {}
    for kind, v in pm.items():
        v["M"].to_csv(out / f"profile_matrix_{kind}.csv")
        summ[f"profile_{kind}"] = {k: v[k] for k in ["diag_mean", "offdiag_mean", "perm_p", "n_diag_rowmax"]}
    # cohort-conditional: M[a, b] = corr(sim of a applied to cohort b controls, observed profile of cohort b)
    if len(cond):
        Mc = pd.DataFrame(np.nan, index=GL, columns=GL)
        for _, rr in cond.iterrows():
            if rr.cohort in obs.index:
                Mc.loc[rr.applied, rr.cohort] = _corr(rr[list(obs.columns)].values.astype(float), obs.loc[rr.cohort].values)
        Mc.to_csv(out / "profile_matrix_conditional.csv")
        dg = np.diag(Mc.values)
        summ["profile_conditional"] = dict(diag_mean=float(np.nanmean(dg)),
                                           offdiag_mean=float(np.nanmean(Mc.values[~np.eye(5, dtype=bool)])),
                                           n_diag_colmax=int(sum(np.nanargmax(Mc.values[:, j]) == j for j in range(5)
                                                                 if np.isfinite(Mc.values[:, j]).any())))

    # 5. generic-direction analysis
    fin = {g: v[:, -1] for g, v in lat.items()}
    mean_eff = {g: v.mean(0) for g, v in fin.items()}
    cos_lat = pd.DataFrame([[rct.cosine(mean_eff[a], mean_eff[b]) for b in GL] for a in GL], index=GL, columns=GL)
    cos_lat.to_csv(out / "effect_cosine_latent.csv")
    cos_dec = pd.DataFrame([[rct.cosine(sim_pop.loc[a].values, sim_pop.loc[b].values) for b in GL] for a in GL],
                           index=GL, columns=GL)
    cos_dec.to_csv(out / "effect_cosine_decoded.csv")
    sev = {}
    for name, pr in probes.items():
        w = pr["clf"].coef_[0] / pr["scaler"].scale_          # direction of increasing risk in z-space
        sev[name] = w
    zc = co["Z"][pop_idx, R]
    trA = sp["train"].act["act5"]
    mag = []
    for d, g in enumerate(GL):
        e = mean_eff[g]
        row = dict(action=g, latent_effect_norm=float(np.linalg.norm(e)),
                   per_patient_effect_norm=float(np.linalg.norm(fin[g], axis=1).mean()),
                   decoded_effect_norm=float(np.linalg.norm(np.nan_to_num(sim_pop.loc[g].values))),
                   prevalence_bins=float(trA[:, d].mean()), latent_state_sd=float(np.linalg.norm(zc.std(0))))
        for name, w in sev.items():
            row[f"cos_with_{name}_direction"] = rct.cosine(e, w)
        mag.append(row)
    pd.DataFrame(mag).to_csv(out / "effect_magnitude.csv", index=False)
    common = np.mean([mean_eff[g] / (np.linalg.norm(mean_eff[g]) + 1e-9) for g in GL], 0)
    summ["generic_direction"] = dict(mean_offdiag_cos_latent=float(np.nanmean(cos_lat.values[~np.eye(5, dtype=bool)])),
                                     mean_offdiag_cos_decoded=float(np.nanmean(cos_dec.values[~np.eye(5, dtype=bool)])),
                                     share_of_common_direction={g: float(rct.cosine(mean_eff[g], common) ** 2) for g in GL})

    # 6. clinical expectations
    exp_rows = []
    for g, ex in EXPECTED.items():
        for var, sgn in ex.items():
            if var not in sim_pop.columns:
                continue
            s_ = sim_pop.loc[g, var]; o_ = obs.loc[g, var] if g in obs.index else np.nan
            rr = sens[(sens.action == g) & (sens.variable == var) & (sens.hour == sens.hour.max())].iloc[0]
            exp_rows.append(dict(action=g, variable=var, expected=sgn, simulated=s_, sim_lo=rr.lo_z, sim_hi=rr.hi_z,
                                 sim_native=rr.delta_native, unit=rr.unit, observed=o_,
                                 sim_agrees=bool(np.sign(s_) == sgn), obs_agrees=bool(np.sign(o_) == sgn) if np.isfinite(o_) else None,
                                 sim_significant=bool((rr.lo_z > 0) or (rr.hi_z < 0))))
    exp = pd.DataFrame(exp_rows)
    exp.to_csv(out / "clinical_expectations.csv", index=False)
    summ["clinical"] = dict(n=len(exp), sim_agree=float(exp.sim_agrees.mean()) if len(exp) else np.nan,
                            sim_agree_and_significant=float((exp.sim_agrees & exp.sim_significant).mean()) if len(exp) else np.nan,
                            obs_agree=float(exp.obs_agrees.dropna().astype(float).mean()) if len(exp) else np.nan)
    summ.update(site=site, model=model_name, checkpoint=str(ck), minutes=(time.time() - t0) / 60,
                probe_auroc={k: p["test_auroc"] for k, p in probes.items()})
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    log.info("[%s/%s] done in %.1f min", site, model_name, (time.time() - t0) / 60)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=list(SITES))
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    run(a.site, a.model)
