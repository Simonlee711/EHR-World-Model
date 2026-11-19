"""
Emulated-RCT evaluation of action conditioning.

Question
    If we *apply* an intervention to patients who did not receive it, does the world
    model move their predicted future toward comparable patients who did receive it?

Design (per action group d, held-out admissions with >= 60 h)
    bins 0-3 (0-24 h)  : context - nobody has received d
    bins 4-8 (24-54 h) : intervention window - treated T start d here, controls C never get d
    bin  9   (54-60 h) : read-out
    Cohorts differ in a SINGLE intervention:
      * exact matching on the co-treatment pattern of the other four actions in the window, then
      * 1:1 nearest-neighbour propensity matching (caliper 0.2 SD of the logit, no replacement)
        on pre-treatment covariates (encoder state 18-24 h, mean state 0-24 h, last observed
        value of every vital/lab, age, sex) within each stratum.
    Arms (all start from the patient's own encoder states z_2, z_3, free-running for 6 steps):
      T_fact  treated, observed actions           C_fact  control, observed actions
      C_do    control + d on the matched treated partner's schedule   ("apply")
      T_undo  treated with d removed                                   ("withhold")
    Read-outs
      gap closed      1 - E(C_do, T_fact) / E(C_fact, T_fact)   (energy distance, latent, 54-60 h)
      direction       cos( E[z(C_do) - z(C_fact)],  E[z_T - z_C] observed (encoder) at 54-60 h )
      outcome probes  long LOS / in-hospital mortality probes on states 24-60 h; risk per arm and
                      over time vs the outcome rates actually observed in the matched cohorts
      decoded effects per-variable simulated effect vs observed matched difference
      specificity     apply d' to the cohort defined by d (5x5): gap closed and direction
    Matching removes measured confounding only; these are model-based contrasts.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .actions import GROUP_LABELS, GROUPS
from .data import Split
from .evaluate import decode, encode, padded, rollout
from .features import LABEL

C_CTX, H = 4, 6
R = C_CTX + H - 1                # read-out bin (9)
T_NEED = R + 1
OUTCOMES = {"Long LOS (>7d)": "adm_long_los", "In-hospital mortality": "adm_mortality"}


def feature_index(state_codes: List[str]) -> Dict[str, int]:
    return {LABEL.get(c, c): i for i, c in enumerate(state_codes)}


def cohort_arrays(model, split: Split, action_key: str = "act5") -> Dict:
    idx = np.where(split.adm.n_bins.values >= T_NEED)[0]
    X, A, _ = padded(split, idx, T_NEED, action_key)
    Z = encode(model, X)
    return dict(idx=idx, X=X, A=A, Z=Z, adm=split.adm.iloc[idx].reset_index(drop=True))


def last_observed(X, n_cont, t0, t1):
    v, m = X[..., :n_cont], X[..., n_cont:2 * n_cont] > 0
    out = np.full((len(X), n_cont), np.nan, np.float32)
    for t in range(t0, t1 + 1):
        out = np.where(m[:, t], v[:, t], out)
    return out


def covariates(co, n_cont, fidx, d):
    X, A, Z, adm = co["X"], co["A"], co["Z"], co["adm"]
    lo = last_observed(X, n_cont, 0, C_CTX - 1)
    interp = pd.DataFrame({k: lo[:, i] for k, i in fidx.items()})
    interp["Age"] = adm["age"].values
    interp["Male"] = (adm["sex"] == "M").astype(float).values
    for g in range(A.shape[2]):
        if g != d:
            interp[f"{GROUP_LABELS[GROUPS[g]]} (0-24h)"] = A[:, :C_CTX, g].max(1)
    model_cov = np.concatenate([Z[:, C_CTX - 1], Z[:, :C_CTX].mean(1)], 1)
    filled = interp.fillna(interp.mean()).fillna(0).values
    return interp, np.concatenate([model_cov, filled, interp.isna().values.astype(np.float32)], 1)


def smd(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def match(logit_t, logit_c, strata_t=None, strata_c=None, caliper_sd=0.2, seed=0):
    """Greedy 1:1 NN matching on the logit, without replacement, within exact strata."""
    cal = caliper_sd * np.std(np.concatenate([logit_t, logit_c]))
    rng = np.random.default_rng(seed)
    if strata_t is None:
        strata_t, strata_c = np.zeros(len(logit_t), int), np.zeros(len(logit_c), int)
    avail = np.ones(len(logit_c), bool)
    pairs = []
    for i in rng.permutation(len(logit_t)):
        dist = np.where(avail & (strata_c == strata_t[i]), np.abs(logit_c - logit_t[i]), np.inf)
        j = int(np.argmin(dist))
        if dist[j] <= cal:
            pairs.append((i, j)); avail[j] = False
    return np.array(pairs, dtype=int).reshape(-1, 2)


def match_mahal(logit_t, logit_c, strata_t, strata_c, C_t, C_c, caliper_sd=0.1, seed=0, replace=False):
    """Mahalanobis matching within propensity caliper and exact strata (1:1, no replacement).

    C_t / C_c: standardised, whitened interpretable covariates. Among controls in the same stratum whose
    propensity logit is within caliper_sd * SD, pick the nearest in Mahalanobis distance."""
    cal = caliper_sd * np.std(np.concatenate([logit_t, logit_c]))
    rng = np.random.default_rng(seed)
    avail = np.ones(len(logit_c), bool)
    pairs = []
    # hardest-to-match first (highest propensity), the usual greedy order for treated-rich regions
    for i in np.argsort(-logit_t + 1e-9 * rng.random(len(logit_t))):
        cand = np.flatnonzero(avail & (strata_c == strata_t[i]) & (np.abs(logit_c - logit_t[i]) <= cal))
        if not len(cand):
            continue
        dist = ((C_c[cand] - C_t[i]) ** 2).sum(1)
        j = cand[int(np.argmin(dist))]
        pairs.append((i, j))
        if not replace:
            avail[j] = False
    return np.array(pairs, dtype=int).reshape(-1, 2)


def _whiten(X):
    """Standardise and decorrelate (Mahalanobis metric = Euclidean on the result)."""
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    cov = np.cov(Xs, rowvar=False) + 1e-3 * np.eye(Xs.shape[1])
    w, V = np.linalg.eigh(cov)
    return Xs @ V / np.sqrt(np.maximum(w, 1e-6))


def energy_distance(X, Y):
    md = lambda a, b: cdist(a, b).mean()
    return float(2 * md(X, Y) - md(X, X) - md(Y, Y))


def subsample_ci(stat, idx, rng, B=100, frac=0.5):
    """Point estimate + 95% CI by half-sampling without replacement (rescaled by sqrt(m/n)).
    Avoids the duplicate-sample bias that with-replacement resampling causes for energy distances."""
    pt = np.array(stat(idx), dtype=float)
    m = max(10, int(len(idx) * frac))
    vals = np.array([stat(rng.choice(idx, m, replace=False)) for _ in range(B)], dtype=float)
    med = np.nanmedian(vals, 0)
    sc = np.sqrt(m / len(idx))
    lo = pt - (np.nanpercentile(vals, 97.5, 0) - med) * sc
    hi = pt + (med - np.nanpercentile(vals, 2.5, 0)) * sc
    return pt, lo, hi


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


# ── outcome probes on states 24-60 h ───────────────────────────────────────────

def fit_outcome_probes(model, train: Split, test_co: Dict, outcomes: Dict[str, str] = OUTCOMES) -> Dict:
    tr = cohort_arrays(model, train)
    probes = {}
    for name, col in outcomes.items():
        if col not in tr["adm"]:
            continue
        y = tr["adm"][col].values.astype(float)
        ok = np.isfinite(y)
        Zs = tr["Z"][ok, C_CTX:T_NEED].reshape(-1, tr["Z"].shape[-1])
        ys = np.repeat(y[ok], T_NEED - C_CTX)
        sc = StandardScaler().fit(Zs)
        clf = LogisticRegression(C=0.1, max_iter=500).fit(sc.transform(Zs), ys)
        yt = test_co["adm"][col].values.astype(float)
        okt = np.isfinite(yt)
        auc = roc_auc_score(yt[okt], clf.predict_proba(sc.transform(test_co["Z"][okt, R]))[:, 1])
        probes[name] = dict(scaler=sc, clf=clf, col=col, test_auroc=auc)
    return probes


def probe_risk(probe, Z):
    sh = Z.shape
    return probe["clf"].predict_proba(probe["scaler"].transform(Z.reshape(-1, sh[-1])))[:, 1].reshape(sh[:-1])


# ── experiment ─────────────────────────────────────────────────────────────────

def transplant(A_dst, A_src, d_src, d_dst):
    out = A_dst.copy()
    out[:, C_CTX:R, d_dst] = np.maximum(out[:, C_CTX:R, d_dst], A_src[:, C_CTX:R, d_src])
    return out


def generate(model, Zp, Ap):
    """Arm trajectory for bins 4..9 given the patient's own encoder states Zp (n, 10, D) and actions Ap (n, 10, A):
    free-running rollout seeded with z_2, z_3 (the model never sees the true state after bin 3)."""
    return rollout(model, Zp[:, C_CTX - 2:C_CTX], Ap[:, C_CTX - 1:R])


def run_arms(model, co, pairs, t_idx, c_idx, d, d_apply=None):
    """Predicted trajectories (n_pairs, 6, D) for bins 4..9 per arm, plus encoder states."""
    d_apply = d if d_apply is None else d_apply
    Z, A = co["Z"], co["A"]
    ti, ci = t_idx[pairs[:, 0]], c_idx[pairs[:, 1]]
    A_t, A_c = A[ti], A[ci]
    A_tundo = A_t.copy(); A_tundo[:, C_CTX:R, d] = 0
    arms = {"T_fact": generate(model, Z[ti], A_t), "C_fact": generate(model, Z[ci], A_c),
            "C_do": generate(model, Z[ci], transplant(A_c, A_t, d, d_apply)),
            "T_undo": generate(model, Z[ti], A_tundo),
            "T_real": Z[ti, C_CTX:T_NEED], "C_real": Z[ci, C_CTX:T_NEED]}
    return arms, ti, ci


def one_step_effect(model, co, pairs, t_idx, c_idx, d, probes=None, d_apply=None, n_boot=100, seed=0):
    """One-step emulated RCT at the treated partner's drug-onset bin s.

    For each matched pair: z_T(s+1) is the treated patient's real next state; for the control the model
    predicts the next state from the control's own true history (z_C(s), z_C(s)-z_C(s-1)) with its observed
    action a_C(s) ("fact") and with drug d_apply added ("do"). Returns gap closed toward the treated
    group's real next states, the direction cosine of the simulated vs observed one-step change, and
    outcome-probe risks.
    """
    import torch
    d_apply = d if d_apply is None else d_apply
    Z, A = co["Z"], co["A"]
    ti, ci = t_idx[pairs[:, 0]], c_idx[pairs[:, 1]]
    s = np.array([C_CTX + int(np.argmax(A[i, C_CTX:R, d] > 0)) for i in ti])
    zc, zc_prev, zt_next, zc_next = Z[ci, s], Z[ci, s - 1], Z[ti, s + 1], Z[ci, s + 1]
    zt, zt_prev = Z[ti, s], Z[ti, s - 1]
    a_fact = A[ci, s].copy()
    a_do = a_fact.copy(); a_do[:, d_apply] = 1.0
    dev = next(model.parameters()).device
    T = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(dev)
    with torch.no_grad():
        p_fact = model._step(T(zc), T(zc - zc_prev), T(a_fact))[0].cpu().numpy() if model.action_conditioned else \
            model._step(T(zc), T(zc - zc_prev), T(np.zeros_like(a_fact)))[0].cpu().numpy()
        p_do = model._step(T(zc), T(zc - zc_prev), T(a_do))[0].cpu().numpy() if model.action_conditioned else p_fact.copy()
        p_tfact = model._step(T(zt), T(zt - zt_prev), T(A[ti, s]))[0].cpu().numpy() if model.action_conditioned else \
            model._step(T(zt), T(zt - zt_prev), T(np.zeros_like(a_fact)))[0].cpu().numpy()
    sd = np.concatenate([zt_next, p_fact]).std(0) + 1e-6
    rng = np.random.default_rng(seed)
    n = len(ti)
    sub = rng.choice(n, min(n, 1500), replace=False)

    def stats(ix):
        e0 = energy_distance(p_fact[ix] / sd, zt_next[ix] / sd)
        e1 = energy_distance(p_do[ix] / sd, zt_next[ix] / sd)
        sim = (p_do[ix] - p_fact[ix]).mean(0)
        obs = ((zt_next[ix] - zt[ix]) - (zc_next[ix] - zc[ix])).mean(0)     # observed change difference
        return 1 - e1 / e0, cosine(sim, obs), cosine(sim, (zt_next[ix] - zc_next[ix]).mean(0))
    pt, lo_, hi_ = subsample_ci(stats, sub, rng, max(n_boot, 2))
    q = lambda k: (float(lo_[k]), float(hi_[k]))
    out = dict(gap1=pt[0], gap1_lo=q(0)[0], gap1_hi=q(0)[1], cos1_change=pt[1], cos1_change_lo=q(1)[0],
               cos1_change_hi=q(1)[1], cos1_state=pt[2], cos1_state_lo=q(2)[0], cos1_state_hi=q(2)[1])
    rows = []
    for name, pr in (probes or {}).items():
        rk = {k: probe_risk(pr, v) for k, v in [("C_fact", p_fact), ("C_do", p_do), ("T_fact", p_tfact),
                                                ("T_real", zt_next), ("C_real", zc_next)]}
        b = np.array([rk["C_do"][ix].mean() - rk["C_fact"][ix].mean() for ix in (rng.integers(0, n, n) for _ in range(n_boot))])
        rows.append(dict(outcome=name, **{f"risk1_{k}": float(v.mean()) for k, v in rk.items()},
                         shift1=float(rk["C_do"].mean() - rk["C_fact"].mean()),
                         shift1_lo=float(np.percentile(b, 2.5)), shift1_hi=float(np.percentile(b, 97.5))))
    return out, pd.DataFrame(rows)


def experiment_for_drug(model, co, d, n_cont, fidx, probes, n_boot=100, seed=0, min_obs=20,
                        exact_cotreatment=True, matching="psm") -> Dict:
    """matching: 'psm' (propensity NN, caliper 0.2) or 'mahal' (Mahalanobis within a 0.1-SD propensity caliper)."""
    A = co["A"]
    pre = A[:, :C_CTX, d].max(1) == 0
    in_win = A[:, C_CTX:R, d].max(1) > 0
    t_idx, c_idx = np.where(pre & in_win)[0], np.where(pre & ~in_win)[0]
    others = [g for g in range(A.shape[2]) if g != d]
    if exact_cotreatment:
        bits = A[:, C_CTX:R, others].max(1)          # co-treatment pattern in the window (16 strata)
        strata = (bits * (2 ** np.arange(bits.shape[1]))).sum(1).astype(int)
    else:
        strata = np.zeros(len(A), int)
    interp, P = covariates(co, n_cont, fidx, d)
    y = np.r_[np.ones(len(t_idx)), np.zeros(len(c_idx))]
    Ps = StandardScaler().fit_transform(P[np.r_[t_idx, c_idx]])
    ps = LogisticRegression(C=0.05, max_iter=1000).fit(Ps, y).decision_function(Ps)
    if matching == "psm_r":      # propensity NN within strata, caliper 0.05 SD, with replacement
        pairs = match_mahal(ps[:len(t_idx)], ps[len(t_idx):], strata[t_idx], strata[c_idx],
                            ps[:len(t_idx), None], ps[len(t_idx):, None], caliper_sd=0.05, seed=seed, replace=True)
    elif matching in ("mahal", "core"):
        sub_ = interp.iloc[np.r_[t_idx, c_idx]]
        if matching == "core":   # frequently observed covariates + propensity logit, with replacement
            core = [c for c in interp.columns if sub_[c].notna().mean() >= 0.5]
            M_ = np.concatenate([sub_[core].fillna(sub_[core].mean()).values, 2.0 * ps[:, None]], 1)
            Cw = _whiten(M_)
            pairs = match_mahal(ps[:len(t_idx)], ps[len(t_idx):], strata[t_idx], strata[c_idx],
                                Cw[:len(t_idx)], Cw[len(t_idx):], caliper_sd=0.2, seed=seed, replace=True)
        else:
            Cw = _whiten(np.concatenate([sub_.fillna(sub_.mean()).fillna(0).values,
                                         sub_.isna().values.astype(float)], 1))
            pairs = match_mahal(ps[:len(t_idx)], ps[len(t_idx):], strata[t_idx], strata[c_idx],
                                Cw[:len(t_idx)], Cw[len(t_idx):], seed=seed)
    else:
        pairs = match(ps[:len(t_idx)], ps[len(t_idx):], strata[t_idx], strata[c_idx], seed=seed)
    bal = pd.DataFrame([dict(covariate=c, smd_before=smd(interp[c].values[t_idx], interp[c].values[c_idx]),
                             smd_after=smd(interp[c].values[t_idx[pairs[:, 0]]], interp[c].values[c_idx[pairs[:, 1]]]),
                             n_obs_min=int(min(np.isfinite(interp[c].values[t_idx[pairs[:, 0]]]).sum(),
                                               np.isfinite(interp[c].values[c_idx[pairs[:, 1]]]).sum())) if len(pairs) else 0)
                        for c in interp.columns])
    rel = bal[bal.n_obs_min >= 30]
    res = dict(drug=GROUP_LABELS[GROUPS[d]], d=d, n_treated=len(t_idx), n_control=len(c_idx), n_pairs=len(pairs),
               n_strata=int(len(np.unique(strata[t_idx]))))
    if len(pairs) < 20:
        return dict(summary=res, balance=bal, probes=pd.DataFrame(), traj=pd.DataFrame(), features=pd.DataFrame(),
                    pairs=pairs, t_idx=t_idx, c_idx=c_idx)

    arms, ti, ci = run_arms(model, co, pairs, t_idx, c_idx, d)
    fin = {k: v[:, -1] for k, v in arms.items()}
    sd = np.concatenate([fin["T_fact"], fin["C_fact"]]).std(0) + 1e-6
    S = {k: v / sd for k, v in fin.items()}
    rng = np.random.default_rng(seed)
    n = len(pairs)
    sub = rng.choice(n, min(n, 1500), replace=False)

    def stats(ix):
        e0 = energy_distance(S["C_fact"][ix], S["T_fact"][ix])
        e1 = energy_distance(S["C_do"][ix], S["T_fact"][ix])
        e3 = energy_distance(S["T_undo"][ix], S["C_fact"][ix])
        sim = (fin["C_do"][ix] - fin["C_fact"][ix]).mean(0)
        return (1 - e1 / e0, 1 - e3 / e0,
                cosine(sim, (fin["T_real"][ix] - fin["C_real"][ix]).mean(0)),
                cosine(sim, (fin["T_fact"][ix] - fin["C_fact"][ix]).mean(0)))
    point, lo_, hi_ = subsample_ci(stats, sub, rng, n_boot)
    q = lambda k: (float(lo_[k]), float(hi_[k]))
    eff = np.linalg.norm((fin["C_do"] - fin["C_fact"]).mean(0)) / (np.linalg.norm((fin["T_real"] - fin["C_real"]).mean(0)) + 1e-9)
    res.update(max_smd_before=float(np.nanmax(np.abs(rel.smd_before))), max_smd_after=float(np.nanmax(np.abs(rel.smd_after))),
               mean_smd_after=float(np.nanmean(np.abs(rel.smd_after))),
               frac_smd_lt_0_1=float((np.abs(rel.smd_after) < 0.1).mean()), n_balance_covariates=int(len(rel)),
               gap_closed=point[0], gap_closed_lo=q(0)[0], gap_closed_hi=q(0)[1],
               gap_closed_withhold=point[1], gap_closed_withhold_lo=q(1)[0], gap_closed_withhold_hi=q(1)[1],
               cos_real=point[2], cos_real_lo=q(2)[0], cos_real_hi=q(2)[1], cos_sim=point[3],
               rel_effect_size=float(eff))
    one, one_probe = one_step_effect(model, co, pairs, t_idx, c_idx, d, probes, n_boot=n_boot, seed=seed)
    res.update(one)

    probe_rows, traj_rows = [], []
    for name, pr in probes.items():
        rk = {k: probe_risk(pr, v) for k, v in arms.items()}
        yT = co["adm"][pr["col"]].values[ti].astype(float)
        yC = co["adm"][pr["col"]].values[ci].astype(float)
        bt = np.array([[rk["C_do"][b, -1].mean() - rk["C_fact"][b, -1].mean(),
                        rk["T_undo"][b, -1].mean() - rk["T_fact"][b, -1].mean()]
                       for b in (rng.integers(0, n, n) for _ in range(n_boot))])
        probe_rows.append(dict(drug=res["drug"], outcome=name, obs_rate_T=np.nanmean(yT), obs_rate_C=np.nanmean(yC),
                               **{f"risk_{k}": float(v[:, -1].mean()) for k, v in rk.items()},
                               shift_apply=float(rk["C_do"][:, -1].mean() - rk["C_fact"][:, -1].mean()),
                               shift_apply_lo=float(np.percentile(bt[:, 0], 2.5)),
                               shift_apply_hi=float(np.percentile(bt[:, 0], 97.5)),
                               shift_withhold=float(rk["T_undo"][:, -1].mean() - rk["T_fact"][:, -1].mean()),
                               shift_withhold_lo=float(np.percentile(bt[:, 1], 2.5)),
                               shift_withhold_hi=float(np.percentile(bt[:, 1], 97.5))))
        for k, v in rk.items():
            for s in range(v.shape[1]):
                traj_rows.append(dict(drug=res["drug"], outcome=name, arm=k, hour=6 * (C_CTX + s) + 3,
                                      mean=float(v[:, s].mean()), sem=float(v[:, s].std() / np.sqrt(len(v)))))
    probe_df = pd.DataFrame(probe_rows)
    if len(probe_df) and len(one_probe):
        probe_df = probe_df.merge(one_probe, on="outcome", how="left")
    if len(probe_df):
        probe_df["obs_diff_T_minus_C"] = probe_df.obs_rate_T - probe_df.obs_rate_C
        probe_df["sim_diff_Tfact_minus_Cfact"] = probe_df.risk_T_fact - probe_df.risk_C_fact
        probe_df["frac_gap_closed"] = probe_df.shift_apply / probe_df.sim_diff_Tfact_minus_Cfact
        probe_df["direction_ok"] = np.sign(probe_df.shift_apply) == np.sign(probe_df.obs_diff_T_minus_C)
        if "shift1" in probe_df:
            probe_df["direction1_ok"] = np.sign(probe_df.shift1) == np.sign(probe_df.obs_diff_T_minus_C)

    X = co["X"]
    dec = {k: decode(model, fin[k]) for k in ["C_fact", "C_do", "T_fact", "T_undo"]}
    obs = last_observed(X, n_cont, R - 1, R)
    feat_rows = []
    for name, i in fidx.items():
        oT, oC = obs[ti, i], obs[ci, i]
        if np.isfinite(oT).sum() < min_obs or np.isfinite(oC).sum() < min_obs:
            continue
        feat_rows.append(dict(drug=res["drug"], feature=name, obs_effect=float(np.nanmean(oT) - np.nanmean(oC)),
                              sim_effect_apply=float((dec["C_do"][:, i] - dec["C_fact"][:, i]).mean()),
                              sim_effect_withhold=float((dec["T_fact"][:, i] - dec["T_undo"][:, i]).mean()),
                              n_obs_T=int(np.isfinite(oT).sum()), n_obs_C=int(np.isfinite(oC).sum())))
    feat = pd.DataFrame(feat_rows, columns=["drug", "feature", "obs_effect", "sim_effect_apply",
                                            "sim_effect_withhold", "n_obs_T", "n_obs_C"])
    res.update(decoded_r_apply=np.nan, decoded_sign_agree=np.nan)
    if len(feat) > 3 and feat.sim_effect_apply.std() > 0:
        res["decoded_r_apply"] = float(np.corrcoef(feat.obs_effect, feat.sim_effect_apply)[0, 1])
        res["decoded_sign_agree"] = float((np.sign(feat.obs_effect) == np.sign(feat.sim_effect_apply)).mean())
    return dict(summary=res, balance=bal, probes=probe_df, traj=pd.DataFrame(traj_rows), features=feat,
                pairs=pairs, t_idx=t_idx, c_idx=c_idx)


def specificity_matrix(model, co, results, n_cont, fidx) -> pd.DataFrame:
    """Cohort defined by drug d (rows) x drug d' applied to its controls (cols)."""
    rows = []
    for d, res in results.items():
        pairs, t_idx, c_idx = res["pairs"], res["t_idx"], res["c_idx"]
        if len(pairs) < 20:
            continue
        sub = np.arange(min(len(pairs), 1500))
        for d2 in range(len(GROUPS)):
            arms, _, _ = run_arms(model, co, pairs[sub], t_idx, c_idx, d, d_apply=d2)
            fin = {k: v[:, -1] for k, v in arms.items()}
            sd = np.concatenate([fin["T_fact"], fin["C_fact"]]).std(0) + 1e-6
            e0 = energy_distance(fin["C_fact"] / sd, fin["T_fact"] / sd)
            e1 = energy_distance(fin["C_do"] / sd, fin["T_fact"] / sd)
            sim = (fin["C_do"] - fin["C_fact"]).mean(0)
            one, _ = one_step_effect(model, co, pairs[sub], t_idx, c_idx, d, None, d_apply=d2, n_boot=1)
            rows.append(dict(cohort=GROUP_LABELS[GROUPS[d]], applied=GROUP_LABELS[GROUPS[d2]],
                             gap1=one["gap1"], cos1_change=one["cos1_change"], cos1_state=one["cos1_state"],
                             gap_closed=1 - e1 / e0 if e0 > 0 else np.nan,
                             cos_real=cosine(sim, (fin["T_real"] - fin["C_real"]).mean(0)),
                             effect_norm=float(np.linalg.norm(sim))))
    return pd.DataFrame(rows)
