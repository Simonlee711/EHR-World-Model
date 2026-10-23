"""
Counterfactual treatment experiments (velocity model, action-conditioned).

    # single-drug screen on 50 positive patients, keep top 3 by |effect|
    python scripts/counterfactual.py --data-dir ... --run-dir runs/velocity --experiment screen --task has_Sepsis

    # greedy one-step MPC: null / observed / MPC-Treat / MPC-Withhold
    python scripts/counterfactual.py ... --experiment greedy-mpc --task has_Sepsis

    # random-shooting MPC (H-step lookahead) vs observed treatment
    python scripts/counterfactual.py ... --experiment shooting-mpc --task readmit_30d

Tasks are EHRSHOT-style labels (cohort.extract_ehrshot_labels), e.g.
has_Sepsis, icu_transfer, readmit_30d, long_los. Risk curves are written to
<run-dir>/counterfactual_<experiment>_<task>.csv (one row per patient × arm × step).

All arms are free-running: seeded with the first --context observed steps, then
rolled out autoregressively on the model's own predictions. Steps < context are
identical across arms (``simulated`` = False in the CSV).
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ehr_world_model import (build_cohort, build_patient_trajectories, collect_patient_cf_data, extract_ehrshot_labels,
                             get_device, get_traj_risk_series, greedy_mpc_rollout, load_checkpoint, observed_rollout,
                             screen_single_drugs, shooting_mpc_rollout, counterfactual_rollouts, train_risk_probe)
from ehr_world_model.counterfactual import drug_vec
from ehr_world_model.inference import HOURS_PER_STEP, get_label


def positive_patients(cf_data, col, label_map, min_len):
    return [pid for pid, d in cf_data.items()
            if d["length"] >= min_len and get_label(pid, col, label_map) == 1.0]


def high_risk_spread(cf_data, pids, probe, t_max, n, min_risk=0.55):
    """Positive patients with mean true risk ≥ min_risk, evenly spread across the risk range."""
    pool = []
    for pid in pids:
        T_i = min(cf_data[pid]["length"], t_max)
        mu = float(np.mean(get_traj_risk_series(probe, cf_data[pid]["z_true"][:T_i])))
        if mu >= min_risk:
            pool.append((mu, pid))
    pool.sort(key=lambda x: x[0], reverse=True)
    if len(pool) >= n:
        step = (len(pool) - 1) / max(n - 1, 1)
        return [pool[round(i * step)][1] for i in range(n)]
    return [p[1] for p in pool]


def records(task, pid, arm, risk, context):
    return [dict(task=task, patient_id=pid, intervention=arm, step=i, hour=i * HOURS_PER_STEP,
                 simulated=i >= context, risk=float(r))
            for i, r in enumerate(risk)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--experiment", choices=["screen", "greedy-mpc", "shooting-mpc"], required=True)
    p.add_argument("--task", default="has_Sepsis")
    p.add_argument("--n-patients", type=int, default=None, help="default: 50 (screen/shooting), 5 (greedy)")
    p.add_argument("--t-max", type=int, default=None, help="rollout steps; default 16 (screen) / 14 (MPC)")
    p.add_argument("--context", type=int, default=2, help="observed steps used to seed each rollout (≥2)")
    p.add_argument("--top-k", type=int, default=3, help="screen: drugs to keep")
    p.add_argument("--n-cand", type=int, default=80, help="greedy-mpc: candidate actions per step")
    p.add_argument("--horizon", type=int, default=4, help="shooting-mpc: lookahead H")
    p.add_argument("--n-seqs", type=int, default=200, help="shooting-mpc: sequences per step")
    p.add_argument("--max-train", type=int, default=10000)
    p.add_argument("--max-val", type=int, default=3000)
    p.add_argument("--max-test", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = get_device(args.device)
    run_dir = Path(args.run_dir)

    cohort = build_cohort(args.data_dir, run_dir, device, seed=args.seed,
                          max_patients={"train": args.max_train, "tuning": args.max_val, "held_out": args.max_test})
    id_col = cohort["id_col"]
    model = load_checkpoint(run_dir / "model.pt", device)
    if not hasattr(model, "_step") or not model.action_conditioned:
        sys.exit("Counterfactuals need an action-conditioned velocity model (--arch velocity, without --no-action).")

    # Labels + risk probe trained on train-split encoder trajectories
    ehr_tr = extract_ehrshot_labels(cohort["dfs"]["train"], id_col).set_index(id_col).to_dict(orient="index")
    ehr_te = extract_ehrshot_labels(cohort["dfs"]["test"],  id_col).set_index(id_col).to_dict(orient="index")
    trajs_train = build_patient_trajectories(model, cohort["loaders"]["train_eval"], cohort["ds"]["train"])
    probe = train_risk_probe(args.task, trajs_train, ehr_tr)
    if probe is None:
        sys.exit(f"Too few training labels for '{args.task}'.")

    t_max = args.t_max or (16 if args.experiment == "screen" else 14)
    cf_data = collect_patient_cf_data(model, cohort["loaders"]["test"], cohort["ds"]["test"], max_steps=max(t_max, 16) + 2)
    pos = positive_patients(cf_data, args.task, ehr_te, min_len=max(4, args.context + 2))
    n_act = cohort["manifest"].n_actions
    m = cohort["manifest"]
    code_labels = [c.split("//")[-1][:28] for c in m.med_codes + m.infusion_codes + m.proc_codes]
    logging.info("%d positive test patients for %s", len(pos), args.task)

    rows = []
    if args.experiment == "screen":
        rng = np.random.default_rng(42)
        pids = rng.choice(pos, size=min(args.n_patients or 50, len(pos)), replace=False).tolist()
        res = screen_single_drugs(model, probe, cf_data, pids, n_act, t_max, args.context)
        top = sorted(range(n_act), key=lambda i: abs(res["effects"][i]), reverse=True)[:args.top_k]
        print(f"No-treatment mean risk: {res['null_mean']:.3f}")
        for r, di in enumerate(top):
            print(f"  #{r + 1} [{di:3d}] {code_labels[di]:30s} effect={res['effects'][di]:+.4f}")
        pd.DataFrame({"action_idx": range(n_act), "code": code_labels, "mean_risk": res["drug_means"],
                      "effect": res["effects"]}).to_csv(run_dir / f"counterfactual_screen_effects_{args.task}.csv", index=False)
        for pid in pids:
            d = cf_data[pid]
            T_roll = min(d["length"], t_max); T_steps = T_roll - 1
            ivs = {"null": torch.zeros(T_steps, n_act), "observed": d["actions"][:T_steps]}
            ivs.update({f"top{r}_{code_labels[di]}": drug_vec(di, T_steps, n_act) for r, di in enumerate(top)})
            for arm, z in counterfactual_rollouts(model, d["z_true"][:T_roll], ivs, args.context).items():
                rows += records(args.task, pid, arm, get_traj_risk_series(probe, z), args.context)

    elif args.experiment == "greedy-mpc":
        pids = high_risk_spread(cf_data, pos, probe, t_max, args.n_patients or 5)
        for pid in pids:
            d = cf_data[pid]
            T_roll = min(d["length"], t_max)
            for strat in ["null", "observed", "minimize", "maximize"]:
                z = greedy_mpc_rollout(model, d["z_true"][:T_roll], d["actions"][:T_roll - 1], probe, n_act, strat, args.n_cand, args.context)
                rows += records(args.task, pid, strat, get_traj_risk_series(probe, z), args.context)

    else:  # shooting-mpc
        pids = high_risk_spread(cf_data, pos, probe, t_max, args.n_patients or 50)
        for pid in pids:
            d = cf_data[pid]
            T_roll = min(d["length"], t_max)
            z_obs = observed_rollout(model, d["z_true"][:T_roll], d["actions"][:T_roll - 1], args.context)
            z_mpc = shooting_mpc_rollout(model, d["z_true"][:T_roll], probe, n_act, args.horizon, args.n_seqs, context=args.context)
            rows += records(args.task, pid, "observed", get_traj_risk_series(probe, z_obs), args.context)
            rows += records(args.task, pid, "mpc_treat", get_traj_risk_series(probe, z_mpc), args.context)

    df = pd.DataFrame(rows)
    out = run_dir / f"counterfactual_{args.experiment}_{args.task}.csv"
    df.to_csv(out, index=False)
    print(df[df["simulated"]].groupby("intervention")["risk"].mean().round(3).to_string())
    logging.info("saved %s (%d patients)", out, df["patient_id"].nunique() if len(df) else 0)


if __name__ == "__main__":
    main()
