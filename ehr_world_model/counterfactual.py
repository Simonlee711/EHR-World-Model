"""
Counterfactual treatment simulation with the action-conditioned
VelocityWorldModel (uses ``model._step``; not supported for the transformer).

Free-running (autoregressive) counterfactuals
---------------------------------------------
Each arm is seeded with the patient's first ``context`` observed encoder states
(``context ≥ 2`` so the initial velocity is real), then rolled forward purely on
the model's own predictions — the observed trajectory is never fed back in:

    z_cf[:context] = z_true[:context]
    dz             = z_true[context−1] − z_true[context−2]
    z_cf[t+1], dz  = _step(z_cf[t], dz, a_t)          for t ≥ context−1

All arms of a patient share the same seed, so differences between arms come
only from the actions. Risk is read out with a linear trajectory probe (see
``inference.train_risk_probe``).

Note: seeding with dz = 0 (a single context step) gives a constant trajectory,
because dz_{t+1} = F·dz_t stays at zero.

Experiments
-----------
* ``screen_single_drugs``  — each action tested as a one-hot intervention vs
                             no-treatment; ranks drugs by |mean risk change|.
* ``greedy_mpc_rollout``   — at each step pick, from N candidate action vectors,
                             the one minimising (or maximising) next-step risk.
* ``shooting_mpc_rollout`` — random-shooting MPC with H-step lookahead: sample
                             N action sequences, roll each H steps (batched),
                             execute the first action of the sequence with the
                             lowest cumulative risk (receding horizon).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List

import numpy as np
import torch

from .inference import ema_smooth, get_traj_risk_series

DEFAULT_CONTEXT = 2


# ── Per-patient encoder trajectories + observed actions ───────────────────────

@torch.no_grad()
def collect_patient_cf_data(model, aug_loader, dataset, max_steps: int = 18) -> Dict[str, Dict]:
    """{pid: {'z_true': (T, D), 'actions': (T, n_act), 'length': T}}.

    ``aug_loader`` must be unshuffled.
    """
    dev = next(model.parameters()).device
    model.eval()
    pid_zt, pid_acts = defaultdict(list), defaultdict(list)
    global_i = 0
    for batch in aug_loader:
        s = batch["s"].to(dev)
        a = batch["actions"]
        out = model(s, a.to(dev), tf_ratio=0.0)
        B = s.shape[0]
        for bi in range(B):
            pid = dataset.patient_ids[global_i + bi]
            if pid is None:
                continue
            pid_zt[str(pid)].append(ema_smooth(out["emb"][bi].cpu(), 0.25))
            pid_acts[str(pid)].append(a[bi].cpu())
        global_i += B
    result = {}
    for pid_s in pid_zt:
        zt = torch.cat(pid_zt[pid_s], dim=0)[:max_steps]
        ac = torch.cat(pid_acts[pid_s], dim=0)[:max_steps]
        result[pid_s] = {"z_true": zt, "actions": ac, "length": zt.shape[0]}
    return result


def drug_vec(idx: int, T_steps: int, n_act: int) -> torch.Tensor:
    v = torch.zeros(T_steps, n_act)
    v[:, idx] = 1.0
    return v


# ── Free-running rollout core ─────────────────────────────────────────────────

@torch.no_grad()
def free_rollout(model, z_true_seq: torch.Tensor,
                 choose_action: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor],
                 context: int = DEFAULT_CONTEXT) -> torch.Tensor:
    """Seed with z_true[:context], then roll out autoregressively to len(z_true_seq).

    ``choose_action(t, z_t, dz_t)`` returns the (n_act,) action applied on the
    transition t → t+1, given the model's *own* current state. Returns (T, D) on cpu.
    """
    if context < 2:
        raise ValueError("context must be ≥ 2 (with dz=0 the velocity model stays constant)")
    dev = next(model.parameters()).device
    model.eval()
    T = len(z_true_seq)
    z_seed = z_true_seq[:context].to(dev)
    z, dz = z_seed[-1], z_seed[-1] - z_seed[-2]
    z_cf = list(z_seed.cpu())
    for t in range(context - 1, T - 1):
        a = choose_action(t, z, dz).to(dev)
        z_nx, dz_nx = model._step(z.unsqueeze(0), dz.unsqueeze(0), a.unsqueeze(0))
        z, dz = z_nx.squeeze(0), dz_nx.squeeze(0)
        z_cf.append(z.cpu())
    return torch.stack(z_cf)


@torch.no_grad()
def counterfactual_rollouts(model, z_true_seq: torch.Tensor, interventions: Dict[str, torch.Tensor],
                            context: int = DEFAULT_CONTEXT) -> Dict[str, torch.Tensor]:
    """interventions: {name: (T-1, n_act) action sequence} → {name: (T, D)} free-running rollouts."""
    return {name: free_rollout(model, z_true_seq, lambda t, z, dz, acts=acts: acts[t], context)
            for name, acts in interventions.items()}


@torch.no_grad()
def observed_rollout(model, z_true_seq, obs_acts, context: int = DEFAULT_CONTEXT) -> torch.Tensor:
    """Free-running rollout under the patient's recorded actions."""
    return free_rollout(model, z_true_seq, lambda t, z, dz: obs_acts[t], context)


def screen_single_drugs(model, probe, cf_data: Dict[str, Dict], pids: List[str], n_act: int,
                        t_max: int = 16, context: int = DEFAULT_CONTEXT) -> Dict:
    """Test every action as a single-drug intervention on ``pids``.

    Returns {'null_mean': float, 'drug_means': (n_act,), 'effects': (n_act,)}
    where effect = mean risk under drug − mean risk under no treatment
    (averaged over the simulated steps only, after the shared context).
    """
    null_means, drug_sums, n = [], np.zeros(n_act), 0
    for pid_s in pids:
        pdata = cf_data[pid_s]
        T_roll = min(pdata["length"], t_max)
        T_steps = T_roll - 1
        ivs = {"null": torch.zeros(T_steps, n_act)}
        for di in range(n_act):
            ivs[f"d{di}"] = drug_vec(di, T_steps, n_act)
        cf_zs = counterfactual_rollouts(model, pdata["z_true"][:T_roll], ivs, context)
        null_means.append(float(np.mean(get_traj_risk_series(probe, cf_zs["null"][context:]))))
        for di in range(n_act):
            drug_sums[di] += float(np.mean(get_traj_risk_series(probe, cf_zs[f"d{di}"][context:])))
        n += 1
    null_mean = float(np.mean(null_means))
    drug_means = drug_sums / max(n, 1)
    return {"null_mean": null_mean, "drug_means": drug_means, "effects": drug_means - null_mean}


# ── Greedy one-step MPC ───────────────────────────────────────────────────────

@torch.no_grad()
def _greedy_mpc_action(model, z_t, dz_t, probe, n_act, direction, n_cand):
    """Best of n_cand candidate actions for next-step risk from the current predicted state."""
    dev = next(model.parameters()).device
    z_td, dz_td = z_t.unsqueeze(0), dz_t.unsqueeze(0)

    # Diverse candidates: null, all-ones, random binary vectors of varying density
    cands = [torch.zeros(n_act), torch.ones(n_act)]
    for dens in np.linspace(0.05, 0.60, n_cand - 2):
        cands.append((torch.rand(n_act) < float(dens)).float())

    best_val = float("inf") if direction == "minimize" else float("-inf")
    best_a = cands[0]
    for a in cands:
        z_nx, _ = model._step(z_td, dz_td, a.to(dev).unsqueeze(0))
        risk = float(torch.sigmoid(probe(z_nx.cpu())).mean())
        if (direction == "minimize" and risk < best_val) or (direction == "maximize" and risk > best_val):
            best_val, best_a = risk, a
    return best_a


@torch.no_grad()
def greedy_mpc_rollout(model, z_true_seq, obs_acts, probe, n_act, strategy: str, n_cand: int = 80,
                       context: int = DEFAULT_CONTEXT) -> torch.Tensor:
    """strategy: 'null' | 'observed' | 'minimize' (MPC-Treat) | 'maximize' (MPC-Withhold)."""
    if strategy == "null":
        return free_rollout(model, z_true_seq, lambda t, z, dz: torch.zeros(n_act), context)
    if strategy == "observed":
        return observed_rollout(model, z_true_seq, obs_acts, context)
    return free_rollout(
        model, z_true_seq,
        lambda t, z, dz: _greedy_mpc_action(model, z, dz, probe, n_act, strategy, n_cand),
        context,
    )


# ── Random-shooting MPC with H-step lookahead ─────────────────────────────────

@torch.no_grad()
def _shooting_action(model, z_t, dz_t, probe, n_act, H, N, seed):
    """First action of the best of N random H-step sequences (lowest cumulative risk)."""
    dev = next(model.parameters()).device
    D = model.latent_dim

    rng_s = np.random.default_rng(seed)
    densities = np.linspace(0.05, 0.75, N)
    rng_s.shuffle(densities)
    first_acts = torch.stack([(torch.rand(n_act) < float(d)).float() for d in densities]).to(dev)  # (N, n_act)

    z_c  = z_t.unsqueeze(0).expand(N, -1).clone()
    dz_c = dz_t.unsqueeze(0).expand(N, -1).clone()
    cumulative_risk = torch.zeros(N, device=dev)

    for h in range(H):
        acts_h = first_acts if h == 0 else torch.stack([
            (torch.rand(n_act) < float(rng_s.uniform(0.05, 0.75))).float() for _ in range(N)
        ]).to(dev)
        inp  = torch.cat([z_c, acts_h], dim=-1) if model.action_conditioned else z_c
        F_m  = model.predictor(inp).view(N, D, D)
        dz_n = torch.bmm(F_m, dz_c.unsqueeze(-1)).squeeze(-1)
        z_c  = torch.clamp(z_c + dz_n, -50.0, 50.0)
        dz_c = dz_n
        cumulative_risk += torch.sigmoid(probe(z_c.cpu())).squeeze(-1).to(dev)

    return first_acts[int(cumulative_risk.argmin())].cpu()


@torch.no_grad()
def shooting_mpc_rollout(model, z_true_seq, probe, n_act, H: int = 4, N: int = 200, seed: int = 99,
                         context: int = DEFAULT_CONTEXT) -> torch.Tensor:
    """Free-running rollout where each step's action is chosen by random-shooting MPC."""
    return free_rollout(
        model, z_true_seq,
        lambda t, z, dz: _shooting_action(model, z, dz, probe, n_act, H, N, seed),
        context,
    )
