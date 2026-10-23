"""
Multi-step autoregressive rollout evaluation.

For each test patient:
  1. Deduplicate overlapping windows → one unique timeline
     (first window + the last ``stride`` new steps of each subsequent window).
  2. Encode step 0 → z_0, roll out autoregressively for k = 1..min(T-1, max_rollout)
     using the patient's observed actions.
  3. MSE(pred_k, encoder(true_state_k)) at each step k.

The mean MSE at step k is over all patients with at least k future steps.
Each step is one 6-hour bin (step 24 = 144 h ahead).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .inference import HOURS_PER_STEP

MAX_ROLLOUT = 24


@torch.no_grad()
def patient_timelines(aug_loader, dataset, stride: int) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """{pid: (states (T, state_dim), actions (T, n_actions))} with overlap removed.

    ``aug_loader`` must be unshuffled.
    """
    pid_windows = defaultdict(list)
    global_i = 0
    for batch in aug_loader:
        s = batch["s"]; a = batch["actions"]
        B = s.shape[0]
        for bi in range(B):
            pid = dataset.patient_ids[global_i + bi]
            if pid is not None:
                pid_windows[str(pid)].append((s[bi].cpu(), a[bi].cpu()))
        global_i += B

    timelines = {}
    for pid, windows in pid_windows.items():
        s_parts = [windows[0][0]] + [sw[-stride:] for sw, _ in windows[1:]]
        a_parts = [windows[0][1]] + [aw[-stride:] for _, aw in windows[1:]]
        timelines[pid] = (torch.cat(s_parts, dim=0), torch.cat(a_parts, dim=0))
    return timelines


@torch.no_grad()
def evaluate_multistep_rollout(model, aug_loader, dataset, stride: int,
                               max_rollout: int = MAX_ROLLOUT) -> pd.DataFrame:
    """Returns a DataFrame with columns step, hours, n_patients, mse."""
    dev = next(model.parameters()).device
    model.eval()

    mse_sum = np.zeros(max_rollout)
    mse_cnt = np.zeros(max_rollout)

    for pid, (all_s, all_a) in patient_timelines(aug_loader, dataset, stride).items():
        T = all_s.shape[0]
        if T < 2:
            continue
        true_emb = model.encoder(all_s.to(dev))                    # (T, D)
        n_steps  = min(T - 1, max_rollout)
        roll_a   = all_a[0:n_steps].to(dev).unsqueeze(0)            # (1, n_steps, n_actions)
        pred_emb = model.rollout(all_s[0:1].to(dev), roll_a, n_steps).squeeze(0)  # (n_steps, D)
        for k in range(n_steps):
            mse_sum[k] += F.mse_loss(pred_emb[k:k + 1], true_emb[k + 1:k + 2]).item()
            mse_cnt[k] += 1

    mse = np.where(mse_cnt > 0, mse_sum / np.maximum(mse_cnt, 1), np.nan)
    return pd.DataFrame({
        "step": np.arange(1, max_rollout + 1),
        "hours": np.arange(1, max_rollout + 1) * HOURS_PER_STEP,
        "n_patients": mse_cnt.astype(int),
        "mse": mse,
    })
