"""
Curriculum training for the world models (shared by velocity + transformer).

Scheduled-sampling curriculum:
    stage 1 : tf=1.0, rollout_k=1   (pure teacher forcing, one-step)
    stage 2 : tf=0.5, rollout_k=2
    stage 3 : tf=0.5, rollout_k=4

Each stage early-stops on validation loss (evaluated free-running, tf=0) and
restores its best weights before the next stage starts.
"""
from __future__ import annotations

import copy
import logging
import math
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.optim as optim

logger = logging.getLogger(__name__)

CURRICULUM_STAGES = [
    dict(tf=1.0, rollout_k=1, epochs=30, patience=8),
    dict(tf=0.5, rollout_k=2, epochs=20, patience=6),
    dict(tf=0.5, rollout_k=4, epochs=20, patience=6),
]

# Lighter schedule used for the S / L transformer scaling runs
CURRICULUM_STAGES_FAST = [
    dict(tf=1.0, rollout_k=1, epochs=20, patience=6),
    dict(tf=0.5, rollout_k=2, epochs=15, patience=5),
    dict(tf=0.5, rollout_k=4, epochs=15, patience=5),
]


def train_curriculum(model, train_loader, val_loader, label: str = "model",
                     stages: Optional[List[Dict]] = None, lr: float = 3e-4) -> pd.DataFrame:
    """Train ``model`` in-place through the curriculum. Returns per-epoch history."""
    stages = stages or CURRICULUM_STAGES
    dev = next(model.parameters()).device
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    history = []

    for si, stage in enumerate(stages):
        tf, rk, n_ep, patience = stage["tf"], stage["rollout_k"], stage["epochs"], stage["patience"]
        best_val, pat, best_w = float("inf"), 0, copy.deepcopy(model.state_dict())
        logger.info("[%s] stage %d/%d  tf=%s  rollout_k=%s", label, si + 1, len(stages), tf, rk)

        for ep in range(n_ep):
            model.train()
            tr = 0.0
            for batch in train_loader:
                s = batch["s"].to(dev)
                a = batch["actions"].to(dev)
                opt.zero_grad(set_to_none=True)
                loss = model(s, a, tf_ratio=tf, rollout_k=rk)["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tr += loss.item()

            model.eval()
            vl = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    s = batch["s"].to(dev)
                    a = batch["actions"].to(dev)
                    vl += model(s, a, tf_ratio=0.0, rollout_k=rk)["loss"].item()

            tr_a = tr / max(1, len(train_loader))
            vl_a = vl / max(1, len(val_loader))
            history.append(dict(stage=si, epoch=ep, train_loss=tr_a, val_loss=vl_a))

            if (ep + 1) % 5 == 0:
                logger.info("  ep %3d/%d  train=%.4f  val=%.4f", ep + 1, n_ep, tr_a, vl_a)
            if math.isnan(vl_a):
                logger.warning("  NaN val loss — stopping stage.")
                break
            if vl_a < best_val - 1e-5:
                best_val, pat, best_w = vl_a, 0, copy.deepcopy(model.state_dict())
            else:
                pat += 1
                if pat >= patience:
                    logger.info("  early stop at ep %d", ep + 1)
                    break

        model.load_state_dict(best_w)

    return pd.DataFrame(history)
