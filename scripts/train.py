"""
Train a world model on a MEDS-format MIMIC-IV extract.

    python scripts/train.py --data-dir MIMIC_MEDS_Extract_v0.0.3_test --out-dir runs/velocity
    python scripts/train.py --data-dir ... --out-dir runs/tf_M --arch transformer --size M
    python scripts/train.py --data-dir ... --out-dir runs/velocity_noact --no-action
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ehr_world_model import (CURRICULUM_STAGES, CURRICULUM_STAGES_FAST, build_cohort, build_model,
                             get_device, save_checkpoint, train_curriculum)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--arch", choices=["velocity", "transformer"], default="velocity")
    p.add_argument("--size", choices=["S", "M", "L"], default="M", help="transformer size")
    p.add_argument("--no-action", action="store_true", help="train the no-action ablation")
    p.add_argument("--fast", action="store_true", help="shorter curriculum (used for S/L scaling runs)")
    p.add_argument("--max-train", type=int, default=10000)
    p.add_argument("--max-val", type=int, default=3000)
    p.add_argument("--max-test", type=int, default=30000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.out_dir)

    cohort = build_cohort(args.data_dir, out_dir, device, seed=args.seed,
                          max_patients={"train": args.max_train, "tuning": args.max_val, "held_out": args.max_test})
    cohort["manifest"].save(out_dir / "feature_manifest.json")

    config = dict(arch=args.arch, state_dim=cohort["state_dim"], n_actions=cohort["manifest"].n_actions,
                  n_continuous=cohort["n_cont"], action_conditioned=not args.no_action, size=args.size)
    model = build_model(**config).to(device)
    logging.info("params: %s", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    stages = CURRICULUM_STAGES_FAST if args.fast else CURRICULUM_STAGES
    hist = train_curriculum(model, cohort["loaders"]["train"], cohort["loaders"]["val"],
                            label=args.arch, stages=stages, lr=args.lr)

    save_checkpoint(model, out_dir / "model.pt", **config)
    hist.to_csv(out_dir / "train_history.csv", index=False)
    logging.info("saved %s", out_dir / "model.pt")


if __name__ == "__main__":
    main()
