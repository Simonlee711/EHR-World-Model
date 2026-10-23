"""
Multi-step autoregressive rollout evaluation on the held-out split.

    python scripts/rollout.py --data-dir MIMIC_MEDS_Extract_v0.0.3_test --run-dir runs/velocity
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ehr_world_model import build_cohort, evaluate_multistep_rollout, get_device, load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--run-dir", required=True, help="directory containing model.pt from scripts/train.py")
    p.add_argument("--max-rollout", type=int, default=24, help="steps (6h each)")
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
    model = load_checkpoint(run_dir / "model.pt", device)

    df = evaluate_multistep_rollout(model, cohort["loaders"]["test"], cohort["ds"]["test"],
                                    stride=cohort["stride"], max_rollout=args.max_rollout)
    df.to_csv(run_dir / "rollout_mse.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
