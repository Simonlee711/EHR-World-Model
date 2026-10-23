# EHR World Model

An action-conditioned latent world model for ICU/hospital EHR trajectories
(MIMIC-IV in [MEDS](https://github.com/Medical-Event-Data-Standard/meds) format).
Patient state is binned into 6-hour steps. Clinician actions (medication starts)
condition the latent dynamics. The trained model is used for multi-step rollout
and for counterfactual treatment simulation.

## Layout

```
ehr_world_model/
  config.py, features.py   feature selection (vitals / labs = state, meds = actions)
  data.py                  MEDS → binned enriched transitions → sliding-window sequences
  labels.py                ICD-prefix disease flags
  cohort.py                end-to-end data pipeline: fast MEDS loader, demographics,
                           outcome + EHRSHOT-style temporal labels, cached transitions,
                           DemoAugLoader, build_cohort()
  sigreg.py                Fourier Epps-Pulley SIGReg regulariser
  models/velocity.py       VelocityWorldModel  (z_{t+1} = z_t + F(z_t, a_t)·dz_t)
  models/transformer.py    TransformerWorldModel (bidirectional encoder + causal predictor, S/M/L)
  training.py              scheduled-sampling curriculum (tf 1.0/k1 → 0.5/k2 → 0.5/k4)
  inference.py             build/save/load checkpoints, patient trajectories, risk probes
  rollout.py               multi-step autoregressive rollout evaluation
  counterfactual.py        free-running counterfactuals, drug screening, greedy + shooting MPC
preprocessing/             standalone single-file MEDS preprocessing (no package imports)
scripts/                   train.py, rollout.py, counterfactual.py
```

## Setup

```bash
pip install -r requirements.txt
```

Data: a MEDS extract of MIMIC-IV with `data/{train,tuning,held_out}/*.parquet`
(and optionally `metadata/patient_splits.parquet`, `extract_code_metadata/codes.parquet`).
MIMIC-IV requires PhysioNet credentialed access. Do not commit data or derived outputs.

## Usage

```bash
# 1. Train (velocity model, action-conditioned)
python scripts/train.py --data-dir MIMIC_MEDS_Extract_v0.0.3_test --out-dir runs/velocity

#    transformer variant / no-action ablation
python scripts/train.py --data-dir ... --out-dir runs/tf_M --arch transformer --size M
python scripts/train.py --data-dir ... --out-dir runs/velocity_noact --no-action

# 2. Multi-step rollout MSE (up to 24 × 6h steps)
python scripts/rollout.py --data-dir ... --run-dir runs/velocity

# 3. Counterfactuals (velocity model only)
python scripts/counterfactual.py --data-dir ... --run-dir runs/velocity --experiment screen       --task has_Sepsis
python scripts/counterfactual.py --data-dir ... --run-dir runs/velocity --experiment greedy-mpc   --task has_Sepsis
python scripts/counterfactual.py --data-dir ... --run-dir runs/velocity --experiment shooting-mpc --task readmit_30d
```

Each run directory gets `model.pt` (weights + constructor config),
`feature_manifest.json`, `train_history.csv`, and CSVs from rollout and counterfactual runs.
Transitions are cached as `transitions_<split>_<hash>.pkl`.

Or from Python:

```python
from ehr_world_model import build_cohort, build_model, train_curriculum, get_device

device = get_device()
cohort = build_cohort("MIMIC_MEDS_Extract_v0.0.3_test", "runs/velocity", device)
model = build_model("velocity", cohort["state_dim"], cohort["manifest"].n_actions,
                    cohort["n_cont"]).to(device)
train_curriculum(model, cohort["loaders"]["train"], cohort["loaders"]["val"])
```

## Method notes

**State.** Each 6h bin has `[values | obs-mask | time-since-last-obs | carry-forward] × n_cont` features,
plus a static demographics vector appended by `DemoAugLoader`. The default config keeps the
top 30 vitals, 50 labs and 100 medications, each present in ≥5% of training patients.

**Training loss.** Latent MSE + path (velocity) MSE against stop-gradient encoder targets.
Fourier SIGReg on the embeddings and predictions prevents collapse, and a masked
reconstruction head keeps feature-level information in the latent.

**Rollout.** Each patient's windows are merged into one timeline. The model encodes step 0 and
rolls out autoregressively under the observed actions. MSE against the encoder embedding is
reported per horizon.

**Counterfactuals.** These are free-running: each arm is seeded with the first `--context`
observed encoder states (default 2, so the initial velocity is real), then rolled out
autoregressively on the model's own predictions under the arm's actions. All arms of a
patient share the same seed. Risk comes from a
linear per-timestep probe trained on train-split encoder trajectories, using EHRSHOT-style
labels anchored at first admission.
- `screen`: one-hot single-drug interventions vs. no treatment, ranked by |Δ mean risk|.
- `greedy-mpc`: at each step, pick the best of N candidate action vectors to minimise
  (MPC-Treat) or maximise (MPC-Withhold) next-step risk.
- `shooting-mpc`: random-shooting MPC with H-step lookahead and a receding horizon.

These are model-based simulations, not causal effect estimates.

## Citation

```bibtex
@article{lee2026forecasting,
  title={Forecasting Patient Trajectories from the Electronic Health Records with Action-Conditioned World Models},
  author={Lee, Simon A and Sha, Jiahang Hank and Chung, Soomin and Akamatsu, Kai and Halperin, Eran and Chiang, Jeffrey N},
  journal={arXiv preprint arXiv:2610.23862},
  year={2026}
}
```
