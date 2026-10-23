# EHR Preprocessing

Self-contained preprocessing pipeline for the EHR World Model. Converts raw
[MEDS-format](https://github.com/Medical-Event-Data-Standard/meds) parquet files
into tensors ready for model training.

## What's in this folder

| File | Description |
|---|---|
| `preprocessing.py` | All preprocessing logic (single file, no internal imports) |

## Requirements

```
numpy
pandas
torch
pyarrow   # for reading parquet files
```

Install with:
```bash
pip install numpy pandas torch pyarrow
```

## Supported datasets

| Dataset | Path | Notes |
|---|---|---|
| MIMIC-IV demo | `mimic-iv-demo-data-in-the-medical-event-data-standard-meds-0.0.1/` | 100 patients, single-shard |
| Full MIMIC-IV | `MIMIC_MEDS_Extract_v0.0.3_test/` | Multi-shard, up to ~30k patients |

Both follow the MEDS parquet layout:
```
data_dir/
  data/{train,tuning,held_out}/0.parquet   (demo: one file per split)
  data/{train,tuning,held_out}/*.parquet   (full: multiple shards)
  metadata/patient_splits.parquet          (full only)
  extract_code_metadata/codes.parquet      (optional: human-readable descriptions)
```

## Pipeline overview

```
Raw MEDS parquet
      │
      ▼
load_meds_split / load_meds_cohort
      │
      ▼
build_feature_manifest          ← training split only
  • classify codes by category (vital / lab / medication / infusion / procedure)
  • select top-K per category by subject frequency
  • compute z-score normalization stats
      │
      ▼
build_transitions               ← standard 2-channel state
  OR
build_enriched_transitions      ← 4-channel state (recommended)
  │
  • discretise events into 6-hour bins
  • per bin: state s_t = [values | obs_mask]
             enriched:  [values | mask | time_since_norm | carry_fwd]
  •          action a_t = binary presence of meds/procedures
  • emit (s_t, a_t, d, s_next, m_next) transition dicts
      │
      ▼
extract_ehrshot_labels          ← diseases + outcomes + demographics
      │
      ▼
attach_labels_to_transitions    ← merge labels into transition dicts
      │
      ▼
TransitionDataset / EnrichedSequenceDataset  ← torch.utils.data.Dataset
```

## Quick-start

```python
import logging
from pathlib import Path
from preprocessing.preprocessing import (
    FeatureConfig,
    load_meds_split,
    build_feature_manifest,
    build_transitions,
    build_enriched_transitions,
    extract_ehrshot_labels,
    attach_labels_to_transitions,
    TransitionDataset,
)

logging.basicConfig(level=logging.INFO)

DATA_DIR = Path("../mimic-iv-demo-data-in-the-medical-event-data-standard-meds-0.0.1")
feat_cfg = FeatureConfig()

# Load raw data
train_df = load_meds_split(DATA_DIR, "train")
val_df   = load_meds_split(DATA_DIR, "tuning")
test_df  = load_meds_split(DATA_DIR, "held_out")

id_col = "patient_id" if "patient_id" in train_df.columns else "subject_id"

# Feature manifest (from training split only)
manifest = build_feature_manifest(train_df, feat_cfg)
print(manifest.summary())

# Trajectories
transitions_train = build_transitions(train_df, manifest, feat_cfg)
transitions_val   = build_transitions(val_df,   manifest, feat_cfg)

# EHR-SHOT labels (diseases + outcomes + demographics)
labels_df = extract_ehrshot_labels(train_df, id_col=id_col)
transitions_train = attach_labels_to_transitions(transitions_train, labels_df, id_col=id_col)

# Dataset
ds_train = TransitionDataset(transitions_train)
print(ds_train.stats())
```

## State encoding details

### Standard (2-channel)
```
s_t = [values (n_continuous) | obs_mask (n_continuous)]
```
- `values[i]`: z-score normalised value; 0 if the feature was not observed this bin
- `obs_mask[i]`: 1 if observed, 0 otherwise

### Enriched (4-channel, recommended)
```
s_t = [values | obs_mask | time_since_norm | carry_fwd]   (4 × n_continuous)
```
- `values[i]`: z-score normalised current observation (0 if unobserved)
- `obs_mask[i]`: 1 if observed this bin, 0 otherwise
- `time_since_norm[i]`: hours since last observation / 168 h (7-day cap); 1 if never seen
- `carry_fwd[i]`: exponential-decay EWMA of last known value (τ = 12 h)

The 4-channel encoding lets the model distinguish
"missing but measured 2 hours ago" from "not measured for 3 days".

## EHR-SHOT labels

`extract_ehrshot_labels()` mirrors the label set from `world_model_v6.ipynb` and the
EHR-SHOT benchmark. It returns one row per patient with three groups:

### A. Original chronic-disease flags (`ORIG_DISEASE_COLS`)

| Column | ICD codes |
|---|---|
| `has_HTN` | I10 / 401 |
| `has_DM` | E10, E11 / 250 |
| `has_HF` | I50 / 428 |
| `has_CAD` | I25, I25 / 414, 410, 412 |
| `has_CKD` | N18 / 585 |
| `has_AKI` | N17 / 584 |
| `has_Sepsis` | A41, R652 / 038, 9959 |
| `has_COPD` | J44 / 491, 492, 496 |
| `has_Cancer` | C* / 14–23, 195–199 |
| `ckd_stage` | 0–5 (max N18.x / 585x stage; 0 = none) |
| `primary_disease` | First match: Sepsis > AKI > HF > CKD > COPD > DM > CAD > Cancer > HTN |
| `comorbidity_n` | Number of disease flags set |

### B. Additional EHR-SHOT disease flags (`NEW_DISEASE_COLS`)

| Column | ICD codes |
|---|---|
| `has_acute_mi` | I21, I22 / 410, 411 |
| `has_lupus` | M32 / 7100 |
| `has_hyperlipidemia` | E78 / 272 |
| `has_hypertension` | I10–I15 / 401–405 (expanded) |
| `has_celiac` | K900, K90 / 5790, 579 |
| `has_pancreatic_cancer` | C25 / 157 |

### C. Clinical outcome labels (`OUTCOME_COLS`)

| Column | Definition |
|---|---|
| `in_hospital_mortality` | MEDS_DEATH / DEATH / EXPIR event present |
| `icu_admission` | Any ICU / MICU / SICU / CVICU / CCU transfer code |
| `los_days` | Float — days from first admission to last discharge |
| `los_long` | 1 if `los_days > 7` |
| `readmit_30d` | 1 if second admission within 30 days of first discharge |
| `readmit_any` | 1 if ≥ 2 HOSPITAL_ADMISSION events |

### Demographics (from `compute_demographics`)

| Column | Values |
|---|---|
| `age` | Float years |
| `sex` | M / F / Unknown |
| `bmi` | Float kg/m² |
| `age_group` | <50 / 50-65 / 65-80 / >80 |
| `bmi_group` | Underweight / Normal / Overweight / Obese |

## Configuration

`FeatureConfig` controls feature selection:

```python
@dataclass
class FeatureConfig:
    bin_hours: float = 6.0          # temporal bin width
    top_vitals: int = 10            # max vital codes
    top_labs: int = 16              # max lab codes
    top_meds: int = 25              # max medication codes
    top_infusions: int = 10         # max infusion codes
    top_procedures: int = 8         # max procedure codes
    min_subject_freq: float = 0.05  # code must appear in ≥5% of patients
```

Use `FeatureConfig.large()` for the full MIMIC-IV dataset (relaxed thresholds, more codes).
