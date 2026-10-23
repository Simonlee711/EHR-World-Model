"""
Data loading and preprocessing for the EHR World Model.

Pipeline
--------
  1. Load raw MEDS parquet split (single-file demo or multi-shard full dataset).
  2. For each (patient, admission) pair, discretise the event stream into
     fixed-width time bins.
  3. Inside each bin compute:
       state  s_t  = [normalised vital/lab values, observation mask]
       action a_t  = binary presence of each med/infusion/procedure
  4. Emit consecutive (s_t, a_t, d, s_{t+1}, m_{t+1}) tuples, where d is
     the static demographic context and m_{t+1} is the observation mask for
     the target window.
  5. Wrap in a TransitionDataset (torch.utils.data.Dataset).

Supports both:
  - Demo MEDS dataset (subject_id column, single shard per split)
  - Full MIMIC MEDS dataset (patient_id column, multiple shards per split)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import FeatureConfig, TrainConfig
from .features import FeatureManifest

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column name helpers (demo uses subject_id, full uses patient_id)
# ---------------------------------------------------------------------------

def _id_col(df: pd.DataFrame) -> str:
    """Return the patient identifier column name."""
    if "patient_id" in df.columns:
        return "patient_id"
    if "subject_id" in df.columns:
        return "subject_id"
    raise ValueError("DataFrame has neither 'patient_id' nor 'subject_id' column.")


# ---------------------------------------------------------------------------
# Loading raw MEDS data
# ---------------------------------------------------------------------------

def load_meds_split(data_dir: Path, split: str) -> pd.DataFrame:
    """
    Load one MEDS split.

    Works for the demo dataset (single shard: data/{split}/0.parquet)
    and for the full MIMIC dataset (same single-shard layout per split).
    For multi-shard loading of the full dataset, use load_meds_cohort().
    """
    path = Path(data_dir) / "data" / split / "0.parquet"
    if not path.exists():
        raise FileNotFoundError(f"MEDS data not found at {path}")
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    id_col = _id_col(df)
    log.info(
        "Loaded %s: %d patients, %d events",
        split, df[id_col].nunique(), len(df),
    )
    return df


def load_meds_cohort(
    data_dir: Path,
    split: str,
    max_patients: Optional[int] = None,
    patient_ids: Optional[Set[int]] = None,
) -> pd.DataFrame:
    """
    Load a cohort from a (potentially multi-shard) MEDS split.

    Parameters
    ----------
    data_dir    : Root data directory (contains data/, metadata/).
    split       : 'train', 'tuning', or 'held_out'.
    max_patients: If set, sample this many patients from the split.
    patient_ids : If set, load only these patient IDs (overrides max_patients).

    Returns the concatenated DataFrame for the selected patients.
    """
    data_dir = Path(data_dir)
    shard_dir = data_dir / "data" / split

    if not shard_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {shard_dir}")

    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet files found in {shard_dir}")

    # If patient IDs not specified and max_patients is set, sample from splits file
    if patient_ids is None and max_patients is not None:
        patient_ids = _sample_patient_ids(data_dir, split, max_patients)

    dfs = []
    total_patients_seen: Set[int] = set()

    for shard in shards:
        chunk = pd.read_parquet(shard)
        chunk["time"] = pd.to_datetime(chunk["time"])
        id_col = _id_col(chunk)

        if patient_ids is not None:
            chunk = chunk[chunk[id_col].isin(patient_ids)]

        if len(chunk) > 0:
            dfs.append(chunk)
            total_patients_seen.update(chunk[id_col].unique())

        log.debug("Shard %s: %d events (filtered)", shard.name, len(chunk))

    if not dfs:
        raise ValueError(f"No events found for the requested patients in split '{split}'.")

    df = pd.concat(dfs, ignore_index=True)
    id_col = _id_col(df)
    log.info(
        "Loaded %s: %d patients, %d events (%d shards)",
        split, df[id_col].nunique(), len(df), len(shards),
    )
    return df


def _sample_patient_ids(
    data_dir: Path,
    split: str,
    n: int,
) -> Set[int]:
    """
    Sample up to n patient IDs from the metadata/patient_splits.parquet file.
    Falls back to scanning the first shard if the splits file is missing.
    """
    splits_path = data_dir / "metadata" / "patient_splits.parquet"
    if splits_path.exists():
        splits_df = pd.read_parquet(splits_path)
        id_col = "patient_id" if "patient_id" in splits_df.columns else "subject_id"
        split_col = "split" if "split" in splits_df.columns else splits_df.columns[-1]
        mask = splits_df[split_col] == split
        ids = splits_df.loc[mask, id_col].unique()
        if len(ids) > n:
            rng = np.random.default_rng(42)
            ids = rng.choice(ids, size=n, replace=False)
        log.info("Sampled %d patients from %s split via patient_splits.parquet", len(ids), split)
        return set(ids.tolist())

    # Fallback: scan first shard
    log.warning("patient_splits.parquet not found — scanning first shard for patient IDs")
    first_shard = sorted((data_dir / "data" / split).glob("*.parquet"))[0]
    df = pd.read_parquet(first_shard, columns=[_id_col(pd.read_parquet(first_shard, nrows=1))])
    id_col = _id_col(df)
    ids = df[id_col].unique()
    if len(ids) > n:
        rng = np.random.default_rng(42)
        ids = rng.choice(ids, size=n, replace=False)
    return set(ids.tolist())


# ---------------------------------------------------------------------------
# Demographic vector
# ---------------------------------------------------------------------------

def _encode_demographics(
    admit_row: pd.Series,
    birth_time: Optional[pd.Timestamp],
    admit_time: pd.Timestamp,
    manifest: FeatureManifest,
    sex: Optional[str] = None,
) -> np.ndarray:
    """
    One-hot encode race, normalised age, and one-hot sex.

    Unknown / missing values map to the last (UNK) slot per column.
    Age is clipped to [0, 100] and divided by 100.
    Sex is encoded from GENDER// events (M / F / UNK).
    """
    parts: List[np.ndarray] = []
    for col in manifest.demo_cols:
        cats = manifest.demo_categories[col]
        val = str(admit_row.get(col, "")) if pd.notna(admit_row.get(col, None)) else ""
        vec = np.zeros(len(cats) + 1, dtype=np.float32)
        idx = cats.index(val) if val in cats else len(cats)
        vec[idx] = 1.0
        parts.append(vec)

    # Age
    if birth_time is not None and pd.notna(birth_time):
        age_years = (admit_time - birth_time).total_seconds() / (365.25 * 24 * 3600)
        age_norm = float(np.clip(age_years, 0, 100)) / 100.0
    else:
        age_norm = 0.5
    parts.append(np.array([age_norm], dtype=np.float32))

    # Sex (from GENDER// events)
    sex_cats = manifest.sex_categories or []
    if sex_cats:
        vec = np.zeros(len(sex_cats) + 1, dtype=np.float32)
        idx = sex_cats.index(sex) if sex in sex_cats else len(sex_cats)
        vec[idx] = 1.0
        parts.append(vec)

    return np.concatenate(parts) if parts else np.array([age_norm], dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-bin feature vectors
# ---------------------------------------------------------------------------

def _state_vector(
    bin_df: pd.DataFrame,
    manifest: FeatureManifest,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate events in one time bin into a normalised state vector + mask.

    Returns
    -------
    values : float32 (n_continuous,)  — z-score normalised, 0 if unobserved
    mask   : float32 (n_continuous,)  — 1 if observed, 0 otherwise
    """
    all_codes = manifest.vital_codes + manifest.lab_codes
    n = len(all_codes)
    values = np.zeros(n, dtype=np.float32)
    mask = np.zeros(n, dtype=np.float32)

    if len(bin_df) == 0:
        return values, mask

    num = bin_df[bin_df["numeric_value"].notna()]
    if len(num) == 0:
        return values, mask

    grp = num.groupby("code")["numeric_value"].mean()

    for i, code in enumerate(all_codes):
        if code in grp.index:
            raw = grp[code]
            mu = manifest.state_mean.get(code, 0.0)
            σ = manifest.state_std.get(code, 1.0)
            z = (raw - mu) / σ
            values[i] = float(np.clip(z, -5.0, 5.0))
            mask[i] = 1.0

    return values, mask


def _action_vector(
    bin_df: pd.DataFrame,
    manifest: FeatureManifest,
) -> np.ndarray:
    """
    Binary action vector: 1 if the med/infusion/procedure appears in this bin.
    """
    all_actions = manifest.med_codes + manifest.infusion_codes + manifest.proc_codes
    present = set(bin_df["code"].unique())
    return np.array([1.0 if c in present else 0.0 for c in all_actions], dtype=np.float32)


# ---------------------------------------------------------------------------
# Transition builder (per patient × admission)
# ---------------------------------------------------------------------------

def _admission_transitions(
    admit_df: pd.DataFrame,
    demo_vec: np.ndarray,
    manifest: FeatureManifest,
    bin_hours: float,
    patient_id: Optional[int] = None,
    hadm_id: Optional[int] = None,
) -> List[Dict]:
    """
    Convert one hospital admission's event stream into transition dicts.
    """
    admit_df = admit_df.sort_values("time").copy()
    clinical = admit_df[~admit_df["time"].isna()].copy()

    if len(clinical) < 2:
        return []

    t0 = clinical["time"].iloc[0]
    t_end = clinical["time"].iloc[-1]
    total_hours = (t_end - t0).total_seconds() / 3600.0

    if total_hours < bin_hours:
        return []

    n_bins = max(2, int(np.ceil(total_hours / bin_hours)))
    bin_edges = [t0 + pd.Timedelta(hours=i * bin_hours) for i in range(n_bins + 1)]

    states, masks, actions = [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bdf = clinical[(clinical["time"] >= lo) & (clinical["time"] < hi)]
        s, m = _state_vector(bdf, manifest)
        a = _action_vector(bdf, manifest)
        states.append(s)
        masks.append(m)
        actions.append(a)

    transitions = []
    for i in range(n_bins - 1):
        if masks[i].sum() == 0 and masks[i + 1].sum() == 0:
            continue

        s_t = np.concatenate([states[i], masks[i]])
        s_next = np.concatenate([states[i + 1], masks[i + 1]])

        t = {
            "s_t":    s_t,
            "a_t":    actions[i],
            "d":      demo_vec,
            "s_next": s_next,
            "m_next": masks[i + 1],
        }
        if patient_id is not None:
            t["patient_id"] = patient_id
        if hadm_id is not None:
            t["hadm_id"] = hadm_id

        transitions.append(t)

    return transitions


def build_transitions(
    df: pd.DataFrame,
    manifest: FeatureManifest,
    cfg: FeatureConfig,
) -> List[Dict]:
    """
    Build all transitions from a MEDS split dataframe.

    Handles both demo (subject_id, hadm_id) and full MIMIC (patient_id) layouts.
    Returns a list of transition dicts.  Each dict contains:
      s_t, a_t, d, s_next, m_next, patient_id (optional), hadm_id (optional).
    """
    id_col = _id_col(df)

    # Extract birth dates and sex per patient
    birth_map: Dict[int, pd.Timestamp] = {}
    birth_rows = df[df["code"] == "MEDS_BIRTH"]
    if len(birth_rows) > 0:
        birth_map = birth_rows.groupby(id_col)["time"].first().to_dict()

    sex_map: Dict[int, str] = {}
    gender_rows = df[df["code"].str.startswith("GENDER//")]
    if len(gender_rows) > 0:
        sex_map = (
            gender_rows.groupby(id_col)["code"]
            .first()
            .str.replace("GENDER//", "", regex=False)
            .to_dict()
        )

    all_transitions: List[Dict] = []

    has_hadm = "hadm_id" in df.columns and df["hadm_id"].notna().any()

    for patient_id, sub_df in df.groupby(id_col):
        birth_time = birth_map.get(patient_id)
        sex = sex_map.get(patient_id)

        if has_hadm:
            for hadm_id, admit_df in sub_df[sub_df["hadm_id"].notna()].groupby("hadm_id"):
                admit_event = admit_df[
                    admit_df["code"].str.startswith("HOSPITAL_ADMISSION//")
                ]
                demo_cols_present = [c for c in manifest.demo_cols if c in admit_df.columns]
                if len(admit_event) > 0 and len(demo_cols_present) > 0:
                    admit_row = admit_event.iloc[0]
                else:
                    admit_row = pd.Series(dtype=object)

                admit_time = admit_df["time"].dropna().min()
                demo_vec = _encode_demographics(admit_row, birth_time, admit_time, manifest, sex=sex)
                transitions = _admission_transitions(
                    admit_df, demo_vec, manifest, cfg.bin_hours,
                    patient_id=int(patient_id), hadm_id=int(hadm_id),
                )
                all_transitions.extend(transitions)
        else:
            admit_rows = sub_df[sub_df["code"].str.startswith("HOSPITAL_ADMISSION//")]
            row = admit_rows.iloc[0] if len(admit_rows) > 0 else pd.Series(dtype=object)
            admit_time = sub_df["time"].dropna().min()
            demo_vec = _encode_demographics(row, birth_time, admit_time, manifest, sex=sex)
            transitions = _admission_transitions(
                sub_df, demo_vec, manifest, cfg.bin_hours,
                patient_id=int(patient_id),
            )
            all_transitions.extend(transitions)

    log.info(
        "Built %d transitions from %d patients",
        len(all_transitions), df[id_col].nunique(),
    )
    return all_transitions


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TransitionDataset(Dataset):
    """Stores all state-action-next_state transitions as stacked tensors."""

    def __init__(self, transitions: List[Dict]):
        if not transitions:
            raise ValueError("No transitions found — check data and feature config.")

        def _stack(key: str) -> torch.Tensor:
            return torch.tensor(np.stack([t[key] for t in transitions]), dtype=torch.float32)

        self.s_t    = _stack("s_t")
        self.a_t    = _stack("a_t")
        self.d      = _stack("d")
        self.s_next = _stack("s_next")
        self.m_next = _stack("m_next")

        # Optional per-transition metadata (stored as plain lists)
        self.patient_ids = [t.get("patient_id") for t in transitions]
        self.hadm_ids    = [t.get("hadm_id")    for t in transitions]

    def __len__(self) -> int:
        return len(self.s_t)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "s_t":    self.s_t[idx],
            "a_t":    self.a_t[idx],
            "d":      self.d[idx],
            "s_next": self.s_next[idx],
            "m_next": self.m_next[idx],
        }

    def stats(self) -> str:
        n = len(self)
        obs_rate = self.m_next.mean().item()
        return (
            f"TransitionDataset: {n} transitions | "
            f"state_dim={self.s_t.shape[1]} | "
            f"action_dim={self.a_t.shape[1]} | "
            f"demo_dim={self.d.shape[1]} | "
            f"avg target obs rate={obs_rate:.1%}"
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloaders(
    data_dir: Path,
    manifest: FeatureManifest,
    feat_cfg: FeatureConfig,
    train_cfg: TrainConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) DataLoaders for demo (single-shard) dataset."""
    loaders = []
    for split in ("train", "tuning", "held_out"):
        df = load_meds_split(data_dir, split)
        transitions = build_transitions(df, manifest, feat_cfg)
        ds = TransitionDataset(transitions)
        log.info("%s: %s", split, ds.stats())
        loader = DataLoader(
            ds,
            batch_size=train_cfg.batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=False,
        )
        loaders.append(loader)
    return tuple(loaders)


# ---------------------------------------------------------------------------
# Enriched state encoding (missingness-aware)
# ---------------------------------------------------------------------------
# State = [values | obs_mask | time_since_norm | carry_fwd]  (4 × n_continuous)
#
# • values        : z-score normalised (0 if unobserved in this bin)
# • obs_mask      : 1 if measured in this bin, 0 otherwise
# • time_since_norm: hours since last observation / 168 (7-day cap), 1 if never seen
# • carry_fwd     : exp-decay EWMA of last known value (τ = 12 h); 0 if never seen
#
# These four channels let the model distinguish "missing but recently measured"
# from "missing and not measured in days", and carry a soft estimate of the
# current value even when the lab/vital was not drawn this bin.


def _state_vector_enriched(
    bin_df: pd.DataFrame,
    manifest: FeatureManifest,
    prev_values: np.ndarray,       # (n_continuous,) last known z-score values
    prev_obs_hours: np.ndarray,    # (n_continuous,) hours-since-epoch of last obs; -inf = never
    bin_start_hours: float,        # hours from admission start for this bin
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute enriched state, update carry-forward and last-obs-time trackers.

    Returns
    -------
    state        : (4 * n_continuous,) = [values | mask | time_since_norm | carry_fwd]
    new_values   : updated prev_values for the next bin
    new_obs_hours: updated prev_obs_hours for the next bin
    mask         : (n_continuous,) observation mask
    """
    all_codes = manifest.vital_codes + manifest.lab_codes
    n = len(all_codes)
    values = np.zeros(n, dtype=np.float32)
    mask   = np.zeros(n, dtype=np.float32)

    if len(bin_df) > 0:
        num = bin_df[bin_df["numeric_value"].notna()]
        if len(num) > 0:
            grp = num.groupby("code")["numeric_value"].mean()
            for i, code in enumerate(all_codes):
                if code in grp.index:
                    raw = grp[code]
                    mu  = manifest.state_mean.get(code, 0.0)
                    σ   = manifest.state_std.get(code, 1.0)
                    values[i] = float(np.clip((raw - mu) / σ, -5.0, 5.0))
                    mask[i]   = 1.0

    MAX_HOURS = 168.0   # 7-day cap on time_since
    TAU       = 12.0    # exponential decay half-life (hours)

    time_since_norm = np.ones(n, dtype=np.float32)   # default: never seen → 1.0
    carry_fwd       = np.zeros(n, dtype=np.float32)

    new_values    = prev_values.copy()
    new_obs_hours = prev_obs_hours.copy()

    for i in range(n):
        if mask[i] == 1.0:
            time_since_norm[i]  = 0.0
            carry_fwd[i]        = values[i]
            new_values[i]       = values[i]
            new_obs_hours[i]    = bin_start_hours
        else:
            hrs = bin_start_hours - prev_obs_hours[i]
            if hrs < 0 or np.isinf(hrs):
                # Never observed
                time_since_norm[i] = 1.0
                carry_fwd[i]       = 0.0
            else:
                time_since_norm[i] = float(np.clip(hrs / MAX_HOURS, 0.0, 1.0))
                alpha              = float(np.exp(-hrs / TAU))
                carry_fwd[i]       = alpha * prev_values[i]
                new_values[i]      = carry_fwd[i]   # propagate decayed estimate

    state = np.concatenate([values, mask, time_since_norm, carry_fwd])
    return state, new_values, new_obs_hours, mask


def _admission_enriched_transitions(
    admit_df: pd.DataFrame,
    demo_vec: np.ndarray,
    manifest: FeatureManifest,
    bin_hours: float,
    patient_id: Optional[int] = None,
    hadm_id: Optional[int] = None,
    sex: Optional[str] = None,
    age: Optional[float] = None,
    race: Optional[str] = None,
    bmi: float = float("nan"),
) -> List[Dict]:
    """Build enriched transitions for one hospital admission."""
    admit_df = admit_df.sort_values("time").copy()
    clinical = admit_df[~admit_df["time"].isna()].copy()

    if len(clinical) < 2:
        return []

    t0          = clinical["time"].iloc[0]
    t_end       = clinical["time"].iloc[-1]
    total_hours = (t_end - t0).total_seconds() / 3600.0

    if total_hours < bin_hours:
        return []

    n_bins    = max(2, int(np.ceil(total_hours / bin_hours)))
    bin_edges = [t0 + pd.Timedelta(hours=i * bin_hours) for i in range(n_bins + 1)]

    n              = manifest.n_continuous
    prev_values    = np.zeros(n, dtype=np.float32)
    prev_obs_hours = np.full(n, -np.inf, dtype=np.float32)  # never observed

    states, masks, actions = [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bdf    = clinical[(clinical["time"] >= lo) & (clinical["time"] < hi)]

        state, prev_values, prev_obs_hours, mask = _state_vector_enriched(
            bdf, manifest, prev_values, prev_obs_hours, float(i * bin_hours)
        )
        a = _action_vector(bdf, manifest)
        states.append(state)
        masks.append(mask)
        actions.append(a)

    transitions = []
    for i in range(n_bins - 1):
        if masks[i].sum() == 0 and masks[i + 1].sum() == 0:
            continue
        t = {
            "s_t":     states[i],
            "a_t":     actions[i],
            "d":       demo_vec,
            "s_next":  states[i + 1],
            "m_next":  masks[i + 1],
            "bin_idx": i,
            "bmi":     bmi,
        }
        if patient_id is not None: t["patient_id"] = patient_id
        if hadm_id    is not None: t["hadm_id"]    = hadm_id
        if sex        is not None: t["sex"]         = sex
        if age        is not None: t["age"]         = float(age)
        if race       is not None: t["race"]        = race
        transitions.append(t)

    return transitions


def build_enriched_transitions(
    df: pd.DataFrame,
    manifest: FeatureManifest,
    cfg: FeatureConfig,
) -> List[Dict]:
    """
    Build enriched transitions from a MEDS split dataframe.

    State = [values | obs_mask | time_since_norm | carry_fwd]  (4 × n_continuous).
    Each transition dict carries stratification metadata: sex, age, race.
    """
    id_col = _id_col(df)

    birth_map: Dict = {}
    birth_rows = df[df["code"] == "MEDS_BIRTH"]
    if len(birth_rows) > 0:
        birth_map = birth_rows.groupby(id_col)["time"].first().to_dict()

    sex_map: Dict = {}
    gender_rows = df[df["code"].str.startswith("GENDER//")]
    if len(gender_rows) > 0:
        sex_map = (
            gender_rows.groupby(id_col)["code"]
            .first()
            .str.replace("GENDER//", "", regex=False)
            .to_dict()
        )

    # BMI per patient — used for stratification
    bmi_map: Dict[int, float] = {}
    _BMI_CODE = "BMI (kg/m2)"
    if _BMI_CODE in (manifest.vital_codes + manifest.lab_codes):
        bmi_rows = df[df["code"] == _BMI_CODE]
        if len(bmi_rows) > 0:
            bmi_map = bmi_rows.groupby(id_col)["numeric_value"].first().to_dict()

    all_transitions: List[Dict] = []
    has_hadm = "hadm_id" in df.columns and df["hadm_id"].notna().any()

    for patient_id, sub_df in df.groupby(id_col):
        birth_time = birth_map.get(patient_id)
        pt_sex     = sex_map.get(patient_id)
        pt_bmi     = float(bmi_map.get(patient_id, float("nan")))

        pt_age = None
        if birth_time is not None and pd.notna(birth_time):
            # Age computed later per-admission; pre-compute from birth
            pass

        if has_hadm:
            for hadm_id, admit_df in sub_df[sub_df["hadm_id"].notna()].groupby("hadm_id"):
                admit_event     = admit_df[admit_df["code"].str.startswith("HOSPITAL_ADMISSION//")]
                demo_cols_pres  = [c for c in manifest.demo_cols if c in admit_df.columns]
                admit_row       = admit_event.iloc[0] if len(admit_event) > 0 and demo_cols_pres else pd.Series(dtype=object)

                admit_time = admit_df["time"].dropna().min()
                demo_vec   = _encode_demographics(admit_row, birth_time, admit_time, manifest, sex=pt_sex)

                age  = None
                if birth_time is not None and pd.notna(birth_time):
                    age = (admit_time - birth_time).total_seconds() / (365.25 * 24 * 3600)
                race = str(admit_row.get("race")) if "race" in admit_row.index and pd.notna(admit_row.get("race")) else None

                transitions = _admission_enriched_transitions(
                    admit_df, demo_vec, manifest, cfg.bin_hours,
                    patient_id=int(patient_id), hadm_id=int(hadm_id),
                    sex=pt_sex, age=age, race=race, bmi=pt_bmi,
                )
                all_transitions.extend(transitions)
        else:
            admit_rows = sub_df[sub_df["code"].str.startswith("HOSPITAL_ADMISSION//")]
            row        = admit_rows.iloc[0] if len(admit_rows) > 0 else pd.Series(dtype=object)
            admit_time = sub_df["time"].dropna().min()
            demo_vec   = _encode_demographics(row, birth_time, admit_time, manifest, sex=pt_sex)

            age  = None
            if birth_time is not None and pd.notna(birth_time):
                age = (admit_time - birth_time).total_seconds() / (365.25 * 24 * 3600)
            race = str(row.get("race")) if "race" in row.index and pd.notna(row.get("race")) else None

            transitions = _admission_enriched_transitions(
                sub_df, demo_vec, manifest, cfg.bin_hours,
                patient_id=int(patient_id),
                sex=pt_sex, age=age, race=race, bmi=pt_bmi,
            )
            all_transitions.extend(transitions)

    log.info(
        "Built %d enriched transitions from %d patients",
        len(all_transitions), df[id_col].nunique(),
    )
    return all_transitions


# ---------------------------------------------------------------------------
# Enriched datasets
# ---------------------------------------------------------------------------

class EnrichedTransitionDataset(Dataset):
    """
    Like TransitionDataset but for enriched states (4 × n_continuous).
    Stores stratification metadata (sex, age, race) alongside tensors.
    """

    def __init__(self, transitions: List[Dict]):
        if not transitions:
            raise ValueError("No transitions found.")

        def _s(key: str) -> torch.Tensor:
            return torch.tensor(np.stack([t[key] for t in transitions]), dtype=torch.float32)

        self.s_t    = _s("s_t")
        self.a_t    = _s("a_t")
        self.d      = _s("d")
        self.s_next = _s("s_next")
        self.m_next = _s("m_next")

        self.patient_ids = [t.get("patient_id") for t in transitions]
        self.hadm_ids    = [t.get("hadm_id")    for t in transitions]
        self.sexes       = [t.get("sex")         for t in transitions]
        self.ages        = [t.get("age")         for t in transitions]
        self.races       = [t.get("race")        for t in transitions]
        self.bin_idxs    = [t.get("bin_idx", 0) for t in transitions]

    def __len__(self) -> int:
        return len(self.s_t)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "s_t":    self.s_t[idx],
            "a_t":    self.a_t[idx],
            "d":      self.d[idx],
            "s_next": self.s_next[idx],
            "m_next": self.m_next[idx],
        }

    def stats(self) -> str:
        n = len(self)
        obs_rate = self.m_next.mean().item()
        return (
            f"EnrichedTransitionDataset: {n} transitions | "
            f"state_dim={self.s_t.shape[1]} | action_dim={self.a_t.shape[1]} | "
            f"avg obs rate={obs_rate:.1%}"
        )


def build_enriched_sequence_batches(
    transitions: List[Dict],
    seq_len: int = 8,
    stride: int = 4,
) -> List[Dict]:
    """
    Slide a fixed-length window over each patient-admission's transitions.

    Each output dict has:
        s        : (seq_len, 4*n_cont)   enriched state sequence
        actions  : (seq_len, n_actions)
        s_next   : (seq_len, 4*n_cont)   next-state sequence (for eval)
        m_next   : (seq_len, n_cont)     observation mask sequence
        patient_id, hadm_id, sex, age, race : metadata
    """
    from collections import defaultdict

    groups: dict = defaultdict(list)
    for t in transitions:
        pid = t.get("patient_id", -1)
        hid = t.get("hadm_id",    -1)
        groups[(pid, hid)].append(t)

    sequences = []
    for (pid, hid), trans in groups.items():
        T = len(trans)
        if T < seq_len:
            continue
        for start in range(0, T - seq_len + 1, stride):
            win = trans[start: start + seq_len]
            seq = {
                "s":          np.stack([w["s_t"]    for w in win]),
                "actions":    np.stack([w["a_t"]    for w in win]),
                "s_next":     np.stack([w["s_next"] for w in win]),
                "m_next":     np.stack([w["m_next"] for w in win]),
                "d":          win[0]["d"],
                "bmi":        win[0].get("bmi", float("nan")),
                "patient_id": pid,
                "hadm_id":    hid,
                "sex":        win[0].get("sex"),
                "age":        win[0].get("age"),
                "race":       win[0].get("race"),
            }
            sequences.append(seq)

    log.info(
        "build_enriched_sequence_batches: %d groups → %d sequences (seq_len=%d stride=%d)",
        len(groups), len(sequences), seq_len, stride,
    )
    return sequences


class EnrichedSequenceDataset(torch.utils.data.Dataset):
    """Sequence dataset for enriched states with full metadata for stratification."""

    def __init__(self, sequences: List[Dict]):
        self.s       = torch.tensor(np.stack([s["s"]       for s in sequences]), dtype=torch.float32)
        self.actions = torch.tensor(np.stack([s["actions"] for s in sequences]), dtype=torch.float32)
        self.s_next  = torch.tensor(np.stack([s["s_next"]  for s in sequences]), dtype=torch.float32)
        self.m_next  = torch.tensor(np.stack([s["m_next"]  for s in sequences]), dtype=torch.float32)
        self.d       = torch.tensor(np.stack([s["d"]        for s in sequences]), dtype=torch.float32)

        self.patient_ids = [s.get("patient_id") for s in sequences]
        self.sexes       = [s.get("sex")        for s in sequences]
        self.ages        = [s.get("age")        for s in sequences]
        self.races       = [s.get("race")       for s in sequences]
        self.bmis        = [s.get("bmi", float("nan")) for s in sequences]

    def __len__(self) -> int:
        return len(self.s)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "s":       self.s[idx],
            "actions": self.actions[idx],
            "s_next":  self.s_next[idx],
            "m_next":  self.m_next[idx],
            "d":       self.d[idx],
        }

    def stats(self) -> str:
        return (
            f"EnrichedSequenceDataset: {len(self)} seqs | "
            f"state_dim={self.s.shape[-1]} | action_dim={self.actions.shape[-1]} | "
            f"seq_len={self.s.shape[1]} | demo_dim={self.d.shape[-1]}"
        )


def make_dataloaders_full(
    data_dir: Path,
    manifest: FeatureManifest,
    feat_cfg: FeatureConfig,
    train_cfg: TrainConfig,
    train_patient_ids: Optional[Set[int]] = None,
    val_patient_ids: Optional[Set[int]] = None,
    test_patient_ids: Optional[Set[int]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return (train, val, test) DataLoaders for the full multi-shard MIMIC dataset.

    Loads up to train_cfg.max_patients from each split (or uses provided IDs).
    """
    max_p = train_cfg.max_patients

    splits_info = [
        ("train",    train_patient_ids, max_p),
        ("tuning",   val_patient_ids,   max(100, max_p // 10) if max_p else None),
        ("held_out", test_patient_ids,  max(100, max_p // 10) if max_p else None),
    ]

    loaders = []
    for split, provided_ids, max_n in splits_info:
        df = load_meds_cohort(
            data_dir, split,
            max_patients=max_n if provided_ids is None else None,
            patient_ids=provided_ids,
        )
        transitions = build_transitions(df, manifest, feat_cfg)
        ds = TransitionDataset(transitions)
        log.info("%s: %s", split, ds.stats())
        loader = DataLoader(
            ds,
            batch_size=train_cfg.batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=False,
        )
        loaders.append(loader)

    return tuple(loaders)
