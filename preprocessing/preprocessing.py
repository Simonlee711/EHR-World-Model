"""
EHR World Model — Data Preprocessing & Trajectory Construction
==============================================================

Self-contained module covering:
  1. Configuration
  2. Feature manifest construction (code selection + normalization stats)
  3. Data loading from MEDS-format parquet files
  4. Trajectory construction (6-hour bins → (state, action, next_state) triples)
     - Standard:  state = [z-score values | obs mask]
     - Enriched:  state = [values | obs_mask | time_since_norm | carry_fwd]
  5. Disease label extraction from ICD-9 / ICD-10 diagnosis codes

Supported datasets
------------------
  Demo  : mimic-iv-demo-data-in-the-medical-event-data-standard-meds-0.0.1/
  Full  : MIMIC_MEDS_Extract_v0.0.3_test/

MEDS event format (parquet columns)
------------------------------------
  patient_id | time | code | numeric_value | [demographic columns…]

Usage (quick-start)
--------------------
  from preprocessing.preprocessing import (
      FeatureConfig, load_meds_split, build_feature_manifest,
      build_transitions, build_enriched_transitions,
      extract_patient_labels, attach_labels_to_transitions,
      TransitionDataset,
  )

  feat_cfg = FeatureConfig()
  train_df = load_meds_split(data_dir, "train")
  manifest = build_feature_manifest(train_df, feat_cfg)

  transitions = build_transitions(train_df, manifest, feat_cfg)
  labels_df   = extract_patient_labels(train_df)
  transitions = attach_labels_to_transitions(transitions, labels_df)

  dataset = TransitionDataset(transitions)
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass
class FeatureConfig:
    """Controls which codes are selected and how time is discretised."""

    # Width of each temporal bin (hours)
    bin_hours: float = 6.0

    # How many top codes to retain per category (ranked by # distinct subjects)
    top_vitals: int = 10
    top_labs: int = 16
    top_meds: int = 25
    top_infusions: int = 10
    top_procedures: int = 8

    # Code must appear in at least this fraction of training subjects
    min_subject_freq: float = 0.05

    @classmethod
    def large(cls) -> "FeatureConfig":
        """Larger feature set for the full MIMIC dataset."""
        return cls(
            bin_hours=6.0,
            top_vitals=20,
            top_labs=35,
            top_meds=50,
            top_infusions=15,
            top_procedures=12,
            min_subject_freq=0.02,
        )


# =============================================================================
# 2. Feature manifest
# =============================================================================

def _lab_category(code: str) -> str:
    """Classify a LAB// code as 'vital' or 'lab' based on MIMIC-IV item ID."""
    parts = code.split("//")
    if len(parts) < 2:
        return "other"
    try:
        item_id = int(parts[1])
        if 220_000 <= item_id < 230_000:
            return "vital"
        if 50_000 <= item_id < 60_000:
            return "lab"
    except ValueError:
        pass
    return "other"


_STANDALONE_VITALS = frozenset({
    "Blood Pressure",
    "Blood Pressure Sitting",
    "Blood Pressure Lying",
    "Weight (Lbs)",
    "BMI (kg/m2)",
    "Height (Inches)",
})


def classify_code(code: str) -> str:
    """Return one of: 'vital', 'lab', 'medication', 'infusion', 'procedure', 'other'."""
    if code.startswith("LAB//"):
        return _lab_category(code)
    if code in _STANDALONE_VITALS or code.startswith("SUBJECT_FLUID_OUTPUT//"):
        return "vital"
    if code.startswith("MEDICATION//START//"):
        return "medication"
    if code.startswith("INFUSION_START//"):
        return "infusion"
    if code.startswith("PROCEDURE//START//"):
        return "procedure"
    return "other"


@dataclass
class FeatureManifest:
    """
    Complete description of all features used by the world model.

    State (involuntary, continuous):
        vital_codes + lab_codes  →  n_continuous features
        Standard state_dim  = 2 * n_continuous  (values + obs mask)
        Enriched state_dim  = 4 * n_continuous  (values + mask + time_since + carry_fwd)

    Action (voluntary, binary presence per bin):
        med_codes + infusion_codes + proc_codes  →  n_actions features

    Demographic context (static per admission, one-hot encoded).
    """

    vital_codes: List[str]
    lab_codes: List[str]
    med_codes: List[str]
    infusion_codes: List[str]
    proc_codes: List[str]
    demo_cols: List[str]
    demo_categories: Dict[str, List[str]]
    state_mean: Dict[str, float]
    state_std: Dict[str, float]
    code_descriptions: Dict[str, str]
    sex_categories: List[str] = None

    # ── Derived dimensions ───────────────────────────────────────────────────

    @property
    def n_vitals(self) -> int:
        return len(self.vital_codes)

    @property
    def n_labs(self) -> int:
        return len(self.lab_codes)

    @property
    def n_continuous(self) -> int:
        return self.n_vitals + self.n_labs

    @property
    def state_dim(self) -> int:
        """Standard state dim: [values | obs_mask] = 2 × n_continuous."""
        return 2 * self.n_continuous

    @property
    def enriched_state_dim(self) -> int:
        """Enriched state dim: [values | mask | time_since | carry_fwd] = 4 × n_continuous."""
        return 4 * self.n_continuous

    @property
    def n_actions(self) -> int:
        return len(self.med_codes) + len(self.infusion_codes) + len(self.proc_codes)

    @property
    def demo_dim(self) -> int:
        onehot = sum(len(v) + 1 for v in self.demo_categories.values())
        n_sex = len(self.sex_categories) + 1 if self.sex_categories else 0
        return onehot + 1 + n_sex  # +1 for age scalar

    def summary(self) -> str:
        return (
            f"FeatureManifest\n"
            f"  State  : {self.n_vitals} vitals + {self.n_labs} labs"
            f" = {self.n_continuous} continuous → state_dim={self.state_dim}"
            f" | enriched_state_dim={self.enriched_state_dim}\n"
            f"  Actions: {len(self.med_codes)} meds + {len(self.infusion_codes)} infusions"
            f" + {len(self.proc_codes)} procedures → n_actions={self.n_actions}\n"
            f"  Demo   : {self.demo_cols} → demo_dim={self.demo_dim}"
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "vital_codes": self.vital_codes,
                    "lab_codes": self.lab_codes,
                    "med_codes": self.med_codes,
                    "infusion_codes": self.infusion_codes,
                    "proc_codes": self.proc_codes,
                    "demo_cols": self.demo_cols,
                    "demo_categories": self.demo_categories,
                    "sex_categories": self.sex_categories or [],
                    "state_mean": self.state_mean,
                    "state_std": self.state_std,
                    "code_descriptions": self.code_descriptions,
                },
                indent=2,
            )
        )
        log.info("FeatureManifest saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "FeatureManifest":
        data = json.loads(Path(path).read_text())
        data.setdefault("sex_categories", [])
        return cls(**data)


def build_feature_manifest(
    train_df: pd.DataFrame,
    cfg: FeatureConfig,
    codes_df: Optional[pd.DataFrame] = None,
) -> FeatureManifest:
    """
    Derive the feature manifest from the training split only.

    Parameters
    ----------
    train_df  : Raw MEDS training dataframe.
    cfg       : FeatureConfig (top-k limits, min frequency).
    codes_df  : Optional codes.parquet for human-readable descriptions.

    Returns
    -------
    FeatureManifest ready for use in build_transitions / build_enriched_transitions.
    """
    id_col = "patient_id" if "patient_id" in train_df.columns else "subject_id"
    n_subjects = train_df[id_col].nunique()
    log.info("Building manifest from %d subjects, %d events", n_subjects, len(train_df))

    # Code descriptions
    code_desc: Dict[str, str] = {}
    if codes_df is not None and "description" in codes_df.columns:
        code_desc = (
            codes_df[codes_df["description"].notna()]
            .set_index("code")["description"]
            .to_dict()
        )

    # Per-code frequency statistics
    code_stats = (
        train_df.assign(category=train_df["code"].map(classify_code))
        .groupby(["code", "category"])
        .agg(
            n_subjects=(id_col, "nunique"),
            n_events=(id_col, "count"),
            has_numeric=("numeric_value", lambda s: s.notna().any()),
        )
        .reset_index()
    )
    code_stats["subject_freq"] = code_stats["n_subjects"] / n_subjects
    common = code_stats[code_stats["subject_freq"] >= cfg.min_subject_freq]

    def _top_continuous(cat: str, n: int) -> List[str]:
        sub = common[(common["category"] == cat) & common["has_numeric"].eq(True)]
        return sub.nlargest(n, "n_subjects")["code"].tolist()

    def _top_action(cat: str, n: int) -> List[str]:
        sub = common[common["category"] == cat]
        return sub.nlargest(n, "n_subjects")["code"].tolist()

    vital_codes     = _top_continuous("vital",     cfg.top_vitals)
    lab_codes       = _top_continuous("lab",       cfg.top_labs)
    med_codes       = _top_action("medication",    cfg.top_meds)
    infusion_codes  = _top_action("infusion",      cfg.top_infusions)
    proc_codes      = _top_action("procedure",     cfg.top_procedures)

    log.info(
        "Selected %d vitals, %d labs | %d meds, %d infusions, %d procedures",
        len(vital_codes), len(lab_codes), len(med_codes), len(infusion_codes), len(proc_codes),
    )

    # Demographics from HOSPITAL_ADMISSION rows
    demo_col_candidates = ["race"]
    demo_cols = [c for c in demo_col_candidates if c in train_df.columns]
    admit_rows = train_df[train_df["code"].str.startswith("HOSPITAL_ADMISSION//")]
    demo_categories: Dict[str, List[str]] = {
        col: sorted(admit_rows[col].dropna().astype(str).unique().tolist())
        for col in demo_cols
    }

    # Sex categories from GENDER// events
    gender_rows = train_df[train_df["code"].str.startswith("GENDER//")]
    sex_categories: List[str] = sorted(
        gender_rows["code"].str.replace("GENDER//", "", regex=False).unique().tolist()
    ) if len(gender_rows) > 0 else []

    # Normalization stats (training split only)
    all_continuous = vital_codes + lab_codes
    num_events = train_df[
        train_df["code"].isin(all_continuous) & train_df["numeric_value"].notna()
    ]
    state_mean: Dict[str, float] = {}
    state_std:  Dict[str, float] = {}
    for code in all_continuous:
        vals = num_events.loc[num_events["code"] == code, "numeric_value"]
        state_mean[code] = float(vals.mean()) if len(vals) > 0 else 0.0
        sigma = float(vals.std()) if len(vals) > 1 else 1.0
        if np.isnan(sigma) or sigma < 1e-6:
            sigma = 1.0
        state_std[code] = sigma

    return FeatureManifest(
        vital_codes=vital_codes,
        lab_codes=lab_codes,
        med_codes=med_codes,
        infusion_codes=infusion_codes,
        proc_codes=proc_codes,
        demo_cols=demo_cols,
        demo_categories=demo_categories,
        sex_categories=sex_categories,
        state_mean=state_mean,
        state_std=state_std,
        code_descriptions=code_desc,
    )


# =============================================================================
# 3. Data loading
# =============================================================================

def _id_col(df: pd.DataFrame) -> str:
    if "patient_id" in df.columns:
        return "patient_id"
    if "subject_id" in df.columns:
        return "subject_id"
    raise ValueError("DataFrame has neither 'patient_id' nor 'subject_id' column.")


def load_meds_split(data_dir: Path, split: str) -> pd.DataFrame:
    """
    Load one MEDS split from a single-shard layout.

    Works for the demo dataset (data/{split}/0.parquet).
    For multi-shard loading of the full MIMIC dataset, use load_meds_cohort().
    """
    path = Path(data_dir) / "data" / split / "0.parquet"
    if not path.exists():
        raise FileNotFoundError(f"MEDS data not found at {path}")
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    id_col = _id_col(df)
    log.info("Loaded %s: %d patients, %d events", split, df[id_col].nunique(), len(df))
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
    data_dir     : Root data directory (contains data/ and metadata/).
    split        : 'train', 'tuning', or 'held_out'.
    max_patients : Sample this many patients (uses patient_splits.parquet).
    patient_ids  : Load only these patient IDs (overrides max_patients).
    """
    data_dir = Path(data_dir)
    shard_dir = data_dir / "data" / split

    if not shard_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {shard_dir}")
    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet files in {shard_dir}")

    if patient_ids is None and max_patients is not None:
        patient_ids = _sample_patient_ids(data_dir, split, max_patients)

    dfs = []
    for shard in shards:
        chunk = pd.read_parquet(shard)
        chunk["time"] = pd.to_datetime(chunk["time"])
        id_col = _id_col(chunk)
        if patient_ids is not None:
            chunk = chunk[chunk[id_col].isin(patient_ids)]
        if len(chunk) > 0:
            dfs.append(chunk)

    if not dfs:
        raise ValueError(f"No events found for the requested patients in split '{split}'.")

    df = pd.concat(dfs, ignore_index=True)
    id_col = _id_col(df)
    log.info(
        "Loaded %s: %d patients, %d events (%d shards)",
        split, df[id_col].nunique(), len(df), len(shards),
    )
    return df


def _sample_patient_ids(data_dir: Path, split: str, n: int) -> Set[int]:
    """Sample up to n patient IDs from metadata/patient_splits.parquet."""
    splits_path = data_dir / "metadata" / "patient_splits.parquet"
    if splits_path.exists():
        splits_df = pd.read_parquet(splits_path)
        id_col    = "patient_id" if "patient_id" in splits_df.columns else "subject_id"
        split_col = "split" if "split" in splits_df.columns else splits_df.columns[-1]
        ids = splits_df.loc[splits_df[split_col] == split, id_col].unique()
        if len(ids) > n:
            ids = np.random.default_rng(42).choice(ids, size=n, replace=False)
        log.info("Sampled %d patients from %s split", len(ids), split)
        return set(ids.tolist())

    # Fallback: scan first shard
    log.warning("patient_splits.parquet not found — scanning first shard")
    first = sorted((data_dir / "data" / split).glob("*.parquet"))[0]
    tmp   = pd.read_parquet(first)
    id_col = _id_col(tmp)
    ids    = tmp[id_col].unique()
    if len(ids) > n:
        ids = np.random.default_rng(42).choice(ids, size=n, replace=False)
    return set(ids.tolist())


# =============================================================================
# 4. Trajectory construction
# =============================================================================

# ── Per-bin helpers ──────────────────────────────────────────────────────────

def _state_vector(
    bin_df: pd.DataFrame,
    manifest: FeatureManifest,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate events in one time bin → (z-score values, obs mask).

    Returns
    -------
    values : float32 (n_continuous,)  — z-score normalised, 0 if unobserved
    mask   : float32 (n_continuous,)  — 1 if observed, 0 otherwise
    """
    all_codes = manifest.vital_codes + manifest.lab_codes
    n      = len(all_codes)
    values = np.zeros(n, dtype=np.float32)
    mask   = np.zeros(n, dtype=np.float32)

    if len(bin_df) == 0:
        return values, mask

    num = bin_df[bin_df["numeric_value"].notna()]
    if len(num) == 0:
        return values, mask

    grp = num.groupby("code")["numeric_value"].mean()
    for i, code in enumerate(all_codes):
        if code in grp.index:
            raw      = grp[code]
            mu, sigma = manifest.state_mean.get(code, 0.0), manifest.state_std.get(code, 1.0)
            values[i] = float(np.clip((raw - mu) / sigma, -5.0, 5.0))
            mask[i]   = 1.0

    return values, mask


def _action_vector(bin_df: pd.DataFrame, manifest: FeatureManifest) -> np.ndarray:
    """Binary presence of each med / infusion / procedure in this bin."""
    all_actions = manifest.med_codes + manifest.infusion_codes + manifest.proc_codes
    present     = set(bin_df["code"].unique())
    return np.array([1.0 if c in present else 0.0 for c in all_actions], dtype=np.float32)


def _encode_demographics(
    admit_row: pd.Series,
    birth_time: Optional[pd.Timestamp],
    admit_time: pd.Timestamp,
    manifest: FeatureManifest,
    sex: Optional[str] = None,
) -> np.ndarray:
    """One-hot race + normalised age + one-hot sex → flat float32 vector."""
    parts: List[np.ndarray] = []
    for col in manifest.demo_cols:
        cats = manifest.demo_categories[col]
        val  = str(admit_row.get(col, "")) if pd.notna(admit_row.get(col, None)) else ""
        vec  = np.zeros(len(cats) + 1, dtype=np.float32)
        vec[cats.index(val) if val in cats else len(cats)] = 1.0
        parts.append(vec)

    if birth_time is not None and pd.notna(birth_time):
        age_norm = float(np.clip(
            (admit_time - birth_time).total_seconds() / (365.25 * 24 * 3600), 0, 100
        )) / 100.0
    else:
        age_norm = 0.5
    parts.append(np.array([age_norm], dtype=np.float32))

    sex_cats = manifest.sex_categories or []
    if sex_cats:
        vec = np.zeros(len(sex_cats) + 1, dtype=np.float32)
        vec[sex_cats.index(sex) if sex in sex_cats else len(sex_cats)] = 1.0
        parts.append(vec)

    return np.concatenate(parts) if parts else np.array([age_norm], dtype=np.float32)


# ── Standard transitions ─────────────────────────────────────────────────────

def _admission_transitions(
    admit_df: pd.DataFrame,
    demo_vec: np.ndarray,
    manifest: FeatureManifest,
    bin_hours: float,
    patient_id: Optional[int] = None,
    hadm_id: Optional[int] = None,
) -> List[Dict]:
    """Convert one admission's event stream into (s_t, a_t, d, s_next, m_next) dicts."""
    admit_df = admit_df.sort_values("time").copy()
    clinical  = admit_df[~admit_df["time"].isna()].copy()
    if len(clinical) < 2:
        return []

    t0, t_end   = clinical["time"].iloc[0], clinical["time"].iloc[-1]
    total_hours = (t_end - t0).total_seconds() / 3600.0
    if total_hours < bin_hours:
        return []

    n_bins    = max(2, int(np.ceil(total_hours / bin_hours)))
    bin_edges = [t0 + pd.Timedelta(hours=i * bin_hours) for i in range(n_bins + 1)]

    states, masks, actions = [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bdf    = clinical[(clinical["time"] >= lo) & (clinical["time"] < hi)]
        s, m   = _state_vector(bdf, manifest)
        a      = _action_vector(bdf, manifest)
        states.append(s); masks.append(m); actions.append(a)

    transitions = []
    for i in range(n_bins - 1):
        if masks[i].sum() == 0 and masks[i + 1].sum() == 0:
            continue
        t = {
            "s_t":    np.concatenate([states[i],     masks[i]]),
            "a_t":    actions[i],
            "d":      demo_vec,
            "s_next": np.concatenate([states[i + 1], masks[i + 1]]),
            "m_next": masks[i + 1],
        }
        if patient_id is not None: t["patient_id"] = patient_id
        if hadm_id    is not None: t["hadm_id"]    = hadm_id
        transitions.append(t)

    return transitions


def build_transitions(
    df: pd.DataFrame,
    manifest: FeatureManifest,
    cfg: FeatureConfig,
) -> List[Dict]:
    """
    Build all standard transitions from a MEDS split dataframe.

    State = [z-score values | obs_mask]  (2 × n_continuous).

    Parameters
    ----------
    df       : Raw MEDS events dataframe.
    manifest : FeatureManifest from build_feature_manifest().
    cfg      : FeatureConfig (bin_hours used here).

    Returns
    -------
    List of transition dicts, each containing:
        s_t, a_t, d, s_next, m_next, patient_id, hadm_id (last two optional).
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

    all_transitions: List[Dict] = []
    has_hadm = "hadm_id" in df.columns and df["hadm_id"].notna().any()

    for patient_id, sub_df in df.groupby(id_col):
        birth_time = birth_map.get(patient_id)
        sex        = sex_map.get(patient_id)

        if has_hadm:
            for hadm_id, admit_df in sub_df[sub_df["hadm_id"].notna()].groupby("hadm_id"):
                admit_event = admit_df[admit_df["code"].str.startswith("HOSPITAL_ADMISSION//")]
                demo_cols_p = [c for c in manifest.demo_cols if c in admit_df.columns]
                admit_row   = admit_event.iloc[0] if len(admit_event) > 0 and demo_cols_p else pd.Series(dtype=object)
                admit_time  = admit_df["time"].dropna().min()
                demo_vec    = _encode_demographics(admit_row, birth_time, admit_time, manifest, sex=sex)
                all_transitions.extend(
                    _admission_transitions(
                        admit_df, demo_vec, manifest, cfg.bin_hours,
                        patient_id=int(patient_id), hadm_id=int(hadm_id),
                    )
                )
        else:
            admit_rows = sub_df[sub_df["code"].str.startswith("HOSPITAL_ADMISSION//")]
            row        = admit_rows.iloc[0] if len(admit_rows) > 0 else pd.Series(dtype=object)
            admit_time = sub_df["time"].dropna().min()
            demo_vec   = _encode_demographics(row, birth_time, admit_time, manifest, sex=sex)
            all_transitions.extend(
                _admission_transitions(
                    sub_df, demo_vec, manifest, cfg.bin_hours,
                    patient_id=int(patient_id),
                )
            )

    log.info("Built %d transitions from %d patients", len(all_transitions), df[id_col].nunique())
    return all_transitions


# ── Enriched transitions (4-channel state encoding) ──────────────────────────

def _state_vector_enriched(
    bin_df: pd.DataFrame,
    manifest: FeatureManifest,
    prev_values: np.ndarray,
    prev_obs_hours: np.ndarray,
    bin_start_hours: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute one bin's enriched state and update carry-forward trackers.

    State channels (per feature):
        values        — z-score normalised current value (0 if unobserved)
        obs_mask      — 1 if measured this bin, 0 otherwise
        time_since_norm — hours since last obs / 168 h (7-day cap), 1 if never
        carry_fwd     — exp-decay EWMA of last known value (τ = 12 h)

    Returns
    -------
    state        : (4 * n_continuous,)
    new_values   : updated carry-forward values
    new_obs_hours: updated last-observation timestamps
    mask         : (n_continuous,) raw observation mask
    """
    all_codes = manifest.vital_codes + manifest.lab_codes
    n      = len(all_codes)
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
                    sigma = manifest.state_std.get(code, 1.0)
                    values[i] = float(np.clip((raw - mu) / sigma, -5.0, 5.0))
                    mask[i]   = 1.0

    MAX_HOURS = 168.0  # 7-day cap
    TAU       = 12.0   # EWMA half-life (hours)

    time_since_norm = np.ones(n, dtype=np.float32)
    carry_fwd       = np.zeros(n, dtype=np.float32)
    new_values      = prev_values.copy()
    new_obs_hours   = prev_obs_hours.copy()

    for i in range(n):
        if mask[i] == 1.0:
            time_since_norm[i] = 0.0
            carry_fwd[i]       = values[i]
            new_values[i]      = values[i]
            new_obs_hours[i]   = bin_start_hours
        else:
            hrs = bin_start_hours - prev_obs_hours[i]
            if hrs < 0 or np.isinf(hrs):
                time_since_norm[i] = 1.0
                carry_fwd[i]       = 0.0
            else:
                time_since_norm[i] = float(np.clip(hrs / MAX_HOURS, 0.0, 1.0))
                alpha              = float(np.exp(-hrs / TAU))
                carry_fwd[i]       = alpha * prev_values[i]
                new_values[i]      = carry_fwd[i]

    return np.concatenate([values, mask, time_since_norm, carry_fwd]), new_values, new_obs_hours, mask


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
    """Build enriched 4-channel transitions for one hospital admission."""
    admit_df = admit_df.sort_values("time").copy()
    clinical  = admit_df[~admit_df["time"].isna()].copy()
    if len(clinical) < 2:
        return []

    t0, t_end   = clinical["time"].iloc[0], clinical["time"].iloc[-1]
    total_hours = (t_end - t0).total_seconds() / 3600.0
    if total_hours < bin_hours:
        return []

    n_bins    = max(2, int(np.ceil(total_hours / bin_hours)))
    bin_edges = [t0 + pd.Timedelta(hours=i * bin_hours) for i in range(n_bins + 1)]

    n              = manifest.n_continuous
    prev_values    = np.zeros(n, dtype=np.float32)
    prev_obs_hours = np.full(n, -np.inf, dtype=np.float32)

    states, masks, actions = [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bdf    = clinical[(clinical["time"] >= lo) & (clinical["time"] < hi)]
        state, prev_values, prev_obs_hours, mask = _state_vector_enriched(
            bdf, manifest, prev_values, prev_obs_hours, float(i * bin_hours)
        )
        states.append(state); masks.append(mask)
        actions.append(_action_vector(bdf, manifest))

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
    Each dict carries stratification metadata: sex, age, race, bmi.
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

        if has_hadm:
            for hadm_id, admit_df in sub_df[sub_df["hadm_id"].notna()].groupby("hadm_id"):
                admit_event = admit_df[admit_df["code"].str.startswith("HOSPITAL_ADMISSION//")]
                demo_cols_p = [c for c in manifest.demo_cols if c in admit_df.columns]
                admit_row   = (
                    admit_event.iloc[0]
                    if len(admit_event) > 0 and demo_cols_p
                    else pd.Series(dtype=object)
                )
                admit_time = admit_df["time"].dropna().min()
                demo_vec   = _encode_demographics(admit_row, birth_time, admit_time, manifest, sex=pt_sex)
                age  = (admit_time - birth_time).total_seconds() / (365.25 * 24 * 3600) if birth_time is not None and pd.notna(birth_time) else None
                race = str(admit_row.get("race")) if "race" in admit_row.index and pd.notna(admit_row.get("race")) else None
                all_transitions.extend(
                    _admission_enriched_transitions(
                        admit_df, demo_vec, manifest, cfg.bin_hours,
                        patient_id=int(patient_id), hadm_id=int(hadm_id),
                        sex=pt_sex, age=age, race=race, bmi=pt_bmi,
                    )
                )
        else:
            admit_rows = sub_df[sub_df["code"].str.startswith("HOSPITAL_ADMISSION//")]
            row        = admit_rows.iloc[0] if len(admit_rows) > 0 else pd.Series(dtype=object)
            admit_time = sub_df["time"].dropna().min()
            demo_vec   = _encode_demographics(row, birth_time, admit_time, manifest, sex=pt_sex)
            age  = (admit_time - birth_time).total_seconds() / (365.25 * 24 * 3600) if birth_time is not None and pd.notna(birth_time) else None
            race = str(row.get("race")) if "race" in row.index and pd.notna(row.get("race")) else None
            all_transitions.extend(
                _admission_enriched_transitions(
                    sub_df, demo_vec, manifest, cfg.bin_hours,
                    patient_id=int(patient_id),
                    sex=pt_sex, age=age, race=race, bmi=pt_bmi,
                )
            )

    log.info(
        "Built %d enriched transitions from %d patients",
        len(all_transitions), df[id_col].nunique(),
    )
    return all_transitions


# ── Sequence batching ─────────────────────────────────────────────────────────

def build_sequence_batches(
    transitions: List[Dict],
    seq_len: int = 8,
    stride: int = 4,
) -> List[Dict]:
    """
    Slide a fixed-length window over each (patient, admission) trajectory.

    Returns list of sequence dicts, each with keys:
        s, actions, s_next, m_next  — shape (seq_len, …)
        d, patient_id, hadm_id, sex, age, race, bmi  — metadata
    """
    groups: dict = defaultdict(list)
    for t in transitions:
        groups[(t.get("patient_id", -1), t.get("hadm_id", -1))].append(t)

    sequences = []
    for (pid, hid), trans in groups.items():
        T = len(trans)
        if T < seq_len:
            continue
        for start in range(0, T - seq_len + 1, stride):
            win = trans[start: start + seq_len]
            sequences.append({
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
            })

    log.info(
        "build_sequence_batches: %d admissions → %d sequences (seq_len=%d, stride=%d)",
        len(groups), len(sequences), seq_len, stride,
    )
    return sequences


# =============================================================================
# 5. PyTorch Datasets
# =============================================================================

class TransitionDataset(Dataset):
    """Stores all (s_t, a_t, d, s_next, m_next) transitions as stacked tensors."""

    def __init__(self, transitions: List[Dict]):
        if not transitions:
            raise ValueError("No transitions found — check data and feature config.")

        def _s(key: str) -> torch.Tensor:
            return torch.tensor(np.stack([t[key] for t in transitions]), dtype=torch.float32)

        self.s_t    = _s("s_t")
        self.a_t    = _s("a_t")
        self.d      = _s("d")
        self.s_next = _s("s_next")
        self.m_next = _s("m_next")
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
        return (
            f"TransitionDataset: {len(self)} transitions | "
            f"state_dim={self.s_t.shape[1]} | action_dim={self.a_t.shape[1]} | "
            f"demo_dim={self.d.shape[1]} | avg obs rate={self.m_next.mean():.1%}"
        )


class EnrichedSequenceDataset(Dataset):
    """Sequence dataset for enriched 4-channel states with stratification metadata."""

    def __init__(self, sequences: List[Dict]):
        if not sequences:
            raise ValueError("No sequences found.")

        def _s(key: str) -> torch.Tensor:
            return torch.tensor(np.stack([s[key] for s in sequences]), dtype=torch.float32)

        self.s       = _s("s")
        self.actions = _s("actions")
        self.s_next  = _s("s_next")
        self.m_next  = _s("m_next")
        self.d       = _s("d")
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


# =============================================================================
# 6. EHR-SHOT label extraction
# =============================================================================
# Labels mirror those used in the EHR-SHOT benchmark and world_model_v6.ipynb.
# Three groups:
#   A. Original chronic-disease flags  (HTN, DM, HF, CAD, CKD, AKI, Sepsis, COPD, Cancer)
#   B. Additional ICD disease flags    (acute_mi, lupus, hyperlipidemia, hypertension
#                                       expanded, celiac, pancreatic_cancer)
#   C. Clinical outcome labels         (in_hospital_mortality, icu_admission,
#                                       los_days / los_long, readmit_30d / readmit_any)
#
# Main entry point: extract_ehrshot_labels(df, id_col)
# Returns one row per patient with all three groups + demographics.

# ── A. Original chronic-disease definitions ──────────────────────────────────

ORIG_DISEASE_DEF: Dict[str, List[Tuple[str, str]]] = {
    "HTN":    [("10", "I10"), ("9", "401")],
    "DM":     [("10", "E10"), ("10", "E11"), ("9", "250")],
    "HF":     [("10", "I50"), ("9", "428")],
    "CAD":    [("10", "I25"), ("9", "414"), ("9", "410"), ("9", "412")],
    "CKD":    [("10", "N18"), ("9", "585")],
    "AKI":    [("10", "N17"), ("9", "584")],
    "Sepsis": [("10", "A41"), ("10", "R652"), ("9", "038"), ("9", "9959")],
    "COPD":   [("10", "J44"), ("9", "496"), ("9", "491"), ("9", "492")],
    "Cancer": [
        ("10", "C"),
        ("9", "14"), ("9", "15"), ("9", "16"), ("9", "17"), ("9", "18"),
        ("9", "19"), ("9", "20"), ("9", "21"), ("9", "22"), ("9", "23"),
        ("9", "195"), ("9", "196"), ("9", "197"), ("9", "198"), ("9", "199"),
    ],
}

# CKD staging (ICD-10 N18.x and ICD-9 585x)
CKD_STAGE_MAP: Dict[str, int] = {
    "DIAGNOSIS//ICD//10//N181": 1,
    "DIAGNOSIS//ICD//10//N182": 2,
    "DIAGNOSIS//ICD//10//N183": 3,
    "DIAGNOSIS//ICD//10//N184": 4,
    "DIAGNOSIS//ICD//10//N185": 5,
    "DIAGNOSIS//ICD//10//N186": 5,
    "DIAGNOSIS//ICD//9//5851":  1,
    "DIAGNOSIS//ICD//9//5852":  2,
    "DIAGNOSIS//ICD//9//5853":  3,
    "DIAGNOSIS//ICD//9//5854":  4,
    "DIAGNOSIS//ICD//9//5855":  5,
    "DIAGNOSIS//ICD//9//5856":  5,
}

# Primary disease assignment priority (first match wins)
PRIMARY_PRIORITY = ["Sepsis", "AKI", "HF", "CKD", "COPD", "DM", "CAD", "Cancer", "HTN"]

# ── B. Additional EHR-SHOT disease definitions ───────────────────────────────

NEW_DISEASE_DEF: Dict[str, List[Tuple[str, str]]] = {
    "acute_mi":          [("10", "I21"),  ("10", "I22"),  ("9", "410"),  ("9", "411")],
    "lupus":             [("10", "M32"),  ("9", "7100")],
    "hyperlipidemia":    [("10", "E78"),  ("9", "272")],
    "hypertension":      [("10", "I10"),  ("10", "I11"),  ("10", "I12"),
                          ("10", "I13"),  ("10", "I15"),
                          ("9",  "401"),  ("9",  "402"),  ("9",  "403"),
                          ("9",  "404"),  ("9",  "405")],
    "celiac":            [("10", "K900"), ("10", "K90"),  ("9",  "5790"), ("9", "579")],
    "pancreatic_cancer": [("10", "C25"),  ("9",  "157")],
}

# Column name groups (for downstream use)
ORIG_DISEASE_COLS = [f"has_{d}" for d in ORIG_DISEASE_DEF]
NEW_DISEASE_COLS  = [f"has_{d}" for d in NEW_DISEASE_DEF]
OUTCOME_COLS      = [
    "in_hospital_mortality", "icu_admission",
    "los_long", "readmit_30d", "readmit_any",
]
ALL_EHRSHOT_COLS  = ORIG_DISEASE_COLS + NEW_DISEASE_COLS + OUTCOME_COLS


# ── Shared helper ─────────────────────────────────────────────────────────────

def extract_disease_flags(
    df: pd.DataFrame,
    disease_defs: Dict[str, List[Tuple[str, str]]],
    id_col: str = "patient_id",
) -> pd.DataFrame:
    """
    Vectorised binary disease flag extraction for any ICD prefix definition dict.

    Uses str.startswith(tuple) — no per-patient Python loops.

    Returns
    -------
    DataFrame with one row per patient and columns has_{disease} (int 0/1).
    """
    diag   = df[df["code"].str.startswith("DIAGNOSIS//ICD//")][[id_col, "code"]]
    result = df[[id_col]].drop_duplicates().reset_index(drop=True)

    for disease, prefixes in disease_defs.items():
        meds_prefixes = tuple(f"DIAGNOSIS//ICD//{v}//{p}" for v, p in prefixes)
        hit_pids      = diag.loc[diag["code"].str.startswith(meds_prefixes), id_col]
        result[f"has_{disease}"] = result[id_col].isin(hit_pids).astype(int)

    return result


# ── C. Clinical outcome labels ────────────────────────────────────────────────

def extract_outcome_labels(df: pd.DataFrame, id_col: str = "patient_id") -> pd.DataFrame:
    """
    Compute in-hospital outcome labels for each patient.

    Outcomes
    --------
    in_hospital_mortality : 1 if a MEDS_DEATH / DEATH / EXPIR event is present
    icu_admission         : 1 if any ICU-related transfer code is present
    los_days              : float, days from first admission to last discharge (NaN if unknown)
    los_long              : 1 if los_days > 7, else 0
    readmit_any           : 1 if ≥ 2 HOSPITAL_ADMISSION events
    readmit_30d           : 1 if the gap between first discharge and second admission ≤ 30 days
    """
    adm = (
        df[df["code"].str.startswith("HOSPITAL_ADMISSION//")]
        [[id_col, "time"]].dropna().sort_values([id_col, "time"])
    )
    dis = (
        df[df["code"].str.startswith("HOSPITAL_DISCHARGE//")]
        [[id_col, "time"]].dropna().sort_values([id_col, "time"])
    )

    # Mortality
    dth = df[
        (df["code"].str.upper().str.contains("DEATH|EXPIR|DEAD", na=False)) |
        (df["code"] == "MEDS_DEATH")
    ][[id_col]].drop_duplicates().copy()
    dth["in_hospital_mortality"] = 1

    # ICU admission
    icu_pat  = r"ICU|INTENSIVE_CARE|CRITICAL_CARE|MICU|SICU|CVICU|NICU|CCU|TRANSFER.*ICU"
    icu_pids = (
        df[df["code"].str.upper().str.contains(icu_pat, na=False, regex=True)]
        [[id_col]].drop_duplicates().copy()
    )
    icu_pids["icu_admission"] = 1

    # Length of stay
    first_adm = adm.groupby(id_col)["time"].min().rename("adm_time")
    last_dis  = dis.groupby(id_col)["time"].max().rename("dis_time")
    los_df    = pd.concat([first_adm, last_dis], axis=1).dropna()
    los_df.index.name = id_col
    los_df["los_days"] = (los_df["dis_time"] - los_df["adm_time"]).dt.total_seconds() / 86400.0
    los_df.loc[los_df["los_days"] <= 0, "los_days"] = float("nan")
    los_df["los_long"] = (los_df["los_days"] > 7).astype(float)
    los_df = los_df[["los_days", "los_long"]].reset_index()

    # 30-day readmission
    adm_cnt          = adm.groupby(id_col)["time"].count().rename("n_adm")
    readmit_any_pids = adm_cnt[adm_cnt >= 2].index

    first_dis_ser  = dis.groupby(id_col)["time"].min().rename("first_dis")
    second_adm_ser = (
        adm.sort_values([id_col, "time"])
        .groupby(id_col)["time"]
        .nth(1)
        .rename("second_adm")
    )
    readmit_df = pd.concat([first_dis_ser, second_adm_ser], axis=1).dropna()
    readmit_df.index.name = id_col
    readmit_df = readmit_df[readmit_df.index.isin(readmit_any_pids)].copy()
    readmit_df["readmit_any"] = 1
    gap_days = (readmit_df["second_adm"] - readmit_df["first_dis"]).dt.total_seconds() / 86400.0
    readmit_df["readmit_30d"] = ((gap_days > 0) & (gap_days <= 30)).astype(int)
    readmit_df = readmit_df[["readmit_any", "readmit_30d"]].reset_index()

    # Merge all outcomes
    all_pids = df[[id_col]].drop_duplicates()
    out = (
        all_pids
        .merge(los_df,     on=id_col, how="left")
        .merge(readmit_df, on=id_col, how="left")
        .merge(icu_pids,   on=id_col, how="left")
        .merge(dth,        on=id_col, how="left")
    )
    out["readmit_any"]           = out["readmit_any"].fillna(0).astype(int)
    out["readmit_30d"]           = out["readmit_30d"].fillna(0).astype(int)
    out["icu_admission"]         = out["icu_admission"].fillna(0).astype(int)
    out["in_hospital_mortality"] = out["in_hospital_mortality"].fillna(0).astype(int)
    return out


# ── Demographics ──────────────────────────────────────────────────────────────

def compute_demographics(df: pd.DataFrame, id_col: str = "patient_id") -> pd.DataFrame:
    """
    Compute per-patient demographic summary from the MEDS event stream.

    Returns
    -------
    DataFrame with columns:
        age       — float years (NaN if birth/admission time missing)
        sex       — 'M', 'F', or 'Unknown'
        bmi       — float kg/m² (NaN if weight or height unavailable)
        age_group — '<50', '50-65', '65-80', '>80'
        bmi_group — 'Underweight', 'Normal', 'Overweight', 'Obese'
    """
    birth_map = df[df["code"] == "MEDS_BIRTH"].groupby(id_col)["time"].min()
    admit_map = (
        df[df["code"].str.startswith("HOSPITAL_ADMISSION//")]
        .groupby(id_col)["time"].min()
    )
    sex_map = (
        df[df["code"].str.startswith("GENDER//")]
        .groupby(id_col)["code"]
        .first()
        .str.replace("GENDER//", "", regex=False)
    )

    # Weight / height codes used for BMI (MIMIC-IV LAB item IDs)
    wt_codes = [
        c for c in df["code"].unique()
        if any(c.startswith(f"LAB//{i}") for i in ["224639", "226512", "226531"])
    ]
    ht_codes = [c for c in df["code"].unique() if c.startswith("LAB//226730")]
    wt_map   = df[df["code"].isin(wt_codes)].groupby(id_col)["numeric_value"].median()
    ht_map   = df[df["code"].isin(ht_codes)].groupby(id_col)["numeric_value"].median()

    records = []
    for pid in df[id_col].unique():
        age = np.nan
        if pid in birth_map.index and pid in admit_map.index:
            age = float(
                np.clip(
                    (admit_map[pid] - birth_map[pid]).total_seconds() / (365.25 * 24 * 3600),
                    0, 120,
                )
            )
        wt  = wt_map.get(pid, np.nan)
        ht  = ht_map.get(pid, np.nan)
        bmi = wt / ((ht / 100) ** 2) if (not np.isnan(wt) and not np.isnan(ht) and ht > 50) else np.nan
        records.append({
            id_col: pid,
            "age": age,
            "sex": sex_map.get(pid, "Unknown"),
            "bmi": bmi,
        })

    demo = pd.DataFrame(records)
    demo["age_group"] = pd.cut(
        demo["age"].fillna(65),
        bins=[0, 50, 65, 80, 200],
        labels=["<50", "50-65", "65-80", ">80"],
    ).astype(str)
    demo["bmi_group"] = pd.cut(
        demo["bmi"],
        bins=[0, 18.5, 25, 30, 200],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    ).astype(str)
    return demo


# ── Main label extraction entry point ────────────────────────────────────────

def extract_ehrshot_labels(df: pd.DataFrame, id_col: str = "patient_id") -> pd.DataFrame:
    """
    Extract the full EHR-SHOT label set for each patient.

    Combines:
      • Original chronic-disease flags  (ORIG_DISEASE_COLS)
      • Additional ICD disease flags    (NEW_DISEASE_COLS)
      • Clinical outcome labels         (OUTCOME_COLS)
      • Demographics                    (age, sex, bmi, age_group, bmi_group)
      • CKD staging (ckd_stage 0–5) and primary_disease / comorbidity_n

    Parameters
    ----------
    df     : Raw MEDS events dataframe.
    id_col : Patient identifier column ('patient_id' or 'subject_id').

    Returns
    -------
    DataFrame with one row per patient containing all columns listed above.
    See ALL_EHRSHOT_COLS for the full classification task column list.
    """
    # Original chronic diseases + CKD stage + primary_disease + comorbidity_n
    orig_flags = extract_disease_flags(df, ORIG_DISEASE_DEF, id_col).set_index(id_col)

    diag     = df[df["code"].str.startswith("DIAGNOSIS//ICD//")][[id_col, "code"]]
    ckd_rows = diag[diag["code"].isin(CKD_STAGE_MAP)].copy()
    ckd_rows["stage"] = ckd_rows["code"].map(CKD_STAGE_MAP)
    ckd_stage = ckd_rows.groupby(id_col)["stage"].max()
    orig_flags["ckd_stage"] = orig_flags.index.map(ckd_stage).fillna(0).astype(int)

    conditions = [orig_flags[f"has_{d}"].astype(bool) for d in PRIMARY_PRIORITY]
    orig_flags["primary_disease"] = np.select(conditions, PRIMARY_PRIORITY, default="Other")
    orig_flags["comorbidity_n"]   = orig_flags[ORIG_DISEASE_COLS].sum(axis=1)

    # Additional EHR-SHOT disease flags
    new_flags = extract_disease_flags(df, NEW_DISEASE_DEF, id_col)

    # Clinical outcome labels
    outcomes = extract_outcome_labels(df, id_col)

    # Demographics
    demo = compute_demographics(df, id_col)

    # Merge everything
    base = df[[id_col]].drop_duplicates()
    labels = (
        base
        .merge(orig_flags.reset_index(), on=id_col, how="left")
        .merge(new_flags,                on=id_col, how="left")
        .merge(outcomes,                 on=id_col, how="left")
        .merge(demo,                     on=id_col, how="left", suffixes=("", "_dup"))
    )
    labels = labels[[c for c in labels.columns if not c.endswith("_dup")]]

    log.info(
        "EHR-SHOT labels for %d patients | orig diseases: %s | outcomes: mortality=%d icu=%d readmit30d=%d",
        len(labels),
        {d: int(labels[f"has_{d}"].sum()) for d in list(ORIG_DISEASE_DEF)[:5]},
        int(labels["in_hospital_mortality"].sum()),
        int(labels["icu_admission"].sum()),
        int(labels["readmit_30d"].sum()),
    )
    return labels


# ── Attach labels to transitions ──────────────────────────────────────────────

def attach_labels_to_transitions(
    transitions: List[Dict],
    labels_df: pd.DataFrame,
    id_col: str = "patient_id",
) -> List[Dict]:
    """
    Merge EHR-SHOT labels into each transition dict.

    Transitions must have a 'patient_id' field.
    Any transition whose patient_id is missing from labels_df is returned unchanged.
    """
    label_map = labels_df.set_index(id_col).to_dict(orient="index")
    enriched = []
    for t in transitions:
        pid = t.get("patient_id")
        if pid is not None and pid in label_map:
            t = {**t, **label_map[pid]}
        enriched.append(t)
    return enriched
