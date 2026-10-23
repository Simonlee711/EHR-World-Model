"""
Feature manifest construction for the EHR World Model.

MIMIC-IV MEDS code format
--------------------------
  LAB//{itemid}//{unit}     — chartevents (ICU vitals, itemid 220xxx) and
                               labevents (labs, itemid 50xxx–51xxx)
  MEDICATION//START//{name} — medication order started
  INFUSION_START//{itemid}  — continuous IV infusion started
  PROCEDURE//START//{itemid}— clinical procedure started
  HOSPITAL_ADMISSION//...   — admission event (carries demographics as columns)
  MEDS_BIRTH                — birth date event (for age calculation)
  GENDER//{M|F}             — sex event

Feature taxonomy
----------------
  State (involuntary) — what the patient's body is doing:
    • vitals : LAB// codes with ICU chartevent IDs  (220 000 – 229 999)
    • labs   : LAB// codes with labevent item IDs   (50 000 –  59 999)

  Action (voluntary) — what clinicians choose to do:
    • medications : MEDICATION//START// codes
    • infusions   : INFUSION_START// codes
    • procedures  : PROCEDURE//START// codes

  Context (static demographics):
    • columns: insurance, language, marital_status, race
    • age computed from MEDS_BIRTH vs first admission time

All statistics (mean, std) are computed from the training split only.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import FeatureConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Code classification helpers
# ---------------------------------------------------------------------------

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


# Standalone vital sign codes that don't use the LAB// prefix
_STANDALONE_VITALS = frozenset({
    "Blood Pressure",
    "Blood Pressure Sitting",
    "Blood Pressure Lying",
    "Weight (Lbs)",
    "BMI (kg/m2)",
    "Height (Inches)",
})


def classify_code(code: str) -> str:
    """
    Return one of: 'vital', 'lab', 'medication', 'infusion', 'procedure', 'other'.
    """
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


# ---------------------------------------------------------------------------
# FeatureManifest
# ---------------------------------------------------------------------------

@dataclass
class FeatureManifest:
    """
    Complete description of all features used by the world model.

    State features (involuntary) — continuous, partially observed per bin:
        vital_codes + lab_codes  →  n_continuous = n_vitals + n_labs features
        state vector = [normalised values (n_continuous), observation mask (n_continuous)]
        → total state_dim = 2 * n_continuous

    Action features (voluntary) — binary presence per bin:
        med_codes + infusion_codes + proc_codes  →  n_actions features

    Demographic context — static per admission, one-hot encoded:
        → demo_dim = sum(len(cats)+1 for each demo column) + 1 (for age)
    """

    # State (involuntary)
    vital_codes: List[str]
    lab_codes: List[str]

    # Action (voluntary)
    med_codes: List[str]
    infusion_codes: List[str]
    proc_codes: List[str]

    # Demographics
    demo_cols: List[str]
    demo_categories: Dict[str, List[str]]   # sorted unique values per column

    # Normalization (training set only)
    state_mean: Dict[str, float]
    state_std: Dict[str, float]

    # Optional: human-readable descriptions from codes.parquet
    code_descriptions: Dict[str, str]

    # Sex encoded separately from GENDER// events (default None for back-compat)
    sex_categories: List[str] = None   # e.g. ['F', 'M']

    # ── Derived dimensions ──────────────────────────────────────────────────

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
        """State encoder input dim: values + observation mask."""
        return 2 * self.n_continuous

    @property
    def n_actions(self) -> int:
        return len(self.med_codes) + len(self.infusion_codes) + len(self.proc_codes)

    @property
    def demo_dim(self) -> int:
        """One-hot encoded demographic vector length (includes UNK slot + age + sex)."""
        onehot = sum(len(v) + 1 for v in self.demo_categories.values())
        n_sex = len(self.sex_categories) + 1 if self.sex_categories else 0
        return onehot + 1 + n_sex  # +1 age, +n_sex sex one-hot

    # ── Summary ─────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            "FeatureManifest",
            f"  State (involuntary) : {self.n_vitals} vitals + {self.n_labs} labs"
            f"  = {self.n_continuous} continuous  →  state_dim={self.state_dim}",
            f"  Action (voluntary)  : {len(self.med_codes)} meds"
            f" + {len(self.infusion_codes)} infusions"
            f" + {len(self.proc_codes)} procedures"
            f"  →  n_actions={self.n_actions}",
            f"  Demographics        : {self.demo_cols}  →  demo_dim={self.demo_dim}",
        ]
        return "\n".join(lines)

    # ── Serialisation ────────────────────────────────────────────────────────

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
        log.info("Feature manifest saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "FeatureManifest":
        data = json.loads(Path(path).read_text())
        # Back-compat: fill in fields added after initial release
        data.setdefault("sex_categories", [])
        return cls(**data)


# ---------------------------------------------------------------------------
# Building the manifest from training data
# ---------------------------------------------------------------------------

def build_feature_manifest(
    train_df: pd.DataFrame,
    cfg: FeatureConfig,
    codes_df: Optional[pd.DataFrame] = None,
) -> FeatureManifest:
    """
    Derive the feature manifest from the training split only.

    Args:
        train_df  : Raw MEDS training dataframe.
        cfg       : Feature configuration (top-k limits, min frequency).
        codes_df  : Optional codes.parquet for human-readable descriptions.

    Returns:
        FeatureManifest ready for use in data.py.
    """
    id_col = "patient_id" if "patient_id" in train_df.columns else "subject_id"
    n_subjects = train_df[id_col].nunique()
    log.info("Building manifest from %d subjects, %d events", n_subjects, len(train_df))

    # ── Code descriptions (optional) ─────────────────────────────────────────
    code_desc: Dict[str, str] = {}
    if codes_df is not None and "description" in codes_df.columns:
        code_desc = (
            codes_df[codes_df["description"].notna()]
            .set_index("code")["description"]
            .to_dict()
        )

    # ── Per-code statistics ───────────────────────────────────────────────────
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

    # Keep only codes above the frequency threshold
    common = code_stats[code_stats["subject_freq"] >= cfg.min_subject_freq]

    def _top(cat: str, n: int) -> List[str]:
        sub = common[(common["category"] == cat) & common["has_numeric"].eq(
            cat in {"vital", "lab"}
        )]
        return sub.nlargest(n, "n_subjects")["code"].tolist()

    vital_codes = _top("vital", cfg.top_vitals)
    lab_codes = _top("lab", cfg.top_labs)

    # For actions, has_numeric isn't required
    def _top_action(cat: str, n: int) -> List[str]:
        sub = common[common["category"] == cat]
        return sub.nlargest(n, "n_subjects")["code"].tolist()

    med_codes = _top_action("medication", cfg.top_meds)
    infusion_codes = _top_action("infusion", cfg.top_infusions)
    proc_codes = _top_action("procedure", cfg.top_procedures)

    log.info(
        "Selected %d vitals, %d labs | %d meds, %d infusions, %d procedures",
        len(vital_codes), len(lab_codes),
        len(med_codes), len(infusion_codes), len(proc_codes),
    )

    # ── Demographics from HOSPITAL_ADMISSION rows ────────────────────────────
    demo_col_candidates = ["race"]  # insurance/language/marital_status excluded
    demo_cols = [c for c in demo_col_candidates if c in train_df.columns]

    admit_rows = train_df[train_df["code"].str.startswith("HOSPITAL_ADMISSION//")]
    demo_categories: Dict[str, List[str]] = {}
    for col in demo_cols:
        cats = sorted(admit_rows[col].dropna().astype(str).unique().tolist())
        demo_categories[col] = cats

    # Sex from GENDER// events
    gender_rows = train_df[train_df["code"].str.startswith("GENDER//")]
    sex_categories: List[str] = sorted(
        gender_rows["code"].str.replace("GENDER//", "", regex=False).unique().tolist()
    ) if len(gender_rows) > 0 else []
    log.info("Sex categories: %s", sex_categories)

    # ── Normalization stats (training only) ──────────────────────────────────
    all_continuous = vital_codes + lab_codes
    num_events = train_df[
        train_df["code"].isin(all_continuous) & train_df["numeric_value"].notna()
    ]

    state_mean: Dict[str, float] = {}
    state_std: Dict[str, float] = {}
    for code in all_continuous:
        vals = num_events.loc[num_events["code"] == code, "numeric_value"]
        state_mean[code] = float(vals.mean()) if len(vals) > 0 else 0.0
        σ = float(vals.std()) if len(vals) > 1 else 1.0
        if np.isnan(σ) or σ < 1e-6:
            σ = 1.0
        state_std[code] = σ

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
