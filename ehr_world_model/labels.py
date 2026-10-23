"""
Disease label extraction from MEDS diagnosis codes.

All disease labels are derived from ICD-9 and ICD-10 codes recorded as
DIAGNOSIS//ICD//{9|10}//{code} events in the MEDS event stream.

Supported disease groups
------------------------
  HTN     Hypertension
  DM      Diabetes Mellitus (type 1 and 2)
  HF      Heart Failure
  CAD     Coronary Artery Disease / Ischemic Heart Disease
  CKD     Chronic Kidney Disease (with staging)
  AKI     Acute Kidney Injury
  Sepsis  Sepsis / Septic Shock
  COPD    Chronic Obstructive Pulmonary Disease
  Cancer  Any malignant neoplasm

Subgroup stratification keys
-----------------------------
  disease_group  : primary disease label (one of the above, or "Other")
  age_group      : <50 | 50-65 | 65-80 | >80
  gender         : M | F | Unknown
  comorbidity_n  : number of disease labels (0, 1, 2, 3+)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ICD code prefix definitions
# ---------------------------------------------------------------------------

# Each entry: list of (icd_version_str, prefix_in_MEDS_code)
# MEDS format: DIAGNOSIS//ICD//{9|10}//{code}
DISEASE_PREFIXES: Dict[str, List[Tuple[str, str]]] = {
    "HTN":    [("10", "I10"), ("9", "401")],
    "DM":     [("10", "E10"), ("10", "E11"), ("9", "250")],
    "HF":     [("10", "I50"), ("9", "428")],
    "CAD":    [("10", "I25"), ("9", "414"), ("9", "410"), ("9", "412")],
    "CKD":    [("10", "N18"), ("9", "585")],
    "AKI":    [("10", "N17"), ("9", "584")],
    "Sepsis": [("10", "A41"), ("10", "R652"), ("9", "038"), ("9", "9959")],
    "COPD":   [("10", "J44"), ("9", "496"), ("9", "491"), ("9", "492")],
    "Cancer": [("10", "C"), ("9", "14"), ("9", "15"), ("9", "16"),
               ("9", "17"), ("9", "18"), ("9", "19"), ("9", "20"),
               ("9", "21"), ("9", "22"), ("9", "23"), ("9", "195"),
               ("9", "196"), ("9", "197"), ("9", "198"), ("9", "199")],
}

# CKD staging via ICD-10 N18.x
CKD_STAGE_MAP: Dict[str, int] = {
    "DIAGNOSIS//ICD//10//N181": 1,
    "DIAGNOSIS//ICD//10//N182": 2,
    "DIAGNOSIS//ICD//10//N183": 3,
    "DIAGNOSIS//ICD//10//N184": 4,
    "DIAGNOSIS//ICD//10//N185": 5,
    "DIAGNOSIS//ICD//10//N186": 5,  # ESRD / Stage 5D
}
# ICD-9 CKD staging
CKD_STAGE_MAP_ICD9: Dict[str, int] = {
    "DIAGNOSIS//ICD//9//5851": 1,
    "DIAGNOSIS//ICD//9//5852": 2,
    "DIAGNOSIS//ICD//9//5853": 3,
    "DIAGNOSIS//ICD//9//5854": 4,
    "DIAGNOSIS//ICD//9//5855": 5,
    "DIAGNOSIS//ICD//9//5856": 5,
}

# Priority order for primary disease assignment (first match wins)
PRIMARY_PRIORITY = ["Sepsis", "AKI", "HF", "CKD", "COPD", "DM", "CAD", "Cancer", "HTN"]


# ---------------------------------------------------------------------------
# Vectorised disease flag extraction (shared by all label functions)
# ---------------------------------------------------------------------------

def extract_disease_flags(
    df: pd.DataFrame,
    disease_defs: Dict[str, List[Tuple[str, str]]],
    id_col: str = "patient_id",
) -> pd.DataFrame:
    """
    Vectorised binary disease flag extraction for any ICD prefix definition dict.

    For each disease, uses str.startswith(tuple) + isin — no per-patient Python
    loops. Returns one row per unique patient with has_{disease} columns (int 0/1).
    """
    diag = df[df["code"].str.startswith("DIAGNOSIS//ICD//")][[id_col, "code"]]
    result = df[[id_col]].drop_duplicates().reset_index(drop=True)

    for disease, prefixes in disease_defs.items():
        meds_prefixes = tuple(f"DIAGNOSIS//ICD//{v}//{p}" for v, p in prefixes)
        hit_pids = diag.loc[diag["code"].str.startswith(meds_prefixes), id_col]
        result[f"has_{disease}"] = result[id_col].isin(hit_pids).astype(int)

    return result


# ---------------------------------------------------------------------------
# Patient-level label extraction
# ---------------------------------------------------------------------------

def extract_patient_labels(df: pd.DataFrame, id_col: str = "patient_id") -> pd.DataFrame:
    """
    Extract binary disease labels for each patient from the event stream.

    Args:
        df     : Raw MEDS events dataframe (must contain 'code' and id_col).
        id_col : Patient identifier column name.

    Returns:
        DataFrame with columns:
            has_{disease}  : int 0/1 — patient ever received this ICD code
            ckd_stage      : int 0-5 — maximum CKD stage observed (0 = none)
            primary_disease: str — first matching disease from PRIMARY_PRIORITY
            age_group      : str — '<50', '50-65', '65-80', '>80' (if age available)
            gender         : str — 'M', 'F', 'Unknown'
            comorbidity_n  : int — number of disease flags set
    """
    labels = extract_disease_flags(df, DISEASE_PREFIXES, id_col).set_index(id_col)

    # CKD staging — vectorised via map + groupby max
    diag = df[df["code"].str.startswith("DIAGNOSIS//ICD//")][[id_col, "code"]]
    all_ckd = {**CKD_STAGE_MAP, **CKD_STAGE_MAP_ICD9}
    ckd_rows = diag[diag["code"].isin(all_ckd)].copy()
    ckd_rows["stage"] = ckd_rows["code"].map(all_ckd)
    ckd_stage = ckd_rows.groupby(id_col)["stage"].max()
    labels["ckd_stage"] = labels.index.map(ckd_stage).fillna(0).astype(int)

    # Primary disease — np.select instead of per-row apply
    conditions = [labels[f"has_{d}"].astype(bool) for d in PRIMARY_PRIORITY]
    labels["primary_disease"] = np.select(conditions, PRIMARY_PRIORITY, default="Other")

    # Comorbidity count
    disease_cols = [f"has_{d}" for d in DISEASE_PREFIXES]
    labels["comorbidity_n"] = labels[disease_cols].sum(axis=1)

    # Gender from GENDER// events
    gender_events = df[df["code"].str.startswith("GENDER//")]
    if len(gender_events) > 0:
        gmap = (
            gender_events.groupby(id_col)["code"]
            .first()
            .str.replace("GENDER//", "", regex=False)
        )
        labels["gender"] = labels.index.map(gmap).fillna("Unknown")
    else:
        labels["gender"] = "Unknown"

    # Age from MEDS_BIRTH + first admission time
    birth_events = df[df["code"] == "MEDS_BIRTH"][[id_col, "time"]].dropna()
    admit_events = df[df["code"].str.startswith("HOSPITAL_ADMISSION//")][[id_col, "time"]].dropna()
    if len(birth_events) > 0 and len(admit_events) > 0:
        birth_map = birth_events.groupby(id_col)["time"].min()
        admit_map = admit_events.groupby(id_col)["time"].min()
        age_years = ((admit_map - birth_map).dt.total_seconds() / (365.25 * 24 * 3600)).clip(0, 120)
        labels["age_years"] = labels.index.map(age_years)
        labels["age_group"] = pd.cut(
            labels["age_years"].fillna(65),
            bins=[0, 50, 65, 80, 200],
            labels=["<50", "50-65", "65-80", ">80"],
        ).astype(str)
    else:
        labels["age_years"] = np.nan
        labels["age_group"] = "Unknown"

    log.info(
        "Extracted labels for %d patients | disease prevalence: %s",
        len(labels),
        {d: int(labels[f"has_{d}"].sum()) for d in list(DISEASE_PREFIXES)[:6]},
    )
    return labels.reset_index()


# ---------------------------------------------------------------------------
# Attach labels to patient transitions
# ---------------------------------------------------------------------------

def attach_labels_to_transitions(
    transitions: list,
    labels_df: pd.DataFrame,
    id_col: str = "patient_id",
) -> list:
    """
    Add disease labels to each transition dict.
    Transitions must have a 'patient_id' field.
    """
    label_map = labels_df.set_index(id_col).to_dict(orient="index")
    enriched = []
    for t in transitions:
        pid = t.get("patient_id")
        if pid is not None and pid in label_map:
            t = {**t, **label_map[pid]}
        enriched.append(t)
    return enriched
