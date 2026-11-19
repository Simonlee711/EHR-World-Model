"""
MEDS loading, demographics and patient-level labels for MIMIC-IV.

* ``load_meds_fast``         column-pruned MEDS loader with patient-id filter pushdown
* ``compute_demographics``   age at first admission, sex, BMI
* ``extract_outcome_labels`` in-hospital outcomes (LOS, ICU admission, readmission, mortality)
* ``build_label_table``      disease flags (labels.DISEASE_PREFIXES + NEW_DISEASE_DEF) + outcomes
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import FeatureConfig
from .labels import extract_disease_flags, extract_patient_labels

logger = logging.getLogger(__name__)

# Additional ICD-prefix disease definitions (on top of labels.DISEASE_PREFIXES)
NEW_DISEASE_DEF = {
    "acute_mi":          [("10", "I21"),  ("10", "I22"), ("9", "410"),  ("9", "411")],
    "lupus":             [("10", "M32"),  ("9", "7100")],
    "hyperlipidemia":    [("10", "E78"),  ("9", "272")],
    "hypertension":      [("10", "I10"),  ("10", "I11"), ("10", "I12"),
                          ("10", "I13"),  ("10", "I15"),
                          ("9",  "401"),  ("9",  "402"), ("9",  "403"),
                          ("9",  "404"),  ("9",  "405")],
    "celiac":            [("10", "K900"), ("10", "K90"), ("9", "5790"), ("9", "579")],
    "pancreatic_cancer": [("10", "C25"),  ("9", "157")],
}

OUTCOME_COLS = ["in_hospital_mortality", "icu_admission", "los_long", "readmit_30d", "readmit_any"]

_ICU_PATTERN = r"ICU|INTENSIVE_CARE|CRITICAL_CARE|MICU|SICU|CVICU|NICU|CCU|TRANSFER.*ICU"


def _sample_pids(data_dir, split, n, seed=42):
    """Sample up to n patient_ids from metadata/patient_splits.parquet."""
    splits_path = Path(data_dir) / "metadata" / "patient_splits.parquet"
    if splits_path.exists():
        sp   = pd.read_parquet(splits_path)
        id_c = "patient_id" if "patient_id" in sp.columns else "subject_id"
        sc   = "split" if "split" in sp.columns else sp.columns[-1]
        ids  = sp.loc[sp[sc] == split, id_c].unique()
    else:
        shard = sorted((Path(data_dir) / "data" / split).glob("*.parquet"))[0]
        ids   = pq.read_table(shard, columns=["patient_id"]).to_pandas()["patient_id"].unique()
    if len(ids) > n:
        rng = np.random.default_rng(seed)
        ids = rng.choice(ids, size=n, replace=False)
    return set(ids.tolist())


def load_meds_fast(data_dir, split, max_patients=None, seed=42) -> pd.DataFrame:
    """Column-pruned + filter-pushdown MEDS loader.

    Only reads: patient_id, time, code, numeric_value, hadm_id, race.
    """
    data_dir = Path(data_dir)
    shards   = sorted((data_dir / "data" / split).glob("*.parquet"))
    pids     = _sample_pids(data_dir, split, max_patients, seed) if max_patients else None

    cols = ["patient_id", "time", "code", "numeric_value", "hadm_id", "race"]
    dfs  = []
    for shard in shards:
        filters = [("patient_id", "in", list(pids))] if pids is not None else None
        chunk   = pq.read_table(shard, columns=cols, filters=filters).to_pandas()
        if len(chunk):
            chunk["time"] = pd.to_datetime(chunk["time"])
            dfs.append(chunk)

    if not dfs:
        raise ValueError(f"No events found for split '{split}'.")
    df = pd.concat(dfs, ignore_index=True)
    logger.info("%s: %s patients, %s events", split, f"{df['patient_id'].nunique():,}", f"{len(df):,}")
    return df


def compute_demographics(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Per-patient age at first admission, sex, BMI and binned groups."""
    birth_map = df[df["code"] == "MEDS_BIRTH"].groupby(id_col)["time"].min()
    admit_map = df[df["code"].str.startswith("HOSPITAL_ADMISSION//")].groupby(id_col)["time"].min()
    sex_map   = (df[df["code"].str.startswith("GENDER//")]
                 .groupby(id_col)["code"].first()
                 .str.replace("GENDER//", "", regex=False))
    wt_codes  = [c for c in df["code"].unique() if any(c.startswith(f"LAB//{i}") for i in ["224639", "226512", "226531"])]
    ht_codes  = [c for c in df["code"].unique() if c.startswith("LAB//226730")]
    wt_map    = df[df["code"].isin(wt_codes)].groupby(id_col)["numeric_value"].median()
    ht_map    = df[df["code"].isin(ht_codes)].groupby(id_col)["numeric_value"].median()
    records   = []
    for pid in df[id_col].unique():
        age = np.nan
        if pid in birth_map and pid in admit_map:
            age = float(np.clip((admit_map[pid] - birth_map[pid]).total_seconds() / (365.25 * 24 * 3600), 0, 120))
        wt = wt_map.get(pid, np.nan); ht = ht_map.get(pid, np.nan)
        bmi = wt / ((ht / 100) ** 2) if (not np.isnan(wt) and not np.isnan(ht) and ht > 50) else np.nan
        records.append({id_col: pid, "age": age, "sex": sex_map.get(pid, "Unknown"), "bmi": bmi})
    demo = pd.DataFrame(records)
    demo["age_group"] = pd.cut(demo["age"].fillna(65), bins=[0, 50, 65, 80, 200],
                               labels=["<50", "50-65", "65-80", ">80"]).astype(str)
    demo["bmi_group"] = pd.cut(demo["bmi"], bins=[0, 18.5, 25, 30, 200],
                               labels=["Underweight", "Normal", "Overweight", "Obese"]).astype(str)
    return demo


def extract_outcome_labels(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """In-hospital outcomes per patient: LOS, readmission, ICU admission, mortality."""
    adm = df[df["code"].str.startswith("HOSPITAL_ADMISSION//")][[id_col, "time"]].dropna().sort_values([id_col, "time"])
    dis = df[df["code"].str.startswith("HOSPITAL_DISCHARGE//")][[id_col, "time"]].dropna().sort_values([id_col, "time"])

    dth = df[
        (df["code"].str.upper().str.contains("DEATH|EXPIR|DEAD", na=False)) |
        (df["code"] == "MEDS_DEATH")
    ][[id_col]].drop_duplicates().copy()
    dth["in_hospital_mortality"] = 1

    icu_pids = df[df["code"].str.upper().str.contains(_ICU_PATTERN, na=False, regex=True)][[id_col]].drop_duplicates().copy()
    icu_pids["icu_admission"] = 1

    # LOS
    first_adm = adm.groupby(id_col)["time"].min().rename("adm_time")
    last_dis  = dis.groupby(id_col)["time"].max().rename("dis_time")
    los_df    = pd.concat([first_adm, last_dis], axis=1).dropna()
    los_df.index.name = id_col  # pandas 2.x may drop the name on concat
    los_df["los_days"] = (los_df["dis_time"] - los_df["adm_time"]).dt.total_seconds() / 86400.0
    los_df.loc[los_df["los_days"] <= 0, "los_days"] = float("nan")
    los_df["los_long"] = (los_df["los_days"] > 7).astype(float)
    los_df = los_df[["los_days", "los_long"]].reset_index()

    # Readmission
    adm_cnt          = adm.groupby(id_col)["time"].count().rename("n_adm")
    readmit_any_pids = adm_cnt[adm_cnt >= 2].index
    first_dis_ser    = dis.groupby(id_col)["time"].min().rename("first_dis")
    # 2nd admission per patient (adm is sorted by time)
    second_adm_ser   = adm[adm.groupby(id_col).cumcount() == 1].set_index(id_col)["time"].rename("second_adm")
    readmit_df = pd.concat([first_dis_ser, second_adm_ser], axis=1).dropna()
    readmit_df.index.name = id_col
    readmit_df = readmit_df[readmit_df.index.isin(readmit_any_pids)].copy()
    readmit_df["readmit_any"] = 1
    gap_days = (readmit_df["second_adm"] - readmit_df["first_dis"]).dt.total_seconds() / 86400.0
    readmit_df["readmit_30d"] = ((gap_days > 0) & (gap_days <= 30)).astype(int)
    readmit_df = readmit_df[["readmit_any", "readmit_30d"]].reset_index()

    out = (
        df[[id_col]].drop_duplicates()
        .merge(los_df,     on=id_col, how="left")
        .merge(readmit_df, on=id_col, how="left")
        .merge(icu_pids,   on=id_col, how="left")
        .merge(dth,        on=id_col, how="left")
    )
    for c in ["readmit_any", "readmit_30d", "icu_admission", "in_hospital_mortality"]:
        out[c] = out[c].fillna(0).astype(int)
    return out


def build_label_table(df: pd.DataFrame, id_col: str, demo: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Disease flags (original + NEW_DISEASE_DEF) + outcome labels (+ demographics)."""
    base    = extract_patient_labels(df, id_col)
    new_dis = extract_disease_flags(df, NEW_DISEASE_DEF, id_col)
    new_out = extract_outcome_labels(df, id_col)
    combined = (base
                .merge(new_dis, on=id_col, how="left")
                .merge(new_out, on=id_col, how="left", suffixes=("", "_dup")))
    combined = combined[[c for c in combined.columns if not c.endswith("_dup")]]
    if demo is not None:
        for col in ["age", "sex", "bmi", "age_group", "bmi_group"]:
            if col not in combined.columns and col in demo.columns:
                combined[col] = combined[id_col].map(demo.set_index(id_col)[col])
    return combined


def default_feature_config() -> FeatureConfig:
    """Feature selection used to define the 78-code medication action space."""
    return FeatureConfig(bin_hours=6.0, top_vitals=30, top_labs=50, top_meds=100,
                         top_infusions=0, top_procedures=0, min_subject_freq=0.05)
