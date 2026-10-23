"""
Cohort assembly for the EHR World Model.

End-to-end data pipeline on top of the MEDS loaders in ``data.py``:

    MEDS parquet ─► load_meds_fast ─► demographics + labels
                 ─► feature manifest ─► enriched transitions (cached)
                 ─► sliding-window sequences ─► DemoAugLoader (state ⊕ demo)

Label families
--------------
* ``extract_patient_labels`` (labels.py)  : ever-diagnosed disease flags
* ``extract_outcome_labels``              : in-hospital outcomes (LOS, ICU, readmission, mortality)
* ``extract_ehrshot_labels``              : EHRSHOT-style temporal labels anchored at first
                                            admission (incident disease within 365d, long LOS,
                                            ICU transfer, 30-day readmission). Used by the
                                            counterfactual experiments.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from .config import FeatureConfig
from .data import EnrichedSequenceDataset, build_enriched_sequence_batches, build_enriched_transitions
from .features import build_feature_manifest
from .labels import DISEASE_PREFIXES, attach_labels_to_transitions, extract_disease_flags, extract_patient_labels

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
EHRSHOT_OUTCOME_COLS = ["long_los", "icu_transfer", "readmit_30d"]

_ICU_PATTERN = r"ICU|INTENSIVE_CARE|CRITICAL_CARE|MICU|SICU|CVICU|NICU|CCU|TRANSFER.*ICU"


# ── Fast MEDS loader ──────────────────────────────────────────────────────────
# Column pruning + pyarrow filter pushdown: ~6x faster, ~5x less memory than
# data.load_meds_cohort.

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


# ── Demographics ──────────────────────────────────────────────────────────────

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


# ── Outcome labels ────────────────────────────────────────────────────────────

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
    second_adm_ser   = adm.groupby(id_col)["time"].nth(1).rename("second_adm")
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


def all_clf_cols():
    return ([f"has_{d}" for d in DISEASE_PREFIXES]
            + [f"has_{k}" for k in NEW_DISEASE_DEF]
            + OUTCOME_COLS)


# ── EHRSHOT-style temporal labels ─────────────────────────────────────────────
# Prediction time t = first hospital admission per patient.
#   Disease (incident, 365-day horizon):
#     Positive : first ICD code in [t_admit, t_admit + 365d]
#     Negative : no ICD code before or after t_admit
#     Excluded : any ICD code before t_admit → NaN (prevalent case)
#   long_los     : first discharge − first admission ≥ 7 days
#   icu_transfer : any ICU-pattern event in [t_admit, t_discharge]
#   readmit_30d  : any new admission in (t_discharge, t_discharge + 30d]

def extract_ehrshot_labels(df: pd.DataFrame, id_col: str, horizon_days: int = 365) -> pd.DataFrame:
    adm_df = df[df["code"].str.startswith("HOSPITAL_ADMISSION//")][[id_col, "time"]].dropna()
    dis_df = df[df["code"].str.startswith("HOSPITAL_DISCHARGE//")][[id_col, "time"]].dropna()

    first_adm = adm_df.groupby(id_col)["time"].min().reset_index().rename(columns={"time": "t_admit"})
    dis_merged = dis_df.merge(first_adm, on=id_col, how="inner")
    first_dis = (dis_merged[dis_merged["time"] >= dis_merged["t_admit"]]
                 .groupby(id_col)["time"].min()
                 .reset_index().rename(columns={"time": "t_discharge"}))
    base = first_adm.merge(first_dis, on=id_col, how="left")

    # Long LOS
    base["los_days"] = (base["t_discharge"] - base["t_admit"]).dt.total_seconds() / 86400
    base.loc[base["los_days"] <= 0, "los_days"] = float("nan")
    base["long_los"] = (base["los_days"] >= 7).astype(float)
    base.loc[base["los_days"].isna(), "long_los"] = float("nan")

    # ICU transfer during the admission episode
    icu_df = df[df["code"].str.upper().str.contains(_ICU_PATTERN, na=False, regex=True)][[id_col, "time"]].dropna()
    base_ep = base[[id_col, "t_admit", "t_discharge"]].copy()
    base_ep["t_end"] = base_ep["t_discharge"].fillna(base_ep["t_admit"] + pd.Timedelta(days=30))
    icu_ep = icu_df.merge(base_ep[[id_col, "t_admit", "t_end"]], on=id_col, how="inner")
    icu_during = set(icu_ep[(icu_ep["time"] >= icu_ep["t_admit"]) & (icu_ep["time"] <= icu_ep["t_end"])][id_col])
    base["icu_transfer"] = base[id_col].isin(icu_during).astype(int)

    # 30-day readmission (anchored to first discharge)
    adm_dis = adm_df.merge(base[[id_col, "t_discharge"]], on=id_col, how="inner").dropna(subset=["t_discharge"])
    adm_dis["gap_days"] = (adm_dis["time"] - adm_dis["t_discharge"]).dt.total_seconds() / 86400
    readmit_pids = set(adm_dis[(adm_dis["gap_days"] > 0) & (adm_dis["gap_days"] <= 30)][id_col])
    base["readmit_30d"] = base[id_col].isin(readmit_pids).astype(int)
    base.loc[base["t_discharge"].isna(), "readmit_30d"] = float("nan")

    # Incident disease labels
    diag_df = df[df["code"].str.startswith("DIAGNOSIS//ICD//")][[id_col, "code", "time"]].dropna()
    disease_defs = {f"has_{d}": p for d, p in {**DISEASE_PREFIXES, **NEW_DISEASE_DEF}.items()}
    horizon_td = pd.Timedelta(days=horizon_days)
    base_adm = base[[id_col, "t_admit"]].copy()

    for col_name, prefixes in disease_defs.items():
        meds_pfx = tuple(f"DIAGNOSIS//ICD//{v}//{p}" for v, p in prefixes)
        hits_m = (diag_df[diag_df["code"].str.startswith(meds_pfx)][[id_col, "time"]]
                  .merge(base_adm, on=id_col, how="inner"))
        prevalent = set(hits_m[hits_m["time"] < hits_m["t_admit"]][id_col])
        incident  = set(hits_m[(hits_m["time"] >= hits_m["t_admit"]) &
                               (hits_m["time"] <= hits_m["t_admit"] + horizon_td)][id_col])
        base[col_name] = [float("nan") if pid in prevalent else (1.0 if pid in incident else 0.0)
                          for pid in base[id_col]]

    return base


# ── Transitions (cached) ──────────────────────────────────────────────────────

def _trans_cache_key(df, feat_cfg, tag, id_col):
    pids_str = ",".join(str(p) for p in sorted(df[id_col].unique()))
    return hashlib.md5(f"{tag}|{pids_str}|{feat_cfg}".encode()).hexdigest()[:12]


def get_or_build_transitions(df, manifest, feat_cfg, tag, out_dir, id_col="patient_id"):
    """build_enriched_transitions with an on-disk joblib cache.

    Cache key = hash of patient IDs + FeatureConfig. Delete the .pkl files if
    the feature extraction code changes.
    """
    key   = _trans_cache_key(df, feat_cfg, tag, id_col)
    cache = Path(out_dir) / f"transitions_{tag}_{key}.pkl"
    if cache.exists():
        logger.info("[%s] loading transitions from cache (%s)", tag, cache.name)
        return joblib.load(cache)
    logger.info("[%s] building transitions", tag)
    trans = build_enriched_transitions(df, manifest, feat_cfg)
    joblib.dump(trans, cache)
    return trans


# ── Datasets & loaders ────────────────────────────────────────────────────────

class LabeledSequenceDataset(EnrichedSequenceDataset):
    """EnrichedSequenceDataset with per-window patient labels attached."""

    def __init__(self, seqs, labels_df, id_col):
        super().__init__(seqs)
        lm = labels_df.set_index(id_col).to_dict(orient="index")
        self._labels = [lm.get(p, {}) for p in self.patient_ids]

    def label_array(self, col):
        return np.array([float(l.get(col, float("nan"))) for l in self._labels])

    def strat_array(self, col):
        return np.array([str(l.get(col, "Unknown")) for l in self._labels])


class DemoAugLoader:
    """Wraps a DataLoader and concatenates the per-patient demo vector to the state
    at every step: s_aug = [s_t ; d]  →  state_dim = 4*n_cont + demo_dim."""

    def __init__(self, loader, device):
        self._loader = loader
        self._device = device

    def __iter__(self):
        for b in self._loader:
            s  = b["s"].to(self._device)
            d  = b["d"].to(self._device)
            de = d.unsqueeze(1).expand(-1, s.shape[1], -1)
            yield {"s": torch.cat([s, de], dim=-1),
                   "actions": b["actions"].to(self._device),
                   "patient_ids": b.get("patient_ids", [])}

    def __len__(self):
        return len(self._loader)


# ── One-call pipeline ─────────────────────────────────────────────────────────

def default_feature_config() -> FeatureConfig:
    """Feature config used for the reported v6/v7 models."""
    return FeatureConfig(bin_hours=6.0, top_vitals=30, top_labs=50, top_meds=100,
                         top_infusions=0, top_procedures=0, min_subject_freq=0.05)


def build_cohort(
    data_dir,
    out_dir,
    device,
    feat_cfg: Optional[FeatureConfig] = None,
    max_patients: Dict[str, int] = None,
    seq_len: int = 8,
    stride: int = 4,
    batch_size: int = 64,
    seed: int = 42,
) -> Dict:
    """MEDS directory → manifest, labels, sequence datasets and loaders.

    Returns a dict with keys: manifest, feat_cfg, id_col, state_dim, n_cont,
    labels/{train,val,test}, ds/{train,val,test}, and loaders
    ``train`` (shuffled), ``train_eval`` / ``val`` / ``test`` (unshuffled, so
    index i lines up with ``ds[split].patient_ids[i]``), plus raw ``dfs``.
    """
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_cfg = feat_cfg or default_feature_config()
    max_patients = max_patients or {"train": 10000, "tuning": 3000, "held_out": 30000}

    dfs = {
        "train": load_meds_fast(data_dir, "train",    max_patients.get("train"),    seed),
        "val":   load_meds_fast(data_dir, "tuning",   max_patients.get("tuning"),   seed),
        "test":  load_meds_fast(data_dir, "held_out", max_patients.get("held_out"), seed),
    }
    id_col = "patient_id" if "patient_id" in dfs["train"].columns else "subject_id"

    labels = {}
    for split, df in dfs.items():
        demo = compute_demographics(df, id_col) if split != "val" else None
        labels[split] = build_label_table(df, id_col, demo)

    codes_path = data_dir / "extract_code_metadata" / "codes.parquet"
    codes_df = pd.read_parquet(codes_path) if codes_path.exists() else None
    manifest = build_feature_manifest(dfs["train"], feat_cfg, codes_df=codes_df)
    n_cont = manifest.n_continuous
    state_dim = 4 * n_cont + manifest.demo_dim

    ds, loaders = {}, {}
    for split, df in dfs.items():
        trans = get_or_build_transitions(df, manifest, feat_cfg, split, out_dir, id_col)
        trans = attach_labels_to_transitions(trans, labels[split], id_col)
        seqs  = build_enriched_sequence_batches(trans, seq_len, stride)
        ds[split] = LabeledSequenceDataset(seqs, labels[split], id_col)

    loaders["train"]      = DemoAugLoader(DataLoader(ds["train"], batch_size=batch_size, shuffle=True), device)
    loaders["train_eval"] = DemoAugLoader(DataLoader(ds["train"], batch_size=batch_size, shuffle=False), device)
    loaders["val"]        = DemoAugLoader(DataLoader(ds["val"],   batch_size=batch_size, shuffle=False), device)
    loaders["test"]       = DemoAugLoader(DataLoader(ds["test"],  batch_size=batch_size, shuffle=False), device)

    logger.info("state_dim=%d  n_cont=%d  n_actions=%d  seqs train/val/test=%d/%d/%d",
                state_dim, n_cont, manifest.n_actions, len(ds["train"]), len(ds["val"]), len(ds["test"]))
    return dict(manifest=manifest, feat_cfg=feat_cfg, id_col=id_col, state_dim=state_dim,
                n_cont=n_cont, labels=labels, ds=ds, loaders=loaders, dfs=dfs,
                seq_len=seq_len, stride=stride)
