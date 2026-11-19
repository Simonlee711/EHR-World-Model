"""
MEDS (MIMIC-IV, MEDS_transforms 0.0.3) -> processed world-model tensors.

State representation: 6-hour bins with the enriched state
``[z-value | obs-mask | time-since-obs | decayed carry-forward]`` plus a
race/age/sex demographic vector. The state variables are the curated clinical
concepts of ``experiments.features`` (vitals and labs worth forecasting,
range-cleaned, MIMIC itemids merged per concept, shared with eICU / NWICU).
Two action encodings are stored per bin:

    act5  : 5 drug groups (experiments.actions.GROUPS) built from *all*
            medication starts and ICU infusion starts.
    act78 : binary MEDICATION//START// indicators for the 78 codes selected by
            ``build_feature_manifest`` (top-100, >=5% of training patients).

Output layout (``out_dir``)::

    feature_manifest.json                  state + 78-code action manifest
    action_code_map.parquet                every treatment code -> group
    meds_5action/data/{split}/0.parquet    converted MEDS events (+ ACTION//<GROUP> rows)
    meds_5action/metadata/patient_splits.parquet
    {split}/timelines.npz                  state (Nbins, 4*n_cont) f16, act5, act78 (uint8)
    {split}/admissions.parquet             one row per admission (offset, n_bins, labels)
    {split}/demo.npy                       (n_adm, demo_dim) demographic vectors
    {split}/patient_labels.parquet         patient-level labels (disease flags + outcomes)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ehr_world_model.cohort import (build_label_table, compute_demographics,
                                    default_feature_config, load_meds_fast)
from ehr_world_model.features import FeatureManifest, build_feature_manifest

from .actions import ACTION_CODES, GROUPS, classify_code, drug_display_name
from .features import BY_CODE, meds_to_concepts
from .features import select as select_concepts

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)

SPLITS = {"train": "train", "val": "tuning", "test": "held_out"}
DEFAULT_N = {"train": 10000, "val": 3000, "test": 30000}   # patients sampled per split
BIN_HOURS = 6.0
MAX_BINS = 120             # cap admissions at 30 days
MAX_HOURS_SINCE = 168.0    # time-since-observation cap (hours)
TAU = 12.0                 # carry-forward decay constant (hours)


# ── Action code map ────────────────────────────────────────────────────────────

def build_action_code_map(train_df: pd.DataFrame, codes_df: pd.DataFrame,
                          med78_codes: List[str]) -> pd.DataFrame:
    """One row per treatment code seen in the training cohort."""
    desc = codes_df.dropna(subset=["code"]).set_index("code")["description"].to_dict()
    t = train_df[train_df["code"].str.startswith(("MEDICATION//START//", "INFUSION_START//"))]
    n_pat = train_df["patient_id"].nunique()
    stats = t.groupby("code").agg(n_patients=("patient_id", "nunique"), n_events=("patient_id", "size"))
    rows = []
    med78_rank = {c: i for i, c in enumerate(med78_codes)}
    for code, r in stats.iterrows():
        d = desc.get(code)
        g = classify_code(code, d)
        rows.append(dict(
            code=code,
            source="eMAR/pharmacy" if code.startswith("MEDICATION//") else "ICU infusion",
            name=drug_display_name(code, d),
            group=g if g else "NONE",
            n_patients=int(r.n_patients),
            pct_patients=100.0 * r.n_patients / n_pat,
            n_events=int(r.n_events),
            med78_action=code in med78_rank,
            med78_index=med78_rank.get(code, -1),
        ))
    return pd.DataFrame(rows).sort_values("n_patients", ascending=False).reset_index(drop=True)


def add_action_rows(df: pd.DataFrame, code_map: pd.DataFrame) -> pd.DataFrame:
    """Converted MEDS: original events + one ACTION//<GROUP> row per grouped treatment event."""
    g = code_map.set_index("code")["group"]
    t = df[df["code"].str.startswith(("MEDICATION//START//", "INFUSION_START//"))]
    grp = t["code"].map(g).fillna("NONE")
    act = t[grp != "NONE"].copy()
    act["code"] = "ACTION//" + grp[grp != "NONE"]
    act["numeric_value"] = np.nan
    return pd.concat([df, act], ignore_index=True).sort_values(["patient_id", "time"], kind="stable")


# ── Timelines ──────────────────────────────────────────────────────────────────

def _demo_vectors(adm: pd.DataFrame, manifest: FeatureManifest) -> np.ndarray:
    """Demographic vector: race one-hot(+UNK) | age/100 | sex one-hot(+UNK)."""
    parts = []
    for col in manifest.demo_cols:
        cats = manifest.demo_categories[col]
        idx = adm[col].map({c: i for i, c in enumerate(cats)}).fillna(len(cats)).astype(int).values
        parts.append(np.eye(len(cats) + 1, dtype=np.float32)[idx])
    age = adm["age"].values
    parts.append(np.where(np.isfinite(age), np.clip(age, 0, 100) / 100.0, 0.5).astype(np.float32)[:, None])
    sc = manifest.sex_categories or []
    if sc:
        idx = adm["sex"].map({c: i for i, c in enumerate(sc)}).fillna(len(sc)).astype(int).values
        parts.append(np.eye(len(sc) + 1, dtype=np.float32)[idx])
    return np.concatenate(parts, axis=1)


def build_timelines(df: pd.DataFrame, manifest: FeatureManifest, code_map: pd.DataFrame,
                    bin_hours: float = BIN_HOURS, max_bins: int = MAX_BINS) -> Dict:
    """Bin every (patient, hadm) admission into 6h steps.

    Bins start at the admission's first timed event.
    """
    ev = df[df["hadm_id"].notna() & df["time"].notna()][["patient_id", "hadm_id", "time", "code", "numeric_value"]]
    g = ev.groupby(["patient_id", "hadm_id"])["time"]
    adm = pd.DataFrame({"t0": g.min(), "t_end": g.max(), "n_ev": g.size()}).reset_index()
    adm["total_h"] = (adm["t_end"] - adm["t0"]).dt.total_seconds() / 3600.0
    adm = adm[(adm["n_ev"] >= 2) & (adm["total_h"] >= bin_hours)].reset_index(drop=True)
    adm["n_bins"] = np.clip(np.ceil(adm["total_h"] / bin_hours).astype(int), 2, max_bins)
    adm["offset"] = np.concatenate([[0], np.cumsum(adm["n_bins"].values)[:-1]])
    adm["adm_idx"] = np.arange(len(adm))
    n_rows = int(adm["n_bins"].sum())

    ev = ev.merge(adm[["patient_id", "hadm_id", "t0", "n_bins", "offset", "adm_idx"]],
                  on=["patient_id", "hadm_id"], how="inner")
    ev["bin"] = ((ev["time"] - ev["t0"]).dt.total_seconds() // (bin_hours * 3600)).astype(int)
    ev = ev[ev["bin"] < ev["n_bins"]]
    ev["row"] = ev["offset"] + ev["bin"]

    # values / mask
    state_codes = manifest.vital_codes + manifest.lab_codes
    n = len(state_codes)
    fidx = {c: i for i, c in enumerate(state_codes)}
    num = ev[ev["code"].isin(fidx) & ev["numeric_value"].notna()]
    agg = num.groupby(["row", "code"])["numeric_value"].mean().reset_index()
    mu = agg["code"].map(manifest.state_mean).values
    sd = agg["code"].map(manifest.state_std).values
    values = np.zeros((n_rows, n), np.float32)
    mask = np.zeros((n_rows, n), np.float32)
    r, c = agg["row"].values, agg["code"].map(fidx).values
    values[r, c] = np.clip((agg["numeric_value"].values - mu) / sd, -5, 5)
    mask[r, c] = 1.0

    # time-since-last-observation and decayed carry-forward
    time_since = np.ones((n_rows, n), np.float32)
    carry = np.zeros((n_rows, n), np.float32)
    nb, off = adm["n_bins"].values, adm["offset"].values
    prev_v = np.zeros((len(adm), n), np.float32)
    prev_h = np.full((len(adm), n), -np.inf, np.float32)
    for b in range(int(nb.max())):
        live = np.where(nb > b)[0]
        rows = off[live] + b
        m = mask[rows] > 0
        hrs = b * bin_hours - prev_h[live]
        seen = np.isfinite(hrs)
        alpha = np.exp(-np.where(seen, hrs, 0) / TAU)
        cf = np.where(m, values[rows], np.where(seen, alpha * prev_v[live], 0.0))
        ts = np.where(m, 0.0, np.where(seen, np.clip(hrs / MAX_HOURS_SINCE, 0, 1), 1.0))
        carry[rows], time_since[rows] = cf, ts
        prev_v[live] = np.where(m | seen, cf, prev_v[live])
        prev_h[live] = np.where(m, b * bin_hours, prev_h[live])
    state = np.concatenate([values, mask, time_since, carry], axis=1).astype(np.float16)

    # actions
    act78 = np.zeros((n_rows, len(manifest.med_codes)), np.uint8)
    midx = {c: i for i, c in enumerate(manifest.med_codes)}
    t78 = ev[ev["code"].isin(midx)]
    act78[t78["row"].values, t78["code"].map(midx).values] = 1
    act5 = np.zeros((n_rows, len(GROUPS)), np.uint8)
    gmap = code_map.set_index("code")["group"]
    tt = ev[ev["code"].str.startswith(("MEDICATION//START//", "INFUSION_START//"))]
    grp = tt["code"].map(gmap)
    tt = tt[grp.isin(GROUPS)]
    act5[tt["row"].values, tt["code"].map(gmap).map({g: i for i, g in enumerate(GROUPS)}).values] = 1

    return dict(adm=adm.drop(columns=["n_ev"]), state=state, act5=act5, act78=act78)


def admission_labels(df: pd.DataFrame, adm: pd.DataFrame) -> pd.DataFrame:
    """Admission-level outcomes: LOS, long LOS (>7d), in-hospital death, 30-day readmission."""
    a = df[df["code"].str.startswith("HOSPITAL_ADMISSION//")].groupby(["patient_id", "hadm_id"])["time"].min()
    d = df[df["code"].str.startswith("HOSPITAL_DISCHARGE//")].groupby(["patient_id", "hadm_id"])["time"].max()
    ad = pd.DataFrame({"admit_time": a, "discharge_time": d}).reset_index()
    death = df[df["code"] == "MEDS_DEATH"].groupby("patient_id")["time"].min().rename("death_time")
    icu = df[df["code"].str.startswith("ICU_ADMISSION//")].groupby(["patient_id", "hadm_id"]).size().rename("n_icu")
    out = adm.merge(ad, on=["patient_id", "hadm_id"], how="left") \
             .merge(death, on="patient_id", how="left") \
             .merge(icu.reset_index(), on=["patient_id", "hadm_id"], how="left")
    out["los_days"] = (out["discharge_time"] - out["admit_time"]).dt.total_seconds() / 86400
    out["adm_long_los"] = (out["los_days"] > 7).astype(float).where(out["los_days"].notna())
    out["adm_mortality"] = ((out["death_time"] >= out["admit_time"]) &
                            (out["death_time"] <= out["discharge_time"] + pd.Timedelta(hours=24))).astype(float)
    out["adm_icu"] = (out["n_icu"].fillna(0) > 0).astype(float)
    nxt = ad.sort_values(["patient_id", "admit_time"]).copy()
    nxt["next_admit"] = nxt.groupby("patient_id")["admit_time"].shift(-1)
    gap = (nxt["next_admit"] - nxt["discharge_time"]).dt.total_seconds() / 86400
    nxt["adm_readmit_30d"] = ((gap > 0) & (gap <= 30)).astype(float)
    out = out.merge(nxt[["patient_id", "hadm_id", "adm_readmit_30d"]], on=["patient_id", "hadm_id"], how="left")
    return out.drop(columns=["n_icu"])


def prepare(data_dir, out_dir, n_patients: Dict[str, int] = None, seed: int = 42,
            write_converted_meds: bool = True) -> None:
    """Full conversion of a MEDS extract into the processed layout above."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_patients = n_patients or DEFAULT_N

    codes_df = pd.read_parquet(data_dir / "extract_code_metadata" / "codes.parquet")
    raw = {s: load_meds_fast(data_dir, SPLITS[s], n_patients[s], seed) for s in SPLITS}

    # Manifest on the raw extract: defines the 78-code action space + demographics
    manifest = build_feature_manifest(raw["train"], default_feature_config(), codes_df=codes_df)
    # Curated state: raw LAB//<itemid> rows -> cleaned clinical concepts (experiments.features)
    dfs = {s: meds_to_concepts(df) for s, df in raw.items()}
    del raw
    vit, lab, mu, sd, freq = select_concepts(dfs["train"])
    manifest.vital_codes, manifest.lab_codes = vit, lab
    manifest.state_mean, manifest.state_std = mu, sd
    manifest.code_descriptions = {}
    manifest.save(out_dir / "feature_manifest.json")
    (out_dir / "state_concepts.json").write_text(json.dumps(
        {"codes": vit + lab, "train_patient_freq": {k: freq.get(k, 0.0) for k in vit + lab},
         "log_transformed": [c for c in vit + lab if BY_CODE[c]["log"]]}, indent=1))
    code_map = build_action_code_map(dfs["train"], codes_df, manifest.med_codes)
    code_map.to_parquet(out_dir / "action_code_map.parquet")
    log.info(manifest.summary())

    splits_rows = []
    for split, df in dfs.items():
        sd = out_dir / split
        sd.mkdir(exist_ok=True)
        if write_converted_meds:
            conv = add_action_rows(df, code_map)
            p = out_dir / "meds_5action" / "data" / SPLITS[split]
            p.mkdir(parents=True, exist_ok=True)
            conv.to_parquet(p / "0.parquet", index=False)
            splits_rows.append(pd.DataFrame({"patient_id": df["patient_id"].unique(), "split": SPLITS[split]}))

        tl = build_timelines(df, manifest, code_map)
        race = (df[df["code"].str.startswith("HOSPITAL_ADMISSION//")]
                .groupby(["patient_id", "hadm_id"])["race"].first().rename("race").reset_index())
        demo = compute_demographics(df, "patient_id")[["patient_id", "age", "sex", "bmi"]]
        adm = tl["adm"].merge(race, on=["patient_id", "hadm_id"], how="left") \
                       .merge(demo[["patient_id", "sex"]], on="patient_id", how="left")
        birth = df[df["code"] == "MEDS_BIRTH"].groupby("patient_id")["time"].min()
        adm["age"] = (adm["t0"] - adm["patient_id"].map(birth)).dt.total_seconds() / (365.25 * 86400)
        adm = admission_labels(df, adm)
        np.save(sd / "demo.npy", _demo_vectors(adm, manifest))
        np.savez(sd / "timelines.npz", state=tl["state"], act5=tl["act5"], act78=tl["act78"])
        adm.to_parquet(sd / "admissions.parquet")
        pl = build_label_table(df, "patient_id", demo)
        # patient-level readmission = admission-level label of the first admission
        first = adm.sort_values("admit_time").groupby("patient_id").first()
        pl["readmit_30d"] = pl.patient_id.map(first.adm_readmit_30d).fillna(0).astype(int)
        pl["readmit_any"] = pl.patient_id.map(adm.groupby("patient_id").size() > 1).fillna(False).astype(int)
        pl.to_parquet(sd / "patient_labels.parquet")
        log.info("[%s] %d admissions, %d bins", split, len(adm), len(tl["state"]))

    if write_converted_meds:
        (out_dir / "meds_5action" / "metadata").mkdir(parents=True, exist_ok=True)
        pd.concat(splits_rows).to_parquet(out_dir / "meds_5action" / "metadata" / "patient_splits.parquet")
    (out_dir / "action_groups.json").write_text(json.dumps({"groups": GROUPS, "codes": ACTION_CODES}, indent=2))


# ── Loading processed data / windows ───────────────────────────────────────────

class Split:
    """Processed split: ragged per-admission timelines."""

    def __init__(self, proc_dir, split: str):
        sd = Path(proc_dir) / split
        z = np.load(sd / "timelines.npz")
        self.state = z["state"]
        self.act = {"act5": z["act5"], "act78": z["act78"]}
        self.adm = pd.read_parquet(sd / "admissions.parquet")
        self.demo = np.load(sd / "demo.npy")
        self.patient_labels = pd.read_parquet(sd / "patient_labels.parquet")
        self.name = split

    def timeline(self, i: int, action_key: str = "act5", max_bins=None):
        """(state ⊕ demo  (T, D) float32,  actions (T, A) float32) for admission i."""
        o, n = int(self.adm.offset.iat[i]), int(self.adm.n_bins.iat[i])
        if max_bins:
            n = min(n, max_bins)
        s = self.state[o:o + n].astype(np.float32)
        d = np.repeat(self.demo[i][None], n, axis=0)
        a = self.act[action_key][o:o + n].astype(np.float32)
        return np.concatenate([s, d], axis=1), a

    @property
    def state_dim(self):
        return self.state.shape[1] + self.demo.shape[1]


class WindowDataset(Dataset):
    """Sliding windows (seq_len, stride) over every admission with >= seq_len bins."""

    def __init__(self, split: Split, action_key: str = "act5", seq_len: int = 8, stride: int = 4,
                 max_windows: int = None, seed: int = 0):
        self.sp, self.key, self.L = split, action_key, seq_len
        idx = []
        nc = split.state.shape[1] // 4
        mask = split.state[:, nc:2 * nc]
        for i, (o, n) in enumerate(zip(split.adm.offset.values, split.adm.n_bins.values)):
            for st in range(0, n - seq_len + 1, stride):
                if not mask[o + st:o + st + seq_len].any():
                    continue   # drop windows with no observations at all
                idx.append((i, o + st))
        idx = np.array(idx, dtype=np.int64).reshape(-1, 2)
        if max_windows and len(idx) > max_windows:
            idx = idx[np.random.default_rng(seed).choice(len(idx), max_windows, replace=False)]
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i, r = self.idx[k]
        s = self.sp.state[r:r + self.L].astype(np.float32)
        d = np.repeat(self.sp.demo[i][None], self.L, axis=0)
        a = self.sp.act[self.key][r:r + self.L].astype(np.float32)
        return {"s": torch.from_numpy(np.concatenate([s, d], 1)), "actions": torch.from_numpy(a)}


def make_loader(split: Split, action_key: str, batch_size: int = 64, shuffle: bool = False,
                **kw) -> DataLoader:
    return DataLoader(WindowDataset(split, action_key, **kw), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, drop_last=shuffle)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data/meds/MIMIC_MEDS_Extract_v0.0.3_test"))
    ap.add_argument("--out-dir", default=str(ROOT / "data/processed/mimic_wm"))
    ap.add_argument("--no-meds-copy", action="store_true")
    a = ap.parse_args()
    prepare(a.data_dir, a.out_dir, write_converted_meds=not a.no_meds_copy)
