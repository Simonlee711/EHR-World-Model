"""
Cross-site export (MIMIC-IV, eICU-CRD, NWICU) with ONE shared feature space.

Input : the harmonised MEDS-like extracts in ``data/{mimic,eicu,nwicu}``
        (codes VITAL//<concept>, LAB//<concept>, MED//ATC4//<class>; unit-harmonised).
Output: ``data/processed/xsite/{site}/{train,val,test}/`` in exactly the same
        layout as ``experiments.data`` (timelines.npz / admissions.parquet / demo.npy),
        so ``experiments.data.Split`` / ``WindowDataset`` load every site unchanged.

Shared definitions (``data/processed/xsite/xsite_manifest.json``)
------------------------------------------------------------------
* State features: the curated clinical concepts of ``experiments.features``
  (vitals/labs worth forecasting; same list as the MEDS-based MIMIC model),
  kept if observed in >=5% of MIMIC train subjects, reused for all sites, with
  identical plausibility limits and log transforms. A feature a site never records is simply always unobserved
  (mask = 0, time-since = 1), which the enriched state encodes explicitly.
* Normalisation: MIMIC-train mean / std for every site (single feature space).
  Each site's own train statistics are stored too (``site_stats``) for
  re-normalisation experiments.
* Actions: the 5 groups of ``experiments.actions`` defined by WHO-ATC4 rules
  (``ATC4_GROUP_RULES``) - identical across sites.
* Demographics: age/100 + sex one-hot (F, M, UNK). Race is not harmonised.
* Cohort: ICU encounters (``encounters.is_icu``); 6h bins from hospital
  admission to discharge, capped at 30 days. Events are assigned to encounters
  by time (encounter_id is missing for ~30% of MIMIC lab/med rows).
* Labels: every task in ``data/{site}/labels`` joined on encounter_id
  (columns ``y_<task>``, prediction time 48h after admission; readmission at discharge).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .actions import GROUPS
from .features import BY_CODE, CONCEPTS, harmonised_clean

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
SITES = ["mimic", "eicu", "nwicu"]
SPLIT_NAMES = {"train": "train", "val": "tuning", "test": "held_out"}
BIN_HOURS, MAX_BINS, MAX_HOURS_SINCE, TAU = 6.0, 120, 168.0, 12.0

# WHO ATC level-4 -> action group (prefix match). Mirrors the drug-name lexicon in actions.py.
ATC4_GROUP_RULES: Dict[str, List[str]] = {
    "ANTIBIOTIC":  ["J01", "A07AA", "J04AB"],
    "IV_FLUID":    ["B05AA", "B05BB", "B05CB", "B05CX"],
    "VASOPRESSOR": ["C01CA", "C01CE", "C01CX", "H01BA"],
    "ANALGESIC":   ["N02A", "N02BE", "N01AH", "M01A"],
    "SEDATIVE":    ["N05BA", "N05CA", "N05CC", "N05CD", "N05CF", "N05CM", "N01AX", "N03AA", "N03AE"],
}


def atc4_group(atc4: str) -> Optional[str]:
    for g in GROUPS:
        if any(atc4.startswith(p) for p in ATC4_GROUP_RULES[g]):
            return g
    return None


def _site_dir(site):
    return ROOT / "data" / site


def select_features(min_freq=0.05) -> Dict[str, List[str]]:
    """Curated concepts (experiments.features) observed in >= min_freq of MIMIC *train*
    subjects of the harmonised extract - the same list as the MEDS-based MIMIC model."""
    sd = _site_dir("mimic")
    split = pd.read_parquet(sd / "metadata/subject_splits.parquet")
    train_ids = set(split.loc[split.split == "train", "subject_id"])
    counts: Dict[str, set] = {}
    for f in sorted((sd / "data").glob("*.parquet")):
        t = pq.read_table(f, columns=["subject_id", "code", "numeric_value"]).to_pandas()
        t = t[t.subject_id.isin(train_ids) & t.code.isin(BY_CODE) & t.numeric_value.notna()]
        for code, ids in t.groupby("code").subject_id.unique().items():
            counts.setdefault(code, set()).update(ids)
    freq = {c: len(v) / len(train_ids) for c, v in counts.items()}
    keep = [c["code"] for c in CONCEPTS if freq.get(c["code"], 0) >= min_freq]
    return {"vital_codes": [c for c in keep if c.startswith("VITAL//")],
            "lab_codes": [c for c in keep if c.startswith("LAB//")],
            "log_transformed": [c for c in keep if BY_CODE[c]["log"]],
            "subject_freq": {c: round(freq.get(c, 0), 4) for c in keep}}


def _load_site(site: str, state_codes: List[str]) -> Dict:
    sd = _site_dir(site)
    enc = pd.read_parquet(sd / "metadata/encounters.parquet")
    enc = enc[enc.is_icu & enc.admit_time.notna() & enc.discharge_time.notna()].copy()
    subj = pd.read_parquet(sd / "metadata/subjects.parquet")[["subject_id", "birth_time", "sex", "split"]]
    keep = set(state_codes)
    parts = []
    for f in sorted((sd / "data").glob("*.parquet")):
        t = pq.read_table(f, columns=["subject_id", "time", "code", "numeric_value"],
                          read_dictionary=["code"]).to_pandas()
        cats = t.code.cat.categories
        ok = pd.Series([c in keep or c.startswith("MED//") for c in cats], index=cats)
        t = t[t.time.notna() & t.code.map(ok).astype(bool)]
        t["code"] = t.code.astype(str)
        t = harmonised_clean(t)                      # same range limits / log transform as MIMIC
        parts.append(_assign_encounters(t, enc))     # per shard keeps memory bounded
        log.info("  %s: %d events kept", f.name, len(parts[-1]))
    ev = pd.concat(parts, ignore_index=True)
    return dict(enc=enc, subj=subj, ev=ev)


def _assign_encounters(ev: pd.DataFrame, enc: pd.DataFrame) -> pd.DataFrame:
    """Attach each event to the ICU encounter whose [admit, discharge] contains it."""
    e = enc[["subject_id", "encounter_id", "admit_time", "discharge_time"]].copy()
    for c in ["admit_time", "discharge_time"]:
        e[c] = e[c].astype("datetime64[ns]")
    e = e.sort_values("admit_time")
    ev = ev.assign(time=ev["time"].astype("datetime64[ns]")).sort_values("time")
    m = pd.merge_asof(ev, e, left_on="time", right_on="admit_time", by="subject_id", direction="backward")
    return m[m.encounter_id.notna() & (m.time <= m.discharge_time)]


def _build(ev, enc, state_codes, mean, std):
    n = len(state_codes)
    fidx = {c: i for i, c in enumerate(state_codes)}
    adm = enc.copy().reset_index(drop=True)
    adm["total_h"] = (adm.discharge_time - adm.admit_time).dt.total_seconds() / 3600
    adm = adm[adm.total_h >= BIN_HOURS].reset_index(drop=True)
    adm["n_bins"] = np.clip(np.ceil(adm.total_h / BIN_HOURS).astype(int), 2, MAX_BINS)
    adm["offset"] = np.concatenate([[0], np.cumsum(adm.n_bins.values)[:-1]])
    adm["adm_idx"] = np.arange(len(adm))
    n_rows = int(adm.n_bins.sum())

    ev = ev.merge(adm[["encounter_id", "admit_time", "n_bins", "offset"]].rename(columns={"admit_time": "t0"}),
                  on="encounter_id")
    ev["bin"] = ((ev.time - ev.t0).dt.total_seconds() // (BIN_HOURS * 3600)).astype(int)
    ev = ev[(ev.bin >= 0) & (ev.bin < ev.n_bins)]
    ev["row"] = ev.offset + ev.bin

    num = ev[ev.code.isin(fidx) & ev.numeric_value.notna()]
    agg = num.groupby(["row", "code"]).numeric_value.mean().reset_index()
    values = np.zeros((n_rows, n), np.float32); mask = np.zeros((n_rows, n), np.float32)
    r, c = agg.row.values, agg.code.map(fidx).values
    values[r, c] = np.clip((agg.numeric_value.values - agg.code.map(mean).values) / agg.code.map(std).values, -5, 5)
    mask[r, c] = 1.0

    time_since = np.ones((n_rows, n), np.float32); carry = np.zeros((n_rows, n), np.float32)
    nb, off = adm.n_bins.values, adm.offset.values
    prev_v = np.zeros((len(adm), n), np.float32); prev_h = np.full((len(adm), n), -np.inf, np.float32)
    for b in range(int(nb.max())):
        live = np.where(nb > b)[0]; rows = off[live] + b
        m = mask[rows] > 0
        hrs = b * BIN_HOURS - prev_h[live]; seen = np.isfinite(hrs)
        cf = np.where(m, values[rows], np.where(seen, np.exp(-np.where(seen, hrs, 0) / TAU) * prev_v[live], 0.0))
        ts = np.where(m, 0.0, np.where(seen, np.clip(hrs / MAX_HOURS_SINCE, 0, 1), 1.0))
        carry[rows], time_since[rows] = cf, ts
        prev_v[live] = np.where(m | seen, cf, prev_v[live])
        prev_h[live] = np.where(m, b * BIN_HOURS, prev_h[live])
    state = np.concatenate([values, mask, time_since, carry], 1).astype(np.float16)

    med = ev[ev.code.str.startswith("MED//")]
    grp = med.code.str.split("//").str[-1].map(atc4_group)
    med = med[grp.notna()]
    act5 = np.zeros((n_rows, len(GROUPS)), np.uint8)
    act5[med.row.values, grp[grp.notna()].map({g: i for i, g in enumerate(GROUPS)}).values] = 1
    return adm, state, act5


def prepare_xsite(out_dir=None, sites=SITES) -> None:
    out_dir = Path(out_dir or ROOT / "data/processed/xsite")
    out_dir.mkdir(parents=True, exist_ok=True)
    mpath = out_dir / "xsite_manifest.json"
    if mpath.exists():
        man = json.loads(mpath.read_text())
    else:
        man = select_features()
        man["groups"] = GROUPS
        man["atc4_group_rules"] = ATC4_GROUP_RULES
        man["site_stats"] = {}
    state_codes = man["vital_codes"] + man["lab_codes"]

    for site in sites:
        log.info("[%s] loading", site)
        d = _load_site(site, state_codes)
        ev = d["ev"].merge(d["subj"][["subject_id", "split"]], on="subject_id")
        tr = ev[(ev.split == "train") & ev.code.isin(state_codes) & ev.numeric_value.notna()]
        st = tr.groupby("code").numeric_value.agg(["mean", "std"])
        man["site_stats"][site] = {c: [float(r["mean"]), float(r["std"])] for c, r in st.iterrows()}
        if site == "mimic":   # shared normalisation = MIMIC train statistics
            ms = man["site_stats"]["mimic"]
            man["state_mean"] = {c: ms.get(c, [0.0, 1.0])[0] for c in state_codes}
            man["state_std"] = {c: max(ms.get(c, [0.0, 1.0])[1], 1e-6) for c in state_codes}
        elif "state_mean" not in man:
            raise RuntimeError("process 'mimic' first: it defines the shared normalisation")
        # ATC4 -> group map actually observed at this site
        atc = pd.Series(ev[ev.code.str.startswith("MED//")].code.unique())
        man.setdefault("atc4_map", {}).update({c.split("//")[-1]: atc4_group(c.split("//")[-1]) for c in atc})

        labels = {}
        for f in sorted((_site_dir(site) / "labels").glob("*.parquet")):
            l = pd.read_parquet(f, columns=["encounter_id", "boolean_value"])
            labels[f"y_{f.stem}"] = l.set_index("encounter_id").boolean_value.astype(float)

        enc = d["enc"].merge(d["subj"], on="subject_id")
        for split, sname in SPLIT_NAMES.items():
            e_s = enc[enc.split == sname]
            adm, state, act5 = _build(ev[ev.split == sname], e_s, state_codes, man["state_mean"], man["state_std"])
            adm["age"] = (adm.admit_time - adm.birth_time).dt.total_seconds() / (365.25 * 86400)
            for k, s in labels.items():
                adm[k] = adm.encounter_id.map(s)
            adm = adm.rename(columns={"subject_id": "patient_id"})
            adm["hadm_id"] = adm["encounter_id"]
            sex_idx = adm.sex.map({"F": 0, "M": 1}).fillna(2).astype(int).values
            demo = np.concatenate([np.where(adm.age.notna(), np.clip(adm.age, 0, 100) / 100, 0.5)[:, None],
                                   np.eye(3)[sex_idx]], 1).astype(np.float32)
            sd = out_dir / site / split
            sd.mkdir(parents=True, exist_ok=True)
            np.savez(sd / "timelines.npz", state=state, act5=act5,
                     act78=np.zeros((len(state), 0), np.uint8))
            np.save(sd / "demo.npy", demo)
            adm.to_parquet(sd / "admissions.parquet")
            # patient_labels kept for API compatibility with experiments.data.Split
            adm[["patient_id"] + [c for c in adm.columns if c.startswith("y_")]].to_parquet(sd / "patient_labels.parquet")
            log.info("[%s/%s] %d ICU encounters, %d bins, action rate %s", site, split, len(adm), len(state),
                     np.round(act5.mean(0), 3).tolist())
        mpath.write_text(json.dumps(man, indent=1, default=str))


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", default=SITES)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    prepare_xsite(a.out_dir, a.sites)
