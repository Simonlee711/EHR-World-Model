"""
Curated clinical state concepts shared by every site.

Only variables that change on the 6-hour time scale and are worth forecasting are
kept. Excluded relative to a frequency-ranked top-30 vitals / top-50 labs list: alarm
limits, Braden and Morse nursing scores, "parameters checked", urine dipstick / sediment
items, admission weight, RBC indices (MCH, MCV, MCHC, RDW, RBC count), PT (redundant
with INR) and absolute differential counts (redundant with %).

Each concept has
  code        harmonised code (identical to data/{mimic,eicu,nwicu} extracts)
  mimic       MIMIC-IV itemids in the MEDS extract that are merged into the concept
              (with an optional unit transform), e.g. NIBP + arterial SBP
  lo, hi      physiological plausibility limits in canonical units (values outside dropped)
  log         z-score log(x) instead of x (heavy-tailed laboratory values)
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

F2C: Callable = lambda v: (v - 32.0) * 5.0 / 9.0
FRAC2PCT: Callable = lambda v: np.where(v <= 1.0, v * 100.0, v)

# code, label, unit, {itemid: transform|None}, lo, hi, log
_C = [
    # ── vitals / bedside ──
    ("VITAL//HR", "Heart rate", "bpm", {220045: None}, 20, 300, False),
    ("VITAL//SBP", "Systolic BP", "mmHg", {220179: None, 220050: None}, 40, 300, False),
    ("VITAL//DBP", "Diastolic BP", "mmHg", {220180: None, 220051: None}, 10, 200, False),
    ("VITAL//RR", "Respiratory rate", "/min", {220210: None, 224690: None}, 2, 80, False),
    ("VITAL//SPO2", "SpO2", "%", {220277: None}, 40, 100, False),
    ("VITAL//TEMP", "Temperature", "degC", {223761: F2C, 223762: None}, 25, 45, False),
    ("VITAL//GCS_EYE", "GCS eye", "score", {220739: None}, 1, 4, False),
    ("VITAL//GCS_VERBAL", "GCS verbal", "score", {223900: None}, 1, 5, False),
    ("VITAL//GCS_MOTOR", "GCS motor", "score", {223901: None}, 1, 6, False),
    ("VITAL//FIO2", "FiO2", "%", {223835: FRAC2PCT}, 21, 100, False),
    # ── haematology ──
    ("LAB//HEMOGLOBIN", "Hemoglobin", "g/dL", {51222: None, 50811: None}, 3, 25, False),
    ("LAB//HEMATOCRIT", "Hematocrit", "%", {51221: None, 50810: None}, 10, 70, False),
    ("LAB//PLATELETS", "Platelets", "10^9/L", {51265: None}, 1, 2000, True),
    ("LAB//WBC", "WBC", "10^9/L", {51301: None, 51300: None}, 0.05, 500, True),
    ("LAB//NEUTROPHILS_PCT", "Neutrophils", "%", {51256: None}, 0, 100, False),
    ("LAB//LYMPHOCYTES_PCT", "Lymphocytes", "%", {51244: None}, 0, 100, False),
    # ── chemistry / renal ──
    ("LAB//SODIUM", "Sodium", "mmol/L", {50983: None, 50824: None}, 100, 190, False),
    ("LAB//POTASSIUM", "Potassium", "mmol/L", {50971: None, 50822: None}, 1.5, 10, False),
    ("LAB//CHLORIDE", "Chloride", "mmol/L", {50902: None, 50806: None}, 60, 150, False),
    ("LAB//BICARBONATE", "Bicarbonate", "mmol/L", {50882: None}, 3, 60, False),
    ("LAB//ANION_GAP", "Anion gap", "mmol/L", {50868: None}, -10, 60, False),
    ("LAB//BUN", "BUN", "mg/dL", {51006: None}, 1, 300, True),
    ("LAB//CREATININE", "Creatinine", "mg/dL", {50912: None}, 0.1, 30, True),
    ("LAB//GLUCOSE", "Glucose", "mg/dL", {50931: None, 50809: None}, 10, 2000, True),
    ("LAB//CALCIUM", "Calcium", "mg/dL", {50893: None}, 3, 20, False),
    ("LAB//ION_CALCIUM", "Ionized calcium", "mmol/L", {50808: None}, 0.3, 3, False),
    ("LAB//MAGNESIUM", "Magnesium", "mg/dL", {50960: None}, 0.3, 10, False),
    ("LAB//PHOSPHATE", "Phosphate", "mg/dL", {50970: None}, 0.3, 20, False),
    # ── liver ──
    ("LAB//ALT", "ALT", "U/L", {50861: None}, 1, 20000, True),
    ("LAB//AST", "AST", "U/L", {50878: None}, 1, 30000, True),
    ("LAB//ALP", "Alkaline phosphatase", "U/L", {50863: None}, 5, 5000, True),
    ("LAB//BILI_TOTAL", "Total bilirubin", "mg/dL", {50885: None}, 0.05, 80, True),
    ("LAB//ALBUMIN", "Albumin", "g/dL", {50862: None}, 0.5, 7, False),
    # ── coagulation ──
    ("LAB//INR", "INR", "ratio", {51237: None}, 0.5, 20, True),
    ("LAB//PTT", "aPTT", "s", {51275: None}, 10, 200, True),
    # ── perfusion / blood gas ──
    ("LAB//LACTATE", "Lactate", "mmol/L", {50813: None}, 0.1, 30, True),
    ("LAB//PH", "pH", "", {50820: None}, 6.5, 8.0, False),
    ("LAB//PCO2", "pCO2", "mmHg", {50818: None}, 5, 200, False),
    ("LAB//PO2", "pO2", "mmHg", {50821: None}, 10, 700, True),
    ("LAB//BASE_EXCESS", "Base excess", "mmol/L", {50802: None}, -40, 40, False),
    # ── injury / inflammation ──
    ("LAB//TROPONIN_T", "Troponin T", "ng/mL", {51003: None}, 0.001, 50, True),
    ("LAB//CRP", "C-reactive protein", "mg/L", {50889: None}, 0.1, 600, True),
]

CONCEPTS: List[Dict] = [dict(code=c, label=l, unit=u, mimic=m, lo=lo, hi=hi, log=lg)
                        for c, l, u, m, lo, hi, lg in _C]
BY_CODE = {c["code"]: c for c in CONCEPTS}
LABEL = {c["code"]: c["label"] for c in CONCEPTS}


def _clean(code: str, v: np.ndarray) -> np.ndarray:
    """Range-filter (NaN outside limits) and log-transform in canonical units."""
    c = BY_CODE[code]
    v = np.where((v >= c["lo"]) & (v <= c["hi"]), v, np.nan)
    if c["log"]:
        v = np.log(np.maximum(v, 1e-3))
    return v


def meds_to_concepts(df: pd.DataFrame) -> pd.DataFrame:
    """MIMIC MEDS extract: replace every LAB//<itemid>//<unit> row by curated concept rows."""
    is_lab = df["code"].str.startswith("LAB//")
    lab = df[is_lab & df["numeric_value"].notna()]
    item = pd.to_numeric(lab["code"].str.split("//").str[1], errors="coerce")
    parts = []
    for c in CONCEPTS:
        for itemid, fn in c["mimic"].items():
            r = lab[item == itemid]
            if not len(r):
                continue
            v = r["numeric_value"].to_numpy(np.float64)
            if fn is not None:
                v = fn(v)
            r = r.assign(code=c["code"], numeric_value=_clean(c["code"], v).astype(np.float32))
            parts.append(r[r["numeric_value"].notna()])
    return pd.concat([df[~is_lab]] + parts, ignore_index=True).sort_values(["patient_id", "time"], kind="stable")


def harmonised_clean(ev: pd.DataFrame) -> pd.DataFrame:
    """Harmonised (x-site) extracts: same range limits + log transform for curated codes."""
    out = []
    for code, g in ev[ev["code"].isin(BY_CODE)].groupby("code"):
        out.append(g.assign(numeric_value=_clean(code, g["numeric_value"].to_numpy(np.float64)).astype(np.float32)))
    clean = pd.concat(out) if out else ev.iloc[:0]
    clean = clean[clean["numeric_value"].notna()]
    return pd.concat([ev[~ev["code"].isin(BY_CODE)], clean], ignore_index=True)


def select(train_df: pd.DataFrame, id_col: str = "patient_id", min_freq: float = 0.05):
    """Curated concepts observed in >= min_freq of training patients, with z-score stats
    (computed on the cleaned / log-transformed scale)."""
    n = train_df[id_col].nunique()
    s = train_df[train_df["code"].isin(BY_CODE) & train_df["numeric_value"].notna()]
    freq = s.groupby("code")[id_col].nunique() / n
    keep = [c["code"] for c in CONCEPTS if freq.get(c["code"], 0) >= min_freq]
    st = s[s["code"].isin(keep)].groupby("code")["numeric_value"].agg(["mean", "std"])
    vit = [c for c in keep if c.startswith("VITAL//")]
    lab = [c for c in keep if c.startswith("LAB//")]
    return vit, lab, st["mean"].to_dict(), st["std"].clip(lower=1e-6).to_dict(), freq.to_dict()
