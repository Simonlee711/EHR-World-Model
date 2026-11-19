"""Processed-data, checkpoint and output locations per site."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITES = {
    "mimic": dict(label="MIMIC-IV", proc=ROOT / "data/processed/mimic_wm", runs=ROOT / "runs/mimic",
                  outcomes={"Long LOS (>7d)": "adm_long_los", "In-hospital mortality": "adm_mortality"}),
    "eicu":  dict(label="eICU-CRD", proc=ROOT / "data/processed/xsite/eicu", runs=ROOT / "runs/eicu",
                  outcomes={"Long LOS (>7d)": "y_long_los", "In-hospital mortality": "y_mortality"}),
    "nwicu": dict(label="NWICU", proc=ROOT / "data/processed/xsite/nwicu", runs=ROOT / "runs/nwicu",
                  outcomes={"Long LOS (>7d)": "y_long_los", "In-hospital mortality": "y_mortality"}),
}
OUT = ROOT / "outputs"
