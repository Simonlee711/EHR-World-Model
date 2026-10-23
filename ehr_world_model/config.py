"""
Configuration dataclasses for the EHR World Model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FeatureConfig:
    """Controls which codes are selected and how time is discretised."""

    # Width of each temporal bin
    bin_hours: float = 6.0

    # How many top codes to retain per feature category
    # (ranked by number of distinct subjects they appear in)
    top_vitals: int = 10
    top_labs: int = 16
    top_meds: int = 25
    top_infusions: int = 10
    top_procedures: int = 8

    # A code must appear in at least this fraction of training subjects
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


@dataclass
class TrainConfig:
    """Training loop settings."""

    batch_size: int = 256
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 1e-4

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    seed: int = 42
    device: str = "cpu"

    # Max patients to load (None = all)
    max_patients: int = 5000

    data_dir: Path = Path("MIMIC_MEDS_Extract_v0.0.3_test")
    out_dir: Path = Path("world_model_mimic_out")

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.out_dir = Path(self.out_dir)
