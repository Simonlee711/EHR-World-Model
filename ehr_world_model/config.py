"""
Feature-selection configuration.
"""
from __future__ import annotations

from dataclasses import dataclass


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
