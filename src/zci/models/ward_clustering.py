"""Result container for Ward clustering with robustness assessment (Stage 2).

Extends :class:`~zci.models.clustering.ClusteringResult` with site-level
robustness metrics (silhouette, co-assignment confidence, pvclust AU).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .clustering import ClusteringResult


@dataclass
class WardClusteringResult(ClusteringResult):
    """Ward clustering result with robustness assessment.

    Inherits all fields from :class:`ClusteringResult` and adds:

    Attributes
    ----------
    robustness_table : pd.DataFrame
        One row per reference site with columns: Original_Cluster,
        Branch_AU, Silhouette, Own_Coassign, Best_Alt_Coassign, Margin, Status.
    coassignment_matrix : pd.DataFrame
        Square co-assignment matrix (sites × sites).
    pvclust_summary : pd.DataFrame or None
        Per-cluster AU/BP from pvclust (None if R was unavailable).
    """

    robustness_table: pd.DataFrame
    coassignment_matrix: pd.DataFrame
    pvclust_summary: Optional[pd.DataFrame]

    def status_distribution(self) -> pd.Series:
        """Value-counts of Core / Peripheral / Uncertain."""
        return self.robustness_table["Status"].value_counts()

    def mean_silhouette(self) -> float:
        """Overall mean silhouette width."""
        return float(self.robustness_table["Silhouette"].mean())

    def summary(self) -> str:
        """One-line human-readable summary."""
        base = super().summary()
        avg_sil = self.mean_silhouette()
        status = self.status_distribution()
        status_str = ", ".join(f"{k}: {v}" for k, v in status.items())
        return (
            f"{base}\n"
            f"  Robustness: mean_sil={avg_sil:.3f}, [{status_str}]"
        )
