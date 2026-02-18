"""Structured result container for hierarchical clustering (Stage 2).

Mirrors ``PCAResult`` in purpose — holds every artefact produced by
the taxa-assemblage clustering step, with convenience helpers for
exporting tables and augmented DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# 16 taxa columns from the study data (octave-transformed)
# ------------------------------------------------------------------

TAXA_COLUMNS: list[str] = [
    "Acari",
    "Amphipoda",
    "Caenis",
    "Ceratopogonidae",
    "Chironomidae",
    "Dreissena",
    "Gastropoda",
    "Hexagenia",
    "Hirudinea",
    "Hydropsychidae",
    "Hydrozoa",
    "Nematoda",
    "Oligochaeta",
    "Other Trichoptera",
    "Sphaeriidae",
    "Turbellaria",
]


@dataclass
class ClusteringResult:
    """Container returned by the taxa-assemblage clustering pipeline.

    Attributes
    ----------
    ref_mask : pd.Series[bool]
        Boolean mask over **all** sites (``True`` = reference).
    cluster_labels : pd.Series
        1-indexed cluster labels for **reference sites only**.
    linkage_matrix : np.ndarray
        Scipy linkage matrix (for dendrogram plotting).
    taxa_ref : pd.DataFrame
        Raw (octave-scale) taxa data for reference sites.
    taxa_ref_transformed : pd.DataFrame
        Taxa data after the chosen transform (may equal ``taxa_ref``
        when transform is ``"octave"``/identity).
    n_clusters : int
        Number of clusters that were requested.
    taxa_transform : str
        Name of the transform that was applied (``"octave"`` or
        ``"relative_abundance"``).
    all_site_index : pd.Index
        Full index of the original data (used to build the all-site
        cluster column with ``NaN`` for non-reference sites).
    """

    ref_mask: pd.Series
    cluster_labels: pd.Series
    linkage_matrix: np.ndarray
    taxa_ref: pd.DataFrame
    taxa_ref_transformed: pd.DataFrame
    n_clusters: int
    taxa_transform: str
    all_site_index: pd.Index

    # --- convenience helpers (no I/O, no side-effects) ------------------

    def cluster_all(self) -> pd.Series:
        """Full-length cluster column — ``NaN`` for non-reference sites."""
        s = pd.Series(np.nan, index=self.all_site_index, name="Cluster")
        s.loc[self.cluster_labels.index] = self.cluster_labels
        return s

    def cluster_distribution(self) -> pd.Series:
        """Value-counts of cluster labels (reference sites only)."""
        return self.cluster_labels.value_counts().sort_index()

    def to_ref_table(self) -> pd.DataFrame:
        """Reference-site taxa matrix with a leading ``Cluster`` column.

        Suitable for direct Excel export.
        """
        tbl = self.taxa_ref.copy()
        tbl.insert(0, "Cluster", self.cluster_labels)
        return tbl

    def to_augmented_dataframe(
        self,
        level0: str = "02_taxa_assemblage",
        level1: str = "raw",
    ) -> pd.DataFrame:
        """Build a MultiIndex DataFrame with the all-site cluster column.

        Returns
        -------
        pd.DataFrame
            3-level column MultiIndex:
            ``(level0, level1, "Cluster")``.
        """
        cluster_col = self.cluster_all().to_frame("Cluster")
        tuples = [(level0, level1, c) for c in cluster_col.columns]
        cluster_col.columns = pd.MultiIndex.from_tuples(
            tuples, names=["block", "subblock", "var"]
        )
        return cluster_col

    def summary(self) -> str:
        """One-line human-readable summary."""
        dist = ", ".join(
            f"G{g}: {n}" for g, n in self.cluster_distribution().items()
        )
        return (
            f"ClusteringResult(n_ref={self.ref_mask.sum()}, "
            f"n_clusters={self.n_clusters}, "
            f"transform={self.taxa_transform!r}, "
            f"distribution=[{dist}])"
        )
