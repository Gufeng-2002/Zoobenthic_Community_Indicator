"""Structured result containers for the Bray–Curtis NMDS pipeline (Stage 4).

Mirrors ``models/lda.py`` and ``models/clustering.py`` in purpose —
holds every artefact produced by the NMDS + ZCI pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class ClusterNMDS:
    """NMDS ordination results for a single cluster.

    Attributes
    ----------
    cluster_id : int
        Cluster identifier (1 or 2).
    coords_df : pd.DataFrame
        (n_sites + 2) × 2 — ordination coordinates including endpoints.
    stress : float
        Final normalised stress.
    var_explained : np.ndarray
        Fraction of variance on each PCA-rotated axis.
    wa_df : pd.DataFrame
        Taxa × (WA1, WA2) weighted-average species scores.
    ref_label, deg_label : str
        Row names of the reference / degraded endpoint in *coords_df*.
    ref_ids, deg_ids : pd.Index
        Site IDs used to construct each endpoint.
    n_ep : int
        Number of extreme sites per endpoint.
    flipped : bool
        Whether NMDS1 was sign-flipped.
    """

    cluster_id: int
    coords_df: pd.DataFrame
    stress: float
    var_explained: np.ndarray
    wa_df: pd.DataFrame
    ref_label: str
    deg_label: str
    ref_ids: pd.Index
    deg_ids: pd.Index
    n_ep: int
    flipped: bool

    @property
    def real_site_ids(self) -> list:
        """Site IDs excluding the two endpoint rows."""
        return [s for s in self.coords_df.index
                if s not in (self.ref_label, self.deg_label)]

    @property
    def n_sites(self) -> int:
        return len(self.real_site_ids)


@dataclass
class ClusterZCI:
    """ZCI results for a single cluster.

    Attributes
    ----------
    cluster_id : int
    zci : pd.Series
        Named ``"ZCI"`` — one value per site.
    method : str
        ZCI method used (e.g. ``"BC-Direct"``).
    n_ep : int
        Number of extreme sites used for ZCI endpoint construction.
    r_pearson, p_pearson : float
        Pearson correlation with pollution score.
    r_spearman, p_spearman : float
        Spearman correlation with pollution score.
    """

    cluster_id: int
    zci: pd.Series
    method: str
    n_ep: int
    r_pearson: float
    p_pearson: float
    r_spearman: float
    p_spearman: float

    @property
    def significance_stars(self) -> str:
        if self.p_pearson < 0.001:
            return "***"
        elif self.p_pearson < 0.01:
            return "**"
        elif self.p_pearson < 0.05:
            return "*"
        return ""

    def summary_line(self) -> str:
        return (
            f"Cluster {self.cluster_id}: N_EP={self.n_ep}, "
            f"Method={self.method}, "
            f"r={self.r_pearson:+.4f}{self.significance_stars} "
            f"(p={self.p_pearson:.2e})"
        )


@dataclass
class NMDSPipelineResult:
    """Top-level result container for the Bray–Curtis NMDS pipeline.

    Holds NMDS ordination and ZCI results for every cluster.
    """

    nmds_results: Dict[int, ClusterNMDS]
    zci_results: Dict[int, ClusterZCI]
    clusters: List[int]
    taxa_relabd: pd.DataFrame
    site_meta: pd.DataFrame
    p20: float
    p80: float

    def summary(self) -> str:
        lines = ["Bray–Curtis NMDS Pipeline — Summary", "=" * 50]
        for cl in self.clusters:
            nm = self.nmds_results[cl]
            zc = self.zci_results[cl]
            lines.append(
                f"  Cluster {cl}: n={nm.n_sites}, "
                f"stress={nm.stress:.5f}, "
                f"ZCI {zc.method} (N_EP={zc.n_ep}) "
                f"r={zc.r_pearson:+.4f}{zc.significance_stars}"
            )
        return "\n".join(lines)
