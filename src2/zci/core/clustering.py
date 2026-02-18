"""Hierarchical clustering — pure computation, no plotting, no file paths.

Provides:
- Ward's hierarchical clustering with 1-indexed labels
- Reference-site selection by pollution quantile

Taxa constants live in ``models.clustering``; taxa transforms live
in ``core.transforms``.  This module only does the maths of clustering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering


# ------------------------------------------------------------------
# Hierarchical clustering (Ward + Euclidean)
# ------------------------------------------------------------------


def ward_cluster(
    taxa: pd.DataFrame,
    n_clusters: int = 2,
) -> tuple[pd.Series, np.ndarray]:
    """Perform Ward's hierarchical clustering on a taxa matrix.

    Parameters
    ----------
    taxa : pd.DataFrame
        (Possibly transformed) taxa matrix — sites × taxa.
    n_clusters : int
        Number of groups to cut (default 2).

    Returns
    -------
    labels : pd.Series
        **1-indexed** cluster labels (``1 … n_clusters``),
        index identical to *taxa*.
    linkage_matrix : np.ndarray
        Scipy linkage matrix (for dendrogram plotting).
    """
    # Scipy linkage (for dendrogram)
    Z = linkage(taxa.values, method="ward", metric="euclidean")

    # Sklearn for cluster assignments (consistent with original code)
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward",
        metric="euclidean",
    )
    raw_labels = model.fit_predict(taxa.values)

    # Convert 0-indexed → 1-indexed
    labels = pd.Series(raw_labels + 1, index=taxa.index, name="Cluster")

    return labels, Z


# ------------------------------------------------------------------
# Reference-site selection helper
# ------------------------------------------------------------------


def select_reference_sites(
    pollution_score: pd.Series,
    quantile: float = 0.20,
) -> pd.Series:
    """Boolean mask: *True* for sites in the bottom *quantile* of pollution.

    Parameters
    ----------
    pollution_score : pd.Series
        Composite pollution score (higher = more polluted).
    quantile : float
        Fraction of sites to classify as "reference" (default 0.20 = 20 %).

    Returns
    -------
    pd.Series[bool]
        Same index as *pollution_score*.
    """
    n = int(len(pollution_score) * quantile)
    if n < 1:
        n = 1
    threshold = pollution_score.nsmallest(n).max()
    return pollution_score <= threshold
