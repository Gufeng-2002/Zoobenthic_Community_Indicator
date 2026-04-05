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
# Reference-site selection helpers
# ------------------------------------------------------------------


def resolve_n_ref(threshold: int | float, n_total: int) -> int:
    """Resolve a threshold specification to an absolute site count.

    Parameters
    ----------
    threshold : int or float
        If ``> 1`` → treated as an absolute number of sites.
        If ``<= 1`` → treated as a proportion of *n_total*.
    n_total : int
        Total number of sites available.

    Returns
    -------
    int
        Number of reference sites (at least 1).
    """
    if threshold > 1:
        return max(1, min(int(threshold), n_total))
    n = int(n_total * threshold)
    return max(n, 1)


def ref_site_label(n_ref: int) -> str:
    """Build a human-readable label like ``'lowest 47 sites'``."""
    return f"lowest {n_ref} sites"


def select_reference_sites(
    pollution_score: pd.Series,
    quantile: int | float = 0.20,
) -> pd.Series:
    """Boolean mask: *True* for the least-polluted sites.

    Parameters
    ----------
    pollution_score : pd.Series
        Composite pollution score (higher = more polluted).
    quantile : int or float
        If ``> 1`` → absolute number of reference sites.
        If ``<= 1`` → fraction of sites (default 0.20 = 20 %).

    Returns
    -------
    pd.Series[bool]
        Same index as *pollution_score*.
    """
    n = resolve_n_ref(quantile, len(pollution_score))
    threshold = pollution_score.nsmallest(n).max()
    return pollution_score <= threshold
