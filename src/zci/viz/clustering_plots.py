"""Clustering visualisations — dendrogram and cluster-composition plots.

Every public function returns ``(fig, axes)``; callers decide where to save.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, set_link_color_palette

from .cluster_panel_plot import CLUSTER_COLORS


# ------------------------------------------------------------------
# Horizontal dendrogram  (tree divides from right → leaves on left)
# ------------------------------------------------------------------


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    labels: Sequence[str],
    n_clusters: int = 2,
    *,
    title: str = "Ward's Dendrogram",
    ylabel: str = "Site",
    figsize: tuple[float, float] = (8, 10),
    leaf_fontsize: int = 9,
    title_fontsize: int = 14,
    label_fontsize: int = 12,
    color_threshold: float | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Draw a **horizontal** dendrogram (leaves on the left, merges on the right).

    Parameters
    ----------
    linkage_matrix : np.ndarray
        Output of ``scipy.cluster.hierarchy.linkage``.
    labels : sequence of str
        Site names for leaves.
    n_clusters : int
        Number of clusters — used to set the default *color_threshold*
        so that the branches are coloured by group.
    title : str
        Figure title.
    figsize : tuple
        ``(width, height)`` in inches.
    leaf_fontsize : int
        Font size for site-name labels.
    title_fontsize : int
        Font size for the title.
    label_fontsize : int
        Font size for axis labels.
    color_threshold : float or None
        Explicit cut height.  ``None`` → automatic from the linkage
        matrix for *n_clusters*.

    Returns
    -------
    fig, ax
    """
    # Automatic colour threshold: midpoint between the last two merge heights
    # that would produce exactly *n_clusters* groups.
    if color_threshold is None:
        heights = linkage_matrix[:, 2]
        if n_clusters >= 2 and len(heights) >= n_clusters:
            # Cut between the (n_clusters-1)-th and n_clusters-th last merges
            sorted_h = np.sort(heights)
            cut_lo = sorted_h[-(n_clusters)]
            cut_hi = sorted_h[-(n_clusters - 1)]
            color_threshold = (cut_lo + cut_hi) / 2
        else:
            color_threshold = 0  # every leaf its own colour

    # ── Set dendrogram link colours to match CLUSTER_COLORS ──────────
    # Determine cluster assignments and leaf ordering so that the first
    # cluster encountered in the dendrogram gets the correct colour.
    assignments = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
    leaf_order = leaves_list(linkage_matrix)
    seen: list[int] = []
    for leaf_idx in leaf_order:
        cid = int(assignments[leaf_idx])
        if cid not in seen:
            seen.append(cid)
    palette = [CLUSTER_COLORS[cid - 1] for cid in seen]
    set_link_color_palette(palette)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    dendrogram(
        linkage_matrix,
        orientation="left",
        labels=list(labels),
        leaf_font_size=leaf_fontsize,
        color_threshold=color_threshold,
        ax=ax,
        above_threshold_color="grey",
    )

    # Reset palette to default
    set_link_color_palette(None)

    ax.set_xlabel("Linkage Distance", fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="x", alpha=0.3)

    return fig, ax
