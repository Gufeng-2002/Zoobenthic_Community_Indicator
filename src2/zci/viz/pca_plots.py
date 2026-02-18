"""Visualisations for the pollution-PCA stage.

Every function: data in → (fig, axes) out.  No file I/O.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
import pandas as pd

from ..models.pca import PCAResult


# ---------------------------------------------------------------------------
# 1. Variance-explained bar + cumulative line
# ---------------------------------------------------------------------------


def plot_variance_explained(
    result: PCAResult,
    figsize: Tuple[int, int] = (12, 5),
) -> Tuple[plt.Figure, np.ndarray]:
    """Two-panel chart: individual + cumulative explained variance.

    Parameters
    ----------
    result : PCAResult
        Output of ``core.pca.run_pca``.
    figsize : tuple
        Overall figure size.

    Returns
    -------
    (fig, axes)
    """
    var_ratio = result.variance_info.loc["Proportion of Variance"].values.astype(float)
    cum_ratio = result.variance_info.loc["Cumulative Proportion"].values.astype(float)
    n = result.n_components
    xs = range(1, n + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # left: bar chart
    ax1.bar(xs, var_ratio, color="steelblue")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("Variance Explained per PC", fontweight="bold")
    ax1.set_xticks(list(xs))
    ax1.grid(axis="y", alpha=0.3)

    # right: cumulative line
    ax2.plot(xs, cum_ratio, "o-", color="steelblue", lw=2, ms=6)
    ax2.axhline(0.8, ls="--", color="red", lw=1.2, alpha=0.7, label="80 %")
    ax2.axhline(0.9, ls="--", color="green", lw=1.2, alpha=0.7, label="90 %")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title("Cumulative Variance", fontweight="bold")
    ax2.set_xticks(list(xs))
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig, np.array([ax1, ax2])


# ---------------------------------------------------------------------------
# 2. Ridge plot of PC loadings
# ---------------------------------------------------------------------------


def plot_ridge_loadings(
    result: PCAResult,
    figsize: Tuple[int, int] = (16, 10),
) -> Tuple[plt.Figure, plt.Axes]:
    """Ridge plot of PC loadings with hierarchically-clustered variable order.

    Positive loadings → solid fill.  Negative → hatched.

    Parameters
    ----------
    result : PCAResult
        Output of ``core.pca.run_pca``.
    figsize : tuple
        Figure size.

    Returns
    -------
    (fig, ax)
    """
    loadings = result.loadings

    # cluster variables for display order
    dist = pdist(loadings.values, metric="euclidean")
    order = leaves_list(linkage(dist, method="ward"))
    var_names = [loadings.index[i] for i in order]
    loadings_ordered = loadings.reindex(var_names)

    n_pcs = loadings_ordered.shape[1]
    ridge_h = 0.8
    spacing = 1.0
    offset = 0.1
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_pcs))

    fig, ax = plt.subplots(figsize=figsize)

    for i, pc in enumerate(loadings_ordered.columns):
        vals = loadings_ordered[pc].values
        abs_vals = np.abs(vals)
        norm = (abs_vals / abs_vals.max()) * ridge_h
        y_base = i * spacing

        for j, (v, nv) in enumerate(zip(vals, norm)):
            bot = y_base + offset
            top = bot + nv
            kw = dict(color=colors[i], edgecolor="white", linewidth=0.5)
            if v >= 0:
                ax.fill_between([j - 0.4, j + 0.4], [bot, bot], [top, top],
                                alpha=0.8, **kw)
            else:
                ax.fill_between([j - 0.4, j + 0.4], [bot, bot], [top, top],
                                alpha=0.6, hatch="///", **kw)

        ax.axhline(y_base + offset, color="lightgray", lw=0.5, alpha=0.7)
        ax.text(-2, y_base + offset + ridge_h / 2, pc,
                fontsize=12, fontweight="bold", ha="right", va="center")

    ax.set_xlim(-3, len(var_names))
    ax.set_ylim(-0.2, n_pcs * spacing + 0.5)
    ax.set_xticks(range(len(var_names)))
    ax.set_xticklabels(var_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticks([])
    ax.set_title("PC Loadings Ridge Plot\n(Variables Ordered by Hierarchical Clustering)",
                 fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Chemical Variables (Clustered Order)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    legend = [
        Patch(facecolor=colors[0], alpha=0.8, label="Positive loadings"),
        Patch(facecolor=colors[0], alpha=0.6, hatch="///", label="Negative loadings"),
    ]
    ax.legend(handles=legend, loc="upper right")
    fig.tight_layout()
    return fig, ax
