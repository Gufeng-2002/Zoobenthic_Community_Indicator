"""Cluster analysis figures.

Builds three standalone figures that were previously combined into a
single multi-panel layout:

* geographic map of reference sites coloured by cluster
* z-scored environmental-variable bar chart (± SEM)
* relative-abundance taxa bar chart (± SEM)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats as sp_stats

from .map_plots import plot_corridor_map
from ..core.anova import _stars          # small private helper for stars


# ------------------------------------------------------------------
# Colour palette  (blue / orange / green — matches dendrogram)
# ------------------------------------------------------------------

CLUSTER_COLORS: list[str] = [
    "#1f77b4",   # Cluster 1  — blue
    "#ff7f0e",   # Cluster 2  — orange
    "#2ca02c",   # Cluster 3  — green
    "#4C72B0",   # Cluster 4  — steel blue  (spare)
    "#9370DB",   # Cluster 5  — purple
]

# Taxa x-axis order matching the reference figure
TAXA_DISPLAY_ORDER: list[str] = [
    'Oligochaeta', 
    'Chironomidae', 
    'Nematoda', 
    'Sphaeriidae',
    'Acari', 
    'Hexagenia',
    'Caenis', 
    'Hirudinea', 
    'Turbellaria', 
    'Gastropoda', 
    'Hydrozoa', 
    'Other Trichoptera', 
    'Amphipoda', 
    'Hydropsychidae', 
    'Dreissena', 
    'Ceratopogonidae'
]



# Short display names for environmental variables
_ENV_SHORT: dict[str, str] = {
    "Measured Depth (m)":                "Depth (m)",
    "Velocity  at bottom (m/sec)":       "Velocity",
    "Velocity  at bottom (m/sec)_Imputed": "Velocity",
    "Water DO Bottom (mg/L)":            "DO (mg/L)",
    "Temperature (oC)":                  "Temp (°C)",
    "MPS (Phi)":                         "MPS (Phi)",
    "LOI (%)":                           "LOI (%)",
}


def _cluster_color(cluster_id: int) -> str:
    """Return the canonical display color for a 1-indexed cluster ID."""
    color_idx = max(0, int(cluster_id) - 1)
    if color_idx < len(CLUSTER_COLORS):
        return CLUSTER_COLORS[color_idx]
    return CLUSTER_COLORS[color_idx % len(CLUSTER_COLORS)]


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _bar_with_stars(
    ax: plt.Axes,
    x: np.ndarray,
    cluster_means: np.ndarray,      # shape (n_clusters, n_vars)
    cluster_sems: np.ndarray,       # same shape
    cluster_ids: Sequence[int],
    pvalues: dict[str, float],      # var_name → p
    var_names: Sequence[str],       # original names (for p lookup)
    display_names: Sequence[str],   # tick labels
    *,
    ylabel: str,
    title: str,
    colors: list[str],
    one_sided_error: bool = False,
) -> None:
    """Grouped bar chart with asymmetric error bars + significance stars."""
    n_clusters = len(cluster_ids)
    width = 0.8 / n_clusters

    for i, cid in enumerate(cluster_ids):
        offset = (i - n_clusters / 2 + 0.5) * width
        means = cluster_means[i]
        sems = cluster_sems[i]

        if one_sided_error:
            # positive bars → upper only; negative → lower only
            lo = np.where(means >= 0, 0, sems)
            hi = np.where(means >= 0, sems, 0)
        else:
            lo = hi = sems

        ax.bar(
            x + offset, means, width,
            yerr=np.array([lo, hi]),
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
            label=f"Cluster {cid}",
            error_kw=dict(linewidth=1.2, ecolor="black", capsize=2),
        )

    # ── significance stars ───────────────────────────────────────────
    for j, var in enumerate(var_names):
        p = pvalues.get(var, 1.0)
        s = _stars(p)
        if not s:
            continue

        # find the tallest positive bar (top of bar + SEM)
        tops = []
        for i in range(n_clusters):
            mean_v = cluster_means[i, j]
            sem_v = cluster_sems[i, j]
            if mean_v >= 0:
                tops.append(mean_v + sem_v)
            else:
                tops.append(mean_v - sem_v)
        max_top = max(tops)
        idx_max = int(np.argmax(tops))

        # x position: right edge of the tallest bar
        bar_x = x[j] + (idx_max - n_clusters / 2 + 0.5) * width + width / 2
        star_y = max_top

        ax.text(
            bar_x + 0.02, star_y, s,
            ha="left", va="bottom",
            fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=35, ha="right", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    # ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def plot_cluster_panel(
    cluster_labels: pd.Series,
    lat: pd.Series,
    lon: pd.Series,
    env_data: pd.DataFrame,
    taxa_octave: pd.DataFrame,
    taxa_relabd: pd.DataFrame,
    env_pvalues: dict[str, float],
    taxa_pvalues: dict[str, float],
    maps_dir: str | Path,
    *,
    env_vars: Sequence[str] | None = None,
    taxa_order: Sequence[str] | None = None,
    map_func=None,
    map_figsize: Tuple[float, float] = (8.5, 8.5),
    env_figsize: Tuple[float, float] = (9.5, 5.2),
    taxa_figsize: Tuple[float, float] = (14.0, 6.8),
    label_fontsize: int = 12,
) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
    """Build standalone cluster figures for map, environment, and taxa.

    Parameters
    ----------
    cluster_labels : pd.Series
        1-indexed cluster IDs for reference sites.
    lat, lon : pd.Series
        Coordinates (same index as *cluster_labels*).
    env_data : pd.DataFrame
        Environmental variables for reference sites (original scale).
    taxa_octave : pd.DataFrame
        Taxa data in octave scale (reference sites).
    taxa_relabd : pd.DataFrame
        Taxa data in relative-abundance scale (reference sites).
    env_pvalues : dict
        ``{var_name: p}`` from ANOVA on environmental variables.
    taxa_pvalues : dict
        ``{taxon_name: p}`` from ANOVA on **octave-scale** taxa.
    maps_dir : str or Path
        Shapefile root directory.
    env_vars : list of str, optional
        Which env columns to plot (default: all in *env_data*).
    taxa_order : list of str, optional
        X-axis order for taxa (default: ``TAXA_DISPLAY_ORDER``).
    map_func : callable, optional
        Custom map renderer ``(ax, maps_dir, ...) -> ax``.
        Defaults to :func:`plot_corridor_map`.
    map_figsize / env_figsize / taxa_figsize : tuple
        Figure size for each standalone output.
    label_fontsize : int
        Axis-label font size.

    Returns
    -------
    dict
        Mapping of ``{"map": (fig, ax), "env": (fig, ax), "taxa": (fig, ax)}``.
    """
    cluster_ids = sorted(cluster_labels.unique())
    n_clusters = len(cluster_ids)
    colors = [_cluster_color(cid) for cid in cluster_ids]

    # ── LEFT: map ────────────────────────────────────────────────────
    fig_map, ax_map = plt.subplots(figsize=map_figsize, constrained_layout=True)
    _map_renderer = map_func or plot_corridor_map
    _map_renderer(ax_map, maps_dir, annotate=True)

    for i, cid in enumerate(cluster_ids):
        mask = cluster_labels == cid
        ax_map.scatter(
            lon.loc[mask], lat.loc[mask],
            c=colors[i], s=150, alpha=0.85,
            edgecolors="black", linewidth=0.8,
            label=f"Cluster {cid} (n={mask.sum()})",
            zorder=3,
        )
        # site-name annotations
        for idx in cluster_labels.loc[mask].index:
            ax_map.annotate(
                str(idx), (lon[idx], lat[idx]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=10, alpha=0.7,
            )

    ax_map.set_xlabel("Longitude", fontsize=label_fontsize, fontweight="bold")
    ax_map.set_ylabel("Latitude", fontsize=label_fontsize, fontweight="bold")
    ax_map.legend(loc="upper left", fontsize=16, framealpha=0.9)

    # ── UPPER-RIGHT: env z-score bars ────────────────────────────────
    if env_vars is None:
        env_vars = list(env_data.columns)

    # z-score: centre each variable across *all* ref sites, then compute
    # per-cluster mean and SEM (in z-score units)
    env_means, env_sems = [], []
    for cid in cluster_ids:
        m_row, s_row = [], []
        for var in env_vars:
            vals_all = env_data[var].dropna()
            mu, sigma = vals_all.mean(), vals_all.std(ddof=1)
            if sigma == 0:
                sigma = 1.0
            cluster_vals = env_data.loc[cluster_labels == cid, var].dropna()
            z = (cluster_vals - mu) / sigma
            m_row.append(z.mean())
            s_row.append(z.sem() if len(z) > 1 else 0.0)
        env_means.append(m_row)
        env_sems.append(s_row)
    env_means = np.array(env_means)
    env_sems = np.array(env_sems)
    env_display = [_ENV_SHORT.get(v, v) for v in env_vars]

    fig_env, ax_env = plt.subplots(figsize=env_figsize, constrained_layout=True)

    _bar_with_stars(
        ax_env,
        np.arange(len(env_vars)),
        env_means, env_sems, cluster_ids,
        env_pvalues, list(env_vars), env_display,
        ylabel="Mean z-score (± SEM)",
        title="Standardized Habitat Features Across Clusters",
        colors=colors,
        one_sided_error=True,
    )
    env_handles = [
        Patch(facecolor=_cluster_color(cid), edgecolor="black", label=f"Cluster {cid}")
        for cid in cluster_ids
    ]
    ax_env.legend(
        handles=env_handles,
        loc="upper right",
        framealpha=0.9,
    )

    # ── LOWER-RIGHT: taxa relative-abundance bars ────────────────────
    if taxa_order is None:
        taxa_order = TAXA_DISPLAY_ORDER
    # keep only taxa that exist in the data
    taxa_order = [t for t in taxa_order if t in taxa_relabd.columns]

    tax_means, tax_sems = [], []
    for cid in cluster_ids:
        mask = cluster_labels == cid
        sub = taxa_relabd.loc[mask, taxa_order]
        tax_means.append(sub.mean().values)
        tax_sems.append(sub.sem().values)
    tax_means = np.array(tax_means)
    tax_sems = np.array(tax_sems)

    fig_tax, ax_tax = plt.subplots(figsize=taxa_figsize, constrained_layout=True)

    _bar_with_stars(
        ax_tax,
        np.arange(len(taxa_order)),
        tax_means, tax_sems, cluster_ids,
        taxa_pvalues, taxa_order, taxa_order,
        ylabel="Mean Relative Abundance (± SE)",
        title="Reference Sites: Taxa by Cluster",
        colors=colors,
        one_sided_error=True,
    )
    taxa_handles = [
        Patch(facecolor=_cluster_color(cid), edgecolor="black", label=f"Cluster {cid}")
        for cid in cluster_ids
    ]
    ax_tax.legend(
        handles=taxa_handles,
        loc="upper right",
        framealpha=0.9,
    )

    return {
        "map": (fig_map, ax_map),
        "env": (fig_env, ax_env),
        "taxa": (fig_tax, ax_tax),
    }
