"""Grid trend plots across clusters.

Taxa versions:
  Single-group version: one line per subplot with ANOVA significance.
  Comparison version: ref vs most-polluted grouped bar charts with SE.

Environmental version:
  plot_env_trend_comparison: ref vs most-polluted grouped bar charts with SE.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind

from .cluster_panel_plot import TAXA_DISPLAY_ORDER, CLUSTER_COLORS
from ..core.anova import _stars


# ------------------------------------------------------------------
# Shared styling helper
# ------------------------------------------------------------------

def _despine(ax: plt.Axes) -> None:
    """Remove top and right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ------------------------------------------------------------------
# Single-group taxa trend grid (ref OR polluted)
# ------------------------------------------------------------------

def plot_taxa_trend_grid(
    taxa_relabd: pd.DataFrame,
    cluster_labels: pd.Series,
    *,
    taxa_order: Sequence[str] | None = None,
    title: str = "",
    color: str = "#4878CF",
    figsize: Tuple[float, float] = (14, 12),
    dpi: int = 300,
) -> Tuple[plt.Figure, np.ndarray]:
    """4×4 grid — one taxon per subplot, mean ± SE across clusters.

    Style matches the reference random-walk figure: plain dots + line,
    no fill band, despined axes.
    """
    if taxa_order is None:
        taxa_order = TAXA_DISPLAY_ORDER
    taxa_order = [t for t in taxa_order if t in taxa_relabd.columns]

    common = cluster_labels.index.intersection(taxa_relabd.index)
    taxa_relabd = taxa_relabd.loc[common]
    cluster_labels = cluster_labels.loc[common]

    cluster_ids = sorted(cluster_labels.unique())
    n_clusters = len(cluster_ids)

    nrows, ncols = 4, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    axes_flat = axes.flatten()

    for idx, taxon in enumerate(taxa_order):
        ax = axes_flat[idx]

        means, sems, groups = [], [], []
        for cid in cluster_ids:
            vals = taxa_relabd.loc[cluster_labels == cid, taxon].dropna()
            groups.append(vals.values)
            means.append(vals.mean())
            sems.append(vals.sem() if len(vals) > 1 else 0.0)

        means = np.array(means)
        sems = np.array(sems)
        x = np.arange(1, n_clusters + 1)

        # Simple line + dots with error bars (no fill band)
        ax.errorbar(
            x, means, yerr=sems,
            fmt="-o", color=color,
            ecolor=color, elinewidth=1.0,
            capsize=3, capthick=0.8,
            markersize=6, markeredgecolor=color,
            linewidth=1.5, zorder=3,
        )

        # Dotted zero / reference line at the grand mean
        ax.axhline(means.mean(), color="grey", linewidth=0.8,
                    linestyle="dotted", zorder=1)

        # ANOVA
        valid = [g for g in groups if len(g) >= 2]
        p_val = 1.0
        if len(valid) >= 2:
            try:
                _, p_val = f_oneway(*valid)
            except Exception:
                pass
        stars = _stars(p_val)
        if stars:
            ax.set_title(f"{taxon}  {stars}", fontsize=9, fontweight="bold")
        else:
            ax.set_title(taxon, fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in cluster_ids], fontsize=8)
        ax.set_xlim(0.5, n_clusters + 0.5)
        ax.tick_params(axis="both", labelsize=7)
        _despine(ax)

        if idx % ncols == 0:
            ax.set_ylabel("position", fontsize=8)

    for idx in range(len(taxa_order), nrows * ncols):
        axes_flat[idx].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, axes


# ------------------------------------------------------------------
# Combined ref vs most-polluted taxa trend grid
# ------------------------------------------------------------------

def plot_taxa_trend_comparison(
    taxa_relabd_ref: pd.DataFrame,
    cluster_labels_ref: pd.Series,
    taxa_relabd_polluted: pd.DataFrame,
    cluster_labels_polluted: pd.Series,
    *,
    taxa_order: Sequence[str] | None = None,
    title: str = "",
    label_ref: str = "Least Polluted",
    label_polluted: str = "Most Polluted",
    figsize: Tuple[float, float] = (16, 14),
    dpi: int = 300,
) -> Tuple[plt.Figure, np.ndarray]:
    """4×4 grid of grouped bar charts comparing ref vs most-polluted sites.

    Each subplot shows one taxon with two groups of bars (left = ref,
    right = most polluted), each group containing one bar per cluster
    coloured by ``CLUSTER_COLORS``.  Bars carry upward-only SE error bars.
    A Welch t-test (ref vs polluted, pooled across clusters) determines
    the significance stars shown in the subplot title.
    """
    if taxa_order is None:
        taxa_order = TAXA_DISPLAY_ORDER
    taxa_order = [t for t in taxa_order
                  if t in taxa_relabd_ref.columns
                  and t in taxa_relabd_polluted.columns]

    # Align
    common_ref = cluster_labels_ref.index.intersection(taxa_relabd_ref.index)
    taxa_relabd_ref = taxa_relabd_ref.loc[common_ref]
    cluster_labels_ref = cluster_labels_ref.loc[common_ref]

    common_pol = cluster_labels_polluted.index.intersection(taxa_relabd_polluted.index)
    taxa_relabd_polluted = taxa_relabd_polluted.loc[common_pol]
    cluster_labels_polluted = cluster_labels_polluted.loc[common_pol]

    cluster_ids = sorted(
        set(cluster_labels_ref.unique()) | set(cluster_labels_polluted.unique())
    )
    n_clusters = len(cluster_ids)
    colors = CLUSTER_COLORS[:n_clusters]

    nrows, ncols = 4, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    axes_flat = axes.flatten()

    bar_width = 0.7
    gap = 1.5  # gap between the two groups

    for idx, taxon in enumerate(taxa_order):
        ax = axes_flat[idx]

        ref_means, ref_sems = [], []
        pol_means, pol_sems = [], []
        ref_vals_all, pol_vals_all = [], []
        for cid in cluster_ids:
            r = taxa_relabd_ref.loc[cluster_labels_ref == cid, taxon].dropna()
            p = taxa_relabd_polluted.loc[cluster_labels_polluted == cid, taxon].dropna()
            ref_means.append(r.mean() if len(r) else 0.0)
            ref_sems.append(r.sem() if len(r) > 1 else 0.0)
            pol_means.append(p.mean() if len(p) else 0.0)
            pol_sems.append(p.sem() if len(p) > 1 else 0.0)
            ref_vals_all.extend(r.values)
            pol_vals_all.extend(p.values)

        ref_means = np.array(ref_means)
        ref_sems = np.array(ref_sems)
        pol_means = np.array(pol_means)
        pol_sems = np.array(pol_sems)

        # Bar positions: left group (ref), gap, right group (polluted)
        x_ref = np.arange(n_clusters) * bar_width
        x_pol = x_ref[-1] + gap + np.arange(n_clusters) * bar_width

        for i, cid in enumerate(cluster_ids):
            label_c = f"Cluster {cid}" if idx == 0 else None
            # Reference bar
            ax.bar(x_ref[i], ref_means[i], bar_width * 0.85,
                   yerr=[[0], [ref_sems[i]]],
                   color=colors[i], edgecolor="black", linewidth=0.4,
                   alpha=0.85, label=label_c,
                   error_kw=dict(linewidth=1.0, ecolor="black", capsize=2))
            # Polluted bar
            ax.bar(x_pol[i], pol_means[i], bar_width * 0.85,
                   yerr=[[0], [pol_sems[i]]],
                   color=colors[i], edgecolor="black", linewidth=0.4,
                   alpha=0.85,
                   error_kw=dict(linewidth=1.0, ecolor="black", capsize=2))

        # Group labels
        ref_center = x_ref.mean()
        pol_center = x_pol.mean()
        ax.set_xticks([ref_center, pol_center])
        ax.set_xticklabels([label_ref, label_polluted], fontsize=7)
        ax.set_xlim(x_ref[0] - bar_width, x_pol[-1] + bar_width)

        # Pairwise t-test: ref vs polluted
        ref_arr = np.array(ref_vals_all)
        pol_arr = np.array(pol_vals_all)
        p_val = 1.0
        if len(ref_arr) >= 2 and len(pol_arr) >= 2:
            try:
                _, p_val = ttest_ind(ref_arr, pol_arr, equal_var=False)
            except Exception:
                pass
        stars = _stars(p_val)
        if stars:
            ax.set_title(f"{taxon}  {stars}", fontsize=9, fontweight="bold")
        else:
            ax.set_title(taxon, fontsize=9, fontweight="bold")

        ax.tick_params(axis="both", labelsize=7)
        _despine(ax)

        if idx % ncols == 0:
            ax.set_ylabel("Mean Rel. Abundance", fontsize=8)

        if idx == 0:
            ax.legend(fontsize=7, loc="best", framealpha=0.8)

    for idx in range(len(taxa_order), nrows * ncols):
        axes_flat[idx].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, axes


# ------------------------------------------------------------------
# Environmental feature: ref vs non-ref comparison grid
# ------------------------------------------------------------------

def plot_env_trend_comparison(
    env_ref: pd.DataFrame,
    cluster_labels_ref: pd.Series,
    env_nonref: pd.DataFrame,
    cluster_labels_nonref: pd.Series,
    *,
    env_variables: Sequence[str] | None = None,
    title: str = "",
    label_ref: str = "Least Polluted",
    label_nonref: str = "Most Polluted",
    figsize: Tuple[float, float] | None = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, np.ndarray]:
    """Grid of grouped bar charts comparing two site groups per env variable.

    Each subplot shows one environmental variable with two groups of bars
    (left = reference, right = non-ref / most-polluted), each group
    containing one bar per cluster coloured by ``CLUSTER_COLORS``.
    Bars carry upward-only SE error bars.  A Welch t-test (ref vs
    non-ref, pooled across clusters) determines the significance stars.
    """
    if env_variables is None:
        env_variables = list(env_ref.columns)

    # Align indices
    common_ref = cluster_labels_ref.index.intersection(env_ref.index)
    env_ref = env_ref.loc[common_ref]
    cluster_labels_ref = cluster_labels_ref.loc[common_ref]

    common_nr = cluster_labels_nonref.index.intersection(env_nonref.index)
    env_nonref = env_nonref.loc[common_nr]
    cluster_labels_nonref = cluster_labels_nonref.loc[common_nr]

    cluster_ids = sorted(
        set(cluster_labels_ref.unique()) | set(cluster_labels_nonref.unique())
    )
    n_clusters = len(cluster_ids)
    colors = CLUSTER_COLORS[:n_clusters]
    n_vars = len(env_variables)

    ncols = min(n_vars, 3)
    nrows = math.ceil(n_vars / ncols)
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, squeeze=False)
    axes_flat = axes.flatten()

    bar_width = 0.7
    gap = 1.5

    for idx, var in enumerate(env_variables):
        ax = axes_flat[idx]

        ref_means, ref_sems = [], []
        nonref_means, nonref_sems = [], []
        ref_vals_all, nonref_vals_all = [], []
        for cid in cluster_ids:
            r = env_ref.loc[cluster_labels_ref == cid, var].dropna()
            n = env_nonref.loc[cluster_labels_nonref == cid, var].dropna()
            ref_means.append(r.mean() if len(r) else 0.0)
            ref_sems.append(r.sem() if len(r) > 1 else 0.0)
            nonref_means.append(n.mean() if len(n) else 0.0)
            nonref_sems.append(n.sem() if len(n) > 1 else 0.0)
            ref_vals_all.extend(r.values)
            nonref_vals_all.extend(n.values)

        ref_means = np.array(ref_means)
        ref_sems = np.array(ref_sems)
        nonref_means = np.array(nonref_means)
        nonref_sems = np.array(nonref_sems)

        # Bar positions
        x_ref = np.arange(n_clusters) * bar_width
        x_nr = x_ref[-1] + gap + np.arange(n_clusters) * bar_width

        for i, cid in enumerate(cluster_ids):
            label_c = f"Cluster {cid}" if idx == 0 else None
            ax.bar(x_ref[i], ref_means[i], bar_width * 0.85,
                   yerr=[[0], [ref_sems[i]]],
                   color=colors[i], edgecolor="black", linewidth=0.4,
                   alpha=0.85, label=label_c,
                   error_kw=dict(linewidth=1.0, ecolor="black", capsize=2))
            ax.bar(x_nr[i], nonref_means[i], bar_width * 0.85,
                   yerr=[[0], [nonref_sems[i]]],
                   color=colors[i], edgecolor="black", linewidth=0.4,
                   alpha=0.85,
                   error_kw=dict(linewidth=1.0, ecolor="black", capsize=2))

        # Group labels
        ref_center = x_ref.mean()
        nr_center = x_nr.mean()
        ax.set_xticks([ref_center, nr_center])
        ax.set_xticklabels([label_ref, label_nonref], fontsize=9)
        ax.set_xlim(x_ref[0] - bar_width, x_nr[-1] + bar_width)

        # Welch t-test
        ref_arr = np.array(ref_vals_all)
        nonref_arr = np.array(nonref_vals_all)
        p_val = 1.0
        if len(ref_arr) >= 2 and len(nonref_arr) >= 2:
            try:
                _, p_val = ttest_ind(ref_arr, nonref_arr, equal_var=False)
            except Exception:
                pass
        stars = _stars(p_val)
        label = var
        if stars:
            label = f"{var}  {stars}"
        ax.set_title(label, fontsize=10, fontweight="bold")

        ax.tick_params(axis="both", labelsize=8)
        _despine(ax)

        if idx % ncols == 0:
            ax.set_ylabel("Mean Value", fontsize=9)

        if idx == 0:
            ax.legend(fontsize=8, loc="best", framealpha=0.8)

    for idx in range(n_vars, nrows * ncols):
        axes_flat[idx].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, axes
