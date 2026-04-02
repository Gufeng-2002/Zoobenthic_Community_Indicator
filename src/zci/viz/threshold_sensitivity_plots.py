"""Cut-off sensitivity visualisations.

Public API
----------
plot_cutoff_r2
    Single-panel: Adjusted R² vs cut-off proportion.
plot_cutoff_pseudoF
    Single-panel: Global Pseudo-F (permutation test) vs cut-off proportion.
plot_cutoff_r2_comparison
    Overlay multiple scores on adjusted R² panel.
plot_cutoff_pseudoF_comparison
    Overlay multiple scores on Pseudo-F panel.
plot_mrt_cutoff_sensitivity
    CVRE with error bars vs cut-off, with top axis for tree size.
plot_mrt_cutoff_comparison
    Overlay multiple scores on MRT CVRE panels.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# ─── helpers ─────────────────────────────────────────────────────────

_TRANSFORM_LABELS = {
    "octave": "Octave",
    "relative_abundance": "Relative Abundance",
    "chord": "Chord",
    "hellinger": "Hellinger",
    "log_chord": "Log-Chord",
}


def _transform_label(taxa_transform: str | None) -> str:
    """Human-readable label for a taxa transform key."""
    if taxa_transform is None:
        return ""
    return _TRANSFORM_LABELS.get(taxa_transform, taxa_transform)


def _find_p_crossing(df: pd.DataFrame, p_target: float = 0.05):
    """Find threshold and F where global *p* first drops below *p_target*.

    Returns ``(threshold, F_statistic)`` or ``None``.
    """
    df = df.sort_values("threshold").reset_index(drop=True)
    above = df["global_p"] > p_target
    below = df["global_p"] <= p_target
    if above.all() or below.all():
        return None
    for i in range(1, len(df)):
        if above.iloc[i - 1] and below.iloc[i]:
            t0, t1 = df["threshold"].iloc[i - 1], df["threshold"].iloc[i]
            p0, p1 = df["global_p"].iloc[i - 1], df["global_p"].iloc[i]
            f0, f1 = df["global_F"].iloc[i - 1], df["global_F"].iloc[i]
            frac = (p_target - p0) / (p1 - p0) if p1 != p0 else 0.5
            t_cross = t0 + frac * (t1 - t0)
            f_cross = f0 + frac * (f1 - f0)
            return t_cross, f_cross
    return None


# ─── public API — RDA single-panel plots ────────────────────────────


def plot_cutoff_r2(
    metrics: pd.DataFrame,
    *,
    shade_range: Tuple[float, float] | None = None,
    score_label: str = "",
    taxa_transform: str | None = None,
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Single-panel: Adjusted R² vs cut-off proportion.

    Parameters
    ----------
    metrics : pd.DataFrame
        Output of ``sweep_thresholds()``.
    shade_range : (lo, hi), optional
        Shade a recommended cut-off region.
    score_label : str
        Label for the contamination score.

    Returns
    -------
    fig, ax
    """
    df = metrics.sort_values("threshold").copy()
    thresholds = df["threshold"].values

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(thresholds, df["r2_adj"], "o-", color="#1f77b4", lw=2, ms=6)
    ax.set_ylabel("Adjusted $R^2$", fontsize=12)
    ax.set_xlabel("Cut-off Proportion", fontsize=12)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.3)

    if shade_range is not None:
        ax.axvspan(
            shade_range[0], shade_range[1],
            color="gold", alpha=0.25, zorder=0,
            label=f"Recommended: {shade_range[0]:.0%}\u2013{shade_range[1]:.0%}",
        )
        ax.legend(fontsize=10)

    title = "Adjusted $R^2$ vs Cut-off (RDA)"
    if score_label:
        title += f" \u2014 {score_label}"
    if taxa_transform:
        title += f"\nTaxa transform: {_transform_label(taxa_transform)}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig, ax


def plot_cutoff_pseudoF(
    metrics: pd.DataFrame,
    *,
    shade_range: Tuple[float, float] | None = None,
    score_label: str = "",
    taxa_transform: str | None = None,
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Single-panel: Global Pseudo-F (permutation test) vs cut-off proportion.

    Parameters
    ----------
    metrics : pd.DataFrame
        Output of ``sweep_thresholds()``.
    shade_range : (lo, hi), optional
        Shade a recommended cut-off region.
    score_label : str
        Label for the contamination score.

    Returns
    -------
    fig, ax
    """
    df = metrics.sort_values("threshold").copy()
    thresholds = df["threshold"].values

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(thresholds, df["global_F"], "s-", color="#ff7f0e", lw=2, ms=6)
    ax.set_ylabel("Global Pseudo-$F$", fontsize=12)
    ax.set_xlabel("Cut-off Proportion", fontsize=12)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.3)

    # F-critical line (threshold where p crosses 0.05)
    crossing = _find_p_crossing(df, 0.05)
    if crossing is not None:
        t_cross, f_cross = crossing
        ax.axhline(
            f_cross, ls=":", color="#d62728", lw=1.5, alpha=0.7,
            label=f"$p$=0.05 critical $F$={f_cross:.2f} (at {t_cross:.0%})",
        )
        ax.axvline(t_cross, ls=":", color="#d62728", lw=1.5, alpha=0.7)

    if shade_range is not None:
        ax.axvspan(
            shade_range[0], shade_range[1],
            color="gold", alpha=0.25, zorder=0,
            label=f"Recommended: {shade_range[0]:.0%}\u2013{shade_range[1]:.0%}",
        )

    ax.legend(fontsize=10)

    title = "Global Pseudo-$F$ vs Cut-off (Permutation Test)"
    if score_label:
        title += f" \u2014 {score_label}"
    if taxa_transform:
        title += f"\nTaxa transform: {_transform_label(taxa_transform)}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig, ax


def plot_cutoff_r2_comparison(
    metrics_dict: dict[str, pd.DataFrame],
    *,
    shade_range: Tuple[float, float] | None = None,
    taxa_transform: str | None = None,
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay multiple scores on Adjusted R² panel.

    Parameters
    ----------
    metrics_dict : dict[str, pd.DataFrame]
        ``{score_label: metrics_df}``.
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for i, (label, df) in enumerate(metrics_dict.items()):
        df = df.sort_values("threshold")
        t = df["threshold"].values
        c = colors[i % len(colors)]
        ax.plot(t, df["r2_adj"], "o-", color=c, lw=2, ms=5, label=label)

    ax.set_ylabel("Adjusted $R^2$", fontsize=12)
    ax.set_xlabel("Cut-off Proportion", fontsize=12)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.3)

    if shade_range is not None:
        ax.axvspan(
            shade_range[0], shade_range[1],
            color="gold", alpha=0.25, zorder=0,
            label=f"Recommended: {shade_range[0]:.0%}\u2013{shade_range[1]:.0%}",
        )

    ax.legend(fontsize=10)
    title = "Adjusted $R^2$ vs Cut-off (RDA) \u2014 Score Comparison"
    if taxa_transform:
        title += f"\nTaxa transform: {_transform_label(taxa_transform)}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig, ax


def plot_cutoff_pseudoF_comparison(
    metrics_dict: dict[str, pd.DataFrame],
    *,
    shade_range: Tuple[float, float] | None = None,
    taxa_transform: str | None = None,
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay multiple scores on Global Pseudo-F panel.

    Parameters
    ----------
    metrics_dict : dict[str, pd.DataFrame]
        ``{score_label: metrics_df}``.
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for i, (label, df) in enumerate(metrics_dict.items()):
        df = df.sort_values("threshold")
        t = df["threshold"].values
        c = colors[i % len(colors)]
        ax.plot(t, df["global_F"], "s-", color=c, lw=2, ms=5, label=label)

    ax.set_ylabel("Global Pseudo-$F$", fontsize=12)
    ax.set_xlabel("Cut-off Proportion", fontsize=12)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.3)

    # F-critical lines per score
    for i, (label, df) in enumerate(metrics_dict.items()):
        df = df.sort_values("threshold")
        crossing = _find_p_crossing(df, 0.05)
        if crossing is not None:
            t_cross, f_cross = crossing
            c = colors[i % len(colors)]
            ax.axhline(
                f_cross, ls=":", color=c, lw=1.5, alpha=0.7,
                label=f"{label} $p$=0.05: $F$={f_cross:.2f} (at {t_cross:.0%})",
            )
            ax.axvline(t_cross, ls=":", color=c, lw=1.5, alpha=0.7)

    if shade_range is not None:
        ax.axvspan(
            shade_range[0], shade_range[1],
            color="gold", alpha=0.25, zorder=0,
            label=f"Recommended: {shade_range[0]:.0%}\u2013{shade_range[1]:.0%}",
        )

    ax.legend(fontsize=10)
    title = "Global Pseudo-$F$ vs Cut-off (Permutation Test) \u2014 Score Comparison"
    if taxa_transform:
        title += f"\nTaxa transform: {_transform_label(taxa_transform)}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig, ax


# ─── backward-compatible aliases ─────────────────────────────────────

plot_threshold_sensitivity = plot_cutoff_r2
plot_threshold_comparison = plot_cutoff_r2_comparison
plot_cutoff_sensitivity = plot_cutoff_r2
plot_cutoff_comparison = plot_cutoff_r2_comparison


# ─── MRT cut-off sensitivity plots ──────────────────────────────────


def plot_mrt_cutoff_sensitivity(
    mrt_metrics: pd.DataFrame,
    *,
    shade_range: Tuple[float, float] | None = None,
    score_label: str = "",
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """CVRE with error bars vs cut-off, with a top axis showing best tree size.

    Parameters
    ----------
    mrt_metrics : pd.DataFrame
        Output of ``sweep_cutoffs_mrt()`` with columns:
        cutoff, n_sites, min_cvre, cvre_se, best_tree_size.
    shade_range : (lo, hi), optional
        Shade a recommended cut-off region.
    score_label : str
        Label for the contamination score.
    """
    df = mrt_metrics.sort_values("cutoff").copy()
    df = df.dropna(subset=["min_cvre"])
    cutoffs = df["cutoff"].values
    cvre = df["min_cvre"].values
    se = df["cvre_se"].values
    tree_sizes = df["best_tree_size"].values

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # CVRE with error bars
    ax.errorbar(
        cutoffs, cvre, yerr=se,
        fmt="o-", color="#1f77b4", lw=2, ms=6,
        capsize=4, capthick=1.5, ecolor="#5B9BD5",
        label="Min CVRE ± SE",
    )

    ax.set_ylabel("Cross-Validated Relative Error (CVRE)", fontsize=12)
    ax.set_xlabel("Cut-off Proportion", fontsize=12)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.3)

    if shade_range is not None:
        ax.axvspan(
            shade_range[0], shade_range[1],
            color="gold", alpha=0.25, zorder=0,
            label=f"Recommended: {shade_range[0]:.0%}\u2013{shade_range[1]:.0%}",
        )

    ax.legend(fontsize=10, loc="upper right")

    # Top axis: best tree size
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(cutoffs)
    ax_top.set_xticklabels(
        [f"{int(s)}" if not np.isnan(s) else "" for s in tree_sizes],
        fontsize=8,
    )
    ax_top.set_xlabel("Best Tree Size (leaves)", fontsize=11)

    title = "Reference Cut-off Sensitivity Analysis (MRT)"
    if score_label:
        title += f" \u2014 {score_label}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig, ax


def plot_mrt_cutoff_comparison(
    mrt_metrics_dict: dict[str, pd.DataFrame],
    *,
    shade_range: Tuple[float, float] | None = None,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay CVRE error-bar curves for multiple scores.

    Parameters
    ----------
    mrt_metrics_dict : dict[str, pd.DataFrame]
        ``{score_label: mrt_metrics_df}``.
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for i, (label, df) in enumerate(mrt_metrics_dict.items()):
        df = df.sort_values("cutoff").dropna(subset=["min_cvre"]).copy()
        c = colors[i % len(colors)]
        ax.plot(
            df["cutoff"].values, df["min_cvre"].values,
            "o-", color=c, lw=2, ms=5, alpha=0.8,
            label=label,
        )

    ax.set_ylabel("Cross-Validated Relative Error (CVRE)", fontsize=12)
    ax.set_xlabel("Cut-off Proportion", fontsize=12)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.3)

    if shade_range is not None:
        ax.axvspan(
            shade_range[0], shade_range[1],
            color="gold", alpha=0.25, zorder=0,
            label=f"Recommended: {shade_range[0]:.0%}\u2013{shade_range[1]:.0%}",
        )

    ax.legend(fontsize=10)

    fig.suptitle(
        "Reference Cut-off Sensitivity (MRT) \u2014 Score Comparison",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig, ax
