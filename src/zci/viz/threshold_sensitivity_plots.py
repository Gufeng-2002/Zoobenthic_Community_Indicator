"""Threshold-sensitivity visualisations.

Public API
----------
plot_threshold_sensitivity
    Multi-panel figure: RDA metrics and sample size vs threshold.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# ─── helper ──────────────────────────────────────────────────────────


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


# ─── public API ─────────────────────────────────────────────────────


def plot_threshold_sensitivity(
    metrics: pd.DataFrame,
    *,
    shade_range: Tuple[float, float] | None = None,
    score_label: str = "",
    figsize: Tuple[float, float] = (14, 5),
    dpi: int = 300,
) -> Tuple[plt.Figure, np.ndarray]:
    """Two-panel figure: adj-R² and pseudo-F vs threshold proportion.

    Parameters
    ----------
    metrics : pd.DataFrame
        Output of ``sweep_thresholds()``.
    shade_range : (lo, hi), optional
        Shade a recommended threshold region on both panels.
    score_label : str
        Label for the contamination score (e.g. ``"SumRel"``).
    figsize, dpi : misc
        Figure size and resolution.

    Returns
    -------
    fig, axes
    """
    df = metrics.sort_values("threshold").copy()
    thresholds = df["threshold"].values

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    ax_r2, ax_f = axes

    # ── Panel 1: adj-R² ─────────────────────────────────────────────
    ax_r2.plot(thresholds, df["r2_adj"], "o-", color="#1f77b4", lw=2, ms=6)
    ax_r2.set_ylabel("Adjusted $R^2$", fontsize=12)
    ax_r2.set_title("Adjusted $R^2$ vs Threshold", fontweight="bold")
    ax_r2.set_xlabel("Threshold Proportion", fontsize=12)
    ax_r2.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax_r2.grid(True, alpha=0.3)

    if shade_range is not None:
        ax_r2.axvspan(
            shade_range[0], shade_range[1],
            color="gold", alpha=0.25, zorder=0,
            label=f"Recommended: {shade_range[0]:.0%}\u2013{shade_range[1]:.0%}",
        )
        ax_r2.legend(fontsize=10)

    # ── Panel 2: Pseudo-F ───────────────────────────────────────────
    ax_f.plot(thresholds, df["global_F"], "s-", color="#ff7f0e", lw=2, ms=6)
    ax_f.set_ylabel("Global Pseudo-$F$", fontsize=12)
    ax_f.set_title("Global Pseudo-$F$ vs Threshold", fontweight="bold")
    ax_f.set_xlabel("Threshold Proportion", fontsize=12)
    ax_f.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax_f.grid(True, alpha=0.3)

    # F-critical line (threshold where p crosses 0.05)
    crossing = _find_p_crossing(df, 0.05)
    if crossing is not None:
        t_cross, f_cross = crossing
        ax_f.axhline(
            f_cross, ls=":", color="#d62728", lw=1.5, alpha=0.7,
            label=f"$p$=0.05 critical $F$={f_cross:.2f} (at {t_cross:.0%})",
        )
        ax_f.axvline(t_cross, ls=":", color="#d62728", lw=1.5, alpha=0.7)

    # Shade recommended range on F panel too
    if shade_range is not None:
        ax_f.axvspan(
            shade_range[0], shade_range[1],
            color="gold", alpha=0.25, zorder=0,
            label=f"Recommended: {shade_range[0]:.0%}\u2013{shade_range[1]:.0%}",
        )

    ax_f.legend(fontsize=10)

    # ── Supertitle ──────────────────────────────────────────────────
    title = "Reference-Threshold Sensitivity Analysis"
    if score_label:
        title += f" \u2014 {score_label}"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_threshold_comparison(
    metrics_dict: dict[str, pd.DataFrame],
    *,
    shade_range: Tuple[float, float] | None = None,
    figsize: Tuple[float, float] = (14, 5),
    dpi: int = 300,
) -> Tuple[plt.Figure, np.ndarray]:
    """Overlay multiple scores on two panels: adj-R² and pseudo-F.

    Parameters
    ----------
    metrics_dict : dict[str, pd.DataFrame]
        Mapping ``{score_label: metrics_df}``.
    shade_range : (lo, hi), optional
        Shade a recommended region on both panels.
    figsize, dpi : misc
        Figure aesthetics.

    Returns
    -------
    fig, axes
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    ax_r2, ax_f = axes

    for i, (label, df) in enumerate(metrics_dict.items()):
        df = df.sort_values("threshold")
        t = df["threshold"].values
        c = colors[i % len(colors)]

        ax_r2.plot(t, df["r2_adj"], "o-", color=c, lw=2, ms=5, label=label)
        ax_f.plot(t, df["global_F"], "s-", color=c, lw=2, ms=5, label=label)

    # ── adj-R² panel ────────────────────────────────────────────────
    ax_r2.set_ylabel("Adjusted $R^2$", fontsize=12)
    ax_r2.set_title("Adjusted $R^2$ vs Threshold", fontweight="bold")
    ax_r2.set_xlabel("Threshold Proportion", fontsize=12)
    ax_r2.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax_r2.grid(True, alpha=0.3)
    ax_r2.legend(fontsize=10)

    if shade_range is not None:
        ax_r2.axvspan(
            shade_range[0], shade_range[1],
            color="gold", alpha=0.25, zorder=0,
            label=f"Recommended: {shade_range[0]:.0%}\u2013{shade_range[1]:.0%}",
        )
        ax_r2.legend(fontsize=10)

    # ── Pseudo-F panel ──────────────────────────────────────────────
    ax_f.set_ylabel("Global Pseudo-$F$", fontsize=12)
    ax_f.set_title("Global Pseudo-$F$ vs Threshold", fontweight="bold")
    ax_f.set_xlabel("Threshold Proportion", fontsize=12)
    ax_f.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax_f.grid(True, alpha=0.3)

    # F-critical lines (per score, where p crosses 0.05)
    for i, (label, df) in enumerate(metrics_dict.items()):
        df = df.sort_values("threshold")
        crossing = _find_p_crossing(df, 0.05)
        if crossing is not None:
            t_cross, f_cross = crossing
            c = colors[i % len(colors)]
            ax_f.axhline(
                f_cross, ls=":", color=c, lw=1.5, alpha=0.7,
                label=f"{label} $p$=0.05: $F$={f_cross:.2f} (at {t_cross:.0%})",
            )
            ax_f.axvline(t_cross, ls=":", color=c, lw=1.5, alpha=0.7)

    # Shade recommended range on F panel too
    if shade_range is not None:
        ax_f.axvspan(
            shade_range[0], shade_range[1],
            color="gold", alpha=0.25, zorder=0,
            label=f"Recommended: {shade_range[0]:.0%}\u2013{shade_range[1]:.0%}",
        )

    ax_f.legend(fontsize=10)

    fig.suptitle(
        "Reference-Threshold Sensitivity \u2014 Score Comparison",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig, axes
