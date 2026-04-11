"""Score Focus visualisations — generic for any contamination score.

Comparison plots overlaying Env-predictor vs Stressor-predictor RDA
results under any contamination view (SumRel, MaxRel, etc.).

Public API
----------
plot_r2_env_vs_stressor
    Overlay adj-R² curves (Env vs Stressors).
plot_pseudoF_env_vs_stressor
    Overlay pseudo-F curves (Env vs Stressors).
plot_pvalue_env_vs_stressor
    Overlay global permutation p-value curves (Env vs Stressors).
plot_vif_env_vs_stressor
    Overlay maximum VIF curves (Env vs Stressors).
"""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─── helpers ────────────────────────────────────────────────────────

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
    """Return (threshold, F) where global_p first drops below *p_target*."""
    below = df[df["global_p"] <= p_target]
    if below.empty:
        return None
    first = below.iloc[0]
    return first["threshold"], first["global_F"]


def _taxa_suffix(taxa_transform: str | None) -> str:
    """Return '  (Taxa - Chord)' style suffix for y-axis labels."""
    if taxa_transform:
        return f"  (Taxa - {_transform_label(taxa_transform)})"
    return ""


def _add_score_top_axis(
    ax: plt.Axes,
    pollution_score: pd.Series,
    score_label: str,
) -> None:
    """Add a twin top x-axis mapping site counts to contamination scores."""
    sorted_score = pollution_score.sort_values()
    lo, hi = ax.get_xlim()
    n_sites = len(sorted_score)

    # Tick every 25 sites, starting at 25
    tick_ns = list(range(25, n_sites + 1, 25))
    tick_labels = []
    for n in tick_ns:
        if n < 1 or n > n_sites:
            tick_labels.append("")
        else:
            tick_labels.append(f"{sorted_score.iloc[n - 1]:.2f}")

    ax2 = ax.twiny()
    ax2.set_xlim(lo, hi)
    ax2.set_xticks(tick_ns)
    ax2.set_xticklabels(tick_labels, fontsize=8)
    ax2.set_xlabel(f"{score_label} Scores", fontsize=12)

    # Draw a right-pointing arrow at the right end of the top axis
    ax2.annotate(
        "", xy=(1.01, 1.0), xytext=(0.0, 1.0),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        annotation_clip=False,
    )


# ─── adj-R² comparison ──────────────────────────────────────────────


def plot_r2_env_vs_stressor(
    env_metrics: pd.DataFrame,
    stressor_metrics: pd.DataFrame,
    *,
    score_label: str = "SumRel",
    shade_range: Tuple[float, float] | None = None,
    n_total: int | None = None,
    taxa_transform: str | None = None,
    pollution_score: pd.Series | None = None,
    single_score_metrics: pd.DataFrame | None = None,
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay Env-Taxa and Stressors-Taxa adj-R²."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    env_df = env_metrics.sort_values("threshold")
    str_df = stressor_metrics.sort_values("threshold")

    ax.plot(
        env_df["n_sites"], env_df["r2_adj"],
        "o-", color="#1f77b4", lw=2, ms=5, label="Environmental",
    )
    ax.plot(
        str_df["n_sites"], str_df["r2_adj"],
        "s-", color="#ff7f0e", lw=2, ms=5, label="Stressors",
    )

    if single_score_metrics is not None:
        ss_df = single_score_metrics.sort_values("threshold")
        ax.plot(
            ss_df["n_sites"], ss_df["r2_adj"],
            "^-", color="#2ca02c", lw=2, ms=5, label=score_label,
        )

    ax.set_ylabel(f"Adjusted $R^2${_taxa_suffix(taxa_transform)}", fontsize=12)
    ax.set_xlabel("Number of Sites Passed to RDA Fitting", fontsize=12)
    ax.grid(True, alpha=0.3)

    if shade_range is not None and n_total is not None:
        lo_n = max(int(shade_range[0] * n_total), 1)
        hi_n = max(int(shade_range[1] * n_total), 1)
        ax.axvspan(
            lo_n, hi_n,
            color="gold", alpha=0.25, zorder=0,
            label=f"Cut-off for Ref-Sites ({lo_n}\u2013{hi_n})",
        )

    ax.legend(fontsize=10)
    fig.tight_layout()

    if pollution_score is not None:
        _add_score_top_axis(ax, pollution_score, score_label)

    return fig, ax


# ─── pseudo-F comparison ────────────────────────────────────────────


def plot_pseudoF_env_vs_stressor(
    env_metrics: pd.DataFrame,
    stressor_metrics: pd.DataFrame,
    *,
    score_label: str = "SumRel",
    shade_range: Tuple[float, float] | None = None,
    n_total: int | None = None,
    taxa_transform: str | None = None,
    pollution_score: pd.Series | None = None,
    single_score_metrics: pd.DataFrame | None = None,
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay Env-Taxa and Stressors-Taxa pseudo-F."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    env_df = env_metrics.sort_values("threshold")
    str_df = stressor_metrics.sort_values("threshold")

    ax.plot(
        env_df["n_sites"], env_df["global_F"],
        "o-", color="#1f77b4", lw=2, ms=5, label="Environmental",
    )
    ax.plot(
        str_df["n_sites"], str_df["global_F"],
        "s-", color="#ff7f0e", lw=2, ms=5, label="Stressors",
    )

    if single_score_metrics is not None:
        ss_df = single_score_metrics.sort_values("threshold")
        ax.plot(
            ss_df["n_sites"], ss_df["global_F"],
            "^-", color="#2ca02c", lw=2, ms=5, label=score_label,
        )

    for label_str, df, c in [
        ("Env", env_df, "#1f77b4"),
        ("Stressor", str_df, "#ff7f0e"),
    ]:
        crossing = _find_p_crossing(df, 0.05)
        if crossing is not None:
            t_cross, f_cross = crossing
            n_cross = int(df.loc[(df["threshold"] - t_cross).abs().idxmin(), "n_sites"])
            ax.axhline(
                f_cross, ls=":", color=c, lw=1.5, alpha=0.7,
                label=f"{label_str} $p$=0.05: $F$={f_cross:.2f} (at {n_cross} sites)",
            )

    ax.set_ylabel(f"Global Pseudo-$F${_taxa_suffix(taxa_transform)}", fontsize=12)
    ax.set_xlabel("Number of Sites Passed to RDA Fitting", fontsize=12)
    ax.grid(True, alpha=0.3)

    if shade_range is not None and n_total is not None:
        lo_n = max(int(shade_range[0] * n_total), 1)
        hi_n = max(int(shade_range[1] * n_total), 1)
        ax.axvspan(
            lo_n, hi_n,
            color="gold", alpha=0.25, zorder=0,
            label=f"Cut-off for Ref-Sites ({lo_n}\u2013{hi_n})",
        )

    ax.legend(fontsize=10)
    fig.tight_layout()

    if pollution_score is not None:
        _add_score_top_axis(ax, pollution_score, score_label)

    return fig, ax


# ─── global p-value comparison ──────────────────────────────────────


def plot_pvalue_env_vs_stressor(
    env_metrics: pd.DataFrame,
    stressor_metrics: pd.DataFrame,
    *,
    score_label: str = "SumRel",
    shade_range: Tuple[float, float] | None = None,
    n_total: int | None = None,
    taxa_transform: str | None = None,
    pollution_score: pd.Series | None = None,
    single_score_metrics: pd.DataFrame | None = None,
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay Env-Taxa and Stressors-Taxa global permutation *p*-value."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    env_df = env_metrics.sort_values("threshold")
    str_df = stressor_metrics.sort_values("threshold")

    ax.plot(
        env_df["n_sites"], env_df["global_p"],
        "o-", color="#1f77b4", lw=2, ms=5, label="Environmental",
    )
    ax.plot(
        str_df["n_sites"], str_df["global_p"],
        "s-", color="#ff7f0e", lw=2, ms=5, label="Stressors",
    )

    if single_score_metrics is not None:
        ss_df = single_score_metrics.sort_values("threshold")
        ax.plot(
            ss_df["n_sites"], ss_df["global_p"],
            "^-", color="#2ca02c", lw=2, ms=5, label=score_label,
        )

    ax.axhline(
        0.05, ls="--", color="red", lw=1.5, alpha=0.7,
        label="$p$ = 0.05",
    )

    ax.set_ylabel(f"Global Permutation $p$-value{_taxa_suffix(taxa_transform)}", fontsize=12)
    ax.set_xlabel("Number of Sites Passed to RDA Fitting", fontsize=12)
    ax.grid(True, alpha=0.3)

    if shade_range is not None and n_total is not None:
        lo_n = max(int(shade_range[0] * n_total), 1)
        hi_n = max(int(shade_range[1] * n_total), 1)
        ax.axvspan(
            lo_n, hi_n,
            color="gold", alpha=0.25, zorder=0,
            label=f"Cut-off for Ref-Sites ({lo_n}\u2013{hi_n})",
        )

    ax.legend(fontsize=10)
    fig.tight_layout()

    if pollution_score is not None:
        _add_score_top_axis(ax, pollution_score, score_label)

    return fig, ax


# ─── VIF comparison ─────────────────────────────────────────────────


def plot_vif_env_vs_stressor(
    env_metrics: pd.DataFrame,
    stressor_metrics: pd.DataFrame,
    *,
    score_label: str = "SumRel",
    shade_range: Tuple[float, float] | None = None,
    n_total: int | None = None,
    taxa_transform: str | None = None,
    pollution_score: pd.Series | None = None,
    single_score_metrics: pd.DataFrame | None = None,
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay maximum VIF across cut-offs for Env vs Stressor predictors."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    env_df = env_metrics.sort_values("threshold")
    str_df = stressor_metrics.sort_values("threshold")

    ax.plot(
        env_df["n_sites"], env_df["max_vif"],
        "o-", color="#1f77b4", lw=2, ms=5, label="Environmental",
    )
    ax.plot(
        str_df["n_sites"], str_df["max_vif"],
        "s-", color="#ff7f0e", lw=2, ms=5, label="Stressors",
    )

    if single_score_metrics is not None:
        ss_df = single_score_metrics.sort_values("threshold")
        ax.plot(
            ss_df["n_sites"], ss_df["max_vif"],
            "^-", color="#2ca02c", lw=2, ms=5, label=score_label,
        )

    ax.axhline(
        10, ls="--", color="red", lw=1.5, alpha=0.7,
        label="VIF = 10 (problematic)",
    )
    ax.axhline(
        5, ls=":", color="orange", lw=1.5, alpha=0.5,
        label="VIF = 5 (moderate)",
    )

    ax.set_ylabel(f"Maximum VIF{_taxa_suffix(taxa_transform)}", fontsize=12)
    ax.set_xlabel("Number of Sites Passed to RDA Fitting", fontsize=12)
    ax.grid(True, alpha=0.3)

    if shade_range is not None and n_total is not None:
        lo_n = max(int(shade_range[0] * n_total), 1)
        hi_n = max(int(shade_range[1] * n_total), 1)
        ax.axvspan(
            lo_n, hi_n,
            color="gold", alpha=0.25, zorder=0,
            label=f"Cut-off for Ref-Sites ({lo_n}\u2013{hi_n})",
        )

    ax.legend(fontsize=10)
    fig.tight_layout()

    if pollution_score is not None:
        _add_score_top_axis(ax, pollution_score, score_label)

    return fig, ax


# ─── backward-compatible aliases ────────────────────────────────────
# Keep old names working for any downstream notebooks that imported them.

plot_sumrel_r2_env_vs_stressor = plot_r2_env_vs_stressor
plot_sumrel_pseudoF_env_vs_stressor = plot_pseudoF_env_vs_stressor
plot_sumrel_pvalue_env_vs_stressor = plot_pvalue_env_vs_stressor
plot_sumrel_vif_env_vs_stressor = plot_vif_env_vs_stressor
