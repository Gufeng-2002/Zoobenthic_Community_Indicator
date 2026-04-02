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
import matplotlib.ticker as mtick
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


def _build_title(base: str, score_label: str, taxa_transform: str | None) -> str:
    """Build a two- or three-line suptitle."""
    title = f"{base} — {score_label} (Env vs Stressors)"
    if taxa_transform:
        title += f"\nTaxa transform: {_transform_label(taxa_transform)}"
    return title


# ─── adj-R² comparison ──────────────────────────────────────────────


def plot_r2_env_vs_stressor(
    env_metrics: pd.DataFrame,
    stressor_metrics: pd.DataFrame,
    *,
    score_label: str = "SumRel",
    shade_range: Tuple[float, float] | None = None,
    taxa_transform: str | None = None,
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay Env-Taxa and Stressors-Taxa adj-R²."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    env_df = env_metrics.sort_values("threshold")
    str_df = stressor_metrics.sort_values("threshold")

    ax.plot(
        env_df["threshold"], env_df["r2_adj"],
        "o-", color="#1f77b4", lw=2, ms=5, label="Environmental",
    )
    ax.plot(
        str_df["threshold"], str_df["r2_adj"],
        "s-", color="#ff7f0e", lw=2, ms=5, label="Stressors (PCA)",
    )

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
    fig.suptitle(
        _build_title("Adjusted $R^2$ vs Cut-off", score_label, taxa_transform),
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig, ax


# ─── pseudo-F comparison ────────────────────────────────────────────


def plot_pseudoF_env_vs_stressor(
    env_metrics: pd.DataFrame,
    stressor_metrics: pd.DataFrame,
    *,
    score_label: str = "SumRel",
    shade_range: Tuple[float, float] | None = None,
    taxa_transform: str | None = None,
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay Env-Taxa and Stressors-Taxa pseudo-F."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    env_df = env_metrics.sort_values("threshold")
    str_df = stressor_metrics.sort_values("threshold")

    ax.plot(
        env_df["threshold"], env_df["global_F"],
        "o-", color="#1f77b4", lw=2, ms=5, label="Environmental",
    )
    ax.plot(
        str_df["threshold"], str_df["global_F"],
        "s-", color="#ff7f0e", lw=2, ms=5, label="Stressors (PCA)",
    )

    for label_str, df, c in [
        ("Env", env_df, "#1f77b4"),
        ("Stressor", str_df, "#ff7f0e"),
    ]:
        crossing = _find_p_crossing(df, 0.05)
        if crossing is not None:
            t_cross, f_cross = crossing
            ax.axhline(
                f_cross, ls=":", color=c, lw=1.5, alpha=0.7,
                label=f"{label_str} $p$=0.05: $F$={f_cross:.2f} (at {t_cross:.0%})",
            )

    ax.set_ylabel("Global Pseudo-$F$", fontsize=12)
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
        _build_title("Global Pseudo-$F$ vs Cut-off", score_label, taxa_transform),
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig, ax


# ─── global p-value comparison ──────────────────────────────────────


def plot_pvalue_env_vs_stressor(
    env_metrics: pd.DataFrame,
    stressor_metrics: pd.DataFrame,
    *,
    score_label: str = "SumRel",
    shade_range: Tuple[float, float] | None = None,
    taxa_transform: str | None = None,
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay Env-Taxa and Stressors-Taxa global permutation *p*-value."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    env_df = env_metrics.sort_values("threshold")
    str_df = stressor_metrics.sort_values("threshold")

    ax.plot(
        env_df["threshold"], env_df["global_p"],
        "o-", color="#1f77b4", lw=2, ms=5, label="Environmental",
    )
    ax.plot(
        str_df["threshold"], str_df["global_p"],
        "s-", color="#ff7f0e", lw=2, ms=5, label="Stressors (PCA)",
    )

    ax.axhline(
        0.05, ls="--", color="red", lw=1.5, alpha=0.7,
        label="$p$ = 0.05",
    )

    ax.set_ylabel("Global Permutation $p$-value", fontsize=12)
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
        _build_title("Global Permutation Test $p$-value vs Cut-off", score_label, taxa_transform),
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig, ax


# ─── VIF comparison ─────────────────────────────────────────────────


def plot_vif_env_vs_stressor(
    env_metrics: pd.DataFrame,
    stressor_metrics: pd.DataFrame,
    *,
    score_label: str = "SumRel",
    shade_range: Tuple[float, float] | None = None,
    taxa_transform: str | None = None,
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay maximum VIF across cut-offs for Env vs Stressor predictors."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    env_df = env_metrics.sort_values("threshold")
    str_df = stressor_metrics.sort_values("threshold")

    ax.plot(
        env_df["threshold"], env_df["max_vif"],
        "o-", color="#1f77b4", lw=2, ms=5, label="Environmental",
    )
    ax.plot(
        str_df["threshold"], str_df["max_vif"],
        "s-", color="#ff7f0e", lw=2, ms=5, label="Stressors (PCA)",
    )

    ax.axhline(
        10, ls="--", color="red", lw=1.5, alpha=0.7,
        label="VIF = 10 (problematic)",
    )
    ax.axhline(
        5, ls=":", color="orange", lw=1.5, alpha=0.5,
        label="VIF = 5 (moderate)",
    )

    ax.set_ylabel("Maximum VIF", fontsize=12)
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
        _build_title("Maximum VIF vs Cut-off", score_label, taxa_transform),
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig, ax


# ─── backward-compatible aliases ────────────────────────────────────
# Keep old names working for any downstream notebooks that imported them.

plot_sumrel_r2_env_vs_stressor = plot_r2_env_vs_stressor
plot_sumrel_pseudoF_env_vs_stressor = plot_pseudoF_env_vs_stressor
plot_sumrel_pvalue_env_vs_stressor = plot_pvalue_env_vs_stressor
plot_sumrel_vif_env_vs_stressor = plot_vif_env_vs_stressor
