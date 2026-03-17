"""RDA triplot visualisation — sites, species labels, environmental arrows.

Public API
----------
plot_rda_triplot
    Publication-ready RDA biplot / triplot with optional significance
    annotations on environmental arrows.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.rda import RDA
from ..models.rda import RDAScores


# ------------------------------------------------------------------
# Waterbody visual styles
# ------------------------------------------------------------------

_WATERBODY_STYLE: Dict[str, Dict] = {
    "DR":  {"color": "#1f77b4", "marker": "^", "size": 80,  "label": "Detroit River"},
    "SCR": {"color": "#ff7f0e", "marker": "o", "size": 60,  "label": "St. Clair River"},
    "LSC": {"color": "#2ca02c", "marker": "s", "size": 50,  "label": "Lake St. Clair"},
}

# Fallback group colours for generic site_groups
_GROUP_COLORS: list[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#DC143C", "#9370DB",
]


# ─── public API ─────────────────────────────────────────────────────


def plot_rda_triplot(
    rda_model: RDA,
    *,
    axes: Tuple[int, int] = (1, 2),
    scaling: int = 1,
    show_sites: bool = True,
    show_species: bool = True,
    show_env: bool = True,
    site_groups: Optional[pd.Series] = None,
    terms_test: Optional[pd.DataFrame] = None,
    global_test: Optional[object] = None,
    arrow_scale: float = 1.0,
    species_scale: float = 1.5,
    figsize: Tuple[float, float] = (10, 9),
    dpi: int = 300,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create an RDA triplot with optional performance metrics.

    Parameters
    ----------
    rda_model : RDA
        Already-fitted :class:`~zci.core.rda.RDA` instance.
    axes : tuple of int
        Which RDA axes to plot (1-indexed, e.g. ``(1, 2)``).
    scaling : {1, 2}
        Biplot scaling (1 = distance, 2 = correlation).
    show_sites, show_species, show_env : bool
        Toggle layers.
    site_groups : pd.Series, optional
        Categorical grouping for colouring sites.  If values are
        ``"DR"`` / ``"SCR"`` / ``"LSC"`` the waterbody style is used
        automatically; otherwise generic cluster colours are used.
    terms_test : pd.DataFrame, optional
        Output of ``rda_model.test_terms()`` — used for p-value labels
        on environmental arrows.
    global_test : PermutationTestResult, optional
        Output of ``rda_model.test_global()`` — if provided, global
        pseudo-F and p are shown in the metrics annotation.
    arrow_scale : float
        Multiplier for environmental arrow length.
    species_scale : float
        Multiplier for species-label positions.
    figsize, dpi : tuple, int
        Figure size and resolution.
    title : str, optional
        Optional supertitle.

    Returns
    -------
    fig, ax
    """
    fit = rda_model.fit_
    if fit is None:
        raise RuntimeError("RDA has not been fit.")

    # 0-indexed axis ids
    i0, i1 = axes[0] - 1, axes[1] - 1
    axis_names = [
        fit.constrained_eigenvalues.index[i0],
        fit.constrained_eigenvalues.index[i1],
    ]

    # ── scores (scaling 1) ──────────────────────────────────────────
    scores = rda_model.scores(n_axes=max(axes))
    site_sc = scores.site_scores.iloc[:, [i0, i1]].copy()
    spec_sc = scores.species_scores.iloc[:, [i0, i1]].copy()
    bp_sc = scores.biplot_scores.iloc[:, [i0, i1]].copy()

    # ── apply scaling 2 if requested ────────────────────────────────
    if scaling == 2:
        lam = fit.constrained_eigenvalues.iloc[[i0, i1]].to_numpy()
        sqrt_lam = np.sqrt(np.maximum(lam, 0.0))
        site_sc = site_sc * sqrt_lam
        spec_sc = spec_sc / np.where(sqrt_lam > 0, sqrt_lam, 1.0)
    elif scaling != 1:
        raise ValueError(f"scaling must be 1 or 2, got {scaling}")

    expl_pct = fit.explained_proportion.iloc[[i0, i1]] * 100

    # ── p-value lookup from terms_test ──────────────────────────────
    pval_map: Dict[str, float] = {}
    if terms_test is not None:
        for _, row in terms_test.iterrows():
            pval_map[row["term"]] = float(row["p"])

    # ── figure ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # sites
    if show_sites:
        _plot_sites(ax, site_sc, site_groups)

    # ── set axis limits FIRST so species labels can be clipped ──────
    # Pad based on site range
    site_xmin, site_xmax = site_sc.iloc[:, 0].min(), site_sc.iloc[:, 0].max()
    site_ymin, site_ymax = site_sc.iloc[:, 1].min(), site_sc.iloc[:, 1].max()
    xpad = (site_xmax - site_xmin) * 0.15
    ypad = (site_ymax - site_ymin) * 0.15
    ax.set_xlim(site_xmin - xpad, site_xmax + xpad)
    ax.set_ylim(site_ymin - ypad, site_ymax + ypad)

    # species labels (clipped inside axes)
    if show_species:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for sp in spec_sc.index:
            x, y = spec_sc.loc[sp] * species_scale
            # make the species labels more spread out if they are near the origin 
            if -1 < x < 1:
                x, y = spec_sc.loc[sp] * 2.5
            else:
                x, y = spec_sc.loc[sp] * 1.3
            # Clip to axes bounds
            x = np.clip(x, xlim[0] + 0.05, xlim[1] - 0.05)
            y = np.clip(y, ylim[0] + 0.05, ylim[1] - 0.05)
            ax.text(
                x, y, sp, fontsize=9, color="red", alpha=0.8,
                ha="center", va="center", zorder=2,
                clip_on=True,
            )

    # environmental arrows
    if show_env:
        for var in bp_sc.index:
            x, y = bp_sc.loc[var] * arrow_scale * 2
            p = pval_map.get(var, None)
            _is_sig = p is not None and p <= 0.05

            ls = "solid" if _is_sig or p is None else "dashed"
            alpha_a = 0.7 if _is_sig or p is None else 0.5
            alpha_t = 1.0 if _is_sig or p is None else 0.5

            ax.arrow(
                0, 0, x, y,
                head_width=0.08, head_length=0.08,
                fc="blue", ec="blue",
                alpha=alpha_a, linewidth=2, linestyle=ls, zorder=4,
            )
            label = var
            if p is not None:
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                label += f" ($p = {p:.3f}$)" if not sig else f" ($p = {p:.3f}$){sig}"
            ax.text(
                x * 1.05, y * 1.1, label,
                fontsize=11, color="blue", fontweight="bold",
                ha="center", va="center", alpha=alpha_t, zorder=5,
                clip_on=True,
            )

    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5, zorder=1)
    ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.5, zorder=1)
    ax.set_xlabel(f"{axis_names[0]} ({expl_pct.iloc[0]:.1f}%)",
                  fontsize=15, fontweight="bold")
    ax.set_ylabel(f"{axis_names[1]} ({expl_pct.iloc[1]:.1f}%)",
                  fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3, zorder=0)

    if show_sites:
        ax.legend(loc="upper left", fontsize=12, framealpha=0.8)

    if title is not None:
        ax.set_title(title, fontsize=14, fontweight="bold")

    # ── RDA performance metrics — plain bold text, no box ───────────
    lines = []
    lines.append(f"$R^2 = {fit.r2:.4f}$")
    lines.append(f"adj-$R^2 = {fit.r2_adj:.4f}$")
    lines.append(f"Constrained inertia = {fit.inertia_constrained:.2f}")
    lines.append(f"Total inertia = {fit.inertia_total:.2f}")
    if global_test is not None:
        lines.append(f"Global pseudo-$F = {global_test.statistic:.2f}$")
        lines.append(f"Global $p = {global_test.p_value:.4f}$")
    metrics_text = "\n".join(lines)
    ax.text(
        0.98, 0.02, metrics_text,
        transform=ax.transAxes,
        fontsize=12, fontweight="bold",
        va="bottom", ha="right",
        zorder=10,
    )

    fig.tight_layout()
    return fig, ax


# ─── internal helpers ───────────────────────────────────────────────


def _plot_sites(
    ax: plt.Axes,
    site_sc: pd.DataFrame,
    site_groups: Optional[pd.Series],
) -> None:
    """Scatter sites, coloured/sized by waterbody or generic groups."""
    if site_groups is not None:
        aligned = site_groups.reindex(site_sc.index)
        groups = sorted(aligned.dropna().unique())

        # Detect whether groups are waterbody codes
        is_waterbody = all(g in _WATERBODY_STYLE for g in groups)

        for i, g in enumerate(groups):
            mask = aligned == g
            sub = site_sc.loc[mask]

            if is_waterbody:
                style = _WATERBODY_STYLE[g]
                ax.scatter(
                    sub.iloc[:, 0], sub.iloc[:, 1],
                    c=style["color"], marker=style["marker"],
                    s=style["size"], edgecolors="gray", linewidth=0.3,
                    label=style["label"], zorder=3, alpha=0.8,
                )
            else:
                c = _GROUP_COLORS[i % len(_GROUP_COLORS)]
                ax.scatter(
                    sub.iloc[:, 0], sub.iloc[:, 1],
                    c=c, s=70, edgecolors="gray", linewidth=0.5,
                    label=f"Cluster {g}", zorder=3,
                )

            for idx in sub.index:
                ax.text(
                    sub.loc[idx].iloc[0] + 0.02,
                    sub.loc[idx].iloc[1] + 0.02,
                    str(idx), fontsize=7, color="black", alpha=0.7,
                    ha="left", va="bottom", zorder=4,
                )
    else:
        ax.scatter(
            site_sc.iloc[:, 0], site_sc.iloc[:, 1],
            c="black", s=60, alpha=0.6,
            edgecolors="gray", linewidth=0.5,
            label="Sites", zorder=3,
        )
        for idx in site_sc.index:
            ax.text(
                site_sc.loc[idx].iloc[0] + 0.02,
                site_sc.loc[idx].iloc[1] + 0.02,
                str(idx), fontsize=7, color="black", alpha=0.7,
                ha="left", va="bottom", zorder=4,
            )
