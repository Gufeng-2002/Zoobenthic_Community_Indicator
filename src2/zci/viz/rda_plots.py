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
# Colour palette  (blue / orange / green — matches reference figures)
# ------------------------------------------------------------------

_GROUP_COLORS: list[str] = [
    "#1f77b4",   # Cluster 1  — blue
    "#ff7f0e",   # Cluster 2  — orange
    "#2ca02c",   # Cluster 3  — green
    "#DC143C",   # Cluster 4  — red  (spare)
    "#9370DB",   # Cluster 5  — purple
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
    arrow_scale: float = 1.0,
    species_scale: float = 2,
    figsize: Tuple[float, float] = (10, 9),
    dpi: int = 300,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create an RDA triplot.

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
        Categorical grouping for colouring sites (e.g. cluster labels).
    terms_test : pd.DataFrame, optional
        Output of ``rda_model.test_terms()`` — used for p-value labels
        on environmental arrows.  If ``None``, arrows are drawn without
        p annotations.
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

    # species labels
    if show_species:
        for sp in spec_sc.index:
            x, y = spec_sc.loc[sp] * species_scale
            ax.text(
                x, y, sp, fontsize=12, color="red", alpha=0.8,
                ha="center", va="center", zorder=2,
            )

    # environmental arrows
    if show_env:
        for var in bp_sc.index:
            x, y = bp_sc.loc[var] * arrow_scale
            p = pval_map.get(var, None)
            _is_sig = p is not None and p <= 0.05

            ls = "solid" if _is_sig or p is None else "dashed"
            alpha_a = 0.7 if _is_sig or p is None else 0.5
            alpha_t = 1.0 if _is_sig or p is None else 0.5

            ax.arrow(
                0, 0, x, y,
                head_width=0.1, head_length=0.1,
                fc="blue", ec="blue",
                alpha=alpha_a, linewidth=2, linestyle=ls, zorder=4,
            )
            label = var
            if p is not None:
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                label += f" ($p = {p:.3f}$)" if not sig else f" ($p = {p:.3f}$){sig}"
            ax.text(
                x * 1.05, y * 1.1, label,
                fontsize=13, color="blue", fontweight="bold",
                ha="center", va="center", alpha=alpha_t, zorder=5,
            )

    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5, zorder=1)
    ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.5, zorder=1)
    ax.set_xlabel(f"{axis_names[0]} ({expl_pct.iloc[0]:.1f}%)",
                  fontsize=15, fontweight="bold")
    ax.set_ylabel(f"{axis_names[1]} ({expl_pct.iloc[1]:.1f}%)",
                  fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3, zorder=0)

    if show_sites:
        ax.legend(loc="upper left", fontsize=14)

    if title is not None:
        ax.set_title(title, fontsize=14, fontweight="bold")

    fig.tight_layout()
    return fig, ax


# ─── internal helpers ───────────────────────────────────────────────


def _plot_sites(
    ax: plt.Axes,
    site_sc: pd.DataFrame,
    site_groups: Optional[pd.Series],
) -> None:
    """Scatter sites, optionally coloured by *site_groups*."""
    if site_groups is not None:
        aligned = site_groups.reindex(site_sc.index)
        groups = sorted(aligned.dropna().unique())
        cmap = {g: _GROUP_COLORS[i % len(_GROUP_COLORS)] for i, g in enumerate(groups)}
        
        

        for g in groups:
            mask = aligned == g
            sub = site_sc.loc[mask]
            ax.scatter(
                sub.iloc[:, 0], sub.iloc[:, 1],
                c=[cmap[g]], s=70,
                edgecolors="gray", linewidth=0.5,
                label=f"Cluster {g}", zorder=3,
            )
            for idx in sub.index:
                ax.text(
                    sub.loc[idx].iloc[0] - 0.01,
                    sub.loc[idx].iloc[1] + 0.01,
                    str(idx), fontsize=9, color="black", alpha=0.8,
                    ha="center", va="bottom", zorder=4,
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
                site_sc.loc[idx].iloc[0] - 0.01,
                site_sc.loc[idx].iloc[1] + 0.01,
                str(idx), fontsize=9, color="black", alpha=0.8,
                ha="center", va="bottom", zorder=4,
            )
