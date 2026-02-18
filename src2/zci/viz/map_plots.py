"""Corridor map + ECDF visualisation for pollution-score bifurcation.

Public API
----------
plot_corridor_bifurcation
    Two-panel figure: spatial map (left) + cumulative frequency (right).

plot_corridor_map (helper)
    Render water-body shapefiles on a matplotlib Axes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ---------------------------------------------------------------------------
# Waterbody marker / colour defaults
# ---------------------------------------------------------------------------

_WATERBODY_STYLE: Dict[str, Dict] = {
    "DR":  {"marker": "^", "label": "Detroit River"},
    "SCR": {"marker": "o", "label": "St. Clair River"},
    "LSC": {"marker": "s", "label": "Lake St. Clair"},
}

_BIFURCATION_COLORS = {
    "above": "#d62728",   # red-ish  — top quantile (most polluted)
    "middle": "#cccccc19",  # light gray — middle band (95% transparent)
    "below": "#2ca02c",   # green    — bottom quantile (least polluted)
}


# ---------------------------------------------------------------------------
# Water-body background
# ---------------------------------------------------------------------------


def plot_corridor_map(
    ax: plt.Axes,
    maps_dir: str | Path,
    *,
    annotate: bool = True,
    annotation_fontsize: int = 10,
) -> plt.Axes:
    """Draw the Huron-Erie Corridor water bodies onto *ax*.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes.
    maps_dir : str or Path
        Root directory containing the shapefile sub-folders
        (``lake_stclair/``, ``detroit_river_aoc_shapefile/``, …).
    annotate : bool
        Whether to add italic water-body name labels.
    annotation_fontsize : int
        Font size for annotation labels.

    Returns
    -------
    ax
    """
    import geopandas as gpd

    maps_dir = Path(maps_dir)

    shapefiles = {
        "lake_stclair":    maps_dir / "lake_stclair"    / "lake_stclair.shp",
        "lake_erie":       maps_dir / "lake_erie"       / "lake_erie.shp",
        "lake_huron":      maps_dir / "lake_huron"      / "lake_huron.shp",
        "detroit_river":   maps_dir / "detroit_river_aoc_shapefile" / "AOC_MI_Detroit_2021.shp",
        "stclair_river":   maps_dir / "aoc_mi_stclair_2021"        / "AOC_MI_StClair_2021.shp",
    }

    for name, path in shapefiles.items():
        if path.exists():
            gdf = gpd.read_file(path).to_crs(epsg=4326)
            gdf.plot(ax=ax, color="lightblue", edgecolor="none", alpha=0.5)

    ax.set_ylim(42.0, 43.1)
    ax.set_xlim(-83.3, -82.3)

    if annotate:
        kw = dict(fontsize=annotation_fontsize, color="gray", style="italic")
        ax.text(-83.00, 42.20, "Detroit River",   **kw)
        ax.text(-82.85, 42.90, "St. Clair River", **kw)
        ax.text(-82.55, 42.05, "Lake Erie",       **kw)
        ax.text(-83.00, 42.50, "Lake St. Clair",  **kw)
        ax.text(-82.60, 43.05, "Lake Huron",      **kw)

    return ax


# ---------------------------------------------------------------------------
# Main two-panel figure
# ---------------------------------------------------------------------------


def plot_corridor_bifurcation(
    scores: pd.Series,
    lat: pd.Series,
    lon: pd.Series,
    waterbody: pd.Series,
    maps_dir: str | Path,
    *,
    threshold_quantile: float = 0.20,
    score_label: str = "Pollution Score",
    waterbody_styles: Dict[str, Dict] | None = None,
    above_color: str | None = None,
    middle_color: str | None = None,
    below_color: str | None = None,
    figsize: Tuple[int, int] = (16, 7),
    label_fontsize: int = 14,
) -> Tuple[plt.Figure, np.ndarray]:
    """Two-panel figure: corridor map (left) + cumulative frequency (right).

    Left panel
    ----------
    Sites plotted on the Huron-Erie Corridor.
    **Colour** encodes the three-way split:
      * green  – bottom *threshold_quantile* (least polluted)
      * red    – top *threshold_quantile* (most polluted)
      * gray   – everything in between
    **Marker shape** encodes the water body but is *not* shown in the legend.

    Right panel
    -----------
    Empirical CDF of *scores* with the lower and upper thresholds highlighted.

    Parameters
    ----------
    scores : pd.Series
        Continuous score per site (e.g. composite pollution score).
    lat, lon : pd.Series
        Latitude / Longitude (same index as *scores*).
    waterbody : pd.Series
        Waterbody label per site (``"DR"``, ``"SCR"``, ``"LSC"``).
    maps_dir : str or Path
        Path to ``data/maps/`` folder with shapefiles.
    threshold_quantile : float
        Quantile (0–1) for the lower cut.  The upper cut is
        ``1 − threshold_quantile``.  Default ``0.20``.
    score_label : str
        Axis label for the score variable.
    waterbody_styles : dict, optional
        Override per-waterbody marker mapping.
    above_color, middle_color, below_color : str, optional
        Override tri-colour scheme.
    figsize : tuple
        Overall figure size.
    label_fontsize : int
        Font size for axis / tick labels and legends.

    Returns
    -------
    (fig, [ax_map, ax_ecdf])
    """
    # ── resolve styles ────────────────────────────────────────────────────
    wb_styles = waterbody_styles or _WATERBODY_STYLE
    c_below  = below_color  or _BIFURCATION_COLORS["below"]
    c_middle = middle_color or _BIFURCATION_COLORS["middle"]
    c_above  = above_color  or _BIFURCATION_COLORS["above"]

    # ── thresholds ────────────────────────────────────────────────────────
    lower_q = threshold_quantile
    upper_q = 1.0 - threshold_quantile
    lower_val = scores.quantile(lower_q)
    upper_val = scores.quantile(upper_q)
    pct_lo = f"{lower_q * 100:.0f}%"
    pct_hi = f"{upper_q * 100:.0f}%"

    is_below  = scores <= lower_val
    is_above  = scores >= upper_val
    is_middle = ~is_below & ~is_above

    n_below  = is_below.sum()
    n_middle = is_middle.sum()
    n_above  = is_above.sum()

    # ── figure layout (tight gap) ─────────────────────────────────────────
    fig = plt.figure(figsize=figsize, dpi=300, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.02)
    ax_map  = fig.add_subplot(gs[0])
    ax_ecdf = fig.add_subplot(gs[1])

    # ── LEFT: corridor map ────────────────────────────────────────────────
    plot_corridor_map(ax_map, maps_dir, annotate=True,
                      annotation_fontsize=label_fontsize - 2)

    # assign per-site colour
    site_color = pd.Series(c_middle, index=scores.index)
    site_color[is_below] = c_below
    site_color[is_above] = c_above

    # scatter by waterbody (for marker shape only)
    for wb_key, style in wb_styles.items():
        mask = waterbody == wb_key
        if mask.sum() == 0:
            continue
        ax_map.scatter(
            lon[mask], lat[mask],
            # marker=style["marker"],
            c=site_color[mask],
            s=85,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )

    # legend: colour only
    colour_handles = [
        mlines.Line2D([], [], color=c_below, marker="o", linestyle="None",
                      markersize=9, label=f"Bottom {pct_lo} (n={n_below})"),
        mlines.Line2D([], [], color=c_middle, marker="o", linestyle="None",
                      markersize=9, label=f"Middle (n={n_middle})"),
        mlines.Line2D([], [], color=c_above, marker="o", linestyle="None",
                      markersize=9, label=f"Top {pct_lo} (n={n_above})"),
    ]
    ax_map.legend(
        handles=colour_handles,
        loc="upper left",
        fontsize=label_fontsize - 1,
        framealpha=0.9,
        ncol=1,
    )

    ax_map.set_xlabel("Longitude", fontsize=label_fontsize)
    ax_map.set_ylabel("Latitude", fontsize=label_fontsize)
    ax_map.tick_params(labelsize=label_fontsize - 1)
    ax_map.grid(linestyle="--", alpha=0.4)

    # ── RIGHT: ECDF / cumulative frequency ────────────────────────────────
    sorted_scores = np.sort(scores.dropna().values)
    n = len(sorted_scores)
    cum_prob = np.arange(1, n + 1) / n

    ax_ecdf.step(sorted_scores, cum_prob, where="post", linewidth=2.5,
                 color="darkblue", alpha=0.8, label="ECDF")

    # lower threshold lines
    ax_ecdf.axhline(y=lower_q, color="red", ls="--", lw=2,
                    label=f"{pct_lo} threshold", zorder=4)
    ax_ecdf.axvline(x=lower_val, color="red", ls="--", lw=2,
                    alpha=0.7, zorder=4)

    # upper threshold lines
    ax_ecdf.axhline(y=upper_q, color="red", ls="--", lw=2,
                    label=f"{pct_hi} threshold", zorder=4)
    ax_ecdf.axvline(x=upper_val, color="red", ls="--", lw=2,
                    alpha=0.7, zorder=4)

    # shade bottom region (green)
    ref_scores = sorted_scores[sorted_scores <= lower_val]
    ax_ecdf.fill_between(
        ref_scores, 0, lower_q,
        alpha=0.25, color=c_below,
        label=f"Bottom {pct_lo} (n={n_below})",
        zorder=1,
    )

    # shade top region (red)
    top_scores = sorted_scores[sorted_scores >= upper_val]
    ax_ecdf.fill_between(
        top_scores, upper_q, 1.0,
        alpha=0.20, color=c_above,
        label=f"Top {pct_lo} (n={n_above})",
        zorder=1,
    )

    # percentile guide lines
    for pct in (10, 25, 50, 75, 90):
        ax_ecdf.axhline(y=pct / 100, color="gray", ls=":", alpha=0.4, lw=1)
        ax_ecdf.text(
            sorted_scores[-1] * 1.02, pct / 100,
            f"{pct}%", fontsize=label_fontsize - 2, color="gray", va="center",
        )

    # annotation boxes
    ax_ecdf.annotate(
        f"Lower = {lower_val:.3f}\n({pct_lo} of sites)",
        xy=(lower_val, lower_q),
        xytext=(lower_val - (sorted_scores[-1] - lower_val) * 0.5,
                lower_q + 0.08),
        fontsize=label_fontsize - 2,
        bbox=dict(boxstyle="round,pad=0.35", fc="wheat", alpha=0.85,
                  ec="red", lw=1.5),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
    )
    ax_ecdf.annotate(
        f"Upper = {upper_val:.3f}\n({pct_hi} of sites)",
        xy=(upper_val, upper_q),
        xytext=(upper_val - (sorted_scores[-1] - upper_val) * 1.2,
                upper_q + 0.06),
        fontsize=label_fontsize - 2,
        bbox=dict(boxstyle="round,pad=0.35", fc="wheat", alpha=0.85,
                  ec="red", lw=1.5),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
    )

    ax_ecdf.set_xlabel(score_label, fontsize=label_fontsize)
    ax_ecdf.set_ylabel("Cumulative Probability", fontsize=label_fontsize)
    ax_ecdf.tick_params(labelsize=label_fontsize - 1)
    ax_ecdf.set_ylim(0, 1.05)
    ax_ecdf.set_xlim(sorted_scores[0] - 0.3, sorted_scores[-1] + 0.8)
    ax_ecdf.grid(True, alpha=0.3)
    ax_ecdf.legend(fontsize=label_fontsize - 2, loc="lower right")

    return fig, np.array([ax_map, ax_ecdf])
