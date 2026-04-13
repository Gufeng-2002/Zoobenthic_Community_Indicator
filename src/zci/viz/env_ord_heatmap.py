"""Heatmap-style environmental PCA ordination plots.

Two figure types per PC pair:
1. **ref_ordination** — LDA posterior-probability heatmap background
   with reference sites projected (colour = cluster, shape = Env_Strength).
2. **nonref_projection** — Same heatmap background with non-reference
   sites only (colour = predicted cluster).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgba


# Canonical cluster colours
CLUSTER_COLORS: List[str] = [
    "#1f77b4",   # C1 — blue
    "#ff7f0e",   # C2 — orange
    "#2ca02c",   # C3 — green
    "#4C72B0",   # C4 — spare
    "#9370DB",   # C5 — spare
]


def _ccolor(cluster_id: int) -> str:
    return CLUSTER_COLORS[(cluster_id - 1) % len(CLUSTER_COLORS)]


def _draw_heatmap(
    ax: plt.Axes,
    xx: np.ndarray,
    yy: np.ndarray,
    grid_probs: np.ndarray,
    grid_labels: np.ndarray,
    cluster_ids: list[int],
    *,
    max_intensity: float = 0.35,
) -> None:
    """Draw posterior-probability heatmap onto *ax*.

    Each grid cell is coloured by the predicted class, with saturation
    proportional to the posterior probability (higher → darker).

    Parameters
    ----------
    max_intensity : float
        Cap the colour intensity so that even prob=1.0 stays pastel.
        0 = fully white, 1 = fully saturated base colour.
    """
    # Build an RGBA image
    h, w = xx.shape
    rgba = np.ones((h, w, 4), dtype=float)
    rgba[:, :, 3] = 1.0  # fully opaque

    # Map cluster → index for probs array
    for idx, cid in enumerate(cluster_ids):
        mask = grid_labels == cid
        base = np.array(to_rgba(_ccolor(cid)))[:3]
        prob = grid_probs[:, :, idx]
        # Scale prob so max intensity is capped
        scaled = prob * max_intensity
        # alpha blending with white: rgb = base * scaled + (1-scaled) * 1.0
        for ch in range(3):
            rgba[:, :, ch] = np.where(
                mask,
                base[ch] * scaled + 1.0 * (1.0 - scaled),
                rgba[:, :, ch],
            )

    ax.imshow(
        rgba,
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
        zorder=0,
    )


def plot_ref_ordination(
    xx: np.ndarray,
    yy: np.ndarray,
    grid_labels: np.ndarray,
    grid_probs: np.ndarray,
    cluster_ids: list[int],
    ref_pc: pd.DataFrame,
    ref_clusters: pd.Series,
    ref_env_strength: pd.Series,
    *,
    pc_x_col: str = "PC1",
    pc_y_col: str = "PC2",
    variance_explained: np.ndarray | None = None,
    loadings: pd.DataFrame | None = None,
    env_short_names: Sequence[str] | None = None,
    loading_indices: Tuple[int, int] = (0, 1),
    arrow_scale: float = 1.0,
    title: str = "Reference Sites — LDA Posterior Heatmap",
    figsize: Tuple[float, float] = (11, 9),
    dpi: int = 180,
) -> Tuple[plt.Figure, plt.Axes]:
    """Heatmap background + reference sites (colour=cluster, shape=EnvStrength).

    Shapes:  ▲ (triangle) = Env Strong, ■ (square) = Env Weak.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Heatmap background
    _draw_heatmap(ax, xx, yy, grid_probs, grid_labels, cluster_ids)

    # Reference sites
    marker_map = {"Strong": "^", "Weak": "s"}
    marker_size = {"Strong": 80, "Weak": 65}

    for cid in cluster_ids:
        for strength in ["Strong", "Weak"]:
            mask = (ref_clusters == cid) & (ref_env_strength == strength)
            if not mask.any():
                continue
            ax.scatter(
                ref_pc.loc[mask, pc_x_col],
                ref_pc.loc[mask, pc_y_col],
                c=_ccolor(cid),
                marker=marker_map[strength],
                s=marker_size[strength],
                edgecolors="black",
                linewidths=0.8,
                label=f"C{cid} Env{strength}",
                zorder=4,
            )

    # Loading arrows
    _draw_loadings(ax, ref_pc, loadings, env_short_names,
                   pc_x_col, pc_y_col, loading_indices, arrow_scale)

    # Axes
    _set_axes(ax, variance_explained, pc_x_col, pc_y_col, title)

    # Legend
    handles, labels_lg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_lg, handles))
    # Add shape legend entries
    shape_handles = [
        plt.Line2D([0], [0], marker="^", color="grey", linestyle="",
                    markersize=9, label="Env Strong"),
        plt.Line2D([0], [0], marker="s", color="grey", linestyle="",
                    markersize=8, label="Env Weak"),
    ]
    for sh in shape_handles:
        lbl = sh.get_label()
        if lbl not in by_label:
            by_label[lbl] = sh
    ax.legend(
        by_label.values(), by_label.keys(),
        loc="upper center", bbox_to_anchor=(0.5, -0.08),
        ncol=min(len(by_label), 5),
        frameon=True, framealpha=0.9, fontsize=10,
    )
    ax.grid(alpha=0.12)
    fig.tight_layout()
    return fig, ax


def plot_nonref_projection(
    xx: np.ndarray,
    yy: np.ndarray,
    grid_labels: np.ndarray,
    grid_probs: np.ndarray,
    cluster_ids: list[int],
    nonref_pc: pd.DataFrame,
    nonref_pred_clusters: pd.Series,
    *,
    pc_x_col: str = "PC1",
    pc_y_col: str = "PC2",
    variance_explained: np.ndarray | None = None,
    loadings: pd.DataFrame | None = None,
    env_short_names: Sequence[str] | None = None,
    loading_indices: Tuple[int, int] = (0, 1),
    arrow_scale: float = 1.0,
    title: str = "Non-Reference Sites — LDA Posterior Heatmap",
    figsize: Tuple[float, float] = (11, 9),
    dpi: int = 180,
) -> Tuple[plt.Figure, plt.Axes]:
    """Heatmap background + non-reference sites (colour = predicted cluster)."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    _draw_heatmap(ax, xx, yy, grid_probs, grid_labels, cluster_ids)

    for cid in cluster_ids:
        mask = nonref_pred_clusters == cid
        if not mask.any():
            continue
        ax.scatter(
            nonref_pc.loc[mask, pc_x_col],
            nonref_pc.loc[mask, pc_y_col],
            c=_ccolor(cid),
            marker="o",
            s=40,
            alpha=0.70,
            edgecolors="grey",
            linewidths=0.3,
            label=f"Non-ref C{cid}",
            zorder=3,
        )

    _draw_loadings(ax, nonref_pc, loadings, env_short_names,
                   pc_x_col, pc_y_col, loading_indices, arrow_scale)
    _set_axes(ax, variance_explained, pc_x_col, pc_y_col, title)

    handles, labels_lg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_lg, handles))
    ax.legend(
        by_label.values(), by_label.keys(),
        loc="upper center", bbox_to_anchor=(0.5, -0.08),
        ncol=min(len(by_label), 5),
        frameon=True, framealpha=0.9, fontsize=10,
    )
    ax.grid(alpha=0.12)
    fig.tight_layout()
    return fig, ax


# ─── Helpers ─────────────────────────────────────────────────────────

def _draw_loadings(
    ax, site_pc, loadings, env_short_names,
    pc_x_col, pc_y_col, loading_indices, arrow_scale,
):
    if loadings is None:
        return
    ld_x = loadings.iloc[:, loading_indices[0]].values
    ld_y = loadings.iloc[:, loading_indices[1]].values
    scale = arrow_scale * np.sqrt(
        np.array([np.ptp(site_pc[pc_x_col]), np.ptp(site_pc[pc_y_col])])
    )
    for j in range(len(ld_x)):
        dx = ld_x[j] * scale[0] * 0.55
        dy = ld_y[j] * scale[1] * 0.55
        ax.annotate(
            "", xy=(dx, dy), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=2.2,
                            mutation_scale=14),
            zorder=6,
        )
        name = (
            env_short_names[j]
            if env_short_names and j < len(env_short_names)
            else loadings.index[j]
        )
        ax.text(
            dx * 1.12, dy * 1.12, name,
            fontsize=9, fontweight="bold", color="#222222",
            ha="center", va="center", zorder=7,
        )


def _set_axes(ax, variance_explained, pc_x_col, pc_y_col, title):
    ax.axhline(0, color="grey", linewidth=0.4, zorder=0)
    ax.axvline(0, color="grey", linewidth=0.4, zorder=0)
    if variance_explained is not None and len(variance_explained) >= 2:
        ix = int(pc_x_col.replace("PC", "")) - 1
        iy = int(pc_y_col.replace("PC", "")) - 1
        ax.set_xlabel(f"{pc_x_col} ({variance_explained[ix]:.1f}%)", fontsize=12)
        ax.set_ylabel(f"{pc_y_col} ({variance_explained[iy]:.1f}%)", fontsize=12)
    else:
        ax.set_xlabel(pc_x_col, fontsize=12)
        ax.set_ylabel(pc_y_col, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
