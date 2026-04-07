"""Finalized-model environmental PCA visualizations.

Provides layered plots combining:
- full-model decision regions (conditional on PC1–PC2),
- 2-PC LDA decision regions (native PC1–PC2 LDA),
- reference sites coloured by true Ward cluster,
- non-reference sites coloured by predicted cluster,
- convex hulls or 95 % confidence ellipses around reference clusters,
- 2×2 panel plots with per-cluster and all-ref ellipses,
- PCA loading arrows.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.patches import Ellipse


# Canonical cluster colours (matches cluster_panel_plot.CLUSTER_COLORS)
CLUSTER_COLORS: List[str] = [
    "#1f77b4",   # C1 — blue
    "#ff7f0e",   # C2 — orange
    "#2ca02c",   # C3 — green
    "#4C72B0",   # C4 — spare
    "#9370DB",   # C5 — spare
]


def _ccolor(cluster_id: int) -> str:
    return CLUSTER_COLORS[(cluster_id - 1) % len(CLUSTER_COLORS)]


# ─── Main plotting function ─────────────────────────────────────────


def plot_finalized_ordination(
    site_data: pd.DataFrame,
    *,
    xx: np.ndarray | None = None,
    yy: np.ndarray | None = None,
    grid_labels: np.ndarray | None = None,
    hulls: Dict[int, np.ndarray] | None = None,
    ellipses: Dict[int, Dict[str, Any]] | None = None,
    loadings: pd.DataFrame | None = None,
    variance_explained: np.ndarray | None = None,
    env_short_names: Sequence[str] | None = None,
    arrow_scale: float = 1.0,
    title: str = "Finalized Model — Environmental PCA Ordination",
    figsize: Tuple[float, float] = (11, 9),
    dpi: int = 180,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create the layered PCA ordination plot.

    Parameters
    ----------
    site_data : DataFrame
        From ``build_site_plot_data`` — must have columns:
        is_reference, true_cluster, pred_cluster, ref_correct,
        display_cluster, PC1, PC2.
    xx, yy, grid_labels : optional
        Prediction grid from ``build_prediction_grid``.
    hulls : optional
        Convex hulls from ``compute_cluster_hulls``.
    ellipses : optional
        Ellipse params from ``compute_cluster_ellipses``.
    loadings : optional
        PCA loadings DataFrame (vars × PCs).
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    unique_clusters = sorted(
        site_data["display_cluster"].dropna().unique().astype(int)
    )

    # ── Layer 1: Decision-region background ──────────────────────────
    if xx is not None and yy is not None and grid_labels is not None:
        bg_cmap_colors = [to_rgba(_ccolor(c), alpha=0.18) for c in unique_clusters]
        bg_cmap = ListedColormap(bg_cmap_colors)
        # Map grid labels to 0-based index for the colormap
        label_to_idx = {c: i for i, c in enumerate(unique_clusters)}
        grid_idx = np.vectorize(lambda v: label_to_idx.get(int(v), 0))(grid_labels)
        ax.pcolormesh(
            xx, yy, grid_idx,
            cmap=bg_cmap, shading="auto", zorder=0, rasterized=True,
        )

    # ── Layer 2: Convex hulls ────────────────────────────────────────
    if hulls is not None:
        for cl, verts in hulls.items():
            color = _ccolor(cl)
            ax.fill(
                verts[:, 0], verts[:, 1],
                alpha=0.10, color=color, zorder=1,
            )
            ax.plot(
                verts[:, 0], verts[:, 1],
                color=color, linewidth=1.2, linestyle="--",
                alpha=0.6, zorder=1,
            )

    # ── Layer 2b: Confidence ellipses ────────────────────────────────
    if ellipses is not None:
        for cl, ep in ellipses.items():
            color = _ccolor(cl)
            ell = Ellipse(
                xy=ep["center"],
                width=ep["width"],
                height=ep["height"],
                angle=ep["angle_deg"],
                facecolor=to_rgba(color, alpha=0.08),
                edgecolor=color,
                linewidth=1.4,
                linestyle=":",
                zorder=1,
            )
            ax.add_patch(ell)

    # ── Layer 3: Non-reference sites ─────────────────────────────────
    nonref = site_data[~site_data["is_reference"]]
    for cl in unique_clusters:
        mask = nonref["display_cluster"] == cl
        if not mask.any():
            continue
        ax.scatter(
            nonref.loc[mask, "PC1"], nonref.loc[mask, "PC2"],
            c=_ccolor(cl), marker="o", s=40, alpha=0.55,
            edgecolors="grey", linewidths=0.3,
            label=f"Non-ref C{cl}",
            zorder=3,
        )

    # ── Layer 4: Reference sites (correct) ───────────────────────────
    ref = site_data[site_data["is_reference"]]
    for cl in unique_clusters:
        mask_correct = (ref["true_cluster"] == cl) & (ref["ref_correct"] == True)  # noqa: E712
        if mask_correct.any():
            ax.scatter(
                ref.loc[mask_correct, "PC1"], ref.loc[mask_correct, "PC2"],
                c=_ccolor(cl), marker="^", s=80,
                edgecolors="black", linewidths=0.8,
                label=f"Ref C{cl} (correct)",
                zorder=4,
            )

    # ── Layer 5: Reference sites (misclassified) ────────────────────
    for cl in unique_clusters:
        mask_wrong = (ref["true_cluster"] == cl) & (ref["ref_correct"] == False)  # noqa: E712
        if mask_wrong.any():
            ax.scatter(
                ref.loc[mask_wrong, "PC1"], ref.loc[mask_wrong, "PC2"],
                c=_ccolor(cl), marker="X", s=90,
                edgecolors="red", linewidths=1.2,
                label=f"Ref C{cl} (misclassified)",
                zorder=5,
            )

    # ── Layer 6: Loading arrows ──────────────────────────────────────
    if loadings is not None:
        # Use only PC1, PC2 loadings
        ld = loadings.iloc[:, :2].values  # (p, 2)
        scale = arrow_scale * np.sqrt(
            np.array([
                np.ptp(site_data["PC1"]),
                np.ptp(site_data["PC2"]),
            ])
        )
        for j in range(ld.shape[0]):
            dx = ld[j, 0] * scale[0] * 0.55
            dy = ld[j, 1] * scale[1] * 0.55
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

    # ── Axes, legend, labels ─────────────────────────────────────────
    ax.axhline(0, color="grey", linewidth=0.4, zorder=0)
    ax.axvline(0, color="grey", linewidth=0.4, zorder=0)

    if variance_explained is not None and len(variance_explained) >= 2:
        ax.set_xlabel(f"PC1 ({variance_explained[0]:.1f}%)", fontsize=12)
        ax.set_ylabel(f"PC2 ({variance_explained[1]:.1f}%)", fontsize=12)
    else:
        ax.set_xlabel("PC1", fontsize=12)
        ax.set_ylabel("PC2", fontsize=12)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # De-duplicate legend
    handles, labels_lg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_lg, handles))
    ax.legend(
        by_label.values(), by_label.keys(),
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        frameon=True, framealpha=0.9, fontsize=8,
        title="Symbol key",
    )
    ax.grid(alpha=0.12)
    fig.tight_layout()

    return fig, ax


# ─── Convenience wrappers for hull-only and ellipse-only versions ───


def plot_hull_version(
    site_data: pd.DataFrame,
    *,
    xx: np.ndarray | None = None,
    yy: np.ndarray | None = None,
    grid_labels: np.ndarray | None = None,
    hulls: Dict[int, np.ndarray] | None = None,
    loadings: pd.DataFrame | None = None,
    variance_explained: np.ndarray | None = None,
    env_short_names: Sequence[str] | None = None,
    title: str = "Finalized Model — Convex Hulls",
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Decision regions + convex hulls (no ellipses)."""
    return plot_finalized_ordination(
        site_data,
        xx=xx, yy=yy, grid_labels=grid_labels,
        hulls=hulls, ellipses=None,
        loadings=loadings,
        variance_explained=variance_explained,
        env_short_names=env_short_names,
        title=title,
        **kwargs,
    )


def plot_ellipse_version(
    site_data: pd.DataFrame,
    *,
    xx: np.ndarray | None = None,
    yy: np.ndarray | None = None,
    grid_labels: np.ndarray | None = None,
    ellipses: Dict[int, Dict[str, Any]] | None = None,
    loadings: pd.DataFrame | None = None,
    variance_explained: np.ndarray | None = None,
    env_short_names: Sequence[str] | None = None,
    title: str = "Finalized Model — 95% Confidence Ellipses",
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Decision regions + confidence ellipses (no hulls)."""
    return plot_finalized_ordination(
        site_data,
        xx=xx, yy=yy, grid_labels=grid_labels,
        hulls=None, ellipses=ellipses,
        loadings=loadings,
        variance_explained=variance_explained,
        env_short_names=env_short_names,
        title=title,
        **kwargs,
    )


# ─── 2×2 Ellipse panel: per-cluster + all-ref ───────────────────────


def _draw_single_panel(
    ax: plt.Axes,
    site_data: pd.DataFrame,
    *,
    xx: np.ndarray | None,
    yy: np.ndarray | None,
    grid_labels: np.ndarray | None,
    ellipse_params: Dict[str, Any] | None,
    ellipse_color: str,
    ellipse_label: str,
    highlight_cluster: int | None,
    unique_clusters: list[int],
    variance_explained: np.ndarray | None,
    loadings: pd.DataFrame | None = None,
    env_short_names: Sequence[str] | None = None,
    arrow_scale: float = 1.0,
    title: str = "",
) -> None:
    """Render one panel of the 2×2 ellipse figure onto *ax*."""

    # Decision-region background
    if xx is not None and yy is not None and grid_labels is not None:
        bg_cmap_colors = [to_rgba(_ccolor(c), alpha=0.18) for c in unique_clusters]
        bg_cmap = ListedColormap(bg_cmap_colors)
        label_to_idx = {c: i for i, c in enumerate(unique_clusters)}
        grid_idx = np.vectorize(lambda v: label_to_idx.get(int(v), 0))(grid_labels)
        ax.pcolormesh(xx, yy, grid_idx, cmap=bg_cmap, shading="auto",
                      zorder=0, rasterized=True)

    # Ellipse
    if ellipse_params is not None:
        ell = Ellipse(
            xy=ellipse_params["center"],
            width=ellipse_params["width"],
            height=ellipse_params["height"],
            angle=ellipse_params["angle_deg"],
            facecolor=to_rgba(ellipse_color, alpha=0.12),
            edgecolor=ellipse_color,
            linewidth=1.6,
            linestyle=":",
            zorder=1,
        )
        ax.add_patch(ell)

    # Non-reference sites
    nonref = site_data[~site_data["is_reference"]]
    for cl in unique_clusters:
        mask = nonref["display_cluster"] == cl
        if not mask.any():
            continue
        ax.scatter(
            nonref.loc[mask, "PC1"], nonref.loc[mask, "PC2"],
            c=_ccolor(cl), marker="o", s=25, alpha=0.40,
            edgecolors="grey", linewidths=0.2,
            label=f"Non-ref C{cl}",
            zorder=3,
        )

    # Reference sites
    ref = site_data[site_data["is_reference"]]
    for cl in unique_clusters:
        mask_correct = (ref["true_cluster"] == cl) & (ref["ref_correct"] == True)  # noqa: E712
        if mask_correct.any():
            ax.scatter(
                ref.loc[mask_correct, "PC1"], ref.loc[mask_correct, "PC2"],
                c=_ccolor(cl), marker="^", s=55,
                edgecolors="black", linewidths=0.6,
                label=f"Ref C{cl} (correct)",
                zorder=4,
            )
        mask_wrong = (ref["true_cluster"] == cl) & (ref["ref_correct"] == False)  # noqa: E712
        if mask_wrong.any():
            ax.scatter(
                ref.loc[mask_wrong, "PC1"], ref.loc[mask_wrong, "PC2"],
                c=_ccolor(cl), marker="X", s=60,
                edgecolors="red", linewidths=0.9,
                label=f"Ref C{cl} (misclass.)",
                zorder=5,
            )

    # Loadings
    if loadings is not None:
        ld = loadings.iloc[:, :2].values
        scale = arrow_scale * np.sqrt(
            np.array([np.ptp(site_data["PC1"]), np.ptp(site_data["PC2"])])
        )
        for j in range(ld.shape[0]):
            dx = ld[j, 0] * scale[0] * 0.45
            dy = ld[j, 1] * scale[1] * 0.45
            ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="#333333",
                                        lw=1.8, mutation_scale=12),
                        zorder=6)
            name = (env_short_names[j]
                    if env_short_names and j < len(env_short_names)
                    else loadings.index[j])
            ax.text(dx * 1.14, dy * 1.14, name,
                    fontsize=7.5, fontweight="bold", color="#222222",
                    ha="center", va="center", zorder=7)

    ax.axhline(0, color="grey", linewidth=0.3, zorder=0)
    ax.axvline(0, color="grey", linewidth=0.3, zorder=0)
    if variance_explained is not None and len(variance_explained) >= 2:
        ax.set_xlabel(f"PC1 ({variance_explained[0]:.1f}%)", fontsize=9)
        ax.set_ylabel(f"PC2 ({variance_explained[1]:.1f}%)", fontsize=9)
    else:
        ax.set_xlabel("PC1", fontsize=9)
        ax.set_ylabel("PC2", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(alpha=0.10)


def plot_ellipse_panel_2x2(
    site_data: pd.DataFrame,
    cluster_ellipses: Dict[int, Dict[str, Any]],
    allref_ellipse: Dict[str, Any],
    *,
    xx: np.ndarray | None = None,
    yy: np.ndarray | None = None,
    grid_labels: np.ndarray | None = None,
    loadings: pd.DataFrame | None = None,
    variance_explained: np.ndarray | None = None,
    env_short_names: Sequence[str] | None = None,
    arrow_scale: float = 1.0,
    suptitle: str = "Reference-Cluster Ellipses",
    figsize: Tuple[float, float] = (16, 14),
    dpi: int = 180,
) -> plt.Figure:
    """Create a 2x2 figure: one panel per cluster ellipse + all-ref ellipse.

    For k=3 clusters the layout is:
        [C1 ellipse] [C2 ellipse]
        [C3 ellipse] [All ref ellipse]
    """
    unique_clusters = sorted(cluster_ellipses.keys())
    k = len(unique_clusters)

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    axflat = axes.ravel()

    common_kw = dict(
        xx=xx, yy=yy, grid_labels=grid_labels,
        unique_clusters=unique_clusters,
        variance_explained=variance_explained,
        loadings=loadings,
        env_short_names=env_short_names,
        arrow_scale=arrow_scale,
    )

    # Per-cluster panels
    for i, cl in enumerate(unique_clusters):
        _draw_single_panel(
            axflat[i], site_data,
            ellipse_params=cluster_ellipses[cl],
            ellipse_color=_ccolor(cl),
            ellipse_label=f"Cluster C{cl}",
            highlight_cluster=cl,
            title=f"Cluster C{cl} — 95% Ellipse",
            **common_kw,
        )

    # All-ref panel in the last slot
    _draw_single_panel(
        axflat[k], site_data,
        ellipse_params=allref_ellipse,
        ellipse_color="#555555",
        ellipse_label="All Ref",
        highlight_cluster=None,
        title="All Reference Sites — 95% Ellipse",
        **common_kw,
    )

    # Hide unused axes (if k < 3)
    for i in range(k + 1, 4):
        axflat[i].set_visible(False)

    # Shared legend from the first axis
    handles, labels_lg = axflat[0].get_legend_handles_labels()
    by_label = dict(zip(labels_lg, handles))
    fig.legend(
        by_label.values(), by_label.keys(),
        loc="lower center", ncol=min(len(by_label), 5),
        frameon=True, framealpha=0.9, fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ─── Single all-ref ellipse plot ────────────────────────────────────


def plot_allref_ellipse(
    site_data: pd.DataFrame,
    allref_ellipse: Dict[str, Any],
    *,
    xx: np.ndarray | None = None,
    yy: np.ndarray | None = None,
    grid_labels: np.ndarray | None = None,
    loadings: pd.DataFrame | None = None,
    variance_explained: np.ndarray | None = None,
    env_short_names: Sequence[str] | None = None,
    arrow_scale: float = 1.0,
    title: str = "All Reference Sites — 95% Confidence Ellipse",
    figsize: Tuple[float, float] = (11, 9),
    dpi: int = 180,
) -> Tuple[plt.Figure, plt.Axes]:
    """Single ordination with only the all-reference 95% ellipse."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    unique_clusters = sorted(
        site_data["display_cluster"].dropna().unique().astype(int)
    )

    # Decision-region background
    if xx is not None and yy is not None and grid_labels is not None:
        bg_cmap_colors = [to_rgba(_ccolor(c), alpha=0.18) for c in unique_clusters]
        bg_cmap = ListedColormap(bg_cmap_colors)
        label_to_idx = {c: i for i, c in enumerate(unique_clusters)}
        grid_idx = np.vectorize(lambda v: label_to_idx.get(int(v), 0))(grid_labels)
        ax.pcolormesh(xx, yy, grid_idx, cmap=bg_cmap, shading="auto",
                      zorder=0, rasterized=True)

    # All-ref ellipse
    ell = Ellipse(
        xy=allref_ellipse["center"],
        width=allref_ellipse["width"],
        height=allref_ellipse["height"],
        angle=allref_ellipse["angle_deg"],
        facecolor=to_rgba("#555555", alpha=0.10),
        edgecolor="#555555",
        linewidth=1.8,
        linestyle=":",
        zorder=1,
    )
    ax.add_patch(ell)

    # Non-reference sites
    nonref = site_data[~site_data["is_reference"]]
    for cl in unique_clusters:
        mask = nonref["display_cluster"] == cl
        if not mask.any():
            continue
        ax.scatter(
            nonref.loc[mask, "PC1"], nonref.loc[mask, "PC2"],
            c=_ccolor(cl), marker="o", s=40, alpha=0.55,
            edgecolors="grey", linewidths=0.3,
            label=f"Non-ref C{cl}", zorder=3,
        )

    # Reference sites
    ref = site_data[site_data["is_reference"]]
    for cl in unique_clusters:
        mask_correct = (ref["true_cluster"] == cl) & (ref["ref_correct"] == True)  # noqa: E712
        if mask_correct.any():
            ax.scatter(
                ref.loc[mask_correct, "PC1"], ref.loc[mask_correct, "PC2"],
                c=_ccolor(cl), marker="^", s=80,
                edgecolors="black", linewidths=0.8,
                label=f"Ref C{cl} (correct)", zorder=4,
            )
        mask_wrong = (ref["true_cluster"] == cl) & (ref["ref_correct"] == False)  # noqa: E712
        if mask_wrong.any():
            ax.scatter(
                ref.loc[mask_wrong, "PC1"], ref.loc[mask_wrong, "PC2"],
                c=_ccolor(cl), marker="X", s=90,
                edgecolors="red", linewidths=1.2,
                label=f"Ref C{cl} (misclassified)", zorder=5,
            )

    # Loading arrows
    if loadings is not None:
        ld = loadings.iloc[:, :2].values
        scale = arrow_scale * np.sqrt(
            np.array([np.ptp(site_data["PC1"]), np.ptp(site_data["PC2"])])
        )
        for j in range(ld.shape[0]):
            dx = ld[j, 0] * scale[0] * 0.55
            dy = ld[j, 1] * scale[1] * 0.55
            ax.annotate(
                "", xy=(dx, dy), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=2.2,
                                mutation_scale=14),
                zorder=6,
            )
            name = (env_short_names[j]
                    if env_short_names and j < len(env_short_names)
                    else loadings.index[j])
            ax.text(dx * 1.12, dy * 1.12, name,
                    fontsize=9, fontweight="bold", color="#222222",
                    ha="center", va="center", zorder=7)

    ax.axhline(0, color="grey", linewidth=0.4, zorder=0)
    ax.axvline(0, color="grey", linewidth=0.4, zorder=0)
    if variance_explained is not None and len(variance_explained) >= 2:
        ax.set_xlabel(f"PC1 ({variance_explained[0]:.1f}%)", fontsize=12)
        ax.set_ylabel(f"PC2 ({variance_explained[1]:.1f}%)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    handles, labels_lg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_lg, handles))
    ax.legend(
        by_label.values(), by_label.keys(),
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        frameon=True, framealpha=0.9, fontsize=8,
        title="Symbol key",
    )
    ax.grid(alpha=0.12)
    fig.tight_layout()
    return fig, ax
