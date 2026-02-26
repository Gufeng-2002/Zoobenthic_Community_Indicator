"""NMDS visualisations — biplot, ZCI distribution, and ZCI vs PS scatter.

Public API
----------
plot_nmds_biplot
    Comprehensive 2-D NMDS biplot per cluster (sites, endpoints,
    pollution bins, waterbody shapes, species arrows).
plot_zci_distribution
    ZCI histogram + strip plot per cluster.
plot_zci_vs_pollution
    ZCI vs Pollution Score scatter with regression per cluster.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import linregress

try:
    from adjustText import adjust_text
except ModuleNotFoundError:
    adjust_text = None

from ..models.nmds import ClusterNMDS, ClusterZCI


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

BIN_COLORS = {
    "Reference": "forestgreen",
    "Test":      "silver",
    "Degraded":  "indianred",
}
BIN_ORDER = ["Reference", "Test", "Degraded"]

WB_MARKER_MAP = {"DR": "o", "SCR": "s", "LSC": "^"}


def _pollution_bin(ps_val: float, p20: float, p80: float) -> str:
    if ps_val <= p20:
        return "Reference"
    elif ps_val >= p80:
        return "Degraded"
    return "Test"


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


# ═══════════════════════════════════════════════════════════════════
# 1) NMDS Biplot
# ═══════════════════════════════════════════════════════════════════


def plot_nmds_biplot(
    nmds_results: Dict[int, ClusterNMDS],
    site_meta: pd.DataFrame,
    clusters: Sequence[int],
    p20: float,
    p80: float,
    *,
    figsize_per_cluster: Tuple[float, float] = (10, 9),
    dpi: int = 150,
    arrow_scale: float = 0.35,
    suptitle: str = "2-D NMDS Biplot on Relative-Abundance Taxa "
                    "(Bray–Curtis, per cluster)",
) -> Tuple[plt.Figure, np.ndarray]:
    """Comprehensive NMDS biplot per cluster.

    Each panel shows sites coloured by pollution bin, shaped by
    waterbody, endpoint stars, and species-score arrows.

    Parameters
    ----------
    nmds_results : dict[int, ClusterNMDS]
    site_meta : pd.DataFrame
        Must contain columns: Pollution_Score, Waterbody, PS_Bin.
    clusters : sequence of int
    p20, p80 : float
        Pollution-score percentile cut-offs.
    """
    n_cl = len(clusters)
    fig, axs = plt.subplots(
        1, n_cl,
        figsize=(figsize_per_cluster[0] * n_cl, figsize_per_cluster[1]),
        dpi=dpi,
    )
    if n_cl == 1:
        axs = [axs]

    # Ensure PS_Bin column exists
    if "PS_Bin" not in site_meta.columns:
        site_meta = site_meta.copy()
        site_meta["PS_Bin"] = site_meta["Pollution_Score"].apply(
            lambda x: _pollution_bin(x, p20, p80)
        )

    wb_unique = sorted(site_meta["Waterbody"].dropna().unique())

    for ax, cl in zip(axs, clusters):
        res = nmds_results[cl]
        coords   = res.coords_df
        wa_df    = res.wa_df
        stress   = res.stress
        ref_lab  = res.ref_label
        deg_lab  = res.deg_label
        var_exp  = res.var_explained

        real_ids = res.real_site_ids
        meta_cl  = site_meta.loc[real_ids]

        # 1) Sites: colour = pollution bin, shape = waterbody
        for wb in wb_unique:
            marker = WB_MARKER_MAP.get(wb, "D")
            for bn in BIN_ORDER:
                mask = (meta_cl["Waterbody"] == wb) & (meta_cl["PS_Bin"] == bn)
                ids = mask.index[mask]
                if len(ids) == 0:
                    continue
                ax.scatter(
                    coords.loc[ids, "NMDS1"],
                    coords.loc[ids, "NMDS2"],
                    marker=marker, s=60,
                    c=BIN_COLORS[bn],
                    edgecolors="k", linewidths=0.45,
                    alpha=0.75, zorder=3,
                )

        # 2) Endpoints as large stars
        ax.scatter(*coords.loc[ref_lab], marker="*", s=450,
                   c="green", edgecolors="k", linewidths=1.0, zorder=6)
        ax.scatter(*coords.loc[deg_lab], marker="*", s=450,
                   c="red", edgecolors="k", linewidths=1.0, zorder=6)
        ax.annotate("REF", xy=tuple(coords.loc[ref_lab]),
                    xytext=(10, 10), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="green",
                    arrowprops=dict(arrowstyle="->", color="green", lw=1.2))
        ax.annotate("DEG", xy=tuple(coords.loc[deg_lab]),
                    xytext=(10, -14), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="red",
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
        # Gradient axis dashed line
        ax.plot(
            [coords.loc[ref_lab, "NMDS1"], coords.loc[deg_lab, "NMDS1"]],
            [coords.loc[ref_lab, "NMDS2"], coords.loc[deg_lab, "NMDS2"]],
            ls="--", color="gray", lw=1.2, alpha=0.5, zorder=2,
        )

        # 3) Species arrows from centroid
        origin = np.array([wa_df["WA1"].mean(), wa_df["WA2"].mean()])
        x_range = coords.loc[real_ids, "NMDS1"].max() - coords.loc[real_ids, "NMDS1"].min()
        y_range = coords.loc[real_ids, "NMDS2"].max() - coords.loc[real_ids, "NMDS2"].min()
        max_arrow = np.sqrt(
            (wa_df["WA1"] - origin[0]).max()**2 +
            (wa_df["WA2"] - origin[1]).max()**2
        ) + 1e-12
        scale = arrow_scale * max(x_range, y_range) / max_arrow

        texts = []
        for taxon in wa_df.index:
            dx = (wa_df.loc[taxon, "WA1"] - origin[0]) * scale
            dy = (wa_df.loc[taxon, "WA2"] - origin[1]) * scale
            tip_x = origin[0] + dx
            tip_y = origin[1] + dy
            ax.annotate(
                "", xy=(tip_x, tip_y), xytext=(origin[0], origin[1]),
                arrowprops=dict(arrowstyle="-|>", color="steelblue",
                                lw=1.0, alpha=0.7),
                zorder=4,
            )
            t = ax.text(
                tip_x, tip_y, taxon,
                fontsize=6.5, fontweight="bold", color="steelblue",
                alpha=0.9, ha="center", va="bottom", zorder=5,
            )
            texts.append(t)

        if adjust_text is not None:
            try:
                adjust_text(
                    texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="steelblue",
                                    lw=0.3, alpha=0.4),
                )
            except Exception:
                pass

        # 4) Legend
        h_bin = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=BIN_COLORS[bn], markeredgecolor="k",
                   markersize=9, label=f"{bn} (PS)")
            for bn in BIN_ORDER
        ]
        h_wb = [
            Line2D([0], [0], marker=WB_MARKER_MAP.get(wb, "D"), color="w",
                   markerfacecolor="gray", markeredgecolor="k",
                   markersize=8, label=wb)
            for wb in wb_unique
        ]
        h_ep = [
            Line2D([0], [0], marker="*", color="w", markerfacecolor="green",
                   markeredgecolor="k", markersize=14, label="REF endpoint"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
                   markeredgecolor="k", markersize=14, label="DEG endpoint"),
        ]
        ax.legend(
            handles=h_bin + h_wb + h_ep, loc="best", fontsize=7,
            framealpha=0.88, ncol=2,
            title="Pollution bin  ·  Waterbody  ·  Endpoints",
            title_fontsize=7.5,
        )

        ax.set_xlabel(f"NMDS1 (PCA-rot, {var_exp[0]:.1%} var)", fontsize=10)
        ax.set_ylabel(f"NMDS2 (PCA-rot, {var_exp[1]:.1%} var)", fontsize=10)
        ax.set_title(
            f"Cluster {cl}  (n = {len(real_ids)})  │  "
            f"stress = {stress:.4f}\n"
            f"colour = Pollution Bin (P20={p20:.2f}, P80={p80:.2f})  │  "
            f"shape = Waterbody  │  arrows = species",
            fontsize=10, fontweight="bold",
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig, np.array(axs)


# ═══════════════════════════════════════════════════════════════════
# 2) ZCI Distribution (histogram + strip)
# ═══════════════════════════════════════════════════════════════════


def plot_zci_distribution(
    zci_results: Dict[int, ClusterZCI],
    site_meta: pd.DataFrame,
    clusters: Sequence[int],
    *,
    figsize_per_cluster: Tuple[float, float] = (8, 12),
    dpi: int = 150,
    suptitle: str = "ZCI Distribution per Cluster",
) -> Tuple[plt.Figure, np.ndarray]:
    """Histogram + strip plot of ZCI per cluster.

    Row 1: histogram with REF/DEG reference lines.
    Row 2: strip plot coloured by Pollution Score.
    """
    n_cl = len(clusters)
    fig, axes = plt.subplots(
        2, n_cl,
        figsize=(figsize_per_cluster[0] * n_cl, figsize_per_cluster[1]),
        dpi=dpi,
    )
    if n_cl == 1:
        axes = axes.reshape(2, 1)

    for ci, cl in enumerate(clusters):
        zc  = zci_results[cl]
        zci = zc.zci
        mc  = site_meta.loc[zci.index]
        ps  = mc["Pollution_Score"]
        ref_mask = mc["Is_Reference"] == 1

        # Row 1: Histogram
        ax1 = axes[0, ci]
        ax1.hist(zci, bins=25, color="steelblue", edgecolor="k", alpha=0.7)
        ax1.axvline(1.0, color="green", ls="--", lw=2, label="REF (ZCI = 1)")
        ax1.axvline(0.0, color="red",   ls="--", lw=2, label="DEG (ZCI = 0)")
        ax1.set_xlabel("ZCI")
        ax1.set_ylabel("Count")
        ax1.set_title(
            f"Cluster {cl} — ZCI Histogram\n"
            f"(N_EP = {zc.n_ep},  method = {zc.method})",
            fontweight="bold",
        )
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.25)

        # Row 2: Strip plot coloured by PS
        ax2 = axes[1, ci]
        rng = np.random.default_rng(42)
        ax2.scatter(
            zci[ref_mask.index[ref_mask]],
            rng.uniform(-0.3, 0.3, ref_mask.sum()),
            c=ps[ref_mask], cmap="RdYlGn_r", marker="o",
            s=60, edgecolors="k", linewidths=0.5, alpha=0.8,
            vmin=ps.min(), vmax=ps.max(),
        )
        sc2 = ax2.scatter(
            zci[ref_mask.index[~ref_mask]],
            np.random.default_rng(43).uniform(-0.3, 0.3, (~ref_mask).sum()),
            c=ps[~ref_mask], cmap="RdYlGn_r", marker="s",
            s=60, edgecolors="k", linewidths=0.5, alpha=0.8,
            vmin=ps.min(), vmax=ps.max(),
        )
        ax2.axvline(1.0, color="green", ls="--", lw=2)
        ax2.axvline(0.0, color="red",   ls="--", lw=2)
        ax2.set_xlabel("ZCI")
        ax2.set_yticks([])
        ax2.set_title(
            f"Cluster {cl} — ZCI strip plot  (colour = Pollution Score)",
            fontweight="bold",
        )
        cb = fig.colorbar(sc2, ax=ax2, shrink=0.6, pad=0.02)
        cb.set_label("Pollution Score", fontsize=8)
        ax2.grid(axis="x", alpha=0.25)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, axes


# ═══════════════════════════════════════════════════════════════════
# 3) ZCI vs Pollution Score scatter + regression
# ═══════════════════════════════════════════════════════════════════


def plot_zci_vs_pollution(
    zci_results: Dict[int, ClusterZCI],
    site_meta: pd.DataFrame,
    clusters: Sequence[int],
    *,
    figsize_per_cluster: Tuple[float, float] = (8, 7),
    dpi: int = 150,
    suptitle: str = "ZCI vs Pollution Score",
) -> Tuple[plt.Figure, np.ndarray]:
    """Scatter plot of ZCI vs PS with regression line + 95 % CI band."""
    n_cl = len(clusters)
    fig, axes = plt.subplots(
        1, n_cl,
        figsize=(figsize_per_cluster[0] * n_cl, figsize_per_cluster[1]),
        dpi=dpi,
    )
    if n_cl == 1:
        axes = np.array([axes])

    for ci, cl in enumerate(clusters):
        zc  = zci_results[cl]
        zci = zc.zci
        mc  = site_meta.loc[zci.index]
        ps  = mc["Pollution_Score"]
        ref_mask = mc["Is_Reference"] == 1

        ax = axes[ci]
        # Reference vs non-reference
        ax.scatter(
            ps[ref_mask], zci[ref_mask.index[ref_mask]],
            c="forestgreen", marker="o", s=55, edgecolors="k",
            linewidths=0.5, alpha=0.7, label="Reference",
        )
        ax.scatter(
            ps[~ref_mask], zci[ref_mask.index[~ref_mask]],
            c="salmon", marker="s", s=55, edgecolors="k",
            linewidths=0.5, alpha=0.7, label="Non-Reference",
        )

        # Regression + CI band
        lr = linregress(ps, zci.loc[ps.index])
        xf = np.linspace(ps.min(), ps.max(), 200)
        yf = lr.slope * xf + lr.intercept
        ax.plot(xf, yf, "--", color="navy", lw=2,
                label=f"slope = {lr.slope:.4f}")
        n_s = len(ps)
        xm = ps.mean()
        se = lr.stderr * np.sqrt(
            1 / n_s + (xf - xm) ** 2 / np.sum((ps - xm) ** 2)
        )
        ax.fill_between(xf, yf - 1.96 * se, yf + 1.96 * se,
                        color="navy", alpha=0.10)

        stars = _sig_stars(zc.p_pearson)
        ax.set_xlabel("Pollution Score")
        ax.set_ylabel("ZCI")
        ax.set_title(
            f"Cluster {cl} — ZCI vs Pollution Score\n"
            f"$r_P$ = {zc.r_pearson:.4f}{stars}  "
            f"($p$ = {zc.p_pearson:.1e}),   "
            f"$r_S$ = {zc.r_spearman:.4f}",
            fontweight="bold", fontsize=10,
        )
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig, axes
