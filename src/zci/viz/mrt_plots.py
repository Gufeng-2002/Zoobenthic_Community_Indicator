"""Two-panel MRT classifier figure: CP profile (left) + rpart-style tree (right)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..models.mrt import MRTResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TAXA_DISPLAY_ORDER: list[str] = [
    "Oligochaeta",
    "Chironomidae",
    "Nematoda",
    "Sphaeriidae",
    "Acari",
    "Hexagenia",
    "Caenis",
    "Hirudinea",
    "Turbellaria",
    "Gastropoda",
    "Hydrozoa",
    "Other Trichoptera",
    "Amphipoda",
    "Hydropsychidae",
    "Dreissena",
    "Ceratopogonidae",
]

_LEAF_BAR_COLORS = [
    "#1f3864", "#2a4d8f", "#3566b5", "#4472c4",
    "#5b8ed0", "#72a8dc", "#8faadc", "#a6c0e8",
    "#1f3864", "#2a4d8f", "#3566b5", "#4472c4",
    "#5b8ed0", "#72a8dc", "#8faadc", "#a6c0e8",
]

_GRAY = "#b0b0b0"
_ABUNDANCE_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cp_tick_labels_subset(cp_values: pd.Series, n_ticks: int = 8) -> tuple[list[int], list[str]]:
    """Return at most *n_ticks* evenly spaced indices and 4-decimal labels."""
    n = len(cp_values)
    if n <= n_ticks:
        idxs = list(range(n))
    else:
        idxs = np.round(np.linspace(0, n - 1, n_ticks)).astype(int).tolist()
    labels: list[str] = []
    for i in idxs:
        v = cp_values.iloc[i]
        if i == 0:
            labels.append("Inf")
        elif v == 0:
            labels.append("0")
        else:
            labels.append(f"{float(v):.4f}")
    return idxs, labels


def _draw_rpart_tree(
    ax,
    model,
    feature_names: list[str],
    selected_row: pd.Series,
    n_ref: int,
    taxa_ref_octave: pd.DataFrame,
    cluster_labels_ref: pd.Series,
    leaf_membership: pd.DataFrame,
    response_transform: str = "",
) -> None:
    """Draw an rpart-style tree with branch labels and 16-taxa leaf barplots."""
    tree_ = model.tree_

    # ---- taxa order -------------------------------------------------------
    taxa_order = [t for t in _TAXA_DISPLAY_ORDER if t in taxa_ref_octave.columns]
    missing = [t for t in taxa_ref_octave.columns if t not in taxa_order]
    taxa_order.extend(missing)
    n_taxa = len(taxa_order)
    bar_colors = (_LEAF_BAR_COLORS * ((n_taxa // len(_LEAF_BAR_COLORS)) + 1))[:n_taxa]

    # ---- topology ---------------------------------------------------------
    def _leaves(node: int) -> list[int]:
        if tree_.children_left[node] == -1:
            return [node]
        return _leaves(tree_.children_left[node]) + _leaves(tree_.children_right[node])

    leaves = _leaves(0)
    n_leaves = len(leaves)
    leaf_pos = {leaf: float(i) for i, leaf in enumerate(leaves)}

    xpos: dict[int, float] = {}
    def _fill_x(node: int) -> None:
        if node in leaf_pos:
            xpos[node] = leaf_pos[node]
        else:
            _fill_x(tree_.children_left[node])
            _fill_x(tree_.children_right[node])
            xpos[node] = (xpos[tree_.children_left[node]]
                          + xpos[tree_.children_right[node]]) / 2
    _fill_x(0)

    depth: dict[int, int] = {}
    def _fill_d(node: int, d: int = 0) -> None:
        depth[node] = d
        if tree_.children_left[node] != -1:
            _fill_d(tree_.children_left[node], d + 1)
            _fill_d(tree_.children_right[node], d + 1)
    _fill_d(0)

    max_d = max(depth.values()) if depth else 0
    ypos = {n: float(max_d - depth[n]) for n in depth}

    # ---- bar layout constants ---------------------------------------------
    bar_top = -0.15
    bar_h   = 0.65
    bar_bot = bar_top - bar_h

    group_half_w = 0.42

    # ---- draw tree lines --------------------------------------------------
    for node in range(tree_.node_count):
        if tree_.children_left[node] == -1:
            continue
        left = tree_.children_left[node]
        right = tree_.children_right[node]

        xn, yn = xpos[node], ypos[node]
        xl, xr = xpos[left], xpos[right]

        ax.plot([xl, xr], [yn, yn], "k-", linewidth=1.0)

        yl_end = ypos[left] if tree_.children_left[left] != -1 else bar_top
        yr_end = ypos[right] if tree_.children_left[right] != -1 else bar_top
        ax.plot([xl, xl], [yn, yl_end], "k-", linewidth=0.8)
        ax.plot([xr, xr], [yn, yr_end], "k-", linewidth=0.8)

        feat = feature_names[tree_.feature[node]]
        thresh = tree_.threshold[node]
        label_y = yn + 0.15
        ax.text((xn + xl) / 2, label_y, f"{feat}>={thresh:.3g}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text((xn + xr) / 2, label_y, f"{feat}< {thresh:.3g}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # ---- compute mean relative abundance per leaf -------------------------
    leaf_ids = leaf_membership["Leaf"].values
    station_ids = leaf_membership["StationID"].values
    leaf_to_stations: dict[int, list] = {}
    for lid, sid in zip(leaf_ids, station_ids):
        leaf_to_stations.setdefault(int(lid), []).append(sid)

    # ---- barplots at leaves -----------------------------------------------
    for leaf in leaves:
        stations = leaf_to_stations.get(int(leaf), [])
        if stations:
            sub = taxa_ref_octave.loc[
                taxa_ref_octave.index.isin(stations), taxa_order
            ]
            means = sub.mean()
        else:
            means = pd.Series(0.0, index=taxa_order)

        max_val = means.max()
        if max_val <= 0:
            max_val = 1.0

        x_left = xpos[leaf] - group_half_w
        x_right = xpos[leaf] + group_half_w
        bw = (x_right - x_left) / n_taxa

        for j, taxon in enumerate(taxa_order):
            bx = x_left + j * bw + bw / 2
            prop = means[taxon] / max_val if max_val > 0 else 0.0
            bh = prop * bar_h
            color = bar_colors[j] if prop >= _ABUNDANCE_THRESHOLD else _GRAY
            ax.bar(bx, bh, width=bw * 0.85, bottom=bar_bot,
                   color=color, edgecolor="none")

        # bottom edge line
        ax.plot([x_left, x_right], [bar_bot, bar_bot], "k-", linewidth=0.6)

        impurity = tree_.impurity[leaf]
        n_s = tree_.n_node_samples[leaf]
        ax.text(xpos[leaf], bar_bot - 0.06,
                f"{impurity:.4g} : n={n_s}",
                ha="center", va="top", fontsize=9)

    # ---- title ------------------------------------------------------------
    n_leaves_sel = int(selected_row["nsplit"]) + 1
    tfm_label = f" ({response_transform})" if response_transform else ""
    ax.text(0.5, 1.08,
            f"Tree of size {n_leaves_sel}{tfm_label}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=13, fontstyle="italic", color="#D62728",
            fontweight="bold")
    ax.text(0.5, 1.01,
            f"RE : {selected_row['rel error']:.2f}    "
            f"CVRE : {selected_row['xerror']:.3f}    "
            f"SE : {selected_row['xstd']:.3f}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11)

    # ---- axes cleanup -----------------------------------------------------
    pad_x = 0.6
    ax.set_xlim(-pad_x, n_leaves - 1 + pad_x)
    ax.set_ylim(bar_bot - 0.25, max_d + 0.60)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

def save_mrt_cp_tree_figure(
    result: MRTResult,
    output_path: str | Path,
) -> Path:
    """Save the two-panel CVRE / tree figure (rpart style)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cp_table = result.cp_table.copy()
    selected = cp_table.loc[cp_table["nsplit"] == result.pruned_nsplits].iloc[0]
    x = np.arange(1, len(cp_table) + 1)
    sizes = (cp_table["nsplit"].astype(int) + 1).tolist()
    se1_level = result.min_cv_error + result.min_cv_se

    fig = plt.figure(figsize=(20, 7), dpi=180)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.25)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    # 95% CI error bars
    if "xcv_lo" in cp_table.columns and "xcv_hi" in cp_table.columns:
        yerr_lo = (cp_table["xerror"] - cp_table["xcv_lo"]).clip(lower=0).values
        yerr_hi = (cp_table["xcv_hi"] - cp_table["xerror"]).clip(lower=0).values
        yerr = np.vstack([yerr_lo, yerr_hi])
    else:
        yerr = cp_table["xstd"].values

    ax_left.errorbar(
        x, cp_table["xerror"], yerr=yerr,
        fmt="o", color="#5B9BD5", ecolor="#5B9BD5",
        elinewidth=1.4, capsize=3, markersize=6,
        label=r"CVRE $95\%$ CI",
        zorder=3,
    )

    ax_left.plot(
        x, cp_table["rel error"],
        "o--", color="#D62728",
        markerfacecolor="white", markeredgecolor="#D62728",
        linewidth=1.2, markersize=5,
        label="full-data", zorder=2,
    )

    min_idx = int(cp_table["xerror"].to_numpy(dtype=float).argmin())
    ax_left.scatter(
        x[min_idx], cp_table.iloc[min_idx]["xerror"],
        s=140, facecolors="white", edgecolors="#2E7D32",
        linewidths=2.0, label="min CVRE", zorder=4,
    )

    sel_idx = int(cp_table.index[cp_table["nsplit"] == result.pruned_nsplits][0])
    ax_left.scatter(
        sel_idx, selected["xerror"],
        s=60, color="#D62728", label="selected tree", zorder=5,
    )

    ax_left.axhline(se1_level, color="#D62728", linestyle=(0, (5, 4)), linewidth=1.2)
    ax_left.text(
        x[0] + 0.12, se1_level + 0.01, "+1SE",
        color="#D62728", fontsize=10, fontstyle="italic",
    )

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(sizes)
    ax_left.set_xlabel("size of tree", fontsize=11)
    ax_left.set_ylabel("Relative Error", fontsize=12)
    ax_left.set_ylim(top=1.4)
    ax_left.grid(axis="y", linestyle="--", alpha=0.25)
    ax_left.spines["right"].set_visible(False)
    ax_left.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False)

    ax_left.text(
        0.98, 0.06,
        f"n = {len(result.ref_stations)} sites",
        transform=ax_left.transAxes,
        ha="right", va="bottom",
        fontsize=10, color="#4B5563",
    )

    ax_top = ax_left.twiny()
    ax_top.set_xlim(ax_left.get_xlim())
    tick_idxs, tick_labels = _cp_tick_labels_subset(cp_table["CP"], n_ticks=8)
    ax_top.set_xticks([x[i] for i in tick_idxs])
    ax_top.set_xticklabels(tick_labels)
    ax_top.set_xlabel("complexity parameter", fontsize=11, labelpad=8)

    _draw_rpart_tree(
        ax_right,
        result.classifier_model,
        list(result.env_ref.columns),
        selected,
        n_ref=len(result.ref_stations),
        taxa_ref_octave=result.taxa_ref_octave,
        cluster_labels_ref=result.cluster_labels_ref,
        leaf_membership=result.leaf_membership,
        response_transform=result.response_transform,
    )

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
