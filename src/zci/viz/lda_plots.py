"""LDA visualisations — triplot and 4-panel cluster comparison.

Public API
----------
plot_lda_triplot
    Site scores by cluster, environmental arrows with Wilks' Λ significance.
plot_cluster_comparison
    4-panel figure: (A) z-scored env, (B) ref taxa, (C) non-ref taxa,
    (D) difference (non-ref − ref) with one-sample t-test stars.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from sklearn.preprocessing import StandardScaler

from ..core.anova import _stars
from ..core.transforms import octave_to_relative_abundance
from ..models.lda import LDAFit, WilksImportance
from .cluster_panel_plot import CLUSTER_COLORS, TAXA_DISPLAY_ORDER, _ENV_SHORT


# ─── LDA triplot ────────────────────────────────────────────────────


def plot_lda_triplot(
    lda_fit: LDAFit,
    wilks: WilksImportance,
    env_all: pd.DataFrame,
    cluster_all: pd.Series,
    *,
    arrow_scale: float | None = None,
    figsize: Tuple[float, float] = (12, 9),
    dpi: int = 300,
    show_site_labels: bool = True,
    title: str | None = "LDA Triplot: Site Scores and Habitat Vectors",
) -> Tuple[plt.Figure, plt.Axes]:
    """Create an LDA triplot (site scores + environmental arrows).

    Parameters
    ----------
    lda_fit : LDAFit
        Fitted LDA result (contains model, scaler, etc.).
    wilks : WilksImportance
        Wilks' Lambda per-variable significance.
    env_all : pd.DataFrame
        Environmental data for **all** sites to project (original scale).
    cluster_all : pd.Series
        Cluster label for each site (``NaN`` → not plotted).
    arrow_scale : float or None
        Multiplier for arrow length.  ``None`` → auto-compute.
    figsize, dpi : tuple, int
        Figure size / resolution.
    show_site_labels : bool
        Annotate each site with its index.
    title : str or None
        Figure title.

    Returns
    -------
    fig, ax
    """
    model = lda_fit.model
    scaler = lda_fit.scaler
    sig_dict = wilks.significance_dict

    n_components = model.scalings_.shape[1]

    # ── transform all sites to LDA space ────────────────────────────
    X = env_all.copy()
    if scaler is not None:
        arr = scaler.transform(X)
        X = pd.DataFrame(arr, index=X.index, columns=X.columns)

    scores = model.transform(X.values)
    scores_df = pd.DataFrame(
        scores, index=X.index,
        columns=[f"LD{i+1}" for i in range(n_components)],
    )

    # ── habitat vectors (scalings) ──────────────────────────────────
    scalings = model.scalings_       # (n_features, n_components)

    if arrow_scale is None:
        site_range = max(
            scores_df["LD1"].max() - scores_df["LD1"].min(),
            (scores_df["LD2"].max() - scores_df["LD2"].min()) if n_components > 1 else 1.0,
        )
        max_sc = np.abs(scalings).max()
        arrow_scale = site_range * 0.3 / max_sc if max_sc > 0 else 1.0

    vectors = pd.DataFrame(
        scalings * arrow_scale,
        index=env_all.columns,
        columns=[f"LD{i+1}" for i in range(n_components)],
    )

    # ── figure ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    aligned = cluster_all.reindex(scores_df.index)
    unique_clusters = sorted(aligned.dropna().unique())
    colors = {g: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, g in enumerate(unique_clusters)}

    for g in unique_clusters:
        mask = aligned == g
        sub = scores_df.loc[mask]
        ld2 = sub["LD2"] if n_components > 1 else np.zeros(len(sub))
        ax.scatter(
            sub["LD1"], ld2,
            c=colors[g], label=f"Cluster {int(g)}",
            s=100, alpha=0.65, edgecolors="black", linewidth=0.5, zorder=3,
        )
        if show_site_labels:
            for idx in sub.index:
                ax.annotate(
                    str(idx),
                    (sub.loc[idx, "LD1"],
                     sub.loc[idx, "LD2"] if n_components > 1 else 0),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7, alpha=0.7,
                )

    # ── arrows ──────────────────────────────────────────────────────
    for var in vectors.index:
        x = vectors.loc[var, "LD1"]
        y = vectors.loc[var, "LD2"] if n_components > 1 else 0.0
        info = sig_dict.get(var, {"p_value": 0.5, "significance": "ns"})
        p = info["p_value"]
        sig = info["significance"]
        is_sig = p < 0.05

        ls = "solid" if is_sig else "dashed"
        alpha_a = 0.75 if is_sig else 0.45

        ax.plot([0, x], [0, y], color="blue", ls=ls, lw=2, alpha=alpha_a, zorder=4)
        ax.annotate(
            "", xy=(x, y), xytext=(x * 0.85, y * 0.85),
            arrowprops=dict(arrowstyle="->", color="blue", lw=2, mutation_scale=15),
            annotation_clip=False,
        )

        label = f"{var} {sig}" if sig != "ns" else var
        ax.text(
            x * 1.08, y * 1.08, label,
            fontsize=10, color="blue", fontweight="bold",
            ha="center", va="center", alpha=1.0 if is_sig else 0.55, zorder=5,
        )

    # ── decorations ─────────────────────────────────────────────────
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5, zorder=1)
    ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.5, zorder=1)

    ev = lda_fit.explained_variance_ratio
    ax.set_xlabel(f"LD1 ({ev[0]:.1%} of variance)", fontsize=14, fontweight="bold")
    if n_components > 1:
        ax.set_ylabel(f"LD2 ({ev[1]:.1%} of variance)", fontsize=14, fontweight="bold")

    ax.legend(loc="upper left", fontsize=14)
    ax.grid(True, alpha=0.3, zorder=0)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    fig.tight_layout()
    return fig, ax


# ─── 4-panel cluster comparison ─────────────────────────────────────


def _bar_panel(
    ax: plt.Axes,
    x: np.ndarray,
    means: np.ndarray,
    sems: np.ndarray,
    cluster_ids: Sequence[int],
    pvalues: Dict[str, float],
    var_names: List[str],
    display_names: List[str],
    *,
    ylabel: str,
    title: str,
    one_sided_error: bool = True,
    show_legend: bool = False,
) -> None:
    """Internal grouped-bar helper (shared by all four panels)."""
    n_cl = len(cluster_ids)
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(n_cl)]
    width = 0.8 / n_cl

    for i, cid in enumerate(cluster_ids):
        offset = (i - n_cl / 2 + 0.5) * width
        m = means[i]
        s = sems[i]
        if one_sided_error:
            lo = np.where(m >= 0, 0, s)
            hi = np.where(m >= 0, s, 0)
        else:
            lo = hi = s
        ax.bar(
            x + offset, m, width,
            yerr=np.array([lo, hi]),
            color=colors[i], edgecolor="black", linewidth=0.4, alpha=0.85,
            label=f"Cluster {cid}",
            error_kw=dict(linewidth=1.0, ecolor="black", capsize=2),
        )

    # significance stars
    for j, var in enumerate(var_names):
        p = pvalues.get(var, 1.0)
        s = _stars(p)
        if not s:
            continue
        tops = []
        for i in range(n_cl):
            mv = means[i, j]
            sv = sems[i, j]
            tops.append(mv + sv if mv >= 0 else mv - sv)
        max_top = max(tops)
        idx_max = int(np.argmax(tops))
        bar_x = x[j] + (idx_max - n_cl / 2 + 0.5) * width + width / 2
        ax.text(bar_x + 0.02, max_top, s, ha="left", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=35, ha="right", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.grid(axis="y", alpha=0.25, ls="--")
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    if show_legend:
        ax.legend(fontsize=10, loc="upper left")


def plot_cluster_comparison(
    raw_env: pd.DataFrame,
    taxa_octave: pd.DataFrame,
    cluster_labels_all: pd.Series,
    ref_mask: pd.Series,
    env_variables: Sequence[str],
    *,
    taxa_order: Sequence[str] | None = None,
    anova_transform: str = "none",
    figsize: Tuple[float, float] = (20, 14),
    dpi: int = 300,
) -> Tuple[plt.Figure, np.ndarray]:
    """4-panel cluster comparison figure.

    Parameters
    ----------
    raw_env : pd.DataFrame
        Environmental data for ALL sites (original scale).
    taxa_octave : pd.DataFrame
        Taxa data for ALL sites in octave scale.
    cluster_labels_all : pd.Series
        Cluster label for every site (ref = from Stage 2, non-ref = predicted).
    ref_mask : pd.Series[bool]
        ``True`` for reference sites.
    env_variables : list of str
        Which env columns to plot.
    taxa_order : list of str or None
        Display order for taxa (default: ``TAXA_DISPLAY_ORDER``).
    anova_transform : str
        ``"none"`` / ``"log"`` / ``"boxcox"`` for ANOVA pre-transformation.
    figsize, dpi : tuple, int

    Returns
    -------
    fig, axes  (2 × 2)
    """
    if taxa_order is None:
        taxa_order = [t for t in TAXA_DISPLAY_ORDER if t in taxa_octave.columns]
    else:
        taxa_order = [t for t in taxa_order if t in taxa_octave.columns]

    env_vars_present = [v for v in env_variables if v in raw_env.columns]

    # aligned masks
    sites_with_cluster = cluster_labels_all.dropna().index
    ref_idx = ref_mask[ref_mask].index.intersection(sites_with_cluster)
    nonref_idx = ref_mask[~ref_mask].index.intersection(sites_with_cluster)
    cluster_ids = sorted(cluster_labels_all.dropna().unique())

    # ── taxa relative abundance ──────────────────────────────────────
    taxa_relabd = octave_to_relative_abundance(taxa_octave[taxa_order])

    # ── z-scored env (all sites with clusters) ───────────────────────
    env_sub = raw_env.loc[sites_with_cluster, env_vars_present].copy()

    # ── helper: means + sems per cluster for a subset ────────────────
    def _cluster_stats(idx, df):
        ms, ss = [], []
        for cid in cluster_ids:
            cidx = cluster_labels_all.loc[idx]
            c_idx = cidx[cidx == cid].index.intersection(df.index)
            sub = df.loc[c_idx]
            ms.append(sub.mean().values)
            ss.append(sub.sem().fillna(0).values)
        return np.array(ms), np.array(ss)

    # ── ANOVA / t-test helpers ───────────────────────────────────────
    def _anova_pvals(idx, df):
        pvals: Dict[str, float] = {}
        labels = cluster_labels_all.loc[idx]
        for col in df.columns:
            groups = [df.loc[labels[labels == c].index.intersection(df.index), col].dropna().values
                      for c in cluster_ids]
            if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
                try:
                    _, p = sp_stats.f_oneway(*groups)
                    pvals[col] = p
                except Exception:
                    pvals[col] = 1.0
            else:
                pvals[col] = 1.0
        return pvals

    def _ttest_diff_pvals(df):
        """One-sample t-test on (nonref − ref) mean difference per cluster."""
        pvals: Dict[str, float] = {}
        for col in df.columns:
            diffs: list[float] = []
            for cid in cluster_ids:
                r_idx = cluster_labels_all.loc[ref_idx]
                nr_idx = cluster_labels_all.loc[nonref_idx]
                r_vals = df.loc[r_idx[r_idx == cid].index.intersection(df.index), col].dropna()
                nr_vals = df.loc[nr_idx[nr_idx == cid].index.intersection(df.index), col].dropna()
                if len(r_vals) >= 1 and len(nr_vals) >= 1:
                    diffs.append(nr_vals.mean() - r_vals.mean())
            if len(diffs) >= 2:
                try:
                    _, p = sp_stats.ttest_1samp(diffs, 0)
                    pvals[col] = p
                except Exception:
                    pvals[col] = 1.0
            else:
                # fall back to independent t-test pooling across clusters
                r_all = df.loc[ref_idx.intersection(df.index), col].dropna()
                nr_all = df.loc[nonref_idx.intersection(df.index), col].dropna()
                if len(r_all) >= 2 and len(nr_all) >= 2:
                    try:
                        _, p = sp_stats.ttest_ind(nr_all, r_all)
                        pvals[col] = p
                    except Exception:
                        pvals[col] = 1.0
                else:
                    pvals[col] = 1.0
        return pvals

    # ── compute stats ────────────────────────────────────────────────
    # Panel A: z-scored env across ALL clustered sites
    env_z_means, env_z_sems = [], []
    for cid in cluster_ids:
        m_row, s_row = [], []
        for var in env_vars_present:
            vals_all = env_sub[var].dropna()
            mu, sigma = vals_all.mean(), vals_all.std(ddof=1)
            sigma = sigma if sigma > 0 else 1.0
            c_mask = cluster_labels_all.loc[sites_with_cluster] == cid
            c_vals = env_sub.loc[c_mask[c_mask].index.intersection(env_sub.index), var].dropna()
            z = (c_vals - mu) / sigma
            m_row.append(z.mean())
            s_row.append(z.sem() if len(z) > 1 else 0.0)
        env_z_means.append(m_row)
        env_z_sems.append(s_row)
    env_z_means = np.array(env_z_means)
    env_z_sems = np.array(env_z_sems)

    env_pvals = _anova_pvals(sites_with_cluster, env_sub)

    # Panel B: ref taxa relative abundance
    ref_tax_means, ref_tax_sems = _cluster_stats(ref_idx, taxa_relabd)
    ref_tax_pvals = _anova_pvals(ref_idx, taxa_octave[taxa_order])

    # Panel C: non-ref taxa relative abundance
    nonref_tax_means, nonref_tax_sems = _cluster_stats(nonref_idx, taxa_relabd)
    nonref_tax_pvals = _anova_pvals(nonref_idx, taxa_octave[taxa_order])

    # Panel D: difference (non-ref − ref)
    diff_means = nonref_tax_means - ref_tax_means
    diff_sems = np.sqrt(ref_tax_sems ** 2 + nonref_tax_sems ** 2)
    diff_pvals = _ttest_diff_pvals(taxa_relabd)

    # ── create figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)

    env_display = [_ENV_SHORT.get(v, v) for v in env_vars_present]

    _bar_panel(
        axes[0, 0], np.arange(len(env_vars_present)),
        env_z_means, env_z_sems, cluster_ids,
        env_pvals, env_vars_present, env_display,
        ylabel="Mean z-score (± SEM)",
        title="(A) Standardized Habitat Features Across Clusters (All Sites)",
        show_legend=True,
    )

    _bar_panel(
        axes[0, 1], np.arange(len(taxa_order)),
        ref_tax_means, ref_tax_sems, cluster_ids,
        ref_tax_pvals, taxa_order, taxa_order,
        ylabel="Mean Relative Abundance ± SE (%)",
        title="(B) Reference Sites: Taxa by Cluster",
    )

    _bar_panel(
        axes[1, 0], np.arange(len(taxa_order)),
        nonref_tax_means, nonref_tax_sems, cluster_ids,
        nonref_tax_pvals, taxa_order, taxa_order,
        ylabel="Mean Relative Abundance ± SE (%)",
        title="(C) Non-Reference Sites: Taxa by Cluster",
    )

    _bar_panel(
        axes[1, 1], np.arange(len(taxa_order)),
        diff_means, diff_sems, cluster_ids,
        diff_pvals, taxa_order, taxa_order,
        ylabel="Difference of Relative Abundance ± SE (%)",
        title="(D) Average Difference (Non-Ref − Ref)",
        one_sided_error=True,
    )

    # add t-test annotation on panel D
    axes[1, 1].text(
        0.98, 0.95,
        r"(One-Sample $t$-test for $H_0$: $\Delta\mu = 0$)",
        transform=axes[1, 1].transAxes,
        fontsize=11, fontstyle="italic", ha="right", va="top",
    )

    fig.tight_layout()
    return fig, axes
