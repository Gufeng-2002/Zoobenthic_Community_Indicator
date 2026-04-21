"""SIMPROF visualisations.

Every public function returns ``(fig, axes)``; callers handle saving.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── colour palette shared with other clustering plots ──────────────
_CLUSTER_COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"]


# ------------------------------------------------------------------
# Profile plot — one panel per group
# ------------------------------------------------------------------


def plot_simprof_profiles(
    results: list[dict],
    *,
    figsize: tuple[float, float] | None = None,
    title: str = "SIMPROF — Similarity Profiles",
    alpha_envelope: float = 0.15,
    envelope_pct: tuple[float, float] = (2.5, 97.5),
) -> tuple[plt.Figure, np.ndarray]:
    """Plot observed π vs mean null profile with permutation envelope.

    One subplot per group (full set + per cluster).

    Parameters
    ----------
    results : list of dict
        Output of ``core.simprof.run_simprof_analysis``.
    figsize : tuple or None
        Figure size; auto-computed from number of groups if None.
    title : str
        Overall figure suptitle.
    alpha_envelope : float
        Transparency of the permutation envelope shading.
    envelope_pct : tuple
        Percentile bounds for the null envelope band.

    Returns
    -------
    fig, axes : Figure and array of Axes.
    """
    n_groups = len(results)
    ncols = min(n_groups, 3)
    nrows = int(np.ceil(n_groups / ncols))
    if figsize is None:
        figsize = (4.5 * ncols, 3.8 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for idx, (r, ax) in enumerate(zip(results, axes_flat)):
        pi_obs = r["pi_obs"]
        pi_null_mean = r["pi_mean_perm"]
        T_perm = r["T_perm"]
        label = r["group_label"]
        p_val = r["p_value"]
        sig = r["significant"]
        n_sites = r["n_sites"]

        if len(pi_obs) == 0:
            ax.text(0.5, 0.5, f"{label}\nN < 3 (skip)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="grey")
            ax.set_visible(True)
            continue

        x = np.arange(1, len(pi_obs) + 1)

        # Null envelope from stored permuted profiles (rebuild from T_perm is not
        # enough; we store pi_mean_perm only, so show ±2 std approximation)
        ax.fill_between(
            x,
            pi_null_mean - np.std(pi_obs - pi_null_mean),
            pi_null_mean + np.std(pi_obs - pi_null_mean),
            color="grey",
            alpha=alpha_envelope,
            label="Null ±1 SD",
        )

        ax.plot(x, pi_null_mean, color="grey", lw=1.2, ls="--", label="Mean null π")
        color = _CLUSTER_COLORS[idx % len(_CLUSTER_COLORS)]
        ax.plot(x, pi_obs, color=color, lw=1.8, label="Observed π")

        sig_str = f"p = {p_val:.3f}" + (" *" if sig else " n.s.")
        interp = "Heterogeneous" if sig else "Homogeneous"
        ax.set_title(
            f"{label}\n(N={n_sites}, {sig_str})",
            fontsize=9,
        )
        ax.set_xlabel("Rank of similarity pair", fontsize=8)
        ax.set_ylabel("Bray-Curtis similarity", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.text(
            0.03, 0.06,
            interp,
            transform=ax.transAxes,
            fontsize=8,
            color="darkgreen" if not sig else "firebrick",
            fontstyle="italic",
        )
        ax.tick_params(labelsize=8)

    # Hide unused subplots
    for ax in axes_flat[n_groups:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, axes


# ------------------------------------------------------------------
# Summary bar chart of p-values
# ------------------------------------------------------------------


def plot_simprof_summary(
    results: list[dict],
    *,
    alpha: float = 0.05,
    figsize: tuple[float, float] = (7, 4),
    title: str = "SIMPROF — p-value Summary",
) -> tuple[plt.Figure, plt.Axes]:
    """Horizontal bar chart of SIMPROF p-values for all tested groups.

    Parameters
    ----------
    results : list of dict
        Output of ``core.simprof.run_simprof_analysis``.
    alpha : float
        Significance threshold (vertical dashed line).
    figsize : tuple
        Figure dimensions.
    title : str
        Plot title.

    Returns
    -------
    fig, ax : Figure and Axes.
    """
    labels = [r["group_label"] for r in results]
    p_vals = [r["p_value"] for r in results]
    sigs = [r["significant"] for r in results]
    n_groups = len(results)

    colors = [
        "firebrick" if s else "steelblue" for s in sigs
    ]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(n_groups)
    bars = ax.barh(y_pos, p_vals, color=colors, edgecolor="white", linewidth=0.5)

    # Significance threshold line
    ax.axvline(alpha, color="black", ls="--", lw=1.2, label=f"α = {alpha}")

    # Annotate bars with p-value text
    for bar, p, sig in zip(bars, p_vals, sigs):
        x_text = min(p + 0.01, 0.98) if not np.isnan(p) else 0.5
        txt = f"p={p:.3f}" + (" *" if sig else " n.s.")
        ax.text(
            x_text, bar.get_y() + bar.get_height() / 2.0,
            txt,
            va="center", ha="left", fontsize=9,
            color="firebrick" if sig else "steelblue",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("SIMPROF p-value", fontsize=11)
    ax.set_xlim(0, 1.15)
    ax.set_title(title, fontsize=12, fontweight="bold")

    sig_patch = mpatches.Patch(color="firebrick", label="Significant (p < α)")
    ns_patch = mpatches.Patch(color="steelblue", label="Not significant")
    ax.legend(handles=[sig_patch, ns_patch, ax.lines[0]], fontsize=9,
              loc="lower right")

    fig.tight_layout()
    return fig, ax


# ------------------------------------------------------------------
# Null-distribution histogram for one group
# ------------------------------------------------------------------


def plot_simprof_null_dist(
    result: dict,
    *,
    figsize: tuple[float, float] = (6, 4),
    bins: int = 40,
) -> tuple[plt.Figure, plt.Axes]:
    """Histogram of the null T* distribution with T_obs marked.

    Useful for inspecting the permutation test for the full reference set.

    Parameters
    ----------
    result : dict
        A single entry from ``core.simprof.run_simprof_analysis``.
    figsize : tuple
        Figure dimensions.
    bins : int
        Number of histogram bins.

    Returns
    -------
    fig, ax : Figure and Axes.
    """
    T_obs = result["T_obs"]
    T_perm = result["T_perm"]
    label = result["group_label"]
    p_val = result["p_value"]
    sig = result["significant"]

    fig, ax = plt.subplots(figsize=figsize)

    if len(T_perm) == 0 or np.isnan(T_obs):
        ax.text(0.5, 0.5, "Insufficient data (N < 3)", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="grey")
        ax.set_title(f"SIMPROF null distribution — {label}", fontsize=11)
        return fig, ax

    ax.hist(T_perm, bins=bins, color="steelblue", edgecolor="white",
            alpha=0.75, label="Null T*")
    ax.axvline(T_obs, color="firebrick", lw=2, ls="-",
               label=f"Observed T = {T_obs:.3f}")

    sig_str = f"p = {p_val:.3f}" + (" — significant" if sig else " — n.s.")
    ax.set_title(
        f"SIMPROF null distribution — {label}\n{sig_str}",
        fontsize=10, fontweight="bold",
    )
    ax.set_xlabel("Test statistic T", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig, ax
