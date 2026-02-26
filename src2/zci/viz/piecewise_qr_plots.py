"""Visualisation for piecewise quantile regression results.

Accepts computed results, returns ``(fig, axes)`` — no data loading here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

from ..core.piecewise_qr import PiecewiseQRResult, predict, SubsampleResult


# ═════════════════════════════════════════════════════════════════════
# 1.  Error-bar plot: CI of all parameters across quantile levels
# ═════════════════════════════════════════════════════════════════════


def plot_quantile_ci_errorbars(
    qr_results: Dict[float, PiecewiseQRResult],
    *,
    cluster_id: int = 0,
    settings_text: str = "",
    figsize: Tuple[float, float] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Error-bar plot of coefficients + breakpoint CIs across τ levels.

    One subplot per parameter.  Horizontal axis = quantile level τ,
    vertical axis = parameter value with CI error bars.

    Parameters
    ----------
    qr_results : dict  {tau → PiecewiseQRResult}
    cluster_id : int   for the title
    settings_text : str
        Annotation string with bootstrap / grid settings to display.
    figsize : tuple

    Returns
    -------
    (fig, axes)
    """
    taus = sorted(qr_results.keys())
    first = qr_results[taus[0]]
    param_names = first.param_names
    n_params = len(param_names)

    # 2×2 layout when 4 parameters, otherwise single row
    if n_params == 4:
        n_rows, n_cols = 2, 2
        if figsize is None:
            figsize = (12, 9)
    else:
        n_rows, n_cols = 1, n_params
        if figsize is None:
            figsize = (5 * n_params, 4.5)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for j, pname in enumerate(param_names):
        ax = axes_flat[j]
        vals = np.array([qr_results[t].all_params[j] for t in taus])
        lo = np.array([qr_results[t].ci_lower[j] for t in taus])
        hi = np.array([qr_results[t].ci_upper[j] for t in taus])

        err_lo = np.maximum(0, vals - lo)
        err_hi = np.maximum(0, hi - vals)

        ax.errorbar(
            taus, vals,
            yerr=[err_lo, err_hi],
            fmt="o", capsize=3, capthick=1, markersize=4,
            color="tab:blue", ecolor="tab:blue", alpha=0.8,
        )
        ax.set_xlabel("Quantile level τ", fontsize=10)
        ax.set_ylabel(pname, fontsize=10)
        ax.set_title(pname, fontsize=11, fontweight="bold")
        ax.xaxis.set_major_locator(MaxNLocator(integer=False, nbins=8))
        ax.grid(True, ls=":", alpha=0.4)

    # hide unused axes
    for j in range(n_params, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # suptitle with settings
    title = f"Cluster {cluster_id} — Piecewise QR: Parameter CIs across Quantile Levels"
    if settings_text:
        title += f"\n{settings_text}"
    fig.suptitle(title, fontsize=12, y=1.04)

    fig.tight_layout()
    return fig, axes_flat


# ═════════════════════════════════════════════════════════════════════
# 2.  Three-panel QR plot (τ = 0.2, 0.5, 0.8) with CI shadow
# ═════════════════════════════════════════════════════════════════════


def plot_three_quantiles(
    x: np.ndarray,
    y: np.ndarray,
    qr_results: Dict[float, PiecewiseQRResult],
    *,
    highlight_taus: Sequence[float] = (0.20, 0.50, 0.80),
    cluster_id: int = 0,
    settings_text: str = "",
    x_label: str = "Pollution Score",
    y_label: str = "ZCI",
    figsize: Tuple[float, float] = (18, 5.5),
    n_grid: int = 200,
) -> Tuple[plt.Figure, np.ndarray]:
    """Three-panel scatter + piecewise QR fit with CI shadow.

    Each panel shows one quantile level with:
    - scatter of (x, y)
    - fitted piecewise-linear line
    - filled shadow for the bootstrap CI of the fitted curve
    - vertical dashed line at the breakpoint with CI band

    Parameters
    ----------
    x, y : arrays
    qr_results : dict  {tau → PiecewiseQRResult}
    highlight_taus : 3-tuple of float
    cluster_id : int
    x_label, y_label : str
    figsize : tuple
    n_grid : int   grid density for the fitted curve

    Returns
    -------
    (fig, axes)
    """
    colors = {0.20: "tab:blue", 0.50: "tab:green", 0.80: "tab:red"}
    n_panels = len(highlight_taus)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False, sharey=True)
    axes = axes.ravel()

    x_grid = np.linspace(x.min(), x.max(), n_grid)

    for i, tau in enumerate(highlight_taus):
        ax = axes[i]
        color = colors.get(tau, f"C{i}")

        # scatter
        ax.scatter(x, y, c="#bbbbbb", edgecolor="k", linewidth=0.3,
                   s=25, alpha=0.5, zorder=1, label="data")

        res = qr_results.get(tau)
        if res is None:
            ax.set_title(f"τ = {tau}  (not fitted)", fontsize=12)
            continue

        # fitted line
        y_fit = predict(x_grid, res.coefficients, res.breakpoints)
        ax.plot(x_grid, y_fit, color=color, lw=2.5, zorder=3,
                label=f"τ = {tau:.2f}")

        # CI shadow: generate prediction from bootstrap coefficients
        boot_preds = np.empty((res.boot_coefs.shape[0], n_grid))
        for b in range(res.boot_coefs.shape[0]):
            if np.any(np.isnan(res.boot_coefs[b])):
                boot_preds[b] = np.nan
                continue
            boot_preds[b] = predict(x_grid, res.boot_coefs[b], res.boot_bps[b])

        lo = np.nanquantile(boot_preds, 0.05, axis=0)
        hi = np.nanquantile(boot_preds, 0.95, axis=0)
        ax.fill_between(x_grid, lo, hi, color=color, alpha=0.15, zorder=2,
                        label="90 % CI")

        # breakpoint marker + CI band
        for j, bp in enumerate(res.breakpoints):
            bp_idx = len(res.coefficients) + j
            bp_lo = res.ci_lower[bp_idx]
            bp_hi = res.ci_upper[bp_idx]
            ax.axvline(bp, color=color, ls="--", lw=1.5, alpha=0.7, zorder=4)
            ax.axvspan(bp_lo, bp_hi, color=color, alpha=0.08, zorder=0)
            ax.text(bp, ax.get_ylim()[1] * 0.98, f"ψ={bp:.2f}",
                    ha="center", va="top", fontsize=8, color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        ax.set_xlabel(x_label, fontsize=11)
        if i == 0:
            ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(f"τ = {tau:.2f}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        ax.grid(True, ls=":", alpha=0.3)

    # suptitle with settings
    title = (
        f"Cluster {cluster_id} — Piecewise Quantile Regression  "
        f"({y_label} vs {x_label})"
    )
    if settings_text:
        title += f"\n{settings_text}"
    fig.suptitle(title, fontsize=13, y=1.05)

    fig.tight_layout()
    return fig, axes


# ═════════════════════════════════════════════════════════════════════
# 3.  Sensitivity analysis plots
# ═════════════════════════════════════════════════════════════════════


def plot_sensitivity(
    sensitivity_results: Dict[float, SubsampleResult],
    true_params: np.ndarray,
    *,
    cluster_id: int = 0,
    settings_text: str = "",
    figsize: Tuple[float, float] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Sample-size sensitivity: error-bar plots with true parameter marks.

    One subplot per parameter.  X-axis = subsample fraction, Y-axis =
    parameter estimate.  Error bars show the mean CI across repeats,
    shaded band shows the spread across repeats, and the horizontal
    dashed line marks the full-data ("true") estimate.

    Parameters
    ----------
    sensitivity_results : dict  {frac → SubsampleResult}
    true_params : (n_params,) array
    cluster_id : int
    settings_text : str
        Annotation string with settings to display.
    figsize : tuple

    Returns
    -------
    (fig, axes)
    """
    fracs = sorted(sensitivity_results.keys())
    first = sensitivity_results[fracs[0]]
    param_names = first.param_names
    n_params = len(param_names)

    # 2×2 layout when 4 parameters
    if n_params == 4:
        n_cols, n_rows = 2, 2
        if figsize is None:
            figsize = (12, 9)
    else:
        n_cols = min(n_params, 4)
        n_rows = (n_params - 1) // n_cols + 1
        if figsize is None:
            figsize = (5 * n_cols, 4.5 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for j in range(n_params):
        ax = axes_flat[j]
        pname = param_names[j]

        # collect per-fraction statistics
        frac_arr = np.array(fracs)
        medians = np.empty(len(fracs))
        q25 = np.empty(len(fracs))
        q75 = np.empty(len(fracs))
        ci_lo_mean = np.empty(len(fracs))
        ci_hi_mean = np.empty(len(fracs))
        coverage = np.empty(len(fracs))

        for fi, frac in enumerate(fracs):
            sr = sensitivity_results[frac]
            ests = sr.param_estimates[:, j]
            good = ~np.isnan(ests)
            if good.sum() == 0:
                medians[fi] = q25[fi] = q75[fi] = np.nan
                ci_lo_mean[fi] = ci_hi_mean[fi] = np.nan
                coverage[fi] = 0
                continue

            medians[fi] = np.nanmedian(ests)
            q25[fi] = np.nanquantile(ests, 0.25)
            q75[fi] = np.nanquantile(ests, 0.75)
            ci_lo_mean[fi] = np.nanmean(sr.ci_lowers[:, j])
            ci_hi_mean[fi] = np.nanmean(sr.ci_uppers[:, j])

            # coverage: fraction of repeats whose CI contains true_params[j]
            covers = (sr.ci_lowers[:, j] <= true_params[j]) & \
                     (sr.ci_uppers[:, j] >= true_params[j])
            coverage[fi] = np.nanmean(covers)

        # Mean CI as error bars
        err_lo = medians - ci_lo_mean
        err_hi = ci_hi_mean - medians

        ax.errorbar(
            frac_arr * 100, medians,
            yerr=[np.abs(err_lo), np.abs(err_hi)],
            fmt="o", capsize=3, capthick=1, markersize=4,
            color="tab:blue", ecolor="tab:blue", alpha=0.7,
            label="median ± mean CI",
        )

        # True parameter
        ax.axhline(true_params[j], color="tab:red", ls="--", lw=2,
                    label=f"full-data = {true_params[j]:.3f}")

        ax.set_xlabel("Subsample size (%)", fontsize=10)
        ax.set_ylabel(pname, fontsize=10)
        ax.set_title(pname, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="best", framealpha=0.9)
        ax.grid(True, ls=":", alpha=0.3)

    # hide unused axes
    for j in range(n_params, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # suptitle with settings
    title = f"Cluster {cluster_id} — Subsample Sensitivity Analysis"
    if settings_text:
        title += f"\n{settings_text}"
    fig.suptitle(title, fontsize=12, y=1.04)

    fig.tight_layout()
    return fig, axes


def plot_sensitivity_coverage(
    sensitivity_results: Dict[float, SubsampleResult],
    true_params: np.ndarray,
    *,
    cluster_id: int = 0,
    settings_text: str = "",
    figsize: Tuple[float, float] = (8, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Coverage probability plot: fraction of repeats whose CI contains
    the full-data ("true") parameter, as a function of subsample size.

    Returns
    -------
    (fig, ax)
    """
    fracs = sorted(sensitivity_results.keys())
    first = sensitivity_results[fracs[0]]
    param_names = first.param_names
    n_params = len(param_names)

    fig, ax = plt.subplots(figsize=figsize)
    frac_arr = np.array(fracs) * 100

    for j in range(n_params):
        cov = np.empty(len(fracs))
        for fi, frac in enumerate(fracs):
            sr = sensitivity_results[frac]
            covers = (sr.ci_lowers[:, j] <= true_params[j]) & \
                     (sr.ci_uppers[:, j] >= true_params[j])
            cov[fi] = np.nanmean(covers)

        ax.plot(frac_arr, cov * 100, "o-", label=param_names[j],
                markersize=4, alpha=0.8)

    ax.axhline(90, color="grey", ls="--", lw=1, alpha=0.6, label="90 % target")
    ax.set_xlabel("Subsample size (%)", fontsize=11)
    ax.set_ylabel("Coverage probability (%)", fontsize=11)
    # title with settings
    title = f"Cluster {cluster_id} — CI Coverage of Full-Data Parameters"
    if settings_text:
        title += f"\n{settings_text}"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=8, ncol=2, loc="lower right", framealpha=0.9)
    ax.grid(True, ls=":", alpha=0.3)

    fig.tight_layout()
    return fig, ax
