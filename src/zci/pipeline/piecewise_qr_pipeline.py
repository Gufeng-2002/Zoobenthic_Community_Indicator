"""Stage 4 — Piecewise Quantile Regression with Wild-Bootstrap CIs.

Orchestrates:
  read Stage 1 (pollution scores) → read Stage 2 (cluster labels) →
  read Stage 3 (ZCI) → merge → per-cluster piecewise QR at multiple
  quantile levels → wild-bootstrap CIs → sample-size sensitivity →
  save tables, figures, artifacts.

Outputs
-------
tables/
    qr_coefficients_cluster_{cl}.xlsx   (all τ levels with CIs)
    sensitivity_cluster_{cl}.xlsx       (coverage & estimates)
figures/
    qr_ci_errorbars_cluster_{cl}.png
    qr_three_quantiles_cluster_{cl}.png
    sensitivity_cluster_{cl}.png
    sensitivity_coverage_cluster_{cl}.png
artifacts/
    04_updated_data.xlsx
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..io.readers import read_study_data, extract_block
from ..io.writers import save_table, save_figure
from ..core.piecewise_qr import (
    fit_multi_quantile,
    subsample_sensitivity,
    PiecewiseQRResult,
)
from ..models.piecewise_qr import ClusterQRResult, PQRPipelineResult
from ..viz.piecewise_qr_plots import (
    plot_quantile_ci_errorbars,
    plot_three_quantiles,
    plot_sensitivity,
    plot_sensitivity_coverage,
)


# ─── main pipeline ──────────────────────────────────────────────────


def pqr_pipeline(
    stage1_artifact: str | Path,
    stage2_artifact: str | Path,
    stage3_artifact: str | Path,
    output_dir: str | Path,
    *,
    # QR configuration
    n_breakpoints: int = 1,
    taus: Sequence[float] | None = None,
    highlight_taus: Sequence[float] = (0.20, 0.50, 0.80),
    n_boot: int = 500,
    confidence: float = 0.90,
    grid_size: int = 50,
    search_range: tuple = (0.10, 0.90),
    # Sensitivity configuration
    run_sensitivity: bool = True,
    sensitivity_tau: float = 0.50,
    sensitivity_fracs: Sequence[float] | None = None,
    sensitivity_repeats: int = 30,
    sensitivity_boot: int = 200,
    # Misc
    min_cluster_size: int = 15,
    random_state: int = 42,
    save_plots: bool = True,
    figure_formats: Sequence[str] = ("png",),
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> PQRPipelineResult:
    """Run the piecewise quantile regression pipeline.

    Parameters
    ----------
    stage1_artifact : path
        ``01_updated_data.xlsx`` → Pollution_Score
    stage2_artifact : path
        ``02_predicted_data.xlsx`` → Predicted_Cluster
    stage3_artifact : path
        ``03_updated_data.xlsx`` → ZCI
    output_dir : path
        Root for outputs (``tables/``, ``figures/``, ``artifacts/``).
    n_breakpoints : int
        Number of breakpoints in the piecewise model (default 1).
    taus : sequence of float or None
        Quantile levels.  Default: 0.10, 0.15, …, 0.90.
    highlight_taus : tuple
        Quantile levels for the three-panel plot (default 0.20, 0.50, 0.80).
    n_boot : int
        Wild-bootstrap replicates for CI construction.
    confidence : float
        Confidence level (default 0.90 → 90 %).
    grid_size : int
        Grid density for breakpoint profile search.
    search_range : (float, float)
        Quantile range for the breakpoint grid.
    run_sensitivity : bool
        Whether to run the subsample sensitivity analysis.
    sensitivity_tau : float
        Quantile level used for the sensitivity analysis.
    sensitivity_fracs : sequence of float or None
        Subsample fractions.  Default: 0.15, 0.20, …, 1.00.
    sensitivity_repeats : int
        Repeats per subsample fraction (default 30).
    sensitivity_boot : int
        Bootstrap replicates per sensitivity fit (default 200).
    min_cluster_size : int
        Skip clusters with fewer sites.
    random_state : int
    save_plots : bool
    figure_formats, table_formats : sequence of str
    verbose : bool

    Returns
    -------
    PQRPipelineResult
    """
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    artifacts_dir = output_dir / "artifacts"

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # Default taus
    if taus is None:
        taus = list(np.round(np.arange(0.10, 0.91, 0.05), 2))

    # Default sensitivity fracs
    if sensitivity_fracs is None:
        sensitivity_fracs = list(np.round(np.arange(0.15, 1.01, 0.05), 2))

    # Build annotation strings for figures (multi-line for title)
    qr_settings_text = (
        f"n_boot={n_boot},  grid_size={grid_size},  "
        f"search_range={search_range}\n"
        f"confidence={confidence:.0%},  n_breakpoints={n_breakpoints},  "
        f"n_quantiles={len(taus)}"
    )
    sens_settings_text = (
        f"sensitivity_repeats={sensitivity_repeats},  "
        f"sensitivity_boot={sensitivity_boot}\n"
        f"n_fracs={len(sensitivity_fracs)},  grid_size={grid_size},  "
        f"confidence={confidence:.0%}"
    )

    # ── 1. Read artifacts ────────────────────────────────────────────
    _log("[1/7] Reading stage artifacts …")
    s1 = pd.read_excel(stage1_artifact, header=[0, 1, 2], index_col=0)
    s2 = pd.read_excel(stage2_artifact, header=[0, 1, 2], index_col=0)
    s3 = pd.read_excel(stage3_artifact, header=[0, 1, 2], index_col=0)

    pollution = s1.xs("Pollution_Score", level=2, axis=1).iloc[:, 0]
    pollution.name = "Pollution_Score"
    cluster = s2.xs("Predicted_Cluster", level=2, axis=1).iloc[:, 0]
    cluster.name = "Cluster"
    zci = s3.xs("ZCI", level=2, axis=1).iloc[:, 0]
    zci.name = "ZCI"

    # ── 2. Merge ─────────────────────────────────────────────────────
    _log("[2/7] Merging data …")
    merged = pd.concat([pollution, cluster, zci], axis=1).dropna()
    merged["Cluster"] = merged["Cluster"].astype(int)
    _log(f"       {len(merged)} sites, "
         f"clusters: {sorted(merged['Cluster'].unique())}")

    clusters = sorted(merged["Cluster"].unique())
    viable_clusters = [
        cl for cl in clusters if (merged["Cluster"] == cl).sum() >= min_cluster_size
    ]
    _log(f"       viable (≥ {min_cluster_size} sites): {viable_clusters}")

    # ── 3. Per-cluster piecewise QR ──────────────────────────────────
    _log("[3/7] Fitting piecewise quantile regressions per cluster …")
    cluster_results: Dict[int, ClusterQRResult] = {}

    for cl in viable_clusters:
        mask = merged["Cluster"] == cl
        x = merged.loc[mask, "Pollution_Score"].values.astype(float)
        y = merged.loc[mask, "ZCI"].values.astype(float)
        n_sites = len(x)
        _log(f"\n  ── Cluster {cl}  ({n_sites} sites) ──")

        qr = fit_multi_quantile(
            x, y,
            taus=taus,
            n_breakpoints=n_breakpoints,
            n_boot=n_boot,
            confidence=confidence,
            grid_size=grid_size,
            search_range=search_range,
            random_state=random_state,
            verbose=verbose,
        )

        cr = ClusterQRResult(
            cluster_id=cl,
            n_sites=n_sites,
            n_taxa=0,
            quantile_results=qr,
            param_names=qr[taus[0]].param_names,
        )
        cluster_results[cl] = cr

    # ── 4. Save tables (coefficients) ────────────────────────────────
    _log("\n[4/7] Saving coefficient tables …")
    for cl in viable_clusters:
        cr = cluster_results[cl]
        rows = []
        for tau in taus:
            res = cr.quantile_results[tau]
            for j, pname in enumerate(res.param_names):
                rows.append({
                    "tau": tau,
                    "parameter": pname,
                    "estimate": res.all_params[j],
                    "ci_lower": res.ci_lower[j],
                    "ci_upper": res.ci_upper[j],
                })
        tbl = pd.DataFrame(rows)
        save_table(
            tbl, tables_dir / f"qr_coefficients_cluster_{cl}",
            formats=table_formats, verbose=verbose,
        )

    # ── 5. Figures (QR) ──────────────────────────────────────────────
    if save_plots:
        _log("[5/7] Creating QR figures …")
        for cl in viable_clusters:
            cr = cluster_results[cl]
            mask = merged["Cluster"] == cl
            x = merged.loc[mask, "Pollution_Score"].values.astype(float)
            y = merged.loc[mask, "ZCI"].values.astype(float)

            # Error-bar plot
            fig_eb, _ = plot_quantile_ci_errorbars(
                cr.quantile_results, cluster_id=cl,
                settings_text=qr_settings_text,
            )
            save_figure(fig_eb, figures_dir / f"qr_ci_errorbars_cluster_{cl}",
                        formats=figure_formats, verbose=verbose)
            plt.close(fig_eb)

            # Three-panel QR
            fig_3q, _ = plot_three_quantiles(
                x, y, cr.quantile_results,
                highlight_taus=highlight_taus,
                cluster_id=cl,
                settings_text=qr_settings_text,
            )
            save_figure(fig_3q, figures_dir / f"qr_three_quantiles_cluster_{cl}",
                        formats=figure_formats, verbose=verbose)
            plt.close(fig_3q)

    # ── 6. Sensitivity analysis ──────────────────────────────────────
    if run_sensitivity:
        _log("[6/7] Running subsample sensitivity analysis …")
        for cl in viable_clusters:
            mask = merged["Cluster"] == cl
            x = merged.loc[mask, "Pollution_Score"].values.astype(float)
            y = merged.loc[mask, "ZCI"].values.astype(float)

            _log(f"\n  ── Cluster {cl} sensitivity (τ = {sensitivity_tau}) ──")
            sens, true_p = subsample_sensitivity(
                x, y,
                tau=sensitivity_tau,
                n_breakpoints=n_breakpoints,
                fracs=sensitivity_fracs,
                n_repeats=sensitivity_repeats,
                n_boot=sensitivity_boot,
                confidence=confidence,
                grid_size=grid_size,
                search_range=search_range,
                random_state=random_state,
                verbose=verbose,
            )

            cr = cluster_results[cl]
            cr.sensitivity_results = sens
            cr.true_params = true_p
            cr.sensitivity_tau = sensitivity_tau

            # Sensitivity table: coverage per fraction
            rows = []
            for frac in sorted(sens.keys()):
                sr = sens[frac]
                for j, pname in enumerate(sr.param_names):
                    covers = (sr.ci_lowers[:, j] <= true_p[j]) & \
                             (sr.ci_uppers[:, j] >= true_p[j])
                    rows.append({
                        "frac_pct": frac * 100,
                        "n_samples": sr.n_samples,
                        "parameter": pname,
                        "true_value": true_p[j],
                        "median_estimate": np.nanmedian(sr.param_estimates[:, j]),
                        "mean_ci_lower": np.nanmean(sr.ci_lowers[:, j]),
                        "mean_ci_upper": np.nanmean(sr.ci_uppers[:, j]),
                        "coverage_pct": np.nanmean(covers) * 100,
                    })
            tbl_s = pd.DataFrame(rows)
            save_table(
                tbl_s, tables_dir / f"sensitivity_cluster_{cl}",
                formats=table_formats, verbose=verbose,
            )

            # Sensitivity figures
            if save_plots:
                fig_s, _ = plot_sensitivity(
                    sens, true_p, cluster_id=cl,
                    settings_text=sens_settings_text,
                )
                save_figure(fig_s, figures_dir / f"sensitivity_cluster_{cl}",
                            formats=figure_formats, verbose=verbose)
                plt.close(fig_s)

                fig_cov, _ = plot_sensitivity_coverage(
                    sens, true_p, cluster_id=cl,
                    settings_text=sens_settings_text,
                )
                save_figure(fig_cov,
                            figures_dir / f"sensitivity_coverage_cluster_{cl}",
                            formats=figure_formats, verbose=verbose)
                plt.close(fig_cov)
    else:
        _log("[6/7] Sensitivity analysis skipped.")

    # ── 7. Save augmented artifact ───────────────────────────────────
    _log("\n[7/7] Saving augmented artifact …")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Build multi-index columns with breakpoint and slope info at median tau
    cols = pd.MultiIndex.from_tuples([
        ("04_piecewise_qr", "raw", "PQR_Breakpoint"),
        ("04_piecewise_qr", "raw", "PQR_Breakpoint_CI_Lower"),
        ("04_piecewise_qr", "raw", "PQR_Breakpoint_CI_Upper"),
        ("04_piecewise_qr", "raw", "PQR_Slope_Before"),
        ("04_piecewise_qr", "raw", "PQR_Slope_After"),
        ("04_piecewise_qr", "raw", "PQR_Tau"),
    ])
    aug = pd.DataFrame(index=merged.index, columns=cols)

    tau_ref = 0.50  # use median quantile for the artifact
    for cl in viable_clusters:
        cr = cluster_results[cl]
        res = cr.quantile_results.get(tau_ref)
        if res is None:
            continue
        mask = merged["Cluster"] == cl
        idx = merged.index[mask]
        bp = res.breakpoints[0]
        bp_idx = len(res.coefficients)  # index in ci vectors
        aug.loc[idx, ("04_piecewise_qr", "raw", "PQR_Breakpoint")] = bp
        aug.loc[idx, ("04_piecewise_qr", "raw", "PQR_Breakpoint_CI_Lower")] = \
            res.ci_lower[bp_idx]
        aug.loc[idx, ("04_piecewise_qr", "raw", "PQR_Breakpoint_CI_Upper")] = \
            res.ci_upper[bp_idx]
        aug.loc[idx, ("04_piecewise_qr", "raw", "PQR_Slope_Before")] = \
            res.coefficients[1]
        aug.loc[idx, ("04_piecewise_qr", "raw", "PQR_Slope_After")] = \
            res.coefficients[1] + res.coefficients[2]
        aug.loc[idx, ("04_piecewise_qr", "raw", "PQR_Tau")] = tau_ref

    aug_path = artifacts_dir / "04_updated_data.xlsx"
    aug.to_excel(aug_path)
    _log(f"  ✓ Saved augmented data: {aug_path}")

    # ── Return ───────────────────────────────────────────────────────
    result = PQRPipelineResult(
        cluster_results=cluster_results,
        clusters=viable_clusters,
        x_name="Pollution_Score",
        y_name="ZCI",
    )

    _log(f"\n✓ Piecewise QR pipeline complete.\n{result.summary()}")
    return result
