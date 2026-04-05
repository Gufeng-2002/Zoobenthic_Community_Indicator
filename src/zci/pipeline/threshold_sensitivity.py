"""Score-Focus Pipeline.

Orchestrates:
  read data → prepare env / taxa → sweep cut-offs for a single
  contamination score (Env + Stressor predictor sets) → comparison
  plots → variance partitioning → save figures + tables.

Outputs (under ``{score_label}_Focus/``)
-----------------------------------------
tables/
    env_cutoff_metrics.xlsx
    stressor_cutoff_metrics.xlsx
    varpart_summary.xlsx
figures/
    {score_label}_r2_env_vs_stressor.png
    {score_label}_pseudoF_env_vs_stressor.png
    {score_label}_pvalue_env_vs_stressor.png
    {score_label}_vif_env_vs_stressor.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..io.readers import read_study_data, extract_block
from ..io.writers import save_table, save_figure
from ..core.threshold_sensitivity import (
    sweep_thresholds,
    _TRANSFORMS,
)
from ..core.clustering import select_reference_sites


# ─── main pipeline ──────────────────────────────────────────────────


def score_focus_pipeline(
    data_path: str | Path,
    output_dir: str | Path,
    score: pd.Series,
    score_label: str,
    *,
    env_variables: Sequence[str] | None = None,
    stressor_predictors: pd.DataFrame | None = None,
    thresholds: Sequence[float] | None = np.arange(0.05, 1.01, 0.02).round(2),
    rda_threshold: int | float = 0.20,
    taxa_transform: str = "octave",
    shade_range: Tuple[float, float] | None = None,
    standardize_env: bool = False,
    log_transform_env: bool = False,
    n_permutations: int = 999,
    random_state: int | None = 42,
    save_plots: bool = True,
    figure_formats: Sequence[str] = ("png",),
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> Dict[str, float] | None:
    """Score-focused comparison of Env vs Stressor predictors.

    Internally runs ``sweep_thresholds`` for both predictor sets, then
    produces comparison plots and variance partitioning.

    Parameters
    ----------
    data_path : path
        Original 3-level MultiIndex workbook.
    output_dir : path
        Root for outputs (e.g. ``results/01_pollution_assessment/SumRel_Focus``).
    score : pd.Series
        Per-site contamination score (SumRel, MaxRel, …).
    score_label : str
        Human-readable label used in titles and filenames (``"SumRel"``, ``"MaxRel"``).

    Returns
    -------
    dict or None
        Variance-partitioning results (or None if it cannot be computed).
    """
    from ..core.rda import varpart_rda
    from ..viz.sumrel_focus_plots import (
        plot_r2_env_vs_stressor,
        plot_pseudoF_env_vs_stressor,
        plot_pvalue_env_vs_stressor,
        plot_vif_env_vs_stressor,
    )

    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    _log(f"\n{'=' * 60}")
    _log(f"  {score_label} Focus — Env vs Stressor Comparison")
    _log(f"  Taxa transformation: {taxa_transform}")
    _log(f"{'=' * 60}")

    # ── 1. Read data and prepare matrices ────────────────────────────
    _log("  [1] Reading study data …")
    data = read_study_data(data_path)

    if env_variables is None:
        env_variables = [
            "Measured Depth (m)",
            "Water DO Bottom (mg/L)",
            "Temperature (oC)",
            "MPS (Phi)",
            "LOI (%)",
        ]

    env_all = extract_block(data, "environmental", "raw")
    env_vars_present = [v for v in env_variables if v in env_all.columns]
    env_all = env_all[env_vars_present]
    taxa_all = extract_block(data, "taxa", "raw")

    # ── 2. Sweep thresholds for Env predictors ───────────────────────
    _log(f"  [2] Sweeping cut-offs ({score_label} · Env) …")
    env_metrics = sweep_thresholds(
        score, env_all, taxa_all,
        thresholds=thresholds,
        standardize_env=standardize_env,
        log_transform_env=log_transform_env,
        taxa_transform=taxa_transform,
        n_permutations=n_permutations,
        random_state=random_state,
        verbose=verbose,
    )
    save_table(
        env_metrics, tables_dir / "env_cutoff_metrics",
        formats=table_formats, verbose=verbose,
    )

    # ── 3. Sweep thresholds for Stressor predictors ──────────────────
    stressor_metrics = None
    if stressor_predictors is not None:
        _log(f"  [3] Sweeping cut-offs ({score_label} · Stressors) …")
        stressor_metrics = sweep_thresholds(
            score, stressor_predictors, taxa_all,
            thresholds=thresholds,
            standardize_env=False,
            log_transform_env=False,
            taxa_transform=taxa_transform,
            n_permutations=n_permutations,
            random_state=random_state,
            verbose=verbose,
        )
        save_table(
            stressor_metrics, tables_dir / "stressor_cutoff_metrics",
            formats=table_formats, verbose=verbose,
        )

    # ── 4. Comparison plots ──────────────────────────────────────────
    n_total = len(score)
    if save_plots and stressor_metrics is not None:
        _log(f"  [4] Saving {score_label} comparison figures …")

        fig_r2, _ = plot_r2_env_vs_stressor(
            env_metrics, stressor_metrics,
            score_label=score_label,
            shade_range=shade_range, n_total=n_total,
            taxa_transform=taxa_transform,
        )
        save_figure(
            fig_r2, figures_dir / f"{score_label}_r2_env_vs_stressor",
            formats=figure_formats, verbose=verbose,
        )
        plt.close(fig_r2)

        fig_pf, _ = plot_pseudoF_env_vs_stressor(
            env_metrics, stressor_metrics,
            score_label=score_label,
            shade_range=shade_range, n_total=n_total,
            taxa_transform=taxa_transform,
        )
        save_figure(
            fig_pf, figures_dir / f"{score_label}_pseudoF_env_vs_stressor",
            formats=figure_formats, verbose=verbose,
        )
        plt.close(fig_pf)

        fig_pv, _ = plot_pvalue_env_vs_stressor(
            env_metrics, stressor_metrics,
            score_label=score_label,
            shade_range=shade_range, n_total=n_total,
            taxa_transform=taxa_transform,
        )
        save_figure(
            fig_pv, figures_dir / f"{score_label}_pvalue_env_vs_stressor",
            formats=figure_formats, verbose=verbose,
        )
        plt.close(fig_pv)

        if "max_vif" in env_metrics.columns and "max_vif" in stressor_metrics.columns:
            fig_vif, _ = plot_vif_env_vs_stressor(
                env_metrics, stressor_metrics,
                score_label=score_label,
                shade_range=shade_range, n_total=n_total,
                taxa_transform=taxa_transform,
            )
            save_figure(
                fig_vif, figures_dir / f"{score_label}_vif_env_vs_stressor",
                formats=figure_formats, verbose=verbose,
            )
            plt.close(fig_vif)

    # ── 5. Variance partitioning ─────────────────────────────────────
    varpart_result = None
    if stressor_predictors is not None and env_variables is not None:
        _log(f"  [5] Running variance partitioning ({score_label}) …")

        ref_mask = select_reference_sites(score, quantile=rda_threshold)
        env_ref = env_all.loc[ref_mask].copy()
        str_ref = stressor_predictors.loc[ref_mask].copy()
        taxa_ref = taxa_all.loc[ref_mask].copy()

        if taxa_transform in _TRANSFORMS:
            taxa_ref = _TRANSFORMS[taxa_transform](taxa_ref)

        valid = (
            env_ref.dropna().index
            .intersection(str_ref.dropna().index)
            .intersection(taxa_ref.dropna().index)
        )
        env_ref = env_ref.loc[valid]
        str_ref = str_ref.loc[valid]
        taxa_ref = taxa_ref.loc[valid]

        if len(valid) > env_ref.shape[1] + str_ref.shape[1] + 1:
            varpart_result = varpart_rda(
                taxa_ref, env_ref, str_ref,
                center_Y=True, center_X=True, scale_X=False,
            )

            _log(f"    adj-R² Env only:   {varpart_result['r2_adj_X1']:.4f}")
            _log(f"    adj-R² Stressor:   {varpart_result['r2_adj_X2']:.4f}")
            _log(f"    adj-R² Combined:   {varpart_result['r2_adj_X1X2']:.4f}")
            _log(f"    [a] Pure Env:      {varpart_result['a_pure_X1']:.4f}")
            _log(f"    [b] Shared:        {varpart_result['b_shared']:.4f}")
            _log(f"    [c] Pure Stressor: {varpart_result['c_pure_X2']:.4f}")
            _log(f"    [d] Unexplained:   {varpart_result['d_unexplained']:.4f}")

            vp_df = pd.DataFrame([varpart_result])
            save_table(
                vp_df, tables_dir / "varpart_summary",
                formats=table_formats, verbose=verbose,
            )
        else:
            _log(f"    Skipped variance partitioning — too few sites ({len(valid)})")

    _log(f"\n✓ {score_label} Focus pipeline complete.")
    return varpart_result


# ─── backward-compatible alias ───────────────────────────────────────

sumrel_focus_pipeline = score_focus_pipeline
