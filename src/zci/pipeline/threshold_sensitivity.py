"""Threshold-Sensitivity Pipeline.

Orchestrates:
  read data → prepare env / taxa → sweep thresholds for each score
  → detect stable ranges → build summary table → save figures + tables.

Outputs
-------
tables/
    {Prefix}_threshold_metrics.xlsx   — tidy metrics per threshold
    threshold_comparison_summary.xlsx  — side-by-side score comparison
    {Prefix}_rda_axes_summary.xlsx     — RDA axis table at chosen threshold
    {Prefix}_rda_terms_summary.xlsx    — RDA terms table at chosen threshold
figures/
    {Prefix}_threshold_sensitivity.png — per-score 2-panel plot
    threshold_comparison.png           — overlay comparison plot
    {Prefix}_rda_triplot.png           — RDA triplot at chosen threshold
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
    ThresholdSweepResult,
)
from ..core.rda import RDA
from ..core.clustering import select_reference_sites
from ..viz.threshold_sensitivity_plots import (
    plot_threshold_sensitivity,
    plot_threshold_comparison,
)
from ..viz.rda_plots import plot_rda_triplot


# ─── RDA summary-table helpers (mirrored from rda_analysis) ─────────


def _significance(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _build_axes_table(rda: RDA, axes_test: pd.DataFrame) -> pd.DataFrame:
    fit = rda.fit_
    n = len(axes_test)
    return pd.DataFrame({
        "Axis": [f"RDA{i + 1}" for i in range(n)],
        "Eigenvalue": fit.constrained_eigenvalues.iloc[:n].round(2).values,
        "Explained (%)": (fit.explained_proportion.iloc[:n] * 100).round(2).values,
        "Cumulative (%)": (fit.cumulative_explained.iloc[:n] * 100).round(2).values,
        "F-statistic": axes_test["F"].round(2).values,
        "p-value": axes_test["p"].round(2).values,
        "Significance": axes_test["p"].apply(_significance).values,
    })


def _build_terms_table(terms_test: pd.DataFrame, biplot_scores: pd.DataFrame) -> pd.DataFrame:
    df = terms_test.copy()
    n_axes = min(2, biplot_scores.shape[1])
    for k in range(n_axes):
        col = biplot_scores.columns[k]
        df[f"{col} Coefficient"] = df["term"].map(
            lambda t, _c=col: round(biplot_scores.loc[t, _c], 2) if t in biplot_scores.index else np.nan
        )
    df["Significance"] = df["p"].apply(_significance)
    df = df.rename(columns={
        "term": "Environmental Variable",
        "delta_inertia": "Delta Inertia",
        "F": "F-statistic",
        "p": "p-value",
    })
    cols = [
        "Environmental Variable", "Delta Inertia", "F-statistic",
        "p-value", "Significance",
    ] + [c for c in df.columns if "Coefficient" in c]
    return df[cols].round(2)


# ─── main pipeline ──────────────────────────────────────────────────


def threshold_sensitivity_pipeline(
    data_path: str | Path,
    output_dir: str | Path,
    scores: Dict[str, pd.Series],
    *,
    thresholds: Sequence[float] | None = np.arange(0.10, 1.01, 0.02).round(2),
    env_variables: Sequence[str] | None = None,
    taxa_columns: Sequence[str] | None = None,
    standardize_env: bool = True,
    log_transform_env: bool = False,
    taxa_transform: str = "octave",
    n_permutations: int = 999,
    random_state: int | None = 42,
    shade_range: Tuple[float, float] | None = None,
    rda_threshold: float | None = None,
    save_plots: bool = True,
    figure_formats: Sequence[str] = ("png",),
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> Dict[str, ThresholdSweepResult]:
    """Run threshold-sensitivity analysis for one or more contamination scores.

    Parameters
    ----------
    data_path : path
        Original 3-level MultiIndex workbook.
    output_dir : path
        Root for outputs (e.g. ``results/ref_threshold_sensitivity``).
    scores : dict[str, pd.Series]
        Mapping ``{"SumRel": sumrel_series, "MaxRel": maxrel_series, …}``.
    thresholds : sequence of float, optional
        Grid of proportions.  Default: 0.10, 0.12, …, 1.00.
    env_variables : list of str, optional
        Environmental column names.  ``None`` → sensible defaults.
    taxa_columns : list of str, optional
        Taxa column names.  ``None`` → all taxa in the data.
    standardize_env, log_transform_env, taxa_transform : misc
        Passed through to each RDA fit.
    n_permutations : int
        Permutations for the global test at each threshold.
    random_state : int or None
        Base RNG seed.
    highlight_threshold : float or None
        A reference threshold to mark on plots (e.g. 0.20).
    p_threshold : float
        Max p for stable-range detection.
    r2_adj_min : float
        Min adj-R² for stable-range detection.
    min_sites : int
        Min sample size for stable-range detection.
    save_plots : bool
        Whether to produce and save figures.
    figure_formats, table_formats : misc
        Output formats.
    verbose : bool
        Print progress.

    Returns
    -------
    dict[str, ThresholdSweepResult]
        One result per contamination score.
    """
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # ── 1. Read data and prepare matrices ────────────────────────────
    _log("[1/4] Reading study data …")
    data = read_study_data(data_path)
    _log(f"      {data.shape[0]} sites × {data.shape[1]} variables")

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
    if taxa_columns is not None:
        taxa_cols = [c for c in taxa_columns if c in taxa_all.columns]
        taxa_all = taxa_all[taxa_cols]

    _log(f"      Env: {len(env_vars_present)} variables, Taxa: {taxa_all.shape[1]} taxa")

    if thresholds is None:
        thresholds = list(np.arange(0.10, 0.31, 0.02).round(2))

    # ── 2. Sweep for each score ──────────────────────────────────────
    results: Dict[str, ThresholdSweepResult] = {}
    all_metrics: Dict[str, pd.DataFrame] = {}

    for label, score in scores.items():
        _log(f"\n[2/4] Sweeping thresholds for {label} …")
        metrics = sweep_thresholds(
            score, env_all, taxa_all,
            thresholds=thresholds,
            standardize_env=standardize_env,
            log_transform_env=log_transform_env,
            taxa_transform=taxa_transform,
            n_permutations=n_permutations,
            random_state=random_state,
            verbose=verbose,
        )
        all_metrics[label] = metrics

        results[label] = ThresholdSweepResult(
            metrics=metrics,
            stable_ranges=[],
            score_name=label,
            env_variables=list(env_vars_present),
            taxa_columns=list(taxa_all.columns),
        )

    # ── 3. Save tables ───────────────────────────────────────────────────
    _log("\n[3/5] Saving tables …")
    for label, res in results.items():
        save_table(
            res.metrics,
            tables_dir / f"{label}_threshold_metrics",
            formats=table_formats,
            verbose=verbose,
        )

    # Comparison summary: one row per (score, threshold)
    comparison_rows = []
    for label, df in all_metrics.items():
        df_copy = df.copy()
        df_copy.insert(0, "score", label)
        comparison_rows.append(df_copy)
    if comparison_rows:
        comparison = pd.concat(comparison_rows, ignore_index=True)
        save_table(
            comparison,
            tables_dir / "threshold_comparison_summary",
            formats=table_formats,
            verbose=verbose,
        )

    # ── 4. Full RDA at chosen threshold ───────────────────────────────
    if rda_threshold is not None:
        _log(f"\n[4/5] Fitting full RDA at {rda_threshold:.0%} for each score …")
        for label, score in scores.items():
            _log(f"  {label}: fitting RDA at {rda_threshold:.0%} …")
            ref_mask = select_reference_sites(score, quantile=rda_threshold)
            env_ref = env_all.loc[ref_mask].copy()
            taxa_ref = taxa_all.loc[ref_mask].copy()

            if standardize_env:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                arr = scaler.fit_transform(env_ref)
                env_ref = pd.DataFrame(arr, index=env_ref.index, columns=env_ref.columns)

            if taxa_transform == "hellinger":
                row_sums = taxa_ref.sum(axis=1)
                taxa_ref = taxa_ref.div(row_sums, axis=0).fillna(0).apply(np.sqrt)

            valid = env_ref.dropna().index.intersection(taxa_ref.dropna().index)
            env_ref = env_ref.loc[valid]
            taxa_ref = taxa_ref.loc[valid]

            rda = RDA(center_X=True, center_Y=True, scale_X=False, ddof=1)
            rda.fit(env_ref, taxa_ref)
            fit = rda.fit_

            global_test = rda.test_global(n_permutations=n_permutations, random_state=random_state)
            axes_test = rda.test_axes(n_permutations=n_permutations, random_state=random_state)
            terms_test = rda.test_terms(n_permutations=n_permutations, random_state=random_state)
            bp = rda.biplot_scores(n_axes=min(6, len(axes_test)))

            _log(f"    R²={fit.r2:.4f}  adj-R²={fit.r2_adj:.4f}  "
                 f"F={global_test.statistic:.2f}  p={global_test.p_value:.4f}  "
                 f"n={len(valid)}")

            # Build summary tables (same structure as rda_analysis)
            axes_table = _build_axes_table(rda, axes_test)
            terms_table = _build_terms_table(terms_test, bp)

            save_table(axes_table, tables_dir / f"{label}_rda_axes_summary",
                       formats=table_formats, verbose=verbose)
            save_table(terms_table, tables_dir / f"{label}_rda_terms_summary",
                       formats=table_formats, verbose=verbose)

            if save_plots:
                # Waterbody grouping for site colours
                sample_info = extract_block(data, "sample_info", "raw")
                waterbody = sample_info["Waterbody"].reindex(valid)

                triplot_title = f"RDA Triplot — {label} (ref. {rda_threshold:.0%})"
                fig_tri, _ = plot_rda_triplot(
                    rda, axes=(1, 2), scaling=1,
                    site_groups=waterbody,
                    terms_test=terms_test, global_test=global_test,
                    # arrow_scale=2.0,
                    # species_scale=2.0,
                    # figsize=(14, 8), dpi=300, title=triplot_title,
                )
                save_figure(fig_tri, figures_dir / f"{label}_rda_triplot",
                            formats=figure_formats, verbose=verbose)
                plt.close(fig_tri)

    # ── 5. Save figures ──────────────────────────────────────────────────
    if save_plots:
        _log("[5/5] Saving sensitivity figures …")

        # Per-score sensitivity plots
        for label, res in results.items():
            fig, _ = plot_threshold_sensitivity(
                res.metrics,
                shade_range=shade_range,
                score_label=label,
            )
            save_figure(
                fig,
                figures_dir / f"{label}_threshold_sensitivity",
                formats=figure_formats,
                verbose=verbose,
            )
            plt.close(fig)

        # Overlay comparison plot
        if len(all_metrics) > 1:
            fig_cmp, _ = plot_threshold_comparison(
                all_metrics,
                shade_range=shade_range,
            )
            save_figure(
                fig_cmp,
                figures_dir / "threshold_comparison",
                formats=figure_formats,
                verbose=verbose,
            )
            plt.close(fig_cmp)

    _log("\n✓ Threshold-sensitivity pipeline complete.")
    return results
