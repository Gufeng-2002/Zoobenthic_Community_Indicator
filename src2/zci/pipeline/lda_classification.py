"""Stage LDA — Linear Discriminant Analysis classification pipeline.

Orchestrates:
  read original data → merge Stage 1 pollution scores → merge Stage 2
  cluster labels → fit LDA on reference sites → Wilks' Lambda importance
  → Monte Carlo CV → LDA triplot → predict non-reference sites →
  4-panel cluster comparison → save tables & figures.

Outputs
-------
tables/
    lda_classification_report.xlsx
    lda_confusion_matrix.xlsx
    lda_axes_summary.xlsx
    lda_env_significance.xlsx
    mccv_confusion_matrix.xlsx
    mccv_classification_report.xlsx
figures/
    lda_triplot.png
    cluster_comparison.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..io.readers import read_study_data, extract_block
from ..io.writers import save_table, save_figure
from ..core.clustering import select_reference_sites
from ..core.lda import (
    fit_lda,
    wilks_lambda_importance,
    monte_carlo_cv,
    predict_sites,
    build_env_significance_table,
    build_confusion_matrix_table,
    build_classification_report_table,
    build_mccv_classification_report_table,
)
from ..models.lda import LDAResult
from ..models.clustering import TAXA_COLUMNS
from ..viz.lda_plots import plot_lda_triplot, plot_cluster_comparison


# ─── main pipeline ──────────────────────────────────────────────────


def lda_pipeline(
    data_path: str | Path,
    stage1_artifact: str | Path,
    stage2_artifact: str | Path,
    output_dir: str | Path,
    *,
    env_variables: Sequence[str] | None = None,
    taxa_columns: Sequence[str] | None = None,
    reference_quantile: float = 0.20,
    standardize_env: bool = True,
    n_mccv_iterations: int = 1000,
    mccv_test_size: float = 0.2,
    random_state: int | None = 42,
    save_plots: bool = True,
    figure_formats: Sequence[str] = ("png",),
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> LDAResult:
    """Run the complete LDA classification pipeline and save outputs.

    Steps
    -----
     1. Read the original study-data Excel workbook.
     2. Read Stage 1 artifact → Pollution_Score → reference mask.
     3. Read Stage 2 artifact → cluster labels for reference sites.
     4. Extract environmental variables for reference sites.
     5. Fit LDA on full reference data.
     6. Wilks' Lambda per-variable importance.
     7. Monte Carlo Cross-Validation (1 000 iterations).
     8. Build & save all tables.
     9. LDA triplot figure.
    10. Predict non-reference sites.
    11. 4-panel cluster comparison figure.
    12. Save augmented artifact.

    Parameters
    ----------
    data_path : path
        Original 3-level MultiIndex workbook.
    stage1_artifact : path
        ``01_updated_data.xlsx`` from Stage 1 (pollution scores).
    stage2_artifact : path
        ``02_updated_data.xlsx`` from Stage 2 (cluster labels).
    output_dir : path
        Root for LDA outputs (``tables/``, ``figures/``, ``artifacts/``).
    env_variables : list of str, optional
        Environmental column names.  ``None`` → sensible defaults.
    taxa_columns : list of str, optional
        Taxa column names for cluster comparison.  ``None`` → all 16.
    reference_quantile : float
        Fraction of least-polluted sites designated as reference.
    standardize_env : bool
        Z-score environmental variables before LDA.
    n_mccv_iterations : int
        Number of Monte Carlo CV splits.
    mccv_test_size : float
        Test-set proportion per split.
    random_state : int or None
        Seed for reproducibility.
    save_plots : bool
        Whether to write figures to disk.
    figure_formats, table_formats : sequence of str
        File-format lists.
    verbose : bool
        Print progress messages.

    Returns
    -------
    LDAResult
    """
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    artifacts_dir = output_dir / "artifacts"

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # ── 1. Read original data ────────────────────────────────────────
    _log("[1/12] Reading original study data …")
    data = read_study_data(data_path)
    _log(f"       {data.shape[0]} sites × {data.shape[1]} variables")

    # ── 2. Pollution scores → reference mask ─────────────────────────
    _log("[2/12] Reading Stage 1 artifact for pollution scores …")
    stage1 = pd.read_excel(stage1_artifact, header=[0, 1, 2], index_col=0)
    pollution = stage1.loc[
        :, ("01_pollution_assessment", "raw", "Pollution_Score")
    ]
    pollution.name = "Pollution_Score"

    ref_mask = select_reference_sites(pollution, quantile=reference_quantile)
    n_ref = ref_mask.sum()
    _log(f"       {n_ref} reference sites (bottom {reference_quantile*100:.0f} %)")

    # ── 3. Read Stage 2 artifact → cluster labels ────────────────────
    _log("[3/12] Reading Stage 2 artifact for cluster labels …")
    stage2 = pd.read_excel(stage2_artifact, header=[0, 1, 2], index_col=0)
    cluster_col_key = ("02_taxa_assemblage", "raw", "Cluster")
    if cluster_col_key in stage2.columns:
        cluster_series = stage2.loc[:, cluster_col_key].copy()
    else:
        # fallback: search for any column containing "Cluster"
        cand = [c for c in stage2.columns if "Cluster" in str(c)]
        if not cand:
            raise ValueError("Cannot find cluster column in Stage 2 artifact")
        cluster_series = stage2.iloc[:, stage2.columns.get_loc(cand[0])].copy()

    cluster_series.name = "Cluster"
    ref_cluster = cluster_series.loc[ref_mask].dropna()
    _log(f"       {len(ref_cluster)} reference sites with cluster labels")
    for g in sorted(ref_cluster.unique()):
        _log(f"         Cluster {int(g)}: {(ref_cluster == g).sum()} sites")

    # ── 4. Extract environmental data ────────────────────────────────
    if env_variables is None:
        env_variables = [
            "Measured Depth (m)",
            "Water DO Bottom (mg/L)",
            "Temperature (oC)",
            "MPS (Phi)",
            "LOI (%)",
        ]
    _log(f"[4/12] Extracting {len(env_variables)} environmental variables …")

    env_all = extract_block(data, "environmental", "raw")
    env_vars_present = [v for v in env_variables if v in env_all.columns]

    # reference sites only, aligned with cluster labels
    common_ref = ref_cluster.index.intersection(env_all.index)
    env_ref_raw = env_all.loc[common_ref, env_vars_present].copy()
    labels_ref = ref_cluster.loc[common_ref].values

    # drop NaN rows
    valid = env_ref_raw.dropna().index
    env_ref_raw = env_ref_raw.loc[valid]
    labels_ref = ref_cluster.loc[valid].values
    _log(f"       {len(valid)} reference sites after dropping NaN")

    # ── 5. Fit LDA ──────────────────────────────────────────────────
    _log("[5/12] Fitting LDA on reference sites …")
    lda_fit = fit_lda(env_ref_raw, labels_ref, standardize=standardize_env)
    _log(f"       Accuracy = {lda_fit.accuracy:.2%}")
    _log(f"       Explained variance: {', '.join(f'LD{i+1}={v:.1%}' for i, v in enumerate(lda_fit.explained_variance_ratio))}")

    # ── 6. Wilks' Lambda importance ──────────────────────────────────
    _log("[6/12] Computing Wilks' Lambda variable importance …")
    wilks = wilks_lambda_importance(lda_fit)
    _log(f"       Full-model Wilks' Λ = {wilks.wilks_lambda_full:.4f}")
    _log(f"       Overall p = {wilks.overall_significance['p_value']:.6f}")

    # ── 7. Monte Carlo CV ────────────────────────────────────────────
    _log(f"[7/12] Monte Carlo Cross-Validation ({n_mccv_iterations} iterations) …")
    mccv = monte_carlo_cv(
        env_ref_raw, labels_ref,
        standardize=standardize_env,
        n_iterations=n_mccv_iterations,
        test_size=mccv_test_size,
        random_state=random_state,
    )
    _log(f"       Mean accuracy = {mccv.mean_accuracy:.2%} ± {mccv.std_accuracy:.2%}")
    _log(f"       Median = {mccv.median_accuracy:.2%}")

    # ── 8. Build & save tables ───────────────────────────────────────
    _log("[8/12] Saving tables …")

    # a) LDA classification report (single fit)
    tbl_report = build_classification_report_table(lda_fit)
    save_table(tbl_report, tables_dir / "lda_classification_report",
               formats=table_formats, verbose=verbose)

    # b) LDA confusion matrix (single fit)
    tbl_cm = build_confusion_matrix_table(
        lda_fit.confusion_matrix, lda_fit.cluster_names,
    )
    save_table(tbl_cm, tables_dir / "lda_confusion_matrix",
               formats=table_formats, verbose=verbose)

    # c) LDA axes summary
    save_table(wilks.axes_summary, tables_dir / "lda_axes_summary",
               formats=table_formats, verbose=verbose)

    # d) Env significance table (Habitat var | Significance | Cluster means)
    env_sig_table = build_env_significance_table(
        lda_fit, wilks, env_ref_raw, labels_ref,
    )
    save_table(env_sig_table, tables_dir / "lda_env_significance",
               formats=table_formats, verbose=verbose)

    # e) MCCV confusion matrix
    tbl_mccv_cm = build_confusion_matrix_table(
        mccv.aggregate_confusion_matrix, mccv.cluster_names,
        note=f"Combined results from {mccv.n_iterations:,} CV iterations "
             f"(test size: {mccv.test_size*100:.0f}%)",
    )
    save_table(tbl_mccv_cm, tables_dir / "mccv_confusion_matrix",
               formats=table_formats, verbose=verbose)

    # f) MCCV classification report
    tbl_mccv_rpt = build_mccv_classification_report_table(mccv)
    save_table(tbl_mccv_rpt, tables_dir / "mccv_classification_report",
               formats=table_formats, verbose=verbose)

    # ── 9. LDA triplot ──────────────────────────────────────────────
    if save_plots:
        _log("[9/12] Creating LDA triplot …")
        # project ALL sites with cluster labels
        env_for_plot = env_all.loc[
            ref_cluster.dropna().index.intersection(env_all.index),
            env_vars_present,
        ].dropna()
        cluster_for_plot = ref_cluster.loc[env_for_plot.index]

        fig_tri, _ = plot_lda_triplot(
            lda_fit, wilks,
            env_all=env_for_plot,
            cluster_all=cluster_for_plot,
        )
        save_figure(fig_tri, figures_dir / "lda_triplot",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_tri)

    # ── 10. Predict non-reference sites ──────────────────────────────
    _log("[10/12] Predicting non-reference sites …")
    nonref_mask = ~ref_mask
    env_nonref = env_all.loc[nonref_mask, env_vars_present].dropna()
    nonref_preds, nonref_probs = predict_sites(lda_fit, env_nonref)
    _log(f"        Predicted {len(nonref_preds)} non-reference sites")
    for g in sorted(nonref_preds.unique()):
        _log(f"          Cluster {int(g)}: {(nonref_preds == g).sum()} sites")

    # combined cluster labels (ref from Stage 2, non-ref from LDA)
    cluster_all = pd.Series(np.nan, index=data.index, name="Cluster")
    cluster_all.loc[ref_cluster.index] = ref_cluster.values
    cluster_all.loc[nonref_preds.index] = nonref_preds.values

    # ── 11. 4-panel cluster comparison ───────────────────────────────
    if save_plots:
        _log("[11/12] Creating 4-panel cluster comparison …")
        if taxa_columns is None:
            taxa_columns = TAXA_COLUMNS
        taxa_all = extract_block(data, "taxa", "raw")
        taxa_cols_present = [c for c in taxa_columns if c in taxa_all.columns]
        taxa_sub = taxa_all[taxa_cols_present]

        fig_comp, _ = plot_cluster_comparison(
            raw_env=env_all[env_vars_present],
            taxa_octave=taxa_sub,
            cluster_labels_all=cluster_all,
            ref_mask=ref_mask,
            env_variables=env_vars_present,
        )
        save_figure(fig_comp, figures_dir / "cluster_comparison",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_comp)

    # ── 12. Save augmented artifact ──────────────────────────────────
    _log("[12/12] Saving augmented artifact …")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # build 3-level multi-index columns
    cols = pd.MultiIndex.from_tuples(
        [("03_lda_classification", "raw", "Predicted_Cluster"),
         ("03_lda_classification", "raw", "Is_Reference")],
    )
    aug = pd.DataFrame(index=data.index, columns=cols)
    aug[("03_lda_classification", "raw", "Predicted_Cluster")] = cluster_all
    aug[("03_lda_classification", "raw", "Is_Reference")] = ref_mask.astype(int)

    # add probability columns
    for cname in lda_fit.cluster_names:
        col_key = ("03_lda_classification", "raw", f"Prob_{cname}")
        aug[col_key] = np.nan
        if nonref_probs is not None:
            aug.loc[nonref_probs.index, col_key] = nonref_probs[cname].values

    aug_path = artifacts_dir / "03_updated_data.xlsx"
    aug.to_excel(aug_path)
    if verbose:
        print(f"  ✓ Saved augmented data: {aug_path}")

    # ── assemble result ──────────────────────────────────────────────
    result = LDAResult(
        lda_fit=lda_fit,
        wilks=wilks,
        mccv=mccv,
        env_significance_table=env_sig_table,
        nonref_predictions=nonref_preds,
        nonref_probabilities=nonref_probs,
        transformation_info={
            "standardize_env": standardize_env,
            "env_variables": list(env_vars_present),
        },
    )

    _log(f"\n✓ LDA pipeline complete.  {result.summary()}")
    return result
