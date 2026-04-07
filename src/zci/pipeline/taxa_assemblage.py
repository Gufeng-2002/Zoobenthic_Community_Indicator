"""Stage 2 -- LDA Classification Pipeline.

Requires pre-computed Ward clustering labels from the WardsClustering pipeline.

Phase 1 (classifier_training): reference sites only
  use Ward cluster labels -> LDA fit -> Wilks Lambda -> MCCV
  -> tables -> env PCA ordination -> save artifact.

Phase 2 (classifier_prediction): non-reference sites
  LDA predict -> probability table -> 4-panel comparison figure -> artifact.
"""

from __future__ import annotations

from pathlib import Path as _Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..io.readers import read_study_data, extract_block
from ..io.writers import save_table, save_figure
from ..core.transforms import octave_to_relative_abundance
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
from ..models.clustering import TAXA_COLUMNS
from ..models.lda import LDAResult
from ..viz.lda_plots import plot_lda_triplot, plot_cluster_comparison
from ..viz.ordination_plots import save_env_pca_ordination
from ..viz.taxa_trend_grid import plot_taxa_trend_comparison, plot_env_trend_comparison


def taxa_assemblage_pipeline(
    data_path: str | _Path,
    stage1_artifact: str | _Path,
    output_dir: str | _Path,
    maps_dir: str | _Path,
    *,
    site_robustness: pd.DataFrame,
    taxa_columns: Sequence[str] = TAXA_COLUMNS,
    env_variables: Sequence[str] | None = None,
    standardize_env: bool = True,
    n_mccv_iterations: int = 1000,
    mccv_test_size: float = 0.2,
    random_state: int | None = 42,
    save_plots: bool = True,
    figure_formats: Sequence[str] = ("png",),
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the LDA classification pipeline using pre-computed Ward clustering.

    Parameters
    ----------
    site_robustness : pd.DataFrame
        Site robustness table from WardsClustering (index=Site, columns include
        Original_Cluster, Branch_AU, Silhouette, etc.).

    Returns dict with keys: lda, ref_mask, labels_ref, cluster_all.
    """
    output_dir = _Path(output_dir)
    ct_dir = output_dir / "classifier_training"
    cp_dir = output_dir / "classifier_prediction"
    ct_tables = ct_dir / "tables"
    ct_figures = ct_dir / "figures"
    ct_artifacts = ct_dir / "artifacts"
    cp_tables = cp_dir / "tables"
    cp_figures = cp_dir / "figures"
    cp_artifacts = cp_dir / "artifacts"

    if env_variables is None:
        env_variables = [
            "Measured Depth (m)",
            "Water DO Bottom (mg/L)",
            "Temperature (oC)",
            "MPS (Phi)",
            "LOI (%)",
        ]

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # ============================================================
    #  Extract cluster labels from site robustness table
    # ============================================================
    labels_ref = site_robustness["Original_Cluster"].copy()
    n_clusters = int(labels_ref.nunique())
    n_ref = len(labels_ref)

    # ============================================================
    #  PHASE 1: LDA Classifier Training (reference sites)
    # ============================================================
    _log("=" * 60)
    _log(f"  PHASE 1: LDA Classifier Training ({n_ref} reference sites)")
    _log("=" * 60)

    _log(f"       {n_ref} reference sites, {n_clusters} clusters")
    for g in sorted(labels_ref.unique()):
        _log(f"         Group {g}: {(labels_ref == g).sum()} sites")

    # -- 1. Read original data -----------------------------------------
    _log("[1/8] Reading original study data ...")
    data = read_study_data(data_path)
    _log(f"       {data.shape[0]} sites x {data.shape[1]} variables")

    # Build reference mask over all sites
    ref_mask = pd.Series(False, index=data.index, name="Is_Reference")
    ref_mask.loc[ref_mask.index.isin(labels_ref.index)] = True

    # -- 2. Read Stage 1 artifact -> Pollution Score -------------------
    _log("[2/8] Reading Stage 1 artifact for pollution scores ...")
    stage1 = pd.read_excel(stage1_artifact, header=[0, 1, 2], index_col=0)
    score_cols = [
        c for c in stage1.columns
        if c[0] == "01_pollution_assessment" and c[1] == "raw"
        and c[2].endswith("_Score")
    ]
    if not score_cols:
        raise KeyError("No pollution score column found in Stage 1 artifact")
    pollution_score = stage1.loc[:, score_cols[0]]
    pollution_score.name = "Pollution_Score"

    # -- 3. Prepare environmental data for LDA -------------------------
    _log("[3/8] Preparing environmental data ...")
    env_block = extract_block(data, "environmental", "raw")
    taxa_all = extract_block(data, "taxa", "raw")[list(taxa_columns)]
    env_vars_present = [v for v in env_variables if v in env_block.columns]
    env_ref_raw = env_block.loc[labels_ref.index, env_vars_present]
    lda_ref_index = env_ref_raw.dropna().index
    if len(lda_ref_index) < len(env_ref_raw):
        _log(
            f"       Dropping {len(env_ref_raw) - len(lda_ref_index)} reference sites "
            "with missing environmental values."
        )
    env_ref_complete = env_ref_raw.loc[lda_ref_index]
    labels_ref_complete = labels_ref.loc[lda_ref_index]

    # -- 4. Fit LDA on reference sites ---------------------------------
    _log(f"[4/8] Fitting LDA on {len(lda_ref_index)} reference sites ...")
    lda_fit = fit_lda(
        env_ref_complete,
        labels_ref_complete.values,
        standardize=standardize_env,
    )
    _log(f"       Accuracy = {lda_fit.accuracy:.2%}")

    # -- 5. Wilks Lambda variable importance ---------------------------
    _log("[5/8] Computing Wilks Lambda variable importance ...")
    wilks = wilks_lambda_importance(lda_fit)
    _log(f"       Full-model Wilks Lambda = {wilks.wilks_lambda_full:.4f}")

    # -- 6. Monte Carlo Cross-Validation -------------------------------
    _log(f"[6/8] MCCV ({n_mccv_iterations} iterations) ...")
    mccv = monte_carlo_cv(
        env_ref_complete, labels_ref_complete.values,
        standardize=standardize_env,
        n_iterations=n_mccv_iterations,
        test_size=mccv_test_size,
        random_state=random_state,
    )
    _log(f"       Mean accuracy = {mccv.mean_accuracy:.2%} +/- {mccv.std_accuracy:.2%}")

    # -- 7. Save LDA tables --------------------------------------------
    _log("[7/8] Saving LDA tables ...")
    save_table(build_classification_report_table(lda_fit),
               ct_tables / "lda_classification_report",
               formats=table_formats, verbose=verbose)
    save_table(build_confusion_matrix_table(lda_fit.confusion_matrix, lda_fit.cluster_names),
               ct_tables / "lda_confusion_matrix",
               formats=table_formats, verbose=verbose)
    save_table(wilks.axes_summary,
               ct_tables / "lda_axes_summary",
               formats=table_formats, verbose=verbose)
    env_sig_table = build_env_significance_table(
        lda_fit,
        wilks,
        env_ref_complete,
        labels_ref_complete,
    )
    save_table(env_sig_table,
               ct_tables / "lda_env_significance",
               formats=table_formats, verbose=verbose)
    save_table(
        build_confusion_matrix_table(
            mccv.aggregate_confusion_matrix, mccv.cluster_names,
            note=f"Combined results from {mccv.n_iterations:,} CV iterations "
                 f"(test size: {mccv.test_size*100:.0f}%)",
        ),
        ct_tables / "mccv_confusion_matrix",
        formats=table_formats, verbose=verbose,
    )
    save_table(build_mccv_classification_report_table(mccv),
               ct_tables / "mccv_classification_report",
               formats=table_formats, verbose=verbose)

    # -- 8. Figures (LDA triplot + env PCA ordination) -----------------
    if save_plots:
        _log("[8/8] Creating LDA triplot + PCA ordination ...")
        env_for_plot = env_block.loc[
            labels_ref.index.intersection(env_block.index),
            env_vars_present,
        ].dropna()
        cluster_for_plot = labels_ref.loc[env_for_plot.index]
        fig_tri, _ = plot_lda_triplot(
            lda_fit, wilks,
            env_all=env_for_plot,
            cluster_all=cluster_for_plot,
        )
        save_figure(fig_tri, ct_figures / "lda_triplot",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_tri)

        # PCA ordination biplot in environmental space
        env_short_names = [n.split("(")[0].strip() if "(" in n else n for n in env_vars_present]
        lda_pred_ref = pd.Series(
            lda_fit.predictions,
            index=labels_ref_complete.index,
            name="Predicted",
        )
        pca_path = save_env_pca_ordination(
            env_ref=env_ref_complete,
            true_labels=labels_ref_complete,
            predicted_labels=lda_pred_ref,
            output_path=ct_figures / "env_pca_ordination.png",
            env_feature_names=env_short_names,
            title=f"PCA Ordination of {n_ref} Reference Sites in Environmental Space",
        )
        if verbose:
            print(f"  > Saved figure: {pca_path}")

    # -- Save augmented site robustness artifact -----------------------
    ct_artifacts.mkdir(parents=True, exist_ok=True)
    augmented_robustness = site_robustness.copy()

    # Predicted cluster from LDA resubstitution
    lda_pred_ref = pd.Series(
        lda_fit.predictions,
        index=labels_ref_complete.index,
        name="Predicted_Cluster",
    )
    augmented_robustness["Predicted_Cluster"] = lda_pred_ref.reindex(
        augmented_robustness.index
    )

    # Posterior probabilities for each cluster
    _, ref_probs = predict_sites(lda_fit, env_ref_complete)
    for cname in lda_fit.cluster_names:
        augmented_robustness[f"Prob_{cname}"] = (
            ref_probs[cname].reindex(augmented_robustness.index)
        )

    aug_path = ct_artifacts / "site_robustness.xlsx"
    augmented_robustness.to_excel(aug_path)
    if verbose:
        print(f"  > Saved augmented robustness: {aug_path}")

    _log("\n> Phase 1 complete.")

    # ============================================================
    #  PHASE 2: Classifier Prediction (non-reference sites)
    # ============================================================
    _log("\n" + "=" * 60)
    _log("  PHASE 2: LDA Classifier Prediction (non-reference sites)")
    _log("=" * 60)

    # -- P2-1. Predict non-reference sites -----------------------------
    _log("[P2-1/4] Predicting non-reference sites ...")
    nonref_mask = ~ref_mask
    env_nonref = env_block.loc[nonref_mask, env_vars_present].dropna()
    nonref_preds, nonref_probs = predict_sites(lda_fit, env_nonref)
    _log(f"         Predicted {len(nonref_preds)} remaining sites")
    for g in sorted(nonref_preds.unique()):
        _log(f"           Cluster {int(g)}: {(nonref_preds == g).sum()} sites")

    # combined cluster labels (ref from Ward, non-ref from LDA)
    cluster_all = pd.Series(np.nan, index=data.index, name="Cluster")
    cluster_all.loc[labels_ref.index] = labels_ref.values
    cluster_all.loc[nonref_preds.index] = nonref_preds.values

    # -- P2-2. Save prediction tables ----------------------------------
    _log("[P2-2/4] Saving prediction tables ...")
    prob_table = nonref_probs.copy()
    prob_table.insert(0, "Predicted_Cluster", nonref_preds)
    save_table(prob_table, cp_tables / "prediction_probabilities",
               formats=table_formats, verbose=verbose)

    # -- P2-3. 4-panel cluster comparison figure -----------------------
    if save_plots:
        _log("[P2-3/4] Creating 4-panel cluster comparison ...")
        taxa_sub = taxa_all[[c for c in taxa_columns if c in taxa_all.columns]]
        fig_comp, _ = plot_cluster_comparison(
            raw_env=env_block[env_vars_present],
            taxa_octave=taxa_sub,
            cluster_labels_all=cluster_all,
            ref_mask=ref_mask,
            env_variables=env_vars_present,
            n_ref=n_ref,
        )
        save_figure(fig_comp, cp_figures / "cluster_comparison",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_comp)

    # -- P2-4. Save Phase 2 artifact -----------------------------------
    _log("[P2-4/4] Saving prediction artifact ...")
    cp_artifacts.mkdir(parents=True, exist_ok=True)
    cols = pd.MultiIndex.from_tuples([
        ("02_taxa_assemblage", "raw", "Predicted_Cluster"),
        ("02_taxa_assemblage", "raw", "Is_Reference"),
    ])
    aug2 = pd.DataFrame(index=data.index, columns=cols)
    aug2[("02_taxa_assemblage", "raw", "Predicted_Cluster")] = cluster_all
    aug2[("02_taxa_assemblage", "raw", "Is_Reference")] = ref_mask.astype(int)
    for cname in lda_fit.cluster_names:
        col_key = ("02_taxa_assemblage", "raw", f"Prob_{cname}")
        aug2[col_key] = np.nan
        if nonref_probs is not None:
            aug2.loc[nonref_probs.index, col_key] = nonref_probs[cname].values
    aug2_path = cp_artifacts / "02_predicted_data.xlsx"
    aug2.to_excel(aug2_path)
    if verbose:
        print(f"  > Saved prediction artifact: {aug2_path}")

    # ── Taxa trend grid plots ─────────────────────────────────────
    if save_plots:
        _log("[P2-5] Creating taxa trend grid plots ...")
        cp_figures.mkdir(parents=True, exist_ok=True)

        taxa_ref_relabd = octave_to_relative_abundance(taxa_all.loc[labels_ref.index])
        avg_score_ref = pollution_score.loc[labels_ref.index].mean()

        n_ref_int = int(ref_mask.sum())
        top_polluted_idx = pollution_score.nlargest(n_ref_int).index
        top_polluted_idx = top_polluted_idx.intersection(cluster_all.dropna().index)
        if len(top_polluted_idx) > 0:
            top_polluted_labels = cluster_all.loc[top_polluted_idx].astype(int)
            avg_score_pol = pollution_score.loc[top_polluted_idx].mean()
            taxa_top_relabd = octave_to_relative_abundance(
                taxa_all.loc[top_polluted_idx.intersection(taxa_all.index)]
            )
            fig_cmp, _ = plot_taxa_trend_comparison(
                taxa_relabd_ref=taxa_ref_relabd,
                cluster_labels_ref=labels_ref,
                taxa_relabd_polluted=taxa_top_relabd,
                cluster_labels_polluted=top_polluted_labels,
                title=f"Least Polluted {n_ref_int} (avg: {avg_score_ref:.2f}) vs Most Polluted {n_ref_int} (avg: {avg_score_pol:.2f}): Taxa Trends",
                label_ref=f"Least Polluted {n_ref_int}",
                label_polluted=f"Most Polluted {n_ref_int}",
            )
            save_figure(fig_cmp, cp_figures / "taxa_trend_ref_vs_polluted",
                        formats=figure_formats, verbose=verbose)
            plt.close(fig_cmp)
        else:
            _log("      WARNING: No polluted sites with cluster labels found.")

        _log("[P2-6] Creating env trend comparison plot ...")
        env_ref_plot = env_block.loc[labels_ref.index, env_vars_present]
        if len(top_polluted_idx) > 0:
            env_pol_plot = env_block.loc[
                top_polluted_idx.intersection(env_block.index),
                env_vars_present,
            ].dropna()
            labels_pol_plot = cluster_all.loc[env_pol_plot.index].astype(int)
        else:
            nonref_with_labels = nonref_preds.dropna()
            env_pol_plot = env_block.loc[
                nonref_with_labels.index.intersection(env_block.index),
                env_vars_present,
            ].dropna()
            labels_pol_plot = nonref_preds.loc[env_pol_plot.index].astype(int)
        fig_env, _ = plot_env_trend_comparison(
            env_ref=env_ref_plot,
            cluster_labels_ref=labels_ref,
            env_nonref=env_pol_plot,
            cluster_labels_nonref=labels_pol_plot,
            env_variables=env_vars_present,
            title=f"Least Polluted {n_ref_int} vs Most Polluted {n_ref_int}: Env Features Across LDA Clusters",
            label_ref=f"Least Polluted {n_ref_int}",
            label_nonref=f"Most Polluted {n_ref_int}",
        )
        save_figure(fig_env, cp_figures / "env_trend_ref_vs_nonref",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_env)

    # -- Assemble LDA result -------------------------------------------
    lda_result = LDAResult(
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

    _log(f"\n> Pipeline complete.  {n_ref} ref sites, {n_clusters} clusters")
    _log(f"  {lda_result.summary()}")

    return {
        "lda": lda_result,
        "ref_mask": ref_mask,
        "labels_ref": labels_ref,
        "cluster_all": cluster_all,
        "site_robustness_augmented": augmented_robustness,
    }
