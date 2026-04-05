"""Stage 2 -- Ward + LDA Combined Pipeline.

Phase 1 (cluster_classifier): reference sites only
  read data -> pollution scores -> select reference sites -> transform taxa
  -> Ward clustering -> relabel -> ANOVA -> cluster panel -> LDA fit
  -> Wilks Lambda -> MCCV -> tables -> triplot -> save artifact.

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
from ..core.transforms import (
    octave_to_relative_abundance,
    octave_transform,
    octave_to_chord,
    octave_to_hellinger,
    octave_to_log_chord,
)
from ..core.clustering import ward_cluster, select_reference_sites, resolve_n_ref
from ..core.anova import anova_table, extract_pvalues
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
from ..models.clustering import TAXA_COLUMNS, ClusteringResult
from ..models.lda import LDAResult
from ..viz.clustering_plots import plot_dendrogram
from ..viz.cluster_panel_plot import plot_cluster_panel, TAXA_DISPLAY_ORDER
from ..viz.lda_plots import plot_lda_triplot, plot_cluster_comparison
from ..viz.ordination_plots import save_env_pca_ordination
from ..viz.taxa_trend_grid import plot_taxa_trend_comparison, plot_env_trend_comparison


_TRANSFORMS = {
    "octave": octave_transform,
    "relative_abundance": octave_to_relative_abundance,
    "chord": octave_to_chord,
    "hellinger": octave_to_hellinger,
    "log_chord": octave_to_log_chord,
}


def _relabel(labels: pd.Series, label_map: Dict[int, int]) -> pd.Series:
    """Remap cluster labels using sentinel-based swap-safe approach."""
    tmp = labels.copy().astype(float)
    sentinel_map: Dict[float, int] = {}
    for i, (old, new) in enumerate(label_map.items()):
        sentinel = -(i + 1000)
        tmp = tmp.replace({float(old): float(sentinel)})
        sentinel_map[float(sentinel)] = new
    for sentinel, new in sentinel_map.items():
        tmp = tmp.replace({sentinel: float(new)})
    return tmp.astype(int)


def taxa_assemblage_pipeline(
    data_path: str | _Path,
    stage1_artifact: str | _Path,
    output_dir: str | _Path,
    maps_dir: str | _Path,
    *,
    taxa_columns: Sequence[str] = TAXA_COLUMNS,
    reference_quantile: int | float = 0.25,
    taxa_transform: str = "chord",
    n_clusters: int = 3,
    label_map: Dict[int, int] | None = None,
    env_variables: Sequence[str] | None = None,
    anova_transform: str = "none",
    standardize_env: bool = True,
    n_mccv_iterations: int = 1000,
    mccv_test_size: float = 0.2,
    random_state: int | None = 42,
    map_func=None,
    save_plots: bool = True,
    figure_formats: Sequence[str] = ("png",),
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the combined Ward clustering + LDA classification pipeline.

    Returns dict with keys: clustering, lda, ref_mask, labels_ref, cluster_all.
    """
    output_dir = _Path(output_dir)
    cc_dir = output_dir / "cluster_classifier"
    cp_dir = output_dir / "classifier_prediction"
    cc_tables = cc_dir / "tables"
    cc_figures = cc_dir / "figures"
    cc_artifacts = cc_dir / "artifacts"
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

    if label_map is None:
        label_map = {}

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # ============================================================
    #  PHASE 1: Cluster Classifier (reference sites)
    # ============================================================
    _ref_hint = f"Least Polluted {int(reference_quantile)} Sites" if reference_quantile > 1 else "Least Polluted Sites"
    _log("=" * 60)
    _log(f"  PHASE 1: Cluster Classifier ({_ref_hint})")
    _log("=" * 60)

    # -- 1. Read original data -----------------------------------------
    _log("[1/16] Reading original study data ...")
    data = read_study_data(data_path)
    _log(f"       {data.shape[0]} sites x {data.shape[1]} variables")

    # -- 2. Read Stage 1 artifact -> Pollution Score -------------------
    _log("[2/16] Reading Stage 1 artifact for pollution scores ...")
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
    _log(f"       Score range: [{pollution_score.min():.4f}, {pollution_score.max():.4f}]")

    # -- 3. Select reference sites -------------------------------------
    _log(f"[3/16] Selecting least polluted {resolve_n_ref(reference_quantile, len(pollution_score))} sites ...")
    ref_mask = select_reference_sites(pollution_score, quantile=reference_quantile)
    n_ref = ref_mask.sum()
    _log(f"       {n_ref} least polluted sites out of {len(ref_mask)} total")

    # -- 4. Extract taxa block -----------------------------------------
    _log(f"[4/16] Extracting taxa data for least polluted {n_ref} sites ...")
    taxa_all = extract_block(data, "taxa", "raw")[list(taxa_columns)]
    taxa_ref = taxa_all.loc[ref_mask]
    _log(f"       {taxa_ref.shape[0]} sites x {taxa_ref.shape[1]} taxa")

    # -- 5. Transform taxa ---------------------------------------------
    _log(f"[5/16] Applying taxa transform: {taxa_transform!r} ...")
    if taxa_transform not in _TRANSFORMS:
        raise ValueError(
            f"Unknown taxa_transform={taxa_transform!r}. "
            f"Choose from {sorted(_TRANSFORMS)}."
        )
    taxa_transformed = _TRANSFORMS[taxa_transform](taxa_ref)

    # -- 6. Ward clustering --------------------------------------------
    _log(f"[6/16] Ward clustering (n_clusters={n_clusters}) ...")
    labels_ref, Z = ward_cluster(taxa_transformed, n_clusters=n_clusters)
    _log("       Cluster distribution (before relabel):")
    for g in sorted(labels_ref.unique()):
        _log(f"         Group {g}: {(labels_ref == g).sum()} sites")

    # -- 7. Relabel clusters -------------------------------------------
    if label_map:
        _log(f"[7/16] Relabelling clusters: {label_map} ...")
        labels_ref = _relabel(labels_ref, label_map)
        labels_ref.name = "Cluster"
        _log("       Cluster distribution (after relabel):")
        for g in sorted(labels_ref.unique()):
            _log(f"         Group {g}: {(labels_ref == g).sum()} sites")
    else:
        _log("[7/16] No relabelling requested, keeping original labels.")

    # Build ClusteringResult
    result_clustering = ClusteringResult(
        ref_mask=ref_mask,
        cluster_labels=labels_ref,
        linkage_matrix=Z,
        taxa_ref=taxa_ref,
        taxa_ref_transformed=taxa_transformed,
        n_clusters=n_clusters,
        taxa_transform=taxa_transform,
        all_site_index=data.index,
    )

    # -- 8. Save dendrogram -------------------------------------------
    if save_plots:
        _log("[8/16] Saving dendrogram ...")
        fig_dend, _ = plot_dendrogram(
            Z,
            labels=taxa_ref.index.astype(str),
            n_clusters=n_clusters,
            title=(
                f"Ward Dendrogram -- {taxa_transform.replace('_', ' ').title()} "
                f"(k = {n_clusters})"
            ),
            ylabel=f"Least Polluted {n_ref} Sites",
        )
        save_figure(fig_dend, cc_figures / "ward_dendrogram",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_dend)

    # -- 9. ANOVA on environmental and taxa variables ------------------
    _log("[9/16] Running ANOVA tests ...")
    env_block = extract_block(data, "environmental", "raw")
    env_vars_present = [v for v in env_variables if v in env_block.columns]
    env_ref_raw = env_block.loc[labels_ref.index, env_vars_present]
    lda_ref_index = env_ref_raw.dropna().index
    if len(lda_ref_index) < len(env_ref_raw):
        _log(
            "       Dropping "
            f"{len(env_ref_raw) - len(lda_ref_index)} reference sites with missing "
            "environmental values for ANOVA/LDA."
        )

    env_ref_complete = env_ref_raw.loc[lda_ref_index]
    labels_ref_complete = labels_ref.loc[lda_ref_index]
    taxa_ref_complete = taxa_ref.loc[lda_ref_index]

    env_anova = anova_table(
        env_ref_complete, labels_ref_complete, env_vars_present,
        transform=anova_transform, label_col="Variable",
    )
    save_table(env_anova, cc_tables / "anova_env",
               formats=table_formats, verbose=verbose)
    env_pvals = extract_pvalues(env_anova, label_col="Variable")

    taxa_anova = anova_table(
        taxa_ref_complete, labels_ref_complete, list(taxa_ref_complete.columns),
        transform=anova_transform, label_col="Taxon",
    )
    save_table(taxa_anova, cc_tables / "anova_taxa",
               formats=table_formats, verbose=verbose)
    taxa_pvals = extract_pvalues(taxa_anova, label_col="Taxon")

    # -- 10. Cluster panel figure --------------------------------------
    if save_plots:
        _log("[10/16] Saving cluster panel figure ...")
        sample_info = extract_block(data, "sample_info", "raw")
        lat = sample_info.loc[labels_ref_complete.index, "Latitude"]
        lon = sample_info.loc[labels_ref_complete.index, "Longitude"]
        taxa_relabd = octave_to_relative_abundance(taxa_ref_complete)

        panel_figures = plot_cluster_panel(
            cluster_labels=labels_ref_complete,
            lat=lat,
            lon=lon,
            env_data=env_ref_complete,
            taxa_octave=taxa_ref_complete,
            taxa_relabd=taxa_relabd,
            env_pvalues=env_pvals,
            taxa_pvalues=taxa_pvals,
            maps_dir=maps_dir,
            env_vars=env_vars_present,
            taxa_order=TAXA_DISPLAY_ORDER,
            map_func=map_func,
            taxa_title=f"Least Polluted {n_ref} Sites: Taxa by Cluster",
        )
        for suffix, (fig_panel, _) in panel_figures.items():
            save_figure(
                fig_panel,
                cc_figures / f"cluster_{suffix}",
                formats=figure_formats,
                verbose=verbose,
            )
            plt.close(fig_panel)

    # -- 11. Save reference taxa clusters table ------------------------
    _log("[11/16] Saving reference taxa clusters table ...")
    save_table(result_clustering.to_ref_table(),
               cc_tables / "reference_taxa_clusters",
               formats=table_formats, verbose=verbose)

    # -- 12. Fit LDA on reference sites --------------------------------
    _log(f"[12/16] Fitting LDA on least polluted {n_ref} sites ...")
    lda_fit = fit_lda(
        env_ref_complete,
        labels_ref_complete.values,
        standardize=standardize_env,
    )
    _log(f"        Accuracy = {lda_fit.accuracy:.2%}")

    # -- 13. Wilks Lambda variable importance --------------------------
    _log("[13/16] Computing Wilks Lambda variable importance ...")
    wilks = wilks_lambda_importance(lda_fit)
    _log(f"        Full-model Wilks Lambda = {wilks.wilks_lambda_full:.4f}")

    # -- 14. Monte Carlo Cross-Validation ------------------------------
    _log(f"[14/16] MCCV ({n_mccv_iterations} iterations) ...")
    mccv = monte_carlo_cv(
        env_ref_complete, labels_ref_complete.values,
        standardize=standardize_env,
        n_iterations=n_mccv_iterations,
        test_size=mccv_test_size,
        random_state=random_state,
    )
    _log(f"        Mean accuracy = {mccv.mean_accuracy:.2%} +/- {mccv.std_accuracy:.2%}")

    # -- 15. Save LDA tables -------------------------------------------
    _log("[15/16] Saving LDA tables ...")
    save_table(build_classification_report_table(lda_fit),
               cc_tables / "lda_classification_report",
               formats=table_formats, verbose=verbose)
    save_table(build_confusion_matrix_table(lda_fit.confusion_matrix, lda_fit.cluster_names),
               cc_tables / "lda_confusion_matrix",
               formats=table_formats, verbose=verbose)
    save_table(wilks.axes_summary,
               cc_tables / "lda_axes_summary",
               formats=table_formats, verbose=verbose)
    env_sig_table = build_env_significance_table(
        lda_fit,
        wilks,
        env_ref_complete,
        labels_ref_complete,
    )
    save_table(env_sig_table,
               cc_tables / "lda_env_significance",
               formats=table_formats, verbose=verbose)
    save_table(
        build_confusion_matrix_table(
            mccv.aggregate_confusion_matrix, mccv.cluster_names,
            note=f"Combined results from {mccv.n_iterations:,} CV iterations "
                 f"(test size: {mccv.test_size*100:.0f}%)",
        ),
        cc_tables / "mccv_confusion_matrix",
        formats=table_formats, verbose=verbose,
    )
    save_table(build_mccv_classification_report_table(mccv),
               cc_tables / "mccv_classification_report",
               formats=table_formats, verbose=verbose)

    # -- 16. LDA triplot (reference sites only) ------------------------
    if save_plots:
        _log("[16/16] Creating LDA triplot ...")
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
        save_figure(fig_tri, cc_figures / "lda_triplot",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_tri)

        # PCA ordination biplot in environmental space
        _log("  Saving PCA ordination biplot ...")
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
            output_path=cc_figures / "env_pca_ordination.png",
            env_feature_names=env_short_names,
            title=f"PCA Ordination of Least Polluted {n_ref} Sites in Environmental Space",
        )
        if verbose:
            print(f"  > Saved figure: {pca_path}")

    # -- Save Phase 1 artifact -----------------------------------------
    augmented = result_clustering.to_augmented_dataframe()
    cc_artifacts.mkdir(parents=True, exist_ok=True)
    aug_path = cc_artifacts / "02_updated_data.xlsx"
    augmented.to_excel(aug_path)
    if verbose:
        print(f"  > Saved augmented data: {aug_path}")

    _log("\n> Phase 1 complete.")

    # ============================================================
    #  PHASE 2: Classifier Prediction (non-reference sites)
    # ============================================================
    _log("\n" + "=" * 60)
    _log(f"  PHASE 2: Classifier Prediction (Most Polluted {n_ref} Sites)")
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

        # Data for comparison figure
        taxa_ref_relabd = octave_to_relative_abundance(taxa_ref)
        avg_score_ref = pollution_score.loc[labels_ref.index].mean()

        # Most polluted sites: top quantile by pollution score (same n as ref)
        n_ref = int(ref_mask.sum())
        top_polluted_idx = pollution_score.nlargest(n_ref).index
        # Keep only those with predicted cluster labels
        top_polluted_idx = top_polluted_idx.intersection(cluster_all.dropna().index)
        if len(top_polluted_idx) > 0:
            top_polluted_labels = cluster_all.loc[top_polluted_idx].astype(int)
            avg_score_pol = pollution_score.loc[top_polluted_idx].mean()
            taxa_top_relabd = octave_to_relative_abundance(
                taxa_all.loc[top_polluted_idx.intersection(taxa_all.index)]
            )
            # Combined comparison figure: ref vs most polluted
            fig_cmp, _ = plot_taxa_trend_comparison(
                taxa_relabd_ref=taxa_ref_relabd,
                cluster_labels_ref=labels_ref,
                taxa_relabd_polluted=taxa_top_relabd,
                cluster_labels_polluted=top_polluted_labels,
                title=f"Least Polluted {n_ref} (avg: {avg_score_ref:.2f}) vs Most Polluted {n_ref} (avg: {avg_score_pol:.2f}): Taxa Trends",
                label_ref=f"Least Polluted {n_ref}",
                label_polluted=f"Most Polluted {n_ref}",
            )
            save_figure(fig_cmp, cp_figures / "taxa_trend_ref_vs_polluted",
                        formats=figure_formats, verbose=verbose)
            plt.close(fig_cmp)
        else:
            _log("      WARNING: No polluted sites with cluster labels found.")

        # Environmental trend comparison: ref vs most polluted
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
            title=f"Least Polluted {n_ref} vs Most Polluted {n_ref}: Env Features Across Ward/LDA Clusters",
            label_ref=f"Least Polluted {n_ref}",
            label_nonref=f"Most Polluted {n_ref}",
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

    _log(f"\n> Pipeline complete.  {result_clustering.summary()}")
    _log(f"  {lda_result.summary()}")

    return {
        "clustering": result_clustering,
        "lda": lda_result,
        "ref_mask": ref_mask,
        "labels_ref": labels_ref,
        "cluster_all": cluster_all,
    }
