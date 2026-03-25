"""Stage MRT -- Multivariate Regression Tree pipeline.

Phase 1 (cluster_classifier): reference sites only
  read data -> pollution scores -> select ref sites -> transform taxa
  -> fit MRT -> prune -> ANOVA -> cluster panel -> save.

Phase 2 (classifier_prediction): non-reference sites
  MRT predict on non-ref sites -> save tables -> save artifact.
"""

from __future__ import annotations

import pickle
from pathlib import Path as _Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.clustering import select_reference_sites
from ..core.mrt import fit_mrt, _assign_dataframe, _r_vector
from ..core.transforms import (
    octave_to_chord,
    octave_to_relative_abundance,
    octave_to_hellinger,
    octave_to_log_chord,
    octave_transform,
)
from ..core.anova import anova_table, extract_pvalues
from ..io.readers import extract_block, read_study_data
from ..io.writers import save_table, save_figure
from ..models.clustering import TAXA_COLUMNS
from ..models.mrt import MRTResult
from ..viz.mrt_plots import save_mrt_cp_tree_figure
from ..viz.cluster_panel_plot import plot_cluster_panel, TAXA_DISPLAY_ORDER
from ..viz.taxa_trend_grid import plot_taxa_trend_grid, plot_taxa_trend_comparison, plot_env_trend_comparison

from rpy2 import robjects as ro


_TRANSFORMS = {
    "octave": octave_transform,
    "relative_abundance": octave_to_relative_abundance,
    "chord": octave_to_chord,
    "hellinger": octave_to_hellinger,
    "log_chord": octave_to_log_chord,
}


def mrt_pipeline(
    data_path: str | _Path,
    stage1_artifact: str | _Path,
    output_dir: str | _Path,
    maps_dir: str | _Path,
    *,
    output_prefix: str = "",
    taxa_columns: Sequence[str] = TAXA_COLUMNS,
    env_variables: Sequence[str] | None = None,
    env_short: Sequence[str] | None = None,
    response_transform: str = "chord",
    reference_quantile: float = 0.25,
    anova_transform: str = "none",
    k_folds: int = 10,
    cv_perms: int = 100,
    minsplit: int = 5,
    minbucket: int = 2,
    save_plots: bool = True,
    figure_formats: Sequence[str] = ("png",),
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> MRTResult:
    """Run the MRT pipeline with cluster_classifier / classifier_prediction layout."""
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
    if env_short is None:
        env_short = ["Depth", "DO", "Temp", "MPS", "LOI"]

    if len(env_variables) != len(env_short):
        raise ValueError("env_variables and env_short must have the same length")

    # Determine transform function
    if response_transform in _TRANSFORMS:
        response_transform_fn = _TRANSFORMS[response_transform]
    else:
        raise ValueError(
            f"Unknown response_transform={response_transform!r}. "
            f"Choose from {sorted(_TRANSFORMS)}."
        )

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # ============================================================
    #  PHASE 1: Cluster Classifier (reference sites)
    # ============================================================
    _log("=" * 60)
    _log("  MRT PHASE 1: Cluster Classifier (Reference Sites)")
    _log("=" * 60)

    _log("[1/9] Reading original study data...")
    data = read_study_data(data_path)
    _log(f"      {data.shape[0]} sites x {data.shape[1]} variables")

    _log("[2/9] Reading Stage 1 artifact for SumRel scores...")
    stage1 = pd.read_excel(stage1_artifact, header=[0, 1, 2], index_col=0)
    score_cols = [
        c for c in stage1.columns
        if c[0] == "01_pollution_assessment" and c[1] == "raw" and c[2].endswith("_Score")
    ]
    if not score_cols:
        raise KeyError("No pollution score column found in Stage 1 artifact")

    pollution_scores = stage1.loc[:, score_cols[0]].rename("SumRel_Score")
    _log(
        f"      Score range: [{pollution_scores.min():.4f}, {pollution_scores.max():.4f}]"
    )

    _log(f"[3/9] Selecting reference sites (bottom {reference_quantile * 100:.0f}%)...")
    ref_mask = select_reference_sites(pollution_scores, quantile=reference_quantile)
    threshold = float(pollution_scores.loc[ref_mask].max())
    ref_stations = pollution_scores.index[ref_mask]
    _log(
        f"      {len(ref_stations)} reference sites out of {len(ref_mask)} total "
        f"(threshold = {threshold:.4f})"
    )

    _log("[4/9] Preparing data for MRT...")
    taxa_all = extract_block(data, "taxa", "raw")[list(taxa_columns)]
    env_all = extract_block(data, "environmental", "raw")

    taxa_ref_oct = taxa_all.loc[ref_stations].copy()
    taxa_response = response_transform_fn(taxa_ref_oct)
    taxa_response.index = ref_stations
    _log(
        f"      Transform '{response_transform}' applied -> {taxa_response.shape[1]} taxa"
    )

    env_ref = env_all.loc[ref_stations, list(env_variables)].copy()
    env_ref.columns = list(env_short)
    env_ref = env_ref.dropna()
    taxa_response = taxa_response.loc[env_ref.index]
    ref_mask = ref_mask.copy()
    ref_mask.loc[:] = False
    ref_mask.loc[env_ref.index] = True
    ref_stations = env_ref.index

    _log(
        f"      {taxa_response.shape[0]} sites x {taxa_response.shape[1]} taxa x {env_ref.shape[1]} env predictors"
    )

    _log(f"[5/9] Growing full tree ({k_folds}-fold CV, {cv_perms} permutations)...")
    result, runtime = fit_mrt(
        taxa_response,
        env_ref,
        ref_mask=ref_mask,
        ref_stations=ref_stations,
        taxa_ref_octave=taxa_ref_oct.loc[env_ref.index],
        reference_quantile=reference_quantile,
        response_transform=response_transform,
        env_variables=list(env_variables),
        taxa_columns=list(taxa_columns),
        k_folds=k_folds,
        cv_perms=cv_perms,
        minsplit=minsplit,
        minbucket=minbucket,
    )
    _log(f"      Full tree: {result.full_tree_splits} splits, {result.full_tree_leaves} leaves")
    _log(f"      CV: {result.k_folds}-fold x {result.cv_perms} permutations")
    _log("      Variables used in splits:")
    for name, count in result.variable_counts.items():
        _log(f"        {name}: {int(count)} splits")

    _log("[6/9] Pruning tree (minimum CVRE rule)...")
    selected_size = int(result.cp_table.loc[result.cp_table['xerror'].idxmin(), 'nsplit']) + 1
    _log(
        f"      Min CV error: {result.min_cv_error:.4f} (SE={result.min_cv_se:.4f}) at size {selected_size}"
    )
    _log(
        f"      Pruned tree: {result.pruned_leaves} leaves ({result.pruned_nsplits} splits), cp={result.best_cp:.6f}"
    )

    if result.pruned_leaves == 1:
        _log(
            "\n  WARNING: Pruned tree has only 1 leaf (root-only). "
            "All reference sites fall into a single cluster.\n"
            "  ANOVA, cluster panel, and Phase 2 predictions will "
            "reflect a single group.\n"
        )

    _log("[7/9] Saving Phase 1 outputs...")
    # CP/tree figure
    figure_path = save_mrt_cp_tree_figure(result, runtime, cc_figures / f"{output_prefix}mrt_cp_tree.png")
    if verbose:
        print(f"  > Saved figure: {figure_path}")

    # CP table
    save_table(result.cp_table, cc_tables / f"{output_prefix}mrt_cp_table", verbose=verbose)

    # Leaf membership
    leaf_path = cc_tables / f"{output_prefix}mrt_leaf_membership.xlsx"
    leaf_path.parent.mkdir(parents=True, exist_ok=True)
    result.leaf_membership.to_excel(leaf_path, index=False)
    if verbose:
        print(f"  > Saved table: {leaf_path}")

    # Reference taxa clusters table
    ref_table = result.to_ref_table()
    ref_table_path = cc_tables / f"{output_prefix}reference_taxa_clusters.xlsx"
    ref_table.to_excel(ref_table_path)
    if verbose:
        print(f"  > Saved table: {ref_table_path}")

    # Pickle artifact
    cc_artifacts.mkdir(parents=True, exist_ok=True)
    artifact_path = cc_artifacts / f"{output_prefix}mrt_model.pkl"
    with artifact_path.open("wb") as f:
        pickle.dump(result, f)
    if verbose:
        print(f"  > Saved artifact: {artifact_path}")

    # ── ANOVA on MRT leaf clusters ────────────────────────────────
    labels_ref = result.cluster_labels
    _log("[8/9] Running ANOVA tests on MRT clusters ...")

    env_ref_long = env_all.loc[labels_ref.index]
    env_vars_present = [v for v in env_variables if v in env_ref_long.columns]
    env_ref_anova = env_ref_long[env_vars_present]

    taxa_ref_anova = result.taxa_ref_octave

    env_anova = anova_table(
        env_ref_anova, labels_ref, env_vars_present,
        transform=anova_transform, label_col="Variable",
    )
    save_table(env_anova, cc_tables / f"{output_prefix}anova_env",
               formats=table_formats, verbose=verbose)
    env_pvals = extract_pvalues(env_anova, label_col="Variable")

    taxa_anova = anova_table(
        taxa_ref_anova, labels_ref, list(taxa_ref_anova.columns),
        transform=anova_transform, label_col="Taxon",
    )
    save_table(taxa_anova, cc_tables / f"{output_prefix}anova_taxa",
               formats=table_formats, verbose=verbose)
    taxa_pvals = extract_pvalues(taxa_anova, label_col="Taxon")

    # ── Cluster panel figure ──────────────────────────────────────
    if save_plots:
        _log("[9/9] Saving cluster panel figure ...")
        sample_info = extract_block(data, "sample_info", "raw")
        lat = sample_info.loc[labels_ref.index, "Latitude"]
        lon = sample_info.loc[labels_ref.index, "Longitude"]
        taxa_relabd = octave_to_relative_abundance(taxa_ref_anova)

        fig_panel, _ = plot_cluster_panel(
            cluster_labels=labels_ref,
            lat=lat,
            lon=lon,
            env_data=env_ref_anova,
            taxa_octave=taxa_ref_anova,
            taxa_relabd=taxa_relabd,
            env_pvalues=env_pvals,
            taxa_pvalues=taxa_pvals,
            maps_dir=maps_dir,
            env_vars=env_vars_present,
            taxa_order=TAXA_DISPLAY_ORDER,
        )
        save_figure(fig_panel, cc_figures / f"{output_prefix}cluster_panel",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_panel)

    _log("\n> MRT Phase 1 complete.")

    # Print summary
    _log("-" * 50)
    _log(f"  CP Table (Full Tree, {result.k_folds}-fold CV x {result.cv_perms} perms)")
    _log("-" * 50)
    _log(
        f"\nRoot node error: {result.root_node_error:.1f} / "
        f"{taxa_response.shape[0]} = {result.root_node_error / taxa_response.shape[0]:.3f}"
    )
    _log(f"n = {taxa_response.shape[0]}\n")
    if verbose:
        print(result.cp_table)

    _log(f"\n  Pruned tree: {result.pruned_leaves} leaves")
    selected = result.cp_table[result.cp_table["nsplit"] == result.pruned_nsplits].iloc[0]
    _log(
        f"  Error: {selected['rel error']:.3f}   CV Error: {selected['xerror']:.3f}   SE: {selected['xstd']:.3f}"
    )

    # ============================================================
    #  PHASE 2: Classifier Prediction (non-reference sites)
    # ============================================================
    _log("\n" + "=" * 60)
    _log("  MRT PHASE 2: Classifier Prediction (Non-Reference Sites)")
    _log("=" * 60)

    # -- P2-1. Predict non-reference sites using R's predict --------
    _log("[P2-1/3] Predicting non-reference sites via MRT ...")
    nonref_mask = ~ref_mask
    env_nonref = env_all.loc[nonref_mask, list(env_variables)].copy()
    env_nonref.columns = list(env_short)
    env_nonref = env_nonref.dropna()

    _assign_dataframe("zci_mrt_nonref_env", env_nonref)
    ro.r("zci_mrt_nonref_pred <- predict(zci_mrt_pruned, newdata = zci_mrt_nonref_env, type = 'matrix')")
    pred_raw = ro.r("zci_mrt_nonref_pred")
    pred_matrix = np.array(pred_raw)

    # predict() returns a matrix of fitted response values (sites x taxa).
    # Each row is the leaf centroid for the leaf that site falls into.
    # Build leaf centroids from training data to match predicted rows.
    leaf_centroids = {}
    for leaf_id in sorted(result.leaf_membership["Leaf"].unique()):
        leaf_stations = result.leaf_membership.loc[
            result.leaf_membership["Leaf"] == leaf_id, "StationID"
        ].values
        leaf_centroids[leaf_id] = taxa_response.loc[leaf_stations].mean().values

    # Map leaf IDs to sequential cluster labels
    unique_leaves = sorted(leaf_centroids.keys())
    leaf_to_cluster = {leaf: i + 1 for i, leaf in enumerate(unique_leaves)}
    centroid_arr = np.array([leaf_centroids[l] for l in unique_leaves])

    # Match each predicted row to its nearest centroid
    from scipy.spatial.distance import cdist
    if pred_matrix.ndim == 1:
        pred_matrix = pred_matrix.reshape(1, -1)
    # Ensure pred_matrix columns match centroid columns
    if pred_matrix.shape[1] != centroid_arr.shape[1]:
        # R may return a flattened or differently shaped matrix; reshape
        n_taxa = centroid_arr.shape[1]
        if pred_matrix.size % n_taxa == 0:
            pred_matrix = pred_matrix.reshape(-1, n_taxa)
        else:
            raise ValueError(
                f"Prediction matrix shape {pred_matrix.shape} incompatible "
                f"with centroid shape {centroid_arr.shape}"
            )
    dists = cdist(pred_matrix, centroid_arr, metric="euclidean")
    nearest = dists.argmin(axis=1)
    pred_clusters = pd.Series(
        [leaf_to_cluster[unique_leaves[i]] for i in nearest],
        index=env_nonref.index,
        name="Predicted_Cluster",
    )
    _log(f"      Predicted {len(pred_clusters)} non-reference sites")
    for g in sorted(pred_clusters.unique()):
        _log(f"        Cluster {int(g)}: {(pred_clusters == g).sum()} sites")

    # -- P2-2. Save prediction tables ----------------------------------
    _log("[P2-2/3] Saving prediction tables ...")
    pred_table = pd.DataFrame({"Predicted_Cluster": pred_clusters}, index=env_nonref.index)
    save_table(pred_table, cp_tables / f"{output_prefix}mrt_predictions",
               formats=table_formats, verbose=verbose)

    # -- P2-3. Save prediction artifact --------------------------------
    _log("[P2-3/3] Saving prediction artifact ...")
    cp_artifacts.mkdir(parents=True, exist_ok=True)

    cluster_all = pd.Series(np.nan, index=data.index, name="Cluster")
    cluster_all.loc[labels_ref.index] = labels_ref.values
    cluster_all.loc[pred_clusters.index] = pred_clusters.values

    cols = pd.MultiIndex.from_tuples([
        ("02_taxa_assemblage_mrt", "raw", "Predicted_Cluster"),
        ("02_taxa_assemblage_mrt", "raw", "Is_Reference"),
    ])
    aug = pd.DataFrame(index=data.index, columns=cols)
    aug[("02_taxa_assemblage_mrt", "raw", "Predicted_Cluster")] = cluster_all
    aug[("02_taxa_assemblage_mrt", "raw", "Is_Reference")] = ref_mask.astype(int)
    aug_path = cp_artifacts / f"{output_prefix}mrt_predicted_data.xlsx"
    aug.to_excel(aug_path)
    if verbose:
        print(f"  > Saved prediction artifact: {aug_path}")

    # ── Taxa trend grid plots ─────────────────────────────────────
    if save_plots:
        _log("[P2-4] Creating taxa trend grid plots ...")
        cp_figures.mkdir(parents=True, exist_ok=True)

        # Reference sites: taxa trend grid
        taxa_ref_relabd = octave_to_relative_abundance(taxa_all.loc[labels_ref.index])
        avg_score_ref = pollution_scores.loc[labels_ref.index].mean()
        fig_ref_trend, _ = plot_taxa_trend_grid(
            taxa_relabd=taxa_ref_relabd,
            cluster_labels=labels_ref,
            title=f"Reference Sites: Taxa Trends Across MRT Clusters (avg score: {avg_score_ref:.2f})",
        )
        save_figure(fig_ref_trend, cp_figures / f"{output_prefix}taxa_trend_ref",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_ref_trend)

        # Most polluted sites: top quantile by pollution score (same n as ref)
        n_ref = int(ref_mask.sum())
        top_polluted_idx = pollution_scores.nlargest(n_ref).index
        # Keep only those with predicted cluster labels
        top_polluted_idx = top_polluted_idx.intersection(cluster_all.dropna().index)
        if len(top_polluted_idx) > 0:
            top_polluted_labels = cluster_all.loc[top_polluted_idx].astype(int)
            avg_score_pol = pollution_scores.loc[top_polluted_idx].mean()
            taxa_top_relabd = octave_to_relative_abundance(
                taxa_all.loc[top_polluted_idx.intersection(taxa_all.index)]
            )
            fig_top_trend, _ = plot_taxa_trend_grid(
                taxa_relabd=taxa_top_relabd,
                cluster_labels=top_polluted_labels,
                title=f"Most Polluted Sites (Top 25%): Taxa Trends Across MRT Clusters (avg score: {avg_score_pol:.2f})",
            )
            save_figure(fig_top_trend, cp_figures / f"{output_prefix}taxa_trend_most_polluted",
                        formats=figure_formats, verbose=verbose)
            plt.close(fig_top_trend)
            # Combined comparison figure: ref vs most polluted
            fig_cmp, _ = plot_taxa_trend_comparison(
                taxa_relabd_ref=taxa_ref_relabd,
                cluster_labels_ref=labels_ref,
                taxa_relabd_polluted=taxa_top_relabd,
                cluster_labels_polluted=top_polluted_labels,
                title=f"Reference (avg: {avg_score_ref:.2f}) vs Most Polluted (avg: {avg_score_pol:.2f}): Taxa Trends Across MRT Clusters",
            )
            save_figure(fig_cmp, cp_figures / f"{output_prefix}taxa_trend_ref_vs_polluted",
                        formats=figure_formats, verbose=verbose)
            plt.close(fig_cmp)
        else:
            _log("      WARNING: No polluted sites with cluster labels found.")

        # Environmental trend comparison: ref vs non-ref
        _log("[P2-5] Creating env trend comparison plot ...")
        env_ref_plot = env_all.loc[labels_ref.index, list(env_variables)]
        nonref_with_labels = pred_clusters.dropna()
        env_nonref_plot = env_all.loc[
            nonref_with_labels.index.intersection(env_all.index),
            list(env_variables),
        ].dropna()
        labels_nonref_plot = pred_clusters.loc[env_nonref_plot.index].astype(int)
        fig_env, _ = plot_env_trend_comparison(
            env_ref=env_ref_plot,
            cluster_labels_ref=labels_ref,
            env_nonref=env_nonref_plot,
            cluster_labels_nonref=labels_nonref_plot,
            env_variables=list(env_variables),
            title="Reference vs Non-Reference: Env Features Across MRT Clusters",
        )
        save_figure(fig_env, cp_figures / f"{output_prefix}env_trend_ref_vs_nonref",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_env)

    _log("\n" + "=" * 60)
    _log("  MRT pipeline complete.")
    _log("=" * 60)

    return result
