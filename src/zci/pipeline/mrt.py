"""Stage 2 -- MRT Classification Pipeline.

Requires pre-computed Ward clustering labels from the WardsClustering pipeline.

Phase 1 (classifier_training): reference sites only
    use Ward cluster labels -> tree classifier fit/prune by CVRE
    -> env PCA ordination -> save.

Phase 2 (classifier_prediction): non-reference sites
    tree predict on non-ref sites -> save tables -> save artifact.
"""

from __future__ import annotations

import pickle
from pathlib import Path as _Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.mrt import fit_mrt
from ..core.transforms import (
    octave_to_chord,
    octave_to_hellinger,
    octave_to_log_chord,
    octave_to_relative_abundance,
    octave_transform,
)
from ..core.lda import build_confusion_matrix_table
from ..io.readers import extract_block, read_study_data
from ..io.writers import save_table, save_figure
from ..models.clustering import TAXA_COLUMNS
from ..models.mrt import MRTResult
from ..viz.mrt_plots import save_mrt_cp_tree_figure
from ..viz.taxa_trend_grid import plot_taxa_trend_comparison, plot_env_trend_comparison
from ..viz.ordination_plots import save_env_pca_ordination


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
    site_robustness: pd.DataFrame,
    output_prefix: str = "",
    taxa_columns: Sequence[str] = TAXA_COLUMNS,
    env_variables: Sequence[str] | None = None,
    env_short: Sequence[str] | None = None,
    response_transform: str = "chord",
    k_folds: int = 10,
    cv_perms: int = 100,
    minsplit: int = 5,
    minbucket: int = 2,
    random_state: int | None = 42,
    save_plots: bool = True,
    figure_formats: Sequence[str] = ("png",),
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> MRTResult:
    """Run the MRT classifier pipeline using pre-computed Ward clustering.

    Parameters
    ----------
    site_robustness : pd.DataFrame
        Site robustness table from WardsClustering (index=Site, columns include
        Original_Cluster, Branch_AU, Silhouette, etc.).
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
    #  Extract cluster labels from site robustness table
    # ============================================================
    ward_labels_ref = site_robustness["Original_Cluster"].copy()
    n_clusters = int(ward_labels_ref.nunique())

    _log("=" * 60)
    _log("  MRT PHASE 1: Classifier Training")
    _log("=" * 60)

    _log(f"       {len(ward_labels_ref)} reference sites, {n_clusters} clusters (from Ward)")
    for g in sorted(ward_labels_ref.unique()):
        _log(f"         Group {g}: {(ward_labels_ref == g).sum()} sites")

    # -- 1. Read original data -----------------------------------------
    _log("[1/6] Reading original study data...")
    data = read_study_data(data_path)
    _log(f"      {data.shape[0]} sites x {data.shape[1]} variables")

    # Build reference mask over all sites
    ref_mask = pd.Series(False, index=data.index, name="Is_Reference")
    ref_mask.loc[ref_mask.index.isin(ward_labels_ref.index)] = True
    reference_quantile = float(ref_mask.sum()) / len(ref_mask)
    ward_linkage = None

    # -- 2. Read Stage 1 artifact for pollution scores -----------------
    _log("[2/6] Reading Stage 1 artifact for site scores...")
    stage1 = pd.read_excel(stage1_artifact, header=[0, 1, 2], index_col=0)
    score_cols = [
        c for c in stage1.columns
        if c[0] == "01_pollution_assessment" and c[1] == "raw" and c[2].endswith("_Score")
    ]
    if not score_cols:
        raise KeyError("No pollution score column found in Stage 1 artifact")
    pollution_scores = stage1.loc[:, score_cols[0]].rename("Pollution_Score")

    # -- 3. Prepare taxa and environmental data ------------------------
    _log("[3/6] Preparing taxa and environmental data...")
    taxa_all = extract_block(data, "taxa", "raw")[list(taxa_columns)]
    env_all = extract_block(data, "environmental", "raw")

    ref_stations = ward_labels_ref.index
    taxa_ref_oct = taxa_all.loc[ref_stations].copy()
    taxa_response = response_transform_fn(taxa_ref_oct)
    taxa_response.index = ref_stations
    env_ref_raw = env_all.loc[ref_stations, list(env_variables)].copy().dropna()

    # Align to complete-case sites
    taxa_ref_oct = taxa_ref_oct.loc[env_ref_raw.index]
    taxa_response = taxa_response.loc[env_ref_raw.index]
    ward_labels_aligned = ward_labels_ref.loc[env_ref_raw.index]
    ref_mask_aligned = ref_mask.copy()
    ref_mask_aligned.loc[:] = False
    ref_mask_aligned.loc[env_ref_raw.index] = True
    ref_stations = env_ref_raw.index
    env_ref = env_ref_raw.copy()
    env_ref.columns = list(env_short)

    _log(
        f"      Transform '{response_transform}' applied -> {taxa_response.shape[1]} taxa; "
        f"{env_ref.shape[1]} environmental predictors"
    )

    # -- 4. Train decision-tree classifier -----------------------------
    _log(
        f"[4/6] Training decision-tree classifier ({k_folds}-fold CV x {cv_perms} random assignments)..."
    )
    result = fit_mrt(
        taxa_response,
        env_ref,
        cluster_labels=ward_labels_aligned,
        ward_linkage=ward_linkage,
        ref_mask=ref_mask_aligned,
        ref_stations=ref_stations,
        taxa_ref_octave=taxa_ref_oct.loc[env_ref.index],
        reference_quantile=reference_quantile,
        response_transform=response_transform,
        env_variables=list(env_short),
        taxa_columns=list(taxa_columns),
        n_clusters=n_clusters,
        k_folds=k_folds,
        cv_perms=cv_perms,
        minsplit=minsplit,
        minbucket=minbucket,
        random_state=random_state,
    )
    _log(f"      Full tree: {result.full_tree_splits} splits, {result.full_tree_leaves} leaves")
    _log(f"      Effective CV: {result.effective_k_folds}-fold x {result.cv_perms} random assignments")
    _log("      Variables used in full-tree splits:")
    for name, count in result.variable_counts.items():
        _log(f"        {name}: {int(count)} splits")

    # -- 5. Select smallest-CVRE tree ----------------------------------
    _log("[5/6] Selecting smallest-CVRE tree...")
    selected_size = result.pruned_nsplits + 1
    _log(
        f"      Min CVRE: {result.min_cv_error:.4f} (SE={result.min_cv_se:.4f}) at size {selected_size}"
    )
    _log(
        f"      Pruned tree: {result.pruned_leaves} leaves ({result.pruned_nsplits} splits), cp={result.best_cp:.6f}"
    )

    if result.pruned_leaves == 1:
        _log(
            "\n  WARNING: Pruned tree has only 1 leaf (root-only). "
            "All reference sites fall into a single cluster.\n"
        )

    # -- 6. Save classifier training outputs ---------------------------
    _log("[6/6] Saving classifier training outputs...")
    figure_path = save_mrt_cp_tree_figure(result, ct_figures / f"{output_prefix}mrt_cp_tree.png")
    if verbose:
        print(f"  > Saved figure: {figure_path}")

    # CP table
    save_table(result.cp_table, ct_tables / f"{output_prefix}mrt_cp_table", verbose=verbose)

    # Leaf membership
    leaf_path = ct_tables / f"{output_prefix}mrt_leaf_membership.xlsx"
    leaf_path.parent.mkdir(parents=True, exist_ok=True)
    result.leaf_membership.to_excel(leaf_path, index=False)
    if verbose:
        print(f"  > Saved table: {leaf_path}")

    # Reference taxa clusters table
    ref_table = result.to_ref_table()
    ref_table_path = ct_tables / f"{output_prefix}reference_taxa_clusters.xlsx"
    ref_table.to_excel(ref_table_path)
    if verbose:
        print(f"  > Saved table: {ref_table_path}")

    # ── Majority-vote leaf→cluster mapping and merged table ───────
    leaf_mem = result.leaf_membership.copy()
    leaf_mem = leaf_mem.set_index("StationID")
    true_labels_aligned = result.cluster_labels_ref.loc[leaf_mem.index]

    # For each leaf, find the majority Ward cluster
    leaf_to_cluster: dict[int, int] = {}
    for leaf_id in leaf_mem["Leaf"].unique():
        stations_in_leaf = leaf_mem.index[leaf_mem["Leaf"] == leaf_id]
        majority = int(true_labels_aligned.loc[stations_in_leaf].mode().iloc[0])
        leaf_to_cluster[int(leaf_id)] = majority

    # Predicted cluster from majority vote
    pred_cluster = leaf_mem["Leaf"].map(leaf_to_cluster).rename("Pre_Cluster")

    # Merged table: StationID | Ward_Cluster | Pre_Cluster | taxa columns ...
    merged = ref_table.copy()
    merged.insert(0, "Pre_Cluster", pred_cluster.loc[merged.index].values)
    merged.rename(columns={"Cluster": "Ward_Cluster"}, inplace=True)
    merged_path = ct_tables / f"{output_prefix}mrt_cluster_comparison.xlsx"
    merged.to_excel(merged_path)
    if verbose:
        print(f"  > Saved table: {merged_path}")

    # Confusion matrix: Ward_Cluster (true) vs Pre_Cluster (predicted)
    unique_clusters = sorted(merged["Ward_Cluster"].unique())
    cluster_names = [f"Cluster {int(c)}" for c in unique_clusters]
    from sklearn.metrics import confusion_matrix as _sk_cm
    cm = _sk_cm(
        merged["Ward_Cluster"].values,
        merged["Pre_Cluster"].values,
        labels=unique_clusters,
    )
    confusion_df = build_confusion_matrix_table(cm, cluster_names)
    save_table(confusion_df, ct_tables / f"{output_prefix}mrt_confusion_matrix",
               formats=table_formats, verbose=verbose)

    # Pickle artifact
    ct_artifacts.mkdir(parents=True, exist_ok=True)
    artifact_path = ct_artifacts / f"{output_prefix}mrt_model.pkl"
    with artifact_path.open("wb") as f:
        pickle.dump(result, f)
    if verbose:
        print(f"  > Saved artifact: {artifact_path}")

    # Save augmented site robustness table
    augmented_robustness = site_robustness.copy()
    augmented_robustness["Predicted_Cluster"] = pred_cluster.reindex(
        augmented_robustness.index
    )
    rob_path = ct_artifacts / f"{output_prefix}site_robustness.xlsx"
    augmented_robustness.to_excel(rob_path)
    if verbose:
        print(f"  > Saved augmented robustness: {rob_path}")

    # PCA ordination biplot in environmental space
    labels_ref = result.cluster_labels
    if save_plots:
        _log("  Saving PCA ordination biplot ...")
        pca_path = save_env_pca_ordination(
            env_ref=env_ref,
            true_labels=labels_ref,
            predicted_labels=pred_cluster.loc[labels_ref.index],
            output_path=ct_figures / f"{output_prefix}env_pca_ordination.png",
            env_feature_names=list(env_short),
        )
        if verbose:
            print(f"  > Saved figure: {pca_path}")

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

    _log(f"\n  Selected tree: {result.pruned_leaves} leaves")
    selected = result.cp_table[result.cp_table["nsplit"] == result.pruned_nsplits].iloc[0]
    _log(
        f"  RE: {selected['rel error']:.3f}   CVRE: {selected['CV error']:.3f}   SE: {selected['CV std']:.3f}"
    )

    # ============================================================
    #  PHASE 2: Classifier Prediction (Non-Reference Sites)
    # ============================================================
    _log("\n" + "=" * 60)
    _log("  MRT PHASE 2: Classifier Prediction (Non-Reference Sites)")
    _log("=" * 60)

    _log("[P2-1/3] Predicting non-reference sites via classifier tree ...")
    nonref_mask = ~ref_mask_aligned
    env_nonref_raw = env_all.loc[nonref_mask, list(env_variables)].copy().dropna()
    env_nonref = env_nonref_raw.copy()
    env_nonref.columns = list(env_short)

    pred_clusters = pd.Series(
        result.classifier_model.predict(env_nonref),
        index=env_nonref.index,
        name="Predicted_Cluster",
    )
    _log(f"      Predicted {len(pred_clusters)} non-reference sites")
    for g in sorted(pred_clusters.unique()):
        _log(f"        Cluster {int(g)}: {(pred_clusters == g).sum()} sites")

    _log("[P2-2/3] Saving prediction tables ...")
    pred_table = pd.DataFrame({"Predicted_Cluster": pred_clusters}, index=env_nonref.index)
    save_table(pred_table, cp_tables / f"{output_prefix}mrt_predictions",
               formats=table_formats, verbose=verbose)

    _log("[P2-3/3] Saving prediction artifact ...")
    cp_artifacts.mkdir(parents=True, exist_ok=True)

    cluster_all = pd.Series(pd.NA, index=data.index, name="Cluster")
    cluster_all.loc[labels_ref.index] = labels_ref.values
    cluster_all.loc[pred_clusters.index] = pred_clusters.values

    cols = pd.MultiIndex.from_tuples([
        ("02_taxa_assemblage_mrt", "raw", "Predicted_Cluster"),
        ("02_taxa_assemblage_mrt", "raw", "Is_Reference"),
    ])
    aug = pd.DataFrame(index=data.index, columns=cols)
    aug[("02_taxa_assemblage_mrt", "raw", "Predicted_Cluster")] = cluster_all
    aug[("02_taxa_assemblage_mrt", "raw", "Is_Reference")] = ref_mask_aligned.astype(int)
    aug_path = cp_artifacts / f"{output_prefix}mrt_predicted_data.xlsx"
    aug.to_excel(aug_path)
    if verbose:
        print(f"  > Saved prediction artifact: {aug_path}")

    # ── Taxa trend grid plots ─────────────────────────────────────
    if save_plots:
        _log("[P2-4] Creating taxa trend grid plots ...")
        cp_figures.mkdir(parents=True, exist_ok=True)

        taxa_ref_relabd = octave_to_relative_abundance(taxa_all.loc[labels_ref.index])
        avg_score_ref = pollution_scores.loc[labels_ref.index].mean()

        n_ref = int(ref_mask_aligned.sum())
        top_polluted_idx = pollution_scores.nlargest(n_ref).index
        top_polluted_idx = top_polluted_idx.intersection(cluster_all.dropna().index)
        if len(top_polluted_idx) > 0:
            top_polluted_labels = cluster_all.loc[top_polluted_idx].astype(int)
            avg_score_pol = pollution_scores.loc[top_polluted_idx].mean()
            taxa_top_relabd = octave_to_relative_abundance(
                taxa_all.loc[top_polluted_idx.intersection(taxa_all.index)]
            )
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

        _log("[P2-5] Creating env trend comparison plot ...")
        env_ref_plot = env_all.loc[labels_ref.index, list(env_variables)]
        if len(top_polluted_idx) > 0:
            env_pol_plot = env_all.loc[top_polluted_idx.intersection(env_all.index), list(env_variables)].dropna()
            labels_pol_plot = cluster_all.loc[env_pol_plot.index].astype(int)
        else:
            nonref_with_labels = pred_clusters.dropna()
            env_pol_plot = env_all.loc[nonref_with_labels.index.intersection(env_all.index), list(env_variables)].dropna()
            labels_pol_plot = pred_clusters.loc[env_pol_plot.index].astype(int)
        fig_env, _ = plot_env_trend_comparison(
            env_ref=env_ref_plot,
            cluster_labels_ref=labels_ref,
            env_nonref=env_pol_plot,
            cluster_labels_nonref=labels_pol_plot,
            env_variables=list(env_variables),
            title="Least Polluted vs Most Polluted: Env Features Across MRT Clusters",
        )
        save_figure(fig_env, cp_figures / f"{output_prefix}env_trend_ref_vs_nonref",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_env)

    _log("\n" + "=" * 60)
    _log("  MRT pipeline complete.")
    _log("=" * 60)

    return result
