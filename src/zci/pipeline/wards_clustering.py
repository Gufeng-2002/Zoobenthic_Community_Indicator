"""Stage 2 -- Ward's Clustering Pipeline with Robustness Assessment.

Standalone clustering of reference sites:
  read data -> pollution scores -> select reference sites -> transform taxa
  -> Ward clustering -> robustness assessment (silhouette, co-assignment,
     optional pvclust) -> ANOVA -> dendrogram -> cluster panel -> save.

The resulting :class:`WardClusteringResult` is intended to be consumed
downstream by the LDA and MRT classification pipelines.
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
from ..core.ward_robustness import (
    compute_silhouettes,
    compute_coassignment,
    site_confidence,
    run_pvclust,
    build_robustness_table,
)
from ..core.env_robustness import (
    compute_env_silhouettes,
    compute_env_coassignment,
    env_site_confidence,
    build_env_coherence_table,
)
from ..core.cross_support import (
    build_updated_combined_table,
    build_class_count_table,
)
from ..core.threshold_grid_search import (
    apply_threshold_configuration,
    grid_search_thresholds,
)
from ..models.clustering import TAXA_COLUMNS
from ..models.ward_clustering import WardClusteringResult
from ..viz.clustering_plots import plot_dendrogram
from ..viz.cluster_panel_plot import plot_cluster_panel, TAXA_DISPLAY_ORDER


_TRANSFORMS = {
    "octave": octave_transform,
    "relative_abundance": octave_to_relative_abundance,
    "chord": octave_to_chord,
    "hellinger": octave_to_hellinger,
    "log_chord": octave_to_log_chord,
}

_ENV_ASSIGNMENT_COLUMNS = [
    "Env_Silhouette",
    "Env_Own_Coassign",
    "Env_BestAlt_Coassign",
    "Env_Margin",
    "Env_Strength",
]


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


def refresh_combined_robustness_outputs(
    data_path: str | _Path,
    output_dir: str | _Path,
    *,
    env_variables: Sequence[str],
    env_strength_method: str = "threshold",
    env_strength_top_pct: float = 0.60,
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """Reapply optimized thresholds to the saved Ward combined outputs.

    The standalone Ward pipeline persists the combined robustness artifact
    before the later LDA-driven threshold optimization step. This helper
    reruns the threshold search against the saved artifact, rewrites the
    combined robustness workbook, and refreshes the related Ward count tables
    so the Ward outputs match the later 2x2 classification used downstream.
    """

    output_dir = _Path(output_dir)
    combined_path = output_dir / "artifacts" / "combined_robustness.xlsx"
    combined_table = pd.read_excel(combined_path, index_col=0)

    data_full = read_study_data(data_path)
    env_block = extract_block(data_full, "environmental", "raw")
    env_vars_present = [v for v in env_variables if v in env_block.columns]
    env_ref_raw = env_block.loc[combined_table.index, env_vars_present].dropna()
    combined_table = combined_table.loc[env_ref_raw.index].copy()
    labels_for_xs = combined_table["Original_Cluster"]

    best_th, gs_results = grid_search_thresholds(
        combined=combined_table,
        env_strength_df=combined_table,
        labels=labels_for_xs,
        env_raw=env_ref_raw,
        env_strength_method=env_strength_method,
        env_strength_top_pct=env_strength_top_pct,
        verbose=verbose,
    )

    updated_combined = apply_threshold_configuration(
        combined_table,
        tsil=best_th["tsil"],
        tmarg=best_th["tmarg"],
        esil=best_th["esil"],
        emarg=best_th["emarg"],
        env_strength_method=env_strength_method,
        env_strength_top_pct=env_strength_top_pct,
    )
    class_count_table = build_class_count_table(updated_combined)

    save_table(
        updated_combined,
        output_dir / "artifacts" / "combined_robustness",
        formats=table_formats,
        verbose=verbose,
    )
    save_table(
        updated_combined.loc[:, _ENV_ASSIGNMENT_COLUMNS],
        output_dir / "tables" / "env_coherence" / "env_strength_assignments",
        formats=table_formats,
        verbose=verbose,
    )
    save_table(
        class_count_table,
        output_dir / "tables" / "env_coherence" / "taxa_env_class_counts",
        formats=table_formats,
        verbose=verbose,
    )
    save_table(
        class_count_table,
        output_dir / "tables" / "taxa_confidence" / "taxa_env_class_counts",
        formats=table_formats,
        verbose=verbose,
    )
    best_thresholds = pd.DataFrame([best_th], index=["best"])
    save_table(
        best_thresholds,
        output_dir / "artifacts" / "combined_robustness_thresholds",
        formats=table_formats,
        verbose=verbose,
    )

    if verbose:
        print(
            "  Reclassified TaxaEnv_Class distribution:"
            f" {updated_combined['TaxaEnv_Class'].value_counts().to_dict()}"
        )
        print("\n  2x2 TaxaEnv class counts (after grid search):")
        print(class_count_table.to_string())

    return updated_combined, class_count_table, best_th, gs_results


def wards_clustering_pipeline(
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
    # Robustness parameters
    n_boot_coassign: int = 1000,
    coassign_sample_frac: float = 0.8,
    n_boot_pvclust: int = 1000,
    run_pvclust_bootstrap: bool = True,
    sil_threshold: float | None = 0.25,
    margin_threshold: float | None = 0.25,
    # Environmental coherence parameters
    n_boot_env_coassign: int = 1000,
    env_coassign_sample_frac: float = 0.8,
    env_sil_threshold: float | None = 0.0,
    env_margin_threshold: float | None = 0.0,
    env_strength_method: str = "threshold",
    env_strength_top_pct: float = 0.60,
    # General
    random_state: int | None = 42,
    map_func=None,
    save_plots: bool = True,
    figure_formats: Sequence[str] = ("png",),
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> WardClusteringResult:
    """Run Ward's clustering with robustness assessment on reference sites.

    Returns
    -------
    WardClusteringResult
        Contains clustering labels, linkage matrix, robustness table,
        co-assignment matrix, and optionally pvclust summary.
    """
    output_dir = _Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    artifacts_dir = output_dir / "artifacts"

    # None thresholds → -inf  (grid search will reclassify later)
    if sil_threshold is None:
        sil_threshold = -np.inf
    if margin_threshold is None:
        margin_threshold = -np.inf
    if env_sil_threshold is None:
        env_sil_threshold = -np.inf
    if env_margin_threshold is None:
        env_margin_threshold = -np.inf

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

    _log("=" * 60)
    _log("  Ward's Clustering with Robustness Assessment")
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
    n_ref_expected = resolve_n_ref(reference_quantile, len(pollution_score))
    _log(f"[3/16] Selecting least polluted {n_ref_expected} sites ...")
    ref_mask = select_reference_sites(pollution_score, quantile=reference_quantile)
    n_ref = ref_mask.sum()
    _log(f"       {n_ref} least polluted sites out of {len(ref_mask)} total")

    # -- 4. Extract taxa block -----------------------------------------
    _log(f"[4/16] Extracting taxa data for {n_ref} reference sites ...")
    taxa_all = extract_block(data, "taxa", "raw")[list(taxa_columns)]
    # Align to sites present in the Stage 1 artifact (pollution score)
    taxa_all = taxa_all.loc[taxa_all.index.intersection(ref_mask.index)]
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

    # -- 8. Silhouette widths ------------------------------------------
    _log("[8/16] Computing per-site silhouette widths ...")
    silhouettes = compute_silhouettes(taxa_transformed, labels_ref)
    _log(f"       Mean silhouette: {silhouettes.mean():.4f}")
    for g in sorted(labels_ref.unique()):
        mask_g = labels_ref == g
        _log(f"         Cluster {g}: mean = {silhouettes.loc[mask_g].mean():.4f}")

    # -- 9. Bootstrap co-assignment ------------------------------------
    _log(f"[9/16] Computing bootstrap co-assignment ({n_boot_coassign} replicates) ...")
    coassign_matrix = compute_coassignment(
        taxa_transformed,
        labels_ref,
        n_clusters=n_clusters,
        n_boot=n_boot_coassign,
        sample_frac=coassign_sample_frac,
        random_state=random_state,
    )
    confidence = site_confidence(coassign_matrix, labels_ref)
    _log(f"       Mean own-cluster co-assignment: {confidence['Own_Coassign'].mean():.4f}")
    _log(f"       Mean margin: {confidence['Margin'].mean():.4f}")

    # -- 10. pvclust bootstrap (optional) ------------------------------
    pvclust_df = None
    if run_pvclust_bootstrap:
        _log(f"[10/16] Running pvclust bootstrap ({n_boot_pvclust} replicates) ...")
        try:
            pvclust_df = run_pvclust(
                taxa_transformed,
                n_clusters=n_clusters,
                nboot=n_boot_pvclust,
                verbose=verbose,
            )
            _log("        pvclust completed successfully.")
            for _, row in pvclust_df.iterrows():
                _log(f"          Cluster {int(row['Cluster'])}: AU={row['AU']:.4f}, BP={row['BP']:.4f}")
        except Exception as e:
            _log(f"        WARNING: pvclust failed ({e}). Continuing without bootstrap AU.")
    else:
        _log("[10/16] Skipping pvclust bootstrap (disabled).")

    # -- 11. Build taxa robustness table -------------------------------
    _log("[11/16] Building taxa robustness summary table ...")
    robustness_table = build_robustness_table(
        labels=labels_ref,
        pvclust_df=pvclust_df,
        silhouettes=silhouettes,
        confidence=confidence,
        sil_threshold=sil_threshold,
        margin_threshold=margin_threshold,
    )
    status_counts = robustness_table["Status"].value_counts()
    for status, count in status_counts.items():
        _log(f"         {status}: {count} sites")

    # -- 12. Environmental coherence: silhouette -------------------------
    _log("[12/16] Computing environmental coherence diagnostics ...")

    env_block = extract_block(data, "environmental", "raw")
    env_vars_present = [v for v in env_variables if v in env_block.columns]
    env_ref_raw = env_block.loc[labels_ref.index, env_vars_present]
    complete_index = env_ref_raw.dropna().index
    env_ref_complete = env_ref_raw.loc[complete_index]
    labels_ref_complete = labels_ref.loc[complete_index]
    taxa_ref_complete = taxa_ref.loc[complete_index]
    taxa_trans_complete = taxa_transformed.loc[complete_index]

    # Standardize environmental variables for env clustering/distances
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    env_std_values = scaler.fit_transform(env_ref_complete)
    env_std = pd.DataFrame(
        env_std_values,
        index=env_ref_complete.index,
        columns=env_ref_complete.columns,
    )

    # Environmental silhouette (env distances + taxa labels)
    env_silhouettes = compute_env_silhouettes(env_std, labels_ref_complete)
    _log(f"       Mean env silhouette: {env_silhouettes.mean():.4f}")
    for g in sorted(labels_ref_complete.unique()):
        mask_g = labels_ref_complete == g
        _log(f"         Cluster {g}: mean env sil = {env_silhouettes.loc[mask_g].mean():.4f}")

    # -- 13. Environmental co-assignment bootstrap ----------------------
    _log(f"[13/16] Computing env bootstrap co-assignment ({n_boot_env_coassign} replicates) ...")
    env_coassign_matrix = compute_env_coassignment(
        env_std,
        n_clusters=n_clusters,
        n_boot=n_boot_env_coassign,
        sample_frac=env_coassign_sample_frac,
        random_state=random_state,
    )
    env_confidence = env_site_confidence(env_coassign_matrix, labels_ref_complete)
    _log(f"       Mean env own-cluster co-assignment: {env_confidence['Env_Own_Coassign'].mean():.4f}")
    _log(f"       Mean env margin: {env_confidence['Env_Margin'].mean():.4f}")

    # -- 14. Build env coherence table & combined table ----------------
    _log("[14/16] Building environmental coherence summary ...")
    env_coherence = build_env_coherence_table(
        labels=labels_ref_complete,
        env_silhouettes=env_silhouettes,
        env_confidence=env_confidence,
        sil_threshold=env_sil_threshold,
        margin_threshold=env_margin_threshold,
        env_strength_method=env_strength_method,
        env_strength_top_pct=env_strength_top_pct,
    )
    env_strength_counts = env_coherence["Env_Strength"].value_counts()
    for strength, count in env_strength_counts.items():
        _log(f"         Env {strength}: {count} sites")

    # Build combined table (taxa + env) with updated rules and TaxaEnv_Class
    # Use robustness_table restricted to complete_index for alignment
    taxa_rob_complete = robustness_table.loc[complete_index]
    combined_table = build_updated_combined_table(
        taxa_rob_complete, env_coherence,
        taxa_sil_threshold=sil_threshold,
        taxa_margin_threshold=margin_threshold,
    )
    class_counts = combined_table["TaxaEnv_Class"].value_counts()
    _log("       TaxaEnv_Class distribution:")
    for cls, count in class_counts.items():
        _log(f"         {cls}: {count}")

    # 2x2 class count table (Output B)
    class_count_table = build_class_count_table(combined_table)
    _log("       2x2 class-count table (Env_Strength x Taxa_Strength):")
    _log(f"\n{class_count_table.to_string()}")

    # -- 15. ANOVA tests -----------------------------------------------
    _log("[15/16] Computing ANOVA tests ...")
    anova_dir = tables_dir / "anova_tests"

    env_anova = anova_table(
        env_ref_complete, labels_ref_complete, env_vars_present,
        transform=anova_transform, label_col="Variable",
    )
    save_table(env_anova, anova_dir / "anova_env",
               formats=table_formats, verbose=verbose)
    env_pvals = extract_pvalues(env_anova, label_col="Variable")

    taxa_anova = anova_table(
        taxa_trans_complete, labels_ref_complete, list(taxa_trans_complete.columns),
        transform=anova_transform, label_col="Taxon",
    )
    save_table(taxa_anova, anova_dir / "anova_taxa",
               formats=table_formats, verbose=verbose)
    taxa_pvals = extract_pvalues(taxa_anova, label_col="Taxon")

    # Subset ANOVA tables: all_sites, core_only, core_peripheral
    _subsets = {
        "all_sites": None,
        "core_only": ["Core"],
        "core_peripheral": ["Core", "Peripheral"],
    }
    for subset_name, keep_statuses in _subsets.items():
        if keep_statuses is None:
            sub_idx = labels_ref_complete.index
        else:
            sub_mask = robustness_table.loc[labels_ref_complete.index, "Status"].isin(keep_statuses)
            sub_idx = labels_ref_complete.index[sub_mask]

        env_sub = env_ref_complete.loc[sub_idx]
        taxa_sub = taxa_trans_complete.loc[sub_idx]
        labels_sub = labels_ref_complete.loc[sub_idx]

        if len(labels_sub.unique()) < 2:
            _log(f"    Skipping ANOVA for {subset_name}: fewer than 2 clusters")
            continue

        env_anova_sub = anova_table(
            env_sub, labels_sub, env_vars_present,
            transform=anova_transform, label_col="Variable",
        )
        save_table(env_anova_sub, anova_dir / f"anova_env_{subset_name}",
                   formats=table_formats, verbose=verbose)

        taxa_anova_sub = anova_table(
            taxa_sub, labels_sub, list(taxa_sub.columns),
            transform=anova_transform, label_col="Taxon",
        )
        save_table(taxa_anova_sub, anova_dir / f"anova_taxa_{subset_name}",
                   formats=table_formats, verbose=verbose)

    # -- 16. Save all outputs ------------------------------------------
    _log("[16/16] Saving outputs ...")

    # ── tables/env_coherence/ ─────────────────────────────────────────
    env_dir = tables_dir / "env_coherence"

    save_table(env_std, env_dir / "env_standardized_matrix",
               formats=table_formats, verbose=verbose)

    from scipy.spatial.distance import squareform as _squareform, pdist as _pdist
    env_dist_sq = pd.DataFrame(
        _squareform(_pdist(env_std.values, metric="euclidean")),
        index=env_std.index,
        columns=env_std.index,
    )
    save_table(env_dist_sq, env_dir / "env_distance_matrix",
               formats=table_formats, verbose=verbose)

    save_table(env_coassign_matrix, env_dir / "env_coassignment_matrix",
               formats=table_formats, verbose=verbose)

    env_sil_df = env_silhouettes.to_frame()
    save_table(env_sil_df, env_dir / "env_silhouettes",
               formats=table_formats, verbose=verbose)

    save_table(env_confidence, env_dir / "env_coassignment_summaries",
               formats=table_formats, verbose=verbose)

    save_table(env_coherence, env_dir / "env_strength_assignments",
               formats=table_formats, verbose=verbose)

    save_table(class_count_table, env_dir / "taxa_env_class_counts",
               formats=table_formats, verbose=verbose)

    # ── tables/taxa_confidence/ ───────────────────────────────────────
    taxa_dir = tables_dir / "taxa_confidence"

    save_table(robustness_table, taxa_dir / "site_robustness",
               formats=table_formats, verbose=verbose)

    save_table(coassign_matrix, taxa_dir / "coassignment_matrix",
               formats=table_formats, verbose=verbose)

    if pvclust_df is not None:
        save_table(pvclust_df.set_index("Cluster"),
                   taxa_dir / "pvclust_summary",
                   formats=table_formats, verbose=verbose)

    ref_table = taxa_ref.copy()
    ref_table.insert(0, "Cluster", labels_ref)
    save_table(ref_table, taxa_dir / "reference_taxa_clusters",
               formats=table_formats, verbose=verbose)

    # ── Dendrogram ────────────────────────────────────────────────────
    if save_plots:
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
        save_figure(fig_dend, figures_dir / "ward_dendrogram",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_dend)

    # ── Cluster panel ─────────────────────────────────────────────────
    if save_plots:
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
                figures_dir / f"cluster_{suffix}",
                formats=figure_formats,
                verbose=verbose,
            )
            plt.close(fig_panel)

    # ── artifacts/ — only combined_robustness ─────────────────────────
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    combined_table.to_excel(artifacts_dir / "combined_robustness.xlsx")
    if verbose:
        print(f"  > Saved artifact: {artifacts_dir / 'combined_robustness.xlsx'}")

    # Build result
    result = WardClusteringResult(
        ref_mask=ref_mask,
        cluster_labels=labels_ref,
        linkage_matrix=Z,
        taxa_ref=taxa_ref,
        taxa_ref_transformed=taxa_transformed,
        n_clusters=n_clusters,
        taxa_transform=taxa_transform,
        all_site_index=data.index,
        robustness_table=robustness_table,
        coassignment_matrix=coassign_matrix,
        pvclust_summary=pvclust_df,
    )

    _log(f"\n{result.summary()}")
    return result


# ══════════════════════════════════════════════════════════════════════
#  pvclust AU sweep across N_REFERENCE_SITES
# ══════════════════════════════════════════════════════════════════════

def pvclust_au_sweep(
    data_path: str | _Path,
    stage1_artifact: str | _Path,
    output_dir: str | _Path,
    *,
    n_range: tuple[int, int] = (40, 70),
    n_clusters: int = 3,
    taxa_transform: str = "octave",
    label_map: Dict[int, int] | None = None,
    nboot: int = 300,
    file_prefix: str = "",
    figure_formats: Sequence[str] = ("png",),
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> pd.DataFrame:
    """Sweep N_REFERENCE_SITES, apply Ward clustering + pvclust at each N.

    At each sweep point the least-polluted *N* sites are Ward-clustered
    into *n_clusters* branches, pvclust is run, and the per-cluster AU
    values and cluster sizes are recorded.  The function produces:

    * **Table** — one row per sweep point with all *K* AU values,
      cluster sizes, the newly entered site, and the min AU / min size.
    * **Figure** — bottom x-axis = newly entered site ID (annotated with
      total N at both ends), top x-axis = min cluster size, single curve
      of min AU across the sweep.

    Outputs are saved under ``output_dir / figures`` and
    ``output_dir / tables``.

    Returns
    -------
    pd.DataFrame
        Full sweep table (one row per N).
    """
    output_dir = _Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    if label_map is None:
        label_map = {}

    transform_fn = _TRANSFORMS.get(taxa_transform)
    if transform_fn is None:
        raise ValueError(
            f"Unknown taxa_transform={taxa_transform!r}. "
            f"Choose from {list(_TRANSFORMS)}."
        )

    # ── load data once ────────────────────────────────────────────────
    data = read_study_data(data_path)
    stage1 = pd.read_excel(stage1_artifact, header=[0, 1, 2], index_col=0)
    score_cols = [
        c for c in stage1.columns
        if c[0] == "01_pollution_assessment" and c[1] == "raw"
        and c[2].endswith("_Score")
    ]
    if not score_cols:
        raise KeyError("No pollution score column found in Stage 1 artifact")
    pollution_score = stage1.loc[:, score_cols[0]]
    taxa_all = extract_block(data, "taxa", "raw")
    taxa_all = taxa_all.loc[taxa_all.index.intersection(pollution_score.index)]

    # Pre-compute the ordered list of sites from least to most polluted
    # so we can identify the "newly entered" site at each sweep step.
    sorted_sites = pollution_score.sort_values().index.tolist()

    n_min, n_max = n_range
    n_values = list(range(n_min, n_max + 1))
    total_steps = len(n_values)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  pvclust AU sweep: N = {n_min}..{n_max}  (k={n_clusters}, nboot={nboot})")
        print(f"{'='*60}")

    rows: list[dict] = []
    for step_i, n_ref in enumerate(n_values, 1):
        if verbose:
            pct = step_i / total_steps * 100
            print(f"\n  [{step_i}/{total_steps} {pct:5.1f}%]  N = {n_ref} ...", end=" ", flush=True)

        ref_mask = select_reference_sites(pollution_score, quantile=n_ref)
        taxa_ref = taxa_all.loc[ref_mask]
        taxa_transformed = transform_fn(taxa_ref)
        labels_ref, Z = ward_cluster(taxa_transformed, n_clusters=n_clusters)

        if label_map:
            labels_ref = _relabel(labels_ref, label_map)

        cluster_sizes = labels_ref.value_counts().sort_index()

        # Identify newly entered site (the n_ref-th least-polluted site)
        newly_entered = str(sorted_sites[n_ref - 1])

        try:
            pvclust_df = run_pvclust(
                taxa_transformed, n_clusters=n_clusters, nboot=nboot, verbose=False,
            )
            row: dict[str, Any] = {
                "N": n_ref,
                "actual_n": int(ref_mask.sum()),
                "newly_entered_site": newly_entered,
            }
            for _, r in pvclust_df.iterrows():
                k = int(r["Cluster"])
                row[f"AU_C{k}"] = round(r["AU"], 4)
                row[f"BP_C{k}"] = round(r["BP"], 4)
                row[f"size_C{k}"] = int(cluster_sizes.get(k, 0))

            # Compute min AU and min cluster size across all K clusters
            au_vals = [row[f"AU_C{k}"] for k in range(1, n_clusters + 1)
                       if f"AU_C{k}" in row]
            size_vals = [row[f"size_C{k}"] for k in range(1, n_clusters + 1)
                         if f"size_C{k}" in row]
            row["min_AU"] = min(au_vals) if au_vals else np.nan
            row["min_size"] = min(size_vals) if size_vals else np.nan

            rows.append(row)
            if verbose:
                au_str = ", ".join(
                    f"C{k}={row.get(f'AU_C{k}', '?')}" for k in range(1, n_clusters + 1)
                )
                print(f"AU: {au_str}  min_AU={row['min_AU']:.4f}  min_size={int(row['min_size'])}")
        except Exception as e:
            rows.append({
                "N": n_ref,
                "actual_n": int(ref_mask.sum()),
                "newly_entered_site": newly_entered,
                "error": str(e),
            })
            if verbose:
                print(f"ERROR: {e}")

    results = pd.DataFrame(rows)

    # ── save table (drop actual_n since it always equals N) ───────
    save_cols = [c for c in results.columns if c != "actual_n"]
    stem = f"{file_prefix}au_sweep_full" if file_prefix else "au_sweep_full"
    save_table(results[save_cols], tables_dir / stem,
               formats=table_formats, verbose=verbose)

    if verbose:
        print(f"\n{'='*60}")
        print("  AU SWEEP COMPLETE")
        print(f"{'='*60}")
        print(results.to_string(index=False))

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Pop-out AU sweep
# ═══════════════════════════════════════════════════════════════════════

def pvclust_au_sweep_popout(
    data_path: str | _Path,
    stage1_artifact: str | _Path,
    output_dir: str | _Path,
    *,
    n_range: tuple[int, int] = (40, 70),
    n_clusters: int = 3,
    taxa_transform: str = "octave",
    label_map: Dict[int, int] | None = None,
    nboot: int = 300,
    file_prefix: str = "",
    drop_threshold: float = 0.3,
    figure_formats: Sequence[str] = ("png",),
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pop-out variant of the AU sweep.

    Works like :func:`pvclust_au_sweep` but monitors how min AU changes
    as each new site enters.  If including a candidate site drops min AU
    by more than *drop_threshold* relative to the previous accepted
    value, the site is marked **atypical**, excluded from the reference
    set, and the sweep continues with the next candidate in pollution-
    ranked order.

    Returns
    -------
    accepted_df : pd.DataFrame
        Sweep table for accepted (non-atypical) sites.
    atypical_df : pd.DataFrame
        Table of atypical sites that were popped out.
    """
    output_dir = _Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    if label_map is None:
        label_map = {}

    transform_fn = _TRANSFORMS.get(taxa_transform)
    if transform_fn is None:
        raise ValueError(
            f"Unknown taxa_transform={taxa_transform!r}. "
            f"Choose from {list(_TRANSFORMS)}."
        )

    # ── load data once ────────────────────────────────────────────────
    data = read_study_data(data_path)
    stage1 = pd.read_excel(stage1_artifact, header=[0, 1, 2], index_col=0)
    score_cols = [
        c for c in stage1.columns
        if c[0] == "01_pollution_assessment" and c[1] == "raw"
        and c[2].endswith("_Score")
    ]
    if not score_cols:
        raise KeyError("No pollution score column found in Stage 1 artifact")
    pollution_score = stage1.loc[:, score_cols[0]]
    taxa_all = extract_block(data, "taxa", "raw")
    taxa_all = taxa_all.loc[taxa_all.index.intersection(pollution_score.index)]

    sorted_sites = pollution_score.sort_values().index.tolist()
    n_min, n_max = n_range

    # ── helper: run Ward + pvclust on a given site list ───────────────
    def _evaluate(site_list):
        taxa_ref = taxa_all.loc[site_list]
        taxa_t = transform_fn(taxa_ref)
        labels, _ = ward_cluster(taxa_t, n_clusters=n_clusters)
        if label_map:
            labels = _relabel(labels, label_map)
        sizes = labels.value_counts().sort_index()
        pvclust_df = run_pvclust(
            taxa_t, n_clusters=n_clusters, nboot=nboot, verbose=False,
        )
        row: dict[str, Any] = {}
        for _, r in pvclust_df.iterrows():
            k = int(r["Cluster"])
            row[f"AU_C{k}"] = round(r["AU"], 4)
            row[f"BP_C{k}"] = round(r["BP"], 4)
            row[f"size_C{k}"] = int(sizes.get(k, 0))
        au_vals = [row[f"AU_C{k}"] for k in range(1, n_clusters + 1)
                   if f"AU_C{k}" in row]
        size_vals = [row[f"size_C{k}"] for k in range(1, n_clusters + 1)
                     if f"size_C{k}" in row]
        row["min_AU"] = min(au_vals) if au_vals else np.nan
        row["min_size"] = min(size_vals) if size_vals else np.nan
        return row

    total_steps = n_max - n_min + 1

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Pop-out AU sweep: N = {n_min}..{n_max}  "
              f"(k={n_clusters}, nboot={nboot}, drop_thr={drop_threshold})")
        print(f"{'='*60}")

    # ── baseline at n_min ─────────────────────────────────────────────
    current_sites = list(sorted_sites[:n_min])
    step_i = 1

    if verbose:
        pct = step_i / total_steps * 100
        print(f"\n  [{step_i}/{total_steps} {pct:5.1f}%]  N = {n_min} (baseline) ...", end=" ", flush=True)

    baseline = _evaluate(current_sites)
    baseline["N"] = n_min
    baseline["actual_n"] = n_min
    baseline["newly_entered_site"] = str(sorted_sites[n_min - 1])
    baseline["is_atypical"] = False

    prev_min_au = baseline["min_AU"]

    if verbose:
        au_str = ", ".join(
            f"C{k}={baseline.get(f'AU_C{k}', '?')}"
            for k in range(1, n_clusters + 1)
        )
        print(f"AU: {au_str}  min_AU={prev_min_au:.4f}  "
              f"min_size={int(baseline['min_size'])}")

    all_rows = [baseline]
    atypical_rows: list[dict] = []

    # ── sweep remaining candidates ────────────────────────────────────
    for n_ref in range(n_min + 1, n_max + 1):
        candidate = sorted_sites[n_ref - 1]
        step_i += 1

        if verbose:
            pct = min(step_i / total_steps * 100, 100.0)
            print(f"\n  [{step_i}/{total_steps} {pct:5.1f}%]  N = {n_ref} trying {candidate} ...",
                  end=" ", flush=True)

        trial_sites = current_sites + [candidate]

        try:
            trial = _evaluate(trial_sites)
            new_min_au = trial["min_AU"]
            au_drop = prev_min_au - new_min_au

            if au_drop > drop_threshold:
                # ── atypical: pop out ─────────────────────────────────
                atyp_row: dict[str, Any] = {
                    "N": n_ref,
                    "actual_n": len(current_sites),
                    "newly_entered_site": str(candidate),
                    "is_atypical": True,
                }
                atyp_row.update(trial)
                all_rows.append(atyp_row)

                atypical_rows.append({
                    "site": str(candidate),
                    "pollution_rank": n_ref,
                    "effective_N_when_rejected": len(current_sites),
                    "prev_min_AU": round(prev_min_au, 4),
                    "trial_min_AU": round(new_min_au, 4),
                    "AU_drop": round(au_drop, 4),
                })
                if verbose:
                    print(f"ATYPICAL  drop={au_drop:.4f} > {drop_threshold}"
                          f"  (trial_min_AU={new_min_au:.4f}) → skipped")
                continue

            # ── accepted ──────────────────────────────────────────────
            current_sites = trial_sites
            prev_min_au = new_min_au

            trial["N"] = n_ref
            trial["actual_n"] = len(current_sites)
            trial["newly_entered_site"] = str(candidate)
            trial["is_atypical"] = False
            all_rows.append(trial)

            if verbose:
                au_str = ", ".join(
                    f"C{k}={trial.get(f'AU_C{k}', '?')}"
                    for k in range(1, n_clusters + 1)
                )
                print(f"AU: {au_str}  min_AU={new_min_au:.4f}  "
                      f"min_size={int(trial['min_size'])}")

        except Exception as e:
            current_sites.append(candidate)
            all_rows.append({
                "N": n_ref,
                "actual_n": len(current_sites),
                "newly_entered_site": str(candidate),
                "is_atypical": False,
                "error": str(e),
            })
            if verbose:
                print(f"ERROR: {e}")

    full_df = pd.DataFrame(all_rows)
    atypical_df = pd.DataFrame(atypical_rows) if atypical_rows else pd.DataFrame(
        columns=["site", "pollution_rank", "effective_N_when_rejected",
                 "prev_min_AU", "trial_min_AU", "AU_drop"],
    )

    # ── save tables ───────────────────────────────────────────────────
    sweep_stem = f"{file_prefix}popout_au_sweep_full"
    save_table(full_df, tables_dir / sweep_stem,
               formats=table_formats, verbose=verbose)

    atyp_stem = f"{file_prefix}popout_atypical_sites"
    save_table(atypical_df, tables_dir / atyp_stem,
               formats=table_formats, verbose=verbose)

    # ── summary ───────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*60}")
        print(f"  POP-OUT AU SWEEP COMPLETE  "
              f"({len(atypical_df)} atypical sites popped)")
        print(f"{'='*60}")
        if not atypical_df.empty:
            print(f"\n  Atypical sites:")
            print(atypical_df.to_string(index=False))
        print(f"\n  Full sweep (incl. atypical):")
        print(full_df.to_string(index=False))

    return full_df, atypical_df


# ═══════════════════════════════════════════════════════════════════════
#  Combined AU-sweep figure (original + pop-out, rotated 90°)
# ═══════════════════════════════════════════════════════════════════════

def plot_combined_au_sweep(
    original_df: pd.DataFrame,
    popout_df: pd.DataFrame,
    *,
    n_clusters: int,
    taxa_transform: str,
    n_range: tuple[int, int],
    nboot: int,
    drop_threshold: float,
    output_dir: str | _Path,
    file_prefix: str = "",
    figure_formats: Sequence[str] = ("png",),
    verbose: bool = True,
):
    """Two-panel figure: original AU sweep (left) and pop-out sweep (right).

    Both panels are rotated 90° so that site IDs run down the vertical
    axis and min AU is on the top horizontal axis.
    """
    output_dir = _Path(output_dir)
    figures_dir = output_dir / "figures"
    n_min, n_max = n_range

    valid_orig = original_df.dropna(subset=["min_AU"])
    if valid_orig.empty and popout_df.empty:
        return

    n_rows = max(len(valid_orig), len(popout_df))
    fig_height = max(10, n_rows * 0.45)
    fig, (ax_orig, ax_pop) = plt.subplots(
        1, 2, figsize=(14, fig_height),
        gridspec_kw={"wspace": 0.55},
    )

    # ── Left panel: Original AU sweep ─────────────────────────────────
    y_orig = np.arange(len(valid_orig))
    ax_orig.plot(
        valid_orig["min_AU"].values, y_orig,
        marker="o", linewidth=1.5, color="steelblue",
    )
    ax_orig.axvline(0.90, color="orange", linestyle="--", alpha=0.6,
                    label="AU = 0.90")

    ax_orig.set_yticks(y_orig)
    ax_orig.set_yticklabels(
        valid_orig["newly_entered_site"].tolist(), fontsize=7,
    )
    ax_orig.invert_yaxis()
    ax_orig.xaxis.tick_top()
    ax_orig.xaxis.set_label_position("top")
    ax_orig.set_xlabel("min AU")
    ax_orig.set_ylabel("Newly Entered Site ID")
    ax_orig.set_title(
        f"AU Sweep (k={n_clusters})", pad=15, fontsize=10,
    )
    ax_orig.grid(True, alpha=0.3)
    ax_orig.legend(loc="lower right", fontsize=8)

    # Right-side y-axis: min cluster size
    ax_orig_r = ax_orig.twinx()
    ax_orig_r.set_ylim(ax_orig.get_ylim())
    ax_orig_r.set_yticks(y_orig)
    ax_orig_r.set_yticklabels(
        [str(int(s)) for s in valid_orig["min_size"].values], fontsize=7,
    )
    ax_orig_r.set_ylabel("Min Cluster Size")

    # ── Right panel: Pop-out AU sweep ─────────────────────────────────
    y_pop = np.arange(len(popout_df))
    is_atyp = popout_df["is_atypical"].values
    au_vals = popout_df["min_AU"].values

    # Build line segments that skip atypical sites
    segments: list[tuple[list, list]] = []
    seg_x: list[float] = []
    seg_y: list[float] = []
    for i in range(len(popout_df)):
        if not is_atyp[i] and not np.isnan(au_vals[i]):
            seg_x.append(au_vals[i])
            seg_y.append(y_pop[i])
        else:
            if seg_x:
                segments.append((seg_x[:], seg_y[:]))
            seg_x, seg_y = [], []
    if seg_x:
        segments.append((seg_x, seg_y))

    for j, (sx, sy) in enumerate(segments):
        ax_pop.plot(
            sx, sy,
            marker="o", linewidth=1.5, color="steelblue",
            label="min AU (pop-out)" if j == 0 else None,
        )

    ax_pop.axvline(0.90, color="orange", linestyle="--", alpha=0.6,
                   label="AU = 0.90")

    ax_pop.set_yticks(y_pop)
    site_labels_pop = popout_df["newly_entered_site"].tolist()
    ax_pop.set_yticklabels(site_labels_pop, fontsize=7)
    ax_pop.invert_yaxis()
    ax_pop.xaxis.tick_top()
    ax_pop.xaxis.set_label_position("top")
    ax_pop.set_xlabel("min AU")
    ax_pop.set_ylabel("Newly Entered Site ID")
    ax_pop.set_title(
        f"Pop-out AU Sweep (k={n_clusters}, drop>{drop_threshold})",
        pad=15, fontsize=10,
    )
    ax_pop.grid(True, alpha=0.3)
    ax_pop.legend(loc="lower right", fontsize=8)

    # Colour atypical site labels red
    for i, label in enumerate(ax_pop.get_yticklabels()):
        if i < len(is_atyp) and is_atyp[i]:
            label.set_color("red")

    # Right-side y-axis: min cluster size (blank for atypical rows)
    ax_pop_r = ax_pop.twinx()
    ax_pop_r.set_ylim(ax_pop.get_ylim())
    ax_pop_r.set_yticks(y_pop)
    size_labels: list[str] = []
    for _, row in popout_df.iterrows():
        if row["is_atypical"]:
            size_labels.append("")
        elif pd.notna(row.get("min_size")):
            size_labels.append(str(int(row["min_size"])))
        else:
            size_labels.append("")
    ax_pop_r.set_yticklabels(size_labels, fontsize=7)
    ax_pop_r.set_ylabel("Min Cluster Size")

    # ── overall title ─────────────────────────────────────────────────
    fig.suptitle(
        f"AU Sweep: Ward k={n_clusters}, {taxa_transform} "
        f"(N = {n_min}–{n_max}, nboot={nboot})",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    fig_stem = f"{file_prefix}au_sweep_combined" if file_prefix else "au_sweep_combined"
    save_figure(fig, figures_dir / fig_stem,
                formats=figure_formats, verbose=verbose)
    plt.close(fig)
