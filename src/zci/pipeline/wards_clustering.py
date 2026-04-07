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
    sil_threshold: float = 0.25,
    margin_threshold: float = 0.25,
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
    _log("[1/12] Reading original study data ...")
    data = read_study_data(data_path)
    _log(f"       {data.shape[0]} sites x {data.shape[1]} variables")

    # -- 2. Read Stage 1 artifact -> Pollution Score -------------------
    _log("[2/12] Reading Stage 1 artifact for pollution scores ...")
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
    _log(f"[3/12] Selecting least polluted {n_ref_expected} sites ...")
    ref_mask = select_reference_sites(pollution_score, quantile=reference_quantile)
    n_ref = ref_mask.sum()
    _log(f"       {n_ref} least polluted sites out of {len(ref_mask)} total")

    # -- 4. Extract taxa block -----------------------------------------
    _log(f"[4/12] Extracting taxa data for {n_ref} reference sites ...")
    taxa_all = extract_block(data, "taxa", "raw")[list(taxa_columns)]
    taxa_ref = taxa_all.loc[ref_mask]
    _log(f"       {taxa_ref.shape[0]} sites x {taxa_ref.shape[1]} taxa")

    # -- 5. Transform taxa ---------------------------------------------
    _log(f"[5/12] Applying taxa transform: {taxa_transform!r} ...")
    if taxa_transform not in _TRANSFORMS:
        raise ValueError(
            f"Unknown taxa_transform={taxa_transform!r}. "
            f"Choose from {sorted(_TRANSFORMS)}."
        )
    taxa_transformed = _TRANSFORMS[taxa_transform](taxa_ref)

    # -- 6. Ward clustering --------------------------------------------
    _log(f"[6/12] Ward clustering (n_clusters={n_clusters}) ...")
    labels_ref, Z = ward_cluster(taxa_transformed, n_clusters=n_clusters)
    _log("       Cluster distribution (before relabel):")
    for g in sorted(labels_ref.unique()):
        _log(f"         Group {g}: {(labels_ref == g).sum()} sites")

    # -- 7. Relabel clusters -------------------------------------------
    if label_map:
        _log(f"[7/12] Relabelling clusters: {label_map} ...")
        labels_ref = _relabel(labels_ref, label_map)
        labels_ref.name = "Cluster"
        _log("       Cluster distribution (after relabel):")
        for g in sorted(labels_ref.unique()):
            _log(f"         Group {g}: {(labels_ref == g).sum()} sites")
    else:
        _log("[7/12] No relabelling requested, keeping original labels.")

    # -- 8. Silhouette widths ------------------------------------------
    _log("[8/12] Computing per-site silhouette widths ...")
    silhouettes = compute_silhouettes(taxa_transformed, labels_ref)
    _log(f"       Mean silhouette: {silhouettes.mean():.4f}")
    for g in sorted(labels_ref.unique()):
        mask_g = labels_ref == g
        _log(f"         Cluster {g}: mean = {silhouettes.loc[mask_g].mean():.4f}")

    # -- 9. Bootstrap co-assignment ------------------------------------
    _log(f"[9/12] Computing bootstrap co-assignment ({n_boot_coassign} replicates) ...")
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
        _log(f"[10/12] Running pvclust bootstrap ({n_boot_pvclust} replicates) ...")
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
        _log("[10/12] Skipping pvclust bootstrap (disabled).")

    # -- 11. Build robustness table ------------------------------------
    _log("[11/12] Building robustness summary table ...")
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

    # -- 12. ANOVA, dendrogram, cluster panel, save --------------------
    _log("[12/12] Saving outputs ...")

    # ANOVA on environmental variables
    env_block = extract_block(data, "environmental", "raw")
    env_vars_present = [v for v in env_variables if v in env_block.columns]
    env_ref_raw = env_block.loc[labels_ref.index, env_vars_present]
    complete_index = env_ref_raw.dropna().index
    env_ref_complete = env_ref_raw.loc[complete_index]
    labels_ref_complete = labels_ref.loc[complete_index]
    taxa_ref_complete = taxa_ref.loc[complete_index]

    env_anova = anova_table(
        env_ref_complete, labels_ref_complete, env_vars_present,
        transform=anova_transform, label_col="Variable",
    )
    save_table(env_anova, tables_dir / "anova_env",
               formats=table_formats, verbose=verbose)
    env_pvals = extract_pvalues(env_anova, label_col="Variable")

    taxa_anova = anova_table(
        taxa_ref_complete, labels_ref_complete, list(taxa_ref_complete.columns),
        transform=anova_transform, label_col="Taxon",
    )
    save_table(taxa_anova, tables_dir / "anova_taxa",
               formats=table_formats, verbose=verbose)
    taxa_pvals = extract_pvalues(taxa_anova, label_col="Taxon")

    # Robustness table
    save_table(robustness_table, tables_dir / "site_robustness",
               formats=table_formats, verbose=verbose)

    # Co-assignment matrix
    save_table(coassign_matrix, tables_dir / "coassignment_matrix",
               formats=table_formats, verbose=verbose)

    # pvclust summary
    if pvclust_df is not None:
        save_table(pvclust_df.set_index("Cluster"),
                   tables_dir / "pvclust_summary",
                   formats=table_formats, verbose=verbose)

    # Reference taxa clusters table
    ref_table = taxa_ref.copy()
    ref_table.insert(0, "Cluster", labels_ref)
    save_table(ref_table, tables_dir / "reference_taxa_clusters",
               formats=table_formats, verbose=verbose)

    # Dendrogram
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

    # Cluster panel
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

    # Save artifact (augmented data + robustness)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    robustness_table.to_excel(artifacts_dir / "site_robustness.xlsx")
    coassign_matrix.to_excel(artifacts_dir / "coassignment_matrix.xlsx")
    if verbose:
        print(f"  > Saved artifact: {artifacts_dir / 'site_robustness.xlsx'}")
        print(f"  > Saved artifact: {artifacts_dir / 'coassignment_matrix.xlsx'}")

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
