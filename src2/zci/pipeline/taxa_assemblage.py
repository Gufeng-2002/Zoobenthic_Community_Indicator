"""Stage 2 — Taxa Assemblage Clustering Pipeline.

Orchestrates:
  read original data → merge Stage 1 pollution scores
  → select reference sites → extract & transform taxa
  → Ward clustering → dendrogram → ANOVA → cluster panel → assign labels → save.

This is the **only** module that touches both ``io`` and ``core`` for Stage 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import matplotlib.pyplot as plt

from ..io.readers import read_study_data, extract_block
from ..io.writers import save_table, save_figure
from ..core.transforms import octave_to_relative_abundance, octave_transform
from ..core.clustering import ward_cluster, select_reference_sites
from ..core.anova import anova_table, extract_pvalues
from ..models.clustering import TAXA_COLUMNS, ClusteringResult
from ..viz.clustering_plots import plot_dendrogram
from ..viz.cluster_panel_plot import plot_cluster_panel, TAXA_DISPLAY_ORDER


def taxa_assemblage_pipeline(
    data_path: str | Path,
    stage1_artifact: str | Path,
    output_dir: str | Path,
    *,
    taxa_columns: Sequence[str] = TAXA_COLUMNS,
    reference_quantile: float = 0.20,
    taxa_transform: str = "octave",
    n_clusters: int = 2,
    env_variables: Sequence[str] | None = None,
    anova_transform: str = "none",
    maps_dir: str | Path | None = None,
    save_plots: bool = True,
    figure_formats: Sequence[str] = ("png",),
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> ClusteringResult:
    """Run the complete taxa-assemblage clustering stage and save outputs.

    Steps
    -----
    1. Read the original 3-level MultiIndex Excel workbook.
    2. Read the Stage 1 artifact and extract the Pollution_Score.
    3. Select reference sites (bottom *reference_quantile* of pollution).
    4. Extract the 16 taxa columns for reference sites.
    5. Apply the chosen taxa transform.
    6. Perform Ward's hierarchical clustering (Euclidean distance).
    7. Save dendrogram figure.
    8. ANOVA on environmental variables across clusters.
    9. ANOVA on taxa variables (octave) across clusters.
    10. Three-panel cluster figure (map + env bars + taxa bars).
    11. Assign cluster labels and save tables / artifacts.

    Parameters
    ----------
    data_path : str or Path
        Path to the study-data Excel file.
    stage1_artifact : str or Path
        Path to ``results2/01_pollution_assessment/artifacts/01_updated_data.xlsx``.
    output_dir : str or Path
        Root output directory for this stage.
    taxa_columns : sequence of str
        Which taxa columns to use (default: the 16 study taxa).
    reference_quantile : float
        Fraction of least-polluted sites to designate as reference.
    taxa_transform : str
        ``"octave"`` (identity) or ``"relative_abundance"``.
    n_clusters : int
        Number of Ward clusters (default 2).
    env_variables : list of str, optional
        Environmental variable names to include in ANOVA / bar plot.
        ``None`` → sensible defaults.
    anova_transform : str
        Pre-ANOVA transformation: ``"none"`` (default), ``"log"``,
        ``"boxcox"``.
    maps_dir : str, Path, or None
        Path to ``data/maps/`` folder with shapefiles for the cluster
        panel figure.  ``None`` disables the panel figure.
    save_plots : bool
        Whether to save figures.
    figure_formats / table_formats : sequence of str
        File formats for outputs.
    verbose : bool
        Print progress messages.

    Returns
    -------
    ClusteringResult
    """
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    artifacts_dir = output_dir / "artifacts"

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # ── 1. Read original data ────────────────────────────────────────────
    _log("[1/8] Reading original study data …")
    data = read_study_data(data_path)
    _log(f"      {data.shape[0]} sites × {data.shape[1]} variables")

    # ── 2. Read Stage 1 artifact → Pollution_Score ───────────────────────
    _log("[2/8] Reading Stage 1 artifact for pollution scores …")
    stage1 = pd.read_excel(stage1_artifact, header=[0, 1, 2], index_col=0)
    pollution_score = stage1.loc[
        :, ("01_pollution_assessment", "raw", "Pollution_Score")
    ]
    pollution_score.name = "Pollution_Score"
    _log(f"      Pollution score range: "
         f"[{pollution_score.min():.4f}, {pollution_score.max():.4f}]")

    # ── 3. Select reference sites ────────────────────────────────────────
    _log(f"[3/8] Selecting reference sites (bottom {reference_quantile*100:.0f} %) …")
    ref_mask = select_reference_sites(pollution_score, quantile=reference_quantile)
    n_ref = ref_mask.sum()
    _log(f"      {n_ref} reference sites out of {len(ref_mask)} total")

    # ── 4. Extract taxa block for reference sites ────────────────────────
    _log("[4/8] Extracting taxa data for reference sites …")
    taxa_all = extract_block(data, "taxa", "raw")[list(taxa_columns)]
    taxa_ref = taxa_all.loc[ref_mask]
    _log(f"      {taxa_ref.shape[0]} sites × {taxa_ref.shape[1]} taxa")

    # ── 5. Transform taxa ────────────────────────────────────────────────
    _log(f"[5/8] Applying taxa transform: {taxa_transform!r} …")
    if taxa_transform == "octave":
        taxa_transformed = octave_transform(taxa_ref)
    elif taxa_transform == "relative_abundance":
        taxa_transformed = octave_to_relative_abundance(taxa_ref)
    else:
        raise ValueError(
            f"Unknown taxa_transform={taxa_transform!r}. "
            "Use 'octave' or 'relative_abundance'."
        )

    # ── 6. Ward clustering ───────────────────────────────────────────────
    _log(f"[6/8] Ward clustering (n_clusters={n_clusters}) …")
    labels_ref, Z = ward_cluster(taxa_transformed, n_clusters=n_clusters)
    _log("      Cluster distribution:")
    for g in sorted(labels_ref.unique()):
        _log(f"        Group {g}: {(labels_ref == g).sum()} sites")

    # ── Build result container ───────────────────────────────────────────
    result = ClusteringResult(
        ref_mask=ref_mask,
        cluster_labels=labels_ref,
        linkage_matrix=Z,
        taxa_ref=taxa_ref,
        taxa_ref_transformed=taxa_transformed,
        n_clusters=n_clusters,
        taxa_transform=taxa_transform,
        all_site_index=data.index,
    )

    # ── 7. Dendrogram figure ─────────────────────────────────────────────
    if save_plots:
        _log("[7/11] Saving dendrogram …")
        fig_dend, _ = plot_dendrogram(
            Z,
            labels=taxa_ref.index.astype(str),
            n_clusters=n_clusters,
            title=(
                f"Ward's Dendrogram — {taxa_transform.replace('_', ' ').title()} "
                f"(k = {n_clusters})"
            ),
        )
        save_figure(fig_dend, figures_dir / "ward_dendrogram",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_dend)

    # ── 8. ANOVA — environmental variables ───────────────────────────────
    # Default environmental variables
    if env_variables is None:
        env_variables = [
            "Measured Depth (m)",
            "Water DO Bottom (mg/L)",
            "Temperature (oC)",
            "MPS (Phi)",
            "LOI (%)",
        ]

    _log(f"[8/11] ANOVA on {len(env_variables)} environmental variables …")
    env_ref = extract_block(data, "environmental", "raw")
    # keep only columns that actually exist
    env_vars_present = [v for v in env_variables if v in env_ref.columns]
    env_ref = env_ref.loc[ref_mask, env_vars_present]

    env_anova = anova_table(
        env_ref, labels_ref, env_vars_present,
        transform=anova_transform, label_col="Variable",
    )
    save_table(env_anova, tables_dir / "anova_env",
               formats=table_formats, verbose=verbose)
    env_pvals = extract_pvalues(env_anova, label_col="Variable")

    # ── 9. ANOVA — taxa variables (octave scale) ────────────────────────
    _log(f"[9/11] ANOVA on {taxa_ref.shape[1]} taxa (octave scale) …")
    taxa_anova = anova_table(
        taxa_ref, labels_ref, list(taxa_ref.columns),
        transform=anova_transform, label_col="Taxon",
    )
    save_table(taxa_anova, tables_dir / "anova_taxa",
               formats=table_formats, verbose=verbose)
    taxa_pvals = extract_pvalues(taxa_anova, label_col="Taxon")

    # ── 10. Three-panel cluster figure ──────────────────────────────────
    if save_plots and maps_dir is not None:
        _log("[10/11] Saving cluster panel figure …")
        sample_info = extract_block(data, "sample_info", "raw")
        lat = sample_info.loc[ref_mask, "Latitude"]
        lon = sample_info.loc[ref_mask, "Longitude"]

        # relative-abundance version for the taxa bar chart
        taxa_relabd = octave_to_relative_abundance(taxa_ref)

        fig_panel, _ = plot_cluster_panel(
            cluster_labels=labels_ref,
            lat=lat,
            lon=lon,
            env_data=env_ref,
            taxa_octave=taxa_ref,
            taxa_relabd=taxa_relabd,
            env_pvalues=env_pvals,
            taxa_pvalues=taxa_pvals,
            maps_dir=maps_dir,
            env_vars=env_vars_present,
            taxa_order=TAXA_DISPLAY_ORDER,
        )
        save_figure(fig_panel, figures_dir / "cluster_panel",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_panel)

    # ── 11. Save tables & augmented artifact ─────────────────────────────
    _log("[11/11] Assigning cluster labels and saving outputs …")

    save_table(result.to_ref_table(), tables_dir / "reference_taxa_clusters",
               formats=table_formats, verbose=verbose)

    augmented = result.to_augmented_dataframe()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    augmented_path = artifacts_dir / "02_updated_data.xlsx"
    augmented.to_excel(augmented_path)
    if verbose:
        print(f"  ✓ Saved augmented data: {augmented_path}")

    _log(f"\n✓ Taxa assemblage pipeline complete.  {result.summary()}")

    return result
