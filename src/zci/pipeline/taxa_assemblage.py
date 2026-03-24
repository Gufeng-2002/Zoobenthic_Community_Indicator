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
from ..models.clustering import TAXA_COLUMNS, ClusteringResult
from ..viz.clustering_plots import plot_dendrogram


def taxa_assemblage_pipeline(
    data_path: str | Path,
    stage1_artifact: str | Path,
    output_dir: str | Path,
    *,
    taxa_columns: Sequence[str] = TAXA_COLUMNS,
    reference_quantile: float = 0.20,
    taxa_transform: str = "octave",
    n_clusters: int = 2,
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
    8. Assign cluster labels and save tables / artifacts.

    Note: ANOVA tests and the cluster panel figure are deferred to the
    hindsight-relabel step so they reflect the final cluster labels.

    Parameters
    ----------
    data_path : str or Path
        Path to the study-data Excel file.
    stage1_artifact : str or Path
        Path to ``results/01_pollution_assessment/artifacts/01_updated_data.xlsx``.
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

    # ── 2. Read Stage 1 artifact → Pollution Score ────────────────────────
    _log("[2/8] Reading Stage 1 artifact for pollution scores …")
    stage1 = pd.read_excel(stage1_artifact, header=[0, 1, 2], index_col=0)
    # Auto-detect score column (SumRel_Score, MaxRel_Score, or Pollution_Score)
    score_cols = [
        c for c in stage1.columns
        if c[0] == "01_pollution_assessment" and c[1] == "raw"
        and c[2].endswith("_Score")
    ]
    if not score_cols:
        raise KeyError("No pollution score column found in Stage 1 artifact")
    score_key = score_cols[0]
    pollution_score = stage1.loc[:, score_key]
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
        _log("[7/8] Saving dendrogram …")
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

    # ── 8. Save tables & augmented artifact ──────────────────────────────
    _log("[8/8] Assigning cluster labels and saving outputs …")

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
