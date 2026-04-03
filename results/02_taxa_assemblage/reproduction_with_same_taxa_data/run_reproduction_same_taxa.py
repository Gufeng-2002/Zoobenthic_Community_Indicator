#!/usr/bin/env python
"""
Run Reproduction with Same Taxa Data (reference sites only)
=============================================================

Usage (from project root):
    python results/02_taxa_assemblage/reproduction_with_same_taxa_data/run_reproduction_same_taxa.py

Replicates the full Stage 2 + hindsight-relabel workflow on the
311-site dataset (complete_env_taxa.xlsx), filtered to 62 pre-identified
reference sites:

  Step A  — Ward clustering on the 62 reference sites (same as run_stage2)
            → dendrogram, reference_taxa_clusters table
  Step B  — Hindsight relabel: cluster 2 → 3
  Step C  — ANOVA (env + taxa) and cluster panel on relabelled clusters

Writes : results/02_taxa_assemblage/reproduction_with_same_taxa_data/
             tables/reference_taxa_clusters.xlsx
             tables/anova_env.xlsx
             tables/anova_taxa.xlsx
             figures/ward_dendrogram.png
             figures/cluster_map.png
             figures/cluster_env.png
             figures/cluster_taxa.png
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Ensure the zci package in src/ is importable
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zci.io.readers import read_study_data, extract_block
from zci.io.writers import save_table, save_figure
from zci.core.transforms import octave_to_relative_abundance, octave_transform
from zci.core.clustering import ward_cluster
from zci.core.anova import anova_table, extract_pvalues
from zci.viz.clustering_plots import plot_dendrogram
from zci.viz.cluster_panel_plot import plot_cluster_panel, TAXA_DISPLAY_ORDER

DATA_PATH  = PROJECT_ROOT / "data" / "processed" / "complete_env_taxa.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent          # same folder as this script
MAPS_DIR   = PROJECT_ROOT / "data" / "maps"

# ── Configuration ────────────────────────────────────────────────────

# 62 reference sites (from the 2008 study)
REFERENCE_SITES: list[str] = [
    "004ABC", "S24", "S1", "S79", "S81", "S11", "S57", "S22",
    "S36", "S25", "S39", "S43", "S14", "S20", "S82", "073C",
    "A10", "S99", "S53", "S10", "S8", "S18", "S23", "S9",
    "S49", "S65", "S68", "S28", "S44", "S50", "S40", "S67",
    "DCC2", "S21(5)", "S37", "S38", "GL1", "S21", "S4", "109C",
    "A53", "S74", "S69", "S70", "S51", "S60", "122B", "S102",
    "S28(5)", "S54", "S55", "S52", "S58", "S72", "S59", "026C",
    "S101", "47FB", "A5", "S12", "S17", "S19",
]

# Clustering settings (same as src/run_stage2.py)
N_CLUSTERS: int = 3
TAXA_TRANSFORM: str = "octave"

# Hindsight relabel mapping:  cluster 2 → cluster 3
LABEL_MAP: dict[int, int] = {
    2: 1,
}

ENV_VARIABLES: list[str] = [
    "Measured Depth (m)",
    "Water DO Bottom (mg/L)",
    "Temperature (oC)",
    "MPS (Phi)",
    "LOI (%)",
]

TAXA_COLUMNS: list[str] = [
    "Acari", "Amphipoda", "Caenis", "Ceratopogonidae",
    "Chironomidae", "Dreissena", "Gastropoda", "Hexagenia",
    "Hirudinea", "Hydropsychidae", "Hydrozoa", "Nematoda",
    "Oligochaeta", "Other Trichoptera", "Sphaeriidae", "Turbellaria",
]

ANOVA_TRANSFORM: str = "none"
FIGURE_FORMATS: tuple[str, ...] = ("png",)
TABLE_FORMATS:  tuple[str, ...] = ("xlsx",)


# ── main logic ───────────────────────────────────────────────────────

def run_reproduction(*, verbose: bool = True) -> None:
    tables_dir  = OUTPUT_DIR / "tables"
    figures_dir = OUTPUT_DIR / "figures"

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # ==================================================================
    #  STEP A — Ward clustering (mirrors run_stage2)
    # ==================================================================
    _log("=" * 60)
    _log("STEP A — Ward Clustering on 62 Reference Sites")
    _log("=" * 60)

    # ── A1. Read data ────────────────────────────────────────────────
    _log(f"\n[A1] Reading study data: {DATA_PATH.name}")
    data = read_study_data(DATA_PATH)
    _log(f"      {data.shape[0]} sites × {data.shape[1]} variables")

    # ── A2. Filter to reference sites ────────────────────────────────
    _log(f"\n[A2] Filtering to {len(REFERENCE_SITES)} reference sites …")
    present = [s for s in REFERENCE_SITES if s in data.index]
    missing = [s for s in REFERENCE_SITES if s not in data.index]
    if missing:
        _log(f"      ⚠ {len(missing)} sites not found in data: {missing}")
    _log(f"      {len(present)} reference sites matched")

    # ── A3. Extract & transform taxa ─────────────────────────────────
    _log(f"\n[A3] Extracting taxa data for reference sites …")
    taxa_all = extract_block(data, "taxa", "raw")[TAXA_COLUMNS]
    taxa_ref = taxa_all.loc[present]
    _log(f"      {taxa_ref.shape[0]} sites × {taxa_ref.shape[1]} taxa")

    _log(f"      Applying taxa transform: {TAXA_TRANSFORM!r} …")
    if TAXA_TRANSFORM == "octave":
        taxa_transformed = octave_transform(taxa_ref)
    elif TAXA_TRANSFORM == "relative_abundance":
        taxa_transformed = octave_to_relative_abundance(taxa_ref)
    else:
        raise ValueError(f"Unknown TAXA_TRANSFORM={TAXA_TRANSFORM!r}")

    # ── A4. Ward clustering ──────────────────────────────────────────
    _log(f"\n[A4] Ward clustering (n_clusters={N_CLUSTERS}) …")
    labels_ref, Z = ward_cluster(taxa_transformed, n_clusters=N_CLUSTERS)
    _log("      Cluster distribution (before hindsight):")
    for g in sorted(labels_ref.unique()):
        _log(f"        Cluster {g}: {(labels_ref == g).sum()} sites")

    # ── A5. Dendrogram ───────────────────────────────────────────────
    _log(f"\n[A5] Saving dendrogram …")
    fig_dend, _ = plot_dendrogram(
        Z,
        labels=taxa_ref.index.astype(str),
        n_clusters=N_CLUSTERS,
        title=(
            f"Ward's Dendrogram — {TAXA_TRANSFORM.replace('_', ' ').title()} "
            f"(k = {N_CLUSTERS})"
        ),
    )
    save_figure(fig_dend, figures_dir / "ward_dendrogram",
                formats=FIGURE_FORMATS, verbose=verbose)
    plt.close(fig_dend)

    # ── A6. Save reference taxa clusters table ───────────────────────
    _log(f"\n[A6] Saving reference_taxa_clusters table …")
    ref_table = taxa_ref.copy()
    ref_table.insert(0, "Cluster", labels_ref)
    save_table(ref_table, tables_dir / "reference_taxa_clusters",
               formats=TABLE_FORMATS, verbose=verbose)

    # ==================================================================
    #  STEP B — Hindsight relabel (mirrors run_hindsight_relabel)
    # ==================================================================
    _log("\n" + "=" * 60)
    _log(f"STEP B — Hindsight Relabel: {LABEL_MAP}")
    _log("=" * 60)

    # Two-pass sentinel approach to handle swaps safely
    tmp = labels_ref.astype(float).copy()
    sentinel_map: dict[int, float] = {}
    for i, (old, new) in enumerate(LABEL_MAP.items()):
        sentinel = -(i + 1000)
        sentinel_map[sentinel] = new
        tmp = tmp.replace({float(old): float(sentinel)})
    for sentinel, new in sentinel_map.items():
        tmp = tmp.replace({float(sentinel): float(new)})

    labels_relabelled = tmp.astype(int)
    labels_relabelled.name = "Cluster"

    _log("      Cluster distribution (after hindsight):")
    for g in sorted(labels_relabelled.unique()):
        _log(f"        Cluster {g}: {(labels_relabelled == g).sum()} sites")

    # ==================================================================
    #  STEP C — ANOVA + Cluster Panel (on relabelled clusters)
    # ==================================================================
    _log("\n" + "=" * 60)
    _log("STEP C — ANOVA & Cluster Panel (relabelled clusters)")
    _log("=" * 60)

    # ── C1. Environmental ANOVA ──────────────────────────────────────
    env_all = extract_block(data, "environmental", "raw")
    env_vars_present = [v for v in ENV_VARIABLES if v in env_all.columns]
    env_data = env_all.loc[present, env_vars_present]

    _log(f"\n[C1] ANOVA on {len(env_vars_present)} environmental variables …")
    env_anova = anova_table(
        env_data, labels_relabelled, env_vars_present,
        transform=ANOVA_TRANSFORM, label_col="Variable",
    )
    save_table(env_anova, tables_dir / "anova_env",
               formats=TABLE_FORMATS, verbose=verbose)
    env_pvals = extract_pvalues(env_anova, label_col="Variable")

    # ── C2. Taxa ANOVA ───────────────────────────────────────────────
    _log(f"[C2] ANOVA on {taxa_ref.shape[1]} taxa (octave scale) …")
    taxa_anova = anova_table(
        taxa_ref, labels_relabelled, list(taxa_ref.columns),
        transform=ANOVA_TRANSFORM, label_col="Taxon",
    )
    save_table(taxa_anova, tables_dir / "anova_taxa",
               formats=TABLE_FORMATS, verbose=verbose)
    taxa_pvals = extract_pvalues(taxa_anova, label_col="Taxon")

    # ── C3. Standalone cluster figures ───────────────────────────────
    _log(f"\n[C3] Saving cluster figures …")
    sample_info = extract_block(data, "sample_info", "raw")
    lat = sample_info.loc[present, "Latitude"]
    lon = sample_info.loc[present, "Longitude"]

    taxa_relabd = octave_to_relative_abundance(taxa_ref)

    panel_figures = plot_cluster_panel(
        cluster_labels=labels_relabelled,
        lat=lat,
        lon=lon,
        env_data=env_data,
        taxa_octave=taxa_ref,
        taxa_relabd=taxa_relabd,
        env_pvalues=env_pvals,
        taxa_pvalues=taxa_pvals,
        maps_dir=MAPS_DIR,
        env_vars=env_vars_present,
        taxa_order=TAXA_DISPLAY_ORDER,
    )
    for suffix, (fig_panel, _) in panel_figures.items():
        save_figure(
            fig_panel,
            figures_dir / f"cluster_{suffix}",
            formats=FIGURE_FORMATS,
            verbose=verbose,
        )
        plt.close(fig_panel)

    _log(f"\n✓ Reproduction complete.  Results in:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    run_reproduction()
