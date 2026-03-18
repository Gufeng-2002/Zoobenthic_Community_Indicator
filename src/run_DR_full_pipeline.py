#!/usr/bin/env python
"""
Run Full Pipeline – Detroit River (DR) Sites Only
===================================================

Usage (from project root):
    python src/run_DR_full_pipeline.py

Same 7-stage workflow as the SCDRS pipeline, but restricted to Detroit
River sites (``Waterbody == "DR"``).  Differences from SCDRS:

  * Pre-filters the data to DR sites only and writes a separate Excel.
  * From RDA onward, includes the extra environmental variable
    ``"Velocity  at bottom (m/sec)"`` in the environmental block.
  * Stage 5 uses lighter settings (fewer taus, smaller bootstrap,
    fewer sensitivity fractions) for faster turnaround.
  * All results go to ``results/DR_results/`` with the same sub-folder
    structure as the SCDRS results/ layout.

Order:
    1. Pre-filter — extract DR sites, save as new Excel workbook
    2. Stage 1   — Pollution PCA
    3. Stage RDA  — Redundancy Analysis  (with Velocity)
    4. Stage 2   — Taxa Assemblage Clustering
    5. Hindsight  — Remap cluster labels + ANOVA + cluster panel (with Velocity)
    6. Stage 3   — LDA Classification (with Velocity)
    7. Stage 4   — Bray–Curtis NMDS + ZCI
    8. Stage 5   — Piecewise Quantile Regression (fast settings)
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── project root ─────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── shared paths ─────────────────────────────────────────────────────

ORIGINAL_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "complete_env_taxa_chemical_Feb_3.xlsx"
)
DR_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "DR_env_taxa_chemical.xlsx"
)
MAPS_DIR = PROJECT_ROOT / "data" / "maps"

# ── DR result folder (mirrors results/ structure) ──────────────────────

DR_RESULTS = PROJECT_ROOT / "results" / "DR_results"

STAGE1_DIR = DR_RESULTS / "01_pollution_assessment"
RDA_DIR    = DR_RESULTS / "RDA_analysis"
STAGE2_DIR = DR_RESULTS / "02_taxa_assemblage"
STAGE3_DIR = DR_RESULTS / "03_LDA_Classification"
STAGE4_DIR = DR_RESULTS / "04_bray_curtis_NMDS"
STAGE5_DIR = DR_RESULTS / "05_piecewise_qr"

# Artifact paths
STAGE1_ARTIFACT = STAGE1_DIR / "artifacts" / "01_updated_data.xlsx"
STAGE2_ARTIFACT = STAGE2_DIR / "artifacts" / "02_updated_data.xlsx"
STAGE2_HINDSIGHT_ARTIFACT = STAGE2_DIR / "artifacts" / "02_hindsight_updated_data.xlsx"
STAGE3_ARTIFACT = STAGE3_DIR / "artifacts" / "03_updated_data.xlsx"
STAGE4_ARTIFACT = STAGE4_DIR / "artifacts" / "04_updated_data.xlsx"

# ── environmental variables ──────────────────────────────────────────

# Base 5 env variables (same as SCDRS, used in Stage 1 / PCA)
ENV_VARIABLES_BASE = [
    "Measured Depth (m)",
    "Water DO Bottom (mg/L)",
    "Temperature (oC)",
    "MPS (Phi)",
    "LOI (%)",
]

# DR has velocity available for all sites — add it from RDA onward
ENV_VARIABLES_DR = ENV_VARIABLES_BASE + [
    "Velocity  at bottom (m/sec)",
]


# ── imports ──────────────────────────────────────────────────────────

from zci.io.readers import read_study_data, extract_block
from zci.pipeline.pollution_assessment import pollution_pca_pipeline
from zci.pipeline.rda_analysis import rda_pipeline
from zci.pipeline.taxa_assemblage import taxa_assemblage_pipeline
from zci.pipeline.lda_classification import lda_pipeline
from zci.pipeline.bray_curtis_nmds import nmds_pipeline
from zci.pipeline.piecewise_qr_pipeline import pqr_pipeline
from zci.viz.map_plots import plot_dr_bifurcation, plot_dr_map

from run_hindsight_relabel import (
    relabel_clusters,
    run_anova_and_panel,
)


# ── helpers ──────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print("\n")
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _filter_dr_sites(src_path: Path, dst_path: Path) -> int:
    """Read the full SCDRS workbook, keep only DR sites, save to *dst_path*.

    Returns the number of DR sites found.
    """
    _banner("PRE-FILTER — Extract Detroit River (DR) Sites")

    print(f"[1/3] Reading full SCDRS dataset:\n      {src_path}")
    df = read_study_data(src_path)
    print(f"      {len(df)} total sites")

    # Locate the Waterbody column in sample_info block
    sample_info = extract_block(df, "sample_info", "raw")
    if "Waterbody" not in sample_info.columns:
        raise ValueError(
            "Cannot find 'Waterbody' column in sample_info block. "
            f"Available columns: {list(sample_info.columns)}"
        )

    wb = sample_info["Waterbody"]
    dr_mask = wb == "DR"
    n_dr = dr_mask.sum()

    print(f"[2/3] Filtering to Waterbody == 'DR' …")
    print(f"      {n_dr} DR sites out of {len(df)} total")
    print(f"      Other waterbodies: {dict(wb.value_counts())}")

    df_dr = df.loc[dr_mask]

    print(f"[3/3] Saving DR subset:\n      {dst_path}")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df_dr.to_excel(dst_path)

    print(f"\n✓ DR pre-filter complete.  {n_dr} sites saved.")
    return n_dr


# ── main pipeline ────────────────────────────────────────────────────

def run_dr_full_pipeline() -> None:
    t0 = time.time()

    # ==================================================================
    #  0. PRE-FILTER — Extract Detroit River sites
    # ==================================================================
    n_dr = _filter_dr_sites(ORIGINAL_DATA_PATH, DR_DATA_PATH)

    # ==================================================================
    #  1. STAGE 1 — Pollution PCA  (same as SCDRS but on DR data)
    # ==================================================================
    _banner("STAGE 1 — Pollution PCA  (DR sites)")
    pollution_pca_pipeline(
        data_path=DR_DATA_PATH,
        output_dir=STAGE1_DIR,
        n_components=5,
        selected_pcs=None,
        composite_transform="min-max",
        maps_dir=MAPS_DIR,
        threshold_quantile=0.20,
        bifurcation_plot_func=plot_dr_bifurcation,
        save_plots=True,
    )

    # ==================================================================
    #  2. STAGE RDA — Redundancy Analysis  (+Velocity)
    # ==================================================================
    _banner("STAGE RDA — Redundancy Analysis  (DR, +Velocity)")
    rda_pipeline(
        data_path=DR_DATA_PATH,
        stage1_artifact=STAGE1_ARTIFACT,
        output_dir=RDA_DIR,
        env_variables=ENV_VARIABLES_DR,
        reference_quantile=0.20,
        standardize_env=True,
        log_transform_env=False,
        taxa_transform="octave",
        n_permutations=999,
        random_state=42,
        cluster_column="Cluster",
        save_plots=True,
    )

    # ==================================================================
    #  3. STAGE 2 — Taxa Assemblage Clustering
    # ==================================================================
    _banner("STAGE 2 — Taxa Assemblage Clustering  (DR sites)")
    taxa_assemblage_pipeline(
        data_path=DR_DATA_PATH,
        stage1_artifact=STAGE1_ARTIFACT,
        output_dir=STAGE2_DIR,
        reference_quantile=0.20,
        taxa_transform="octave",
        n_clusters=3,
        save_plots=True,
    )

    # ==================================================================
    #  4. HINDSIGHT RELABEL — Remap clusters + ANOVA + panel (+Velocity)
    # ==================================================================
    _banner("HINDSIGHT RELABEL — Remap clusters + ANOVA + Cluster Panel  (DR)")

    # First, inspect cluster distribution to decide the label map.
    # Read the Stage 2 artifact to see the distribution.
    _tmp_df = pd.read_excel(STAGE2_ARTIFACT, header=[0, 1, 2], index_col=0)
    _cluster_key = ("02_taxa_assemblage", "raw", "Cluster")
    if _cluster_key not in _tmp_df.columns:
        _cands = [c for c in _tmp_df.columns if "Cluster" in str(c)]
        _cluster_key = _cands[0] if _cands else _cluster_key
    _clusters = _tmp_df[_cluster_key].dropna()
    _dist = _clusters.value_counts().sort_index()
    print(f"\n  Stage 2 cluster distribution (before relabel):")
    for g, n in _dist.items():
        print(f"    Cluster {int(g)}: {n} sites")

    # Auto-determine label map: merge the smallest cluster into Cluster 1
    # if there are 3 clusters and the smallest is very small.
    _unique = sorted(_dist.index)
    if len(_unique) == 3:
        # Find the cluster with the fewest sites (excluding Cluster 1)
        _others = {int(g): int(n) for g, n in _dist.items() if int(g) != 1}
        _smallest_label = min(_others, key=_others.get)
        _smallest_size = _others[_smallest_label]
        if _smallest_size <= max(5, int(0.15 * len(_clusters))):
            label_map = {_smallest_label: 1}
            print(f"  → Auto label map: {label_map}  "
                  f"(merging {_smallest_size}-site cluster into Cluster 1)")
        else:
            label_map = {}
            print(f"  → No relabel needed (all clusters are substantial)")
    elif len(_unique) == 2:
        label_map = {}
        print(f"  → Two clusters found — no relabel needed")
    else:
        label_map = {}
        print(f"  → {len(_unique)} clusters found — keeping as-is")

    if label_map:
        relabelled_df = relabel_clusters(
            artifact_path=STAGE2_ARTIFACT,
            output_path=STAGE2_HINDSIGHT_ARTIFACT,
            label_map=label_map,
        )
    else:
        # No relabelling — just copy the artifact
        import shutil
        shutil.copy2(STAGE2_ARTIFACT, STAGE2_HINDSIGHT_ARTIFACT)
        relabelled_df = _tmp_df
        print("  → Copied Stage 2 artifact as-is (no relabel).")

    run_anova_and_panel(
        data_path=DR_DATA_PATH,
        relabelled_artifact=relabelled_df,
        output_dir=STAGE2_DIR,
        maps_dir=MAPS_DIR,
        env_variables=ENV_VARIABLES_DR,
        anova_transform="none",
        map_func=plot_dr_map,
        figure_formats=("png",),
        table_formats=("xlsx",),
    )

    # ==================================================================
    #  5. STAGE 3 — LDA Classification  (+Velocity)
    # ==================================================================
    _banner("STAGE 3 — LDA Classification  (DR, +Velocity)")
    lda_pipeline(
        data_path=DR_DATA_PATH,
        stage1_artifact=STAGE1_ARTIFACT,
        stage2_artifact=STAGE2_HINDSIGHT_ARTIFACT,
        output_dir=STAGE3_DIR,
        env_variables=ENV_VARIABLES_DR,
        reference_quantile=0.20,
        standardize_env=True,
        n_mccv_iterations=1000,
        mccv_test_size=0.2,
        random_state=42,
        save_plots=True,
    )

    # ==================================================================
    #  6. STAGE 4 — Bray–Curtis NMDS + ZCI  (same method/settings)
    # ==================================================================
    _banner("STAGE 4 — Bray–Curtis NMDS + ZCI  (DR sites)")
    nmds_pipeline(
        data_path=DR_DATA_PATH,
        stage1_artifact=STAGE1_ARTIFACT,
        stage3_artifact=STAGE3_ARTIFACT,
        output_dir=STAGE4_DIR,
        nmds_n_ep=5,
        zci_config={
            1: {"N_EP": 5, "Method": "BC-Direct"},
            2: {"N_EP": 5, "Method": "BC-Direct"},
            3: {"N_EP": 5, "Method": "BC-Direct"},
        },
        n_components=2,
        n_nmds_iterations=3,
        max_iter_per_run=1000,
        n_init_first=10,
        n_init_subsequent=4,
        random_state=42,
        save_plots=True,
        figure_formats=("png",),
        table_formats=("xlsx",),
    )

    # ==================================================================
    #  7. STAGE 5 — Piecewise Quantile Regression  (fast settings)
    # ==================================================================
    _banner("STAGE 5 — Piecewise Quantile Regression  (DR, fast settings)")
    pqr_pipeline(
        stage1_artifact=STAGE1_ARTIFACT,
        stage3_artifact=STAGE3_ARTIFACT,
        stage4_artifact=STAGE4_ARTIFACT,
        output_dir=STAGE5_DIR,
        n_breakpoints=1,
        # Fewer quantile levels: 7 instead of 17
        taus=[0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90],
        highlight_taus=(0.25, 0.50, 0.75),
        # Smaller bootstrap
        n_boot=200,
        confidence=0.90,
        grid_size=30,
        search_range=(0.4, 0.6),
        # Lighter sensitivity analysis
        run_sensitivity=True,
        sensitivity_tau=0.50,
        sensitivity_fracs=[0.20, 0.40, 0.60, 0.80, 1.00],
        sensitivity_repeats=5,
        sensitivity_boot=100,
        min_cluster_size=15,
        random_state=42,
        save_plots=True,
        figure_formats=("png",),
        table_formats=("xlsx",),
    )

    # ==================================================================
    elapsed = time.time() - t0
    _banner(f"DR FULL PIPELINE COMPLETE — {elapsed:.1f} s")


if __name__ == "__main__":
    run_dr_full_pipeline()
