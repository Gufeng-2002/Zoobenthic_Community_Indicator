#!/usr/bin/env python
"""
Run Full Pipeline: Stage 1 → RDA → Stage 2 → Hindsight Relabel → Stage 3 (LDA) → Stage 3 (NMDS) → Stage 4
================================================================================================

Usage (from project root):
    python src/run_SCDRS_full_pipeline.py

Executes all stages in order, with the same parameters and output
directories as the individual run_stage*.py scripts.  Each stage
overwrites its output folder.

Order:
    1. Stage 1  — Pollution PCA
    2. Stage RDA — Redundancy Analysis
    3. Stage 2  — Taxa Assemblage Clustering
    4. Hindsight Relabel — Remap cluster labels + ANOVA + cluster panel
    5. Stage 3 (LDA) — LDA Classification
    6. Stage 3  — Bray–Curtis NMDS + ZCI
    7. Stage 4  — Piecewise Quantile Regression
"""

import time
from pathlib import Path

import numpy as np

# ── project root ─────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── shared paths ─────────────────────────────────────────────────────

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "complete_env_taxa_chemical_Feb_3.xlsx"
)
MAPS_DIR = PROJECT_ROOT / "data" / "maps"

# Artifact paths (output of one stage → input of the next)
STAGE1_ARTIFACT = (
    PROJECT_ROOT / "results" / "01_pollution_assessment"
    / "contamination_stressors" / "artifacts" / "01_updated_data.xlsx"
)
STAGE2_ARTIFACT = (
    PROJECT_ROOT / "results" / "02_taxa_assemblage"
    / "artifacts" / "02_updated_data.xlsx"
)
STAGE2_HINDSIGHT_ARTIFACT = (
    PROJECT_ROOT / "results" / "02_taxa_assemblage"
    / "artifacts" / "02_hindsight_updated_data.xlsx"
)
STAGE3_ARTIFACT = (
    PROJECT_ROOT / "results" / "03_bray_curtis_NMDS"
    / "artifacts" / "03_updated_data.xlsx"
)


# ── imports (lazy, so module-level path is resolved first) ───────────

from zci.pipeline.pollution_assessment import pollution_pca_pipeline
from zci.pipeline.rda_analysis import rda_pipeline
from zci.pipeline.taxa_assemblage import taxa_assemblage_pipeline
from zci.pipeline.lda_classification import lda_pipeline
from zci.pipeline.bray_curtis_nmds import nmds_pipeline
from zci.pipeline.piecewise_qr_pipeline import pqr_pipeline

from run_hindsight_relabel import (
    relabel_clusters,
    run_anova_and_panel,
)


# ── helper ───────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print("\n")
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ── main ─────────────────────────────────────────────────────────────

def run_full_pipeline() -> None:
    t0 = time.time()

    # ==================================================================
    #  1. STAGE 1 — Pollution PCA
    # ==================================================================
    _banner("STAGE 1 — Pollution PCA")
    pollution_pca_pipeline(
        data_path=DATA_PATH,
        output_dir=PROJECT_ROOT / "results" / "01_pollution_assessment" / "contamination_stressors",
        pollution_standardize=True,
        n_components=5,
        selected_pcs=None,
        composite_transform="min-max",
        maps_dir=MAPS_DIR,
        threshold_quantile=0.20,
        save_plots=True,
    )

    # ==================================================================
    #  2. STAGE RDA — Redundancy Analysis
    # ==================================================================
    _banner("STAGE RDA — Redundancy Analysis")
    rda_pipeline(
        data_path=DATA_PATH,
        stage1_artifact=STAGE1_ARTIFACT,
        output_dir=PROJECT_ROOT / "results" / "RDA_analysis",
        env_variables=[
            "Measured Depth (m)",
            "Water DO Bottom (mg/L)",
            "Temperature (oC)",
            "MPS (Phi)",
            "LOI (%)",
        ],
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
    _banner("STAGE 2 — Taxa Assemblage Clustering")
    taxa_assemblage_pipeline(
        data_path=DATA_PATH,
        stage1_artifact=STAGE1_ARTIFACT,
        output_dir=PROJECT_ROOT / "results" / "02_taxa_assemblage",
        reference_quantile=0.20,
        taxa_transform="octave",
        n_clusters=3,
        save_plots=True,
    )

    # ==================================================================
    #  4. HINDSIGHT RELABEL — Remap clusters + ANOVA + panel
    # ==================================================================
    _banner("HINDSIGHT RELABEL — Remap clusters + ANOVA + Cluster Panel")
    relabelled_df = relabel_clusters(
        artifact_path=STAGE2_ARTIFACT,
        output_path=STAGE2_HINDSIGHT_ARTIFACT,
        label_map={3: 1},
    )
    run_anova_and_panel(
        data_path=DATA_PATH,
        relabelled_artifact=relabelled_df,
        output_dir=PROJECT_ROOT / "results" / "02_taxa_assemblage",
        maps_dir=MAPS_DIR,
        env_variables=[
            "Measured Depth (m)",
            "Water DO Bottom (mg/L)",
            "Temperature (oC)",
            "MPS (Phi)",
            "LOI (%)",
        ],
        anova_transform="none",
        figure_formats=("png",),
        table_formats=("xlsx",),
    )

    # ==================================================================
    #  5. STAGE 3 — LDA Classification
    # ==================================================================
    _banner("STAGE 3 — LDA Classification")
    lda_pipeline(
        data_path=DATA_PATH,
        stage1_artifact=STAGE1_ARTIFACT,
        stage2_artifact=STAGE2_HINDSIGHT_ARTIFACT,
        output_dir=PROJECT_ROOT / "results" / "03_LDA_Classification",
        env_variables=[
            "Measured Depth (m)",
            "Water DO Bottom (mg/L)",
            "Temperature (oC)",
            "MPS (Phi)",
            "LOI (%)",
        ],
        reference_quantile=0.20,
        standardize_env=True,
        n_mccv_iterations=1000,
        mccv_test_size=0.2,
        random_state=42,
        save_plots=True,
    )

    # ==================================================================
    #  6. STAGE 3 — Bray–Curtis NMDS + ZCI
    # ==================================================================
    _banner("STAGE 3 — Bray–Curtis NMDS + ZCI")
    nmds_pipeline(
        data_path=DATA_PATH,
        stage1_artifact=STAGE1_ARTIFACT,
        stage2_artifact=STAGE2_ARTIFACT,
        output_dir=PROJECT_ROOT / "results" / "03_bray_curtis_NMDS",
        nmds_n_ep=5,
        zci_config={
            1: {"N_EP": 15, "Method": "BC-Direct"},
            2: {"N_EP":  3, "Method": "BC-Direct"},
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
    #  7. STAGE 4 — Piecewise Quantile Regression
    # ==================================================================
    # _banner("STAGE 4 — Piecewise Quantile Regression")
    # pqr_pipeline(
    #     stage1_artifact=STAGE1_ARTIFACT,
    #     stage2_artifact=STAGE2_ARTIFACT,
    #     stage3_artifact=STAGE3_ARTIFACT,
    #     output_dir=PROJECT_ROOT / "results" / "04_piecewise_qr",
    #     n_breakpoints=1,
    #     taus=list(np.round(np.arange(0.1, 0.91, 0.05), 2)),
    #     highlight_taus=(0.20, 0.50, 0.80),
    #     n_boot=500,
    #     confidence=0.90,
    #     grid_size=30,
    #     search_range=(0.4, 0.6),
    #     run_sensitivity=True,
    #     sensitivity_tau=0.50,
    #     sensitivity_fracs=list(np.round(np.arange(0.15, 1.01, 0.03), 2)),
    #     sensitivity_repeats=5,
    #     sensitivity_boot=500,
    #     min_cluster_size=15,
    #     random_state=42,
    #     save_plots=True,
    #     figure_formats=("png",),
    #     table_formats=("xlsx",),
    # )

    # ==================================================================
    elapsed = time.time() - t0
    _banner(f"FULL PIPELINE COMPLETE — {elapsed:.1f} s")


if __name__ == "__main__":
    run_full_pipeline()
