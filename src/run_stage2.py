#!/usr/bin/env python
"""
Run Stage 2 -- Full Pipeline: Ward's Clustering → LDA & MRT Classification
============================================================================

Usage (from project root):
    python src/run_stage2.py

Runs three sub-pipelines in sequence:
  1. Ward's Clustering with robustness assessment
  2. Confidence-Aware LDA classification (4 models)
  3. Confidence-Aware MRT classification (4 models)

Global parameters
-----------------
TAXA_TRANSFORM : str
    Transformation applied to raw taxa counts before clustering *and*
    MRT classification.  One of "octave", "chord", "hellinger",
    "log_chord", "relative_abundance".
N_REFERENCE_SITES : int
    Number of least-polluted sites to enter Ward's clustering.  The
    resulting cluster labels and robustness status table then feed
    into the downstream LDA and MRT pipelines.
"""

import pandas as pd
from pathlib import Path

from zci.pipeline.wards_clustering import wards_clustering_pipeline
from zci.pipeline.confidence_lda import confidence_lda_pipeline
from zci.pipeline.confidence_mrt import confidence_mrt_pipeline

# ═══════════════════════════════════════════════════════════════════════
#  GLOBAL PARAMETERS — change these as needed
# ═══════════════════════════════════════════════════════════════════════
TAXA_TRANSFORM: str = "chord"
"""Taxa transformation for Ward's clustering and MRT classification.
One of: "octave", "chord", "hellinger", "log_chord", "relative_abundance".
"""

N_REFERENCE_SITES: int = 50
"""Number of least-polluted reference sites entering Ward's clustering."""

# ═══════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "complete_env_taxa_chemical_Feb_3.xlsx"
STAGE1_ARTIFACT = (
    PROJECT_ROOT / "results" / "01_pollution_assessment"
    / "PCA_Stressors" / "artifacts" / "SumRel_01_updated_data.xlsx"
)
MAPS_DIR = PROJECT_ROOT / "data" / "maps"

WARDS_OUTPUT = PROJECT_ROOT / "results" / "02_taxa_assemblage" / "WardsClustering"
LDA_OUTPUT   = PROJECT_ROOT / "results" / "02_taxa_assemblage" / "LDA_Method"
MRT_OUTPUT   = PROJECT_ROOT / "results" / "02_taxa_assemblage" / "MRT_Method"

# Shared environmental variables for LDA and MRT
ENV_VARIABLES = [
    "Measured Depth (m)",
    "Water DO Bottom (mg/L)",
    "Temperature (oC)",
    "MPS (Phi)",
    "LOI (%)",
]


if __name__ == "__main__":
    print("=" * 70)
    print("  Stage 2 Full Pipeline")
    print("=" * 70)
    print(f"  Taxa transform   : {TAXA_TRANSFORM}")
    print(f"  Reference sites  : {N_REFERENCE_SITES}")
    print("=" * 70)

    # ── 1. Ward's Clustering ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  [1/3] Ward's Clustering")
    print("=" * 70)
    ward_result = wards_clustering_pipeline(
        data_path=DATA_PATH,
        stage1_artifact=STAGE1_ARTIFACT,
        output_dir=WARDS_OUTPUT,
        maps_dir=MAPS_DIR,
        reference_quantile=N_REFERENCE_SITES,
        taxa_transform=TAXA_TRANSFORM,
        n_clusters=3,
        label_map={1: 1, 2: 2, 3: 3},
        env_variables=ENV_VARIABLES,
        # Robustness parameters
        n_boot_coassign=1000,
        coassign_sample_frac=0.8,
        n_boot_pvclust=1000,
        run_pvclust_bootstrap=True,
        sil_threshold=0.25,
        margin_threshold=0.25,
        random_state=42,
        save_plots=True,
    )

    print(f"\n{ward_result.summary()}")
    print(f"\nCluster distribution (reference sites):")
    print(ward_result.cluster_distribution())
    print(f"\nStatus distribution:")
    print(ward_result.status_distribution())
    print(f"\nMean silhouette: {ward_result.mean_silhouette():.4f}")

    # Read the robustness table produced by Ward's clustering
    robustness = pd.read_excel(
        WARDS_OUTPUT / "tables" / "site_robustness.xlsx", index_col=0
    )
    print(f"\n  {len(robustness)} reference sites, "
          f"{robustness['Original_Cluster'].nunique()} clusters")
    print(f"  Status distribution: "
          f"{robustness['Status'].value_counts().to_dict()}")

    # ── 2. Confidence-Aware LDA ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("  [2/3] Confidence-Aware LDA Classification")
    print("=" * 70)
    lda_result = confidence_lda_pipeline(
        data_path=DATA_PATH,
        stage1_artifact=STAGE1_ARTIFACT,
        output_dir=LDA_OUTPUT,
        site_robustness=robustness,
        env_variables=ENV_VARIABLES,
        standardize_env=True,
        cv_folds=5,
        cv_repeats=10,
        random_state=42,
        save_plots=True,
    )

    print(f"\n{lda_result.summary()}")
    print(f"\nLDA model comparison:")
    print(lda_result.summary_table.to_string())

    # ── 3. Confidence-Aware MRT ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("  [3/3] Confidence-Aware MRT Classification")
    print("=" * 70)
    mrt_result = confidence_mrt_pipeline(
        data_path=DATA_PATH,
        stage1_artifact=STAGE1_ARTIFACT,
        output_dir=MRT_OUTPUT,
        site_robustness=robustness,
        env_variables=ENV_VARIABLES,
        response_transform=TAXA_TRANSFORM,
        k_folds=5,
        cv_perms=10,
        minsplit=3,
        minbucket=2,
        random_state=42,
        save_plots=True,
    )

    print(f"\nMRT model comparison:")
    print(mrt_result["summary_table"].to_string())

    # ── Done ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Stage 2 Complete")
    print("=" * 70)
