#!/usr/bin/env python
"""
Run Stage 2 -- Ward's Clustering: Robustness Assessment
========================================================

Usage (from project root):
    python src/run_stage2_WardsClustering.py

Reads  : data/processed/complete_env_taxa_chemical_Feb_3.xlsx
         results/01_pollution_assessment/PCA_Stressors/artifacts/SumRel_01_updated_data.xlsx
Writes : results/02_taxa_assemblage/WardsClustering/{figures,tables,artifacts}

Ward clustering on reference sites with:
  - Silhouette widths (per site)
  - Bootstrap co-assignment matrix and site-level confidence
  - pvclust multiscale bootstrap (AU/BP per cluster branch)
  - Qualitative status (Core / Peripheral / Uncertain)
  - ANOVA on environmental and taxa variables
  - Dendrogram and cluster panel figures

The clustering result (artifact) is consumed by
run_stage2_LDAMethod.py and run_stage2_MRTMethod.py.
"""

from pathlib import Path

from zci.pipeline.wards_clustering import (
    refresh_combined_robustness_outputs,
    wards_clustering_pipeline,
)

# Resolve project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "complete_env_taxa_chemical_Feb_3.xlsx"
STAGE1_ARTIFACT = (
    PROJECT_ROOT / "results" / "01_pollution_assessment"
    / "PCA_Stressors" / "artifacts" / "SumRel_01_updated_data.xlsx"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "02_taxa_assemblage" / "WardsClustering"
MAPS_DIR   = PROJECT_ROOT / "data" / "maps"

ENV_VARIABLES = [
    "Measured Depth (m)",
    "Water DO Bottom (mg/L)",
    "Temperature (oC)",
    "MPS (Phi)",
    "LOI (%)",
]


if __name__ == "__main__":
    result = wards_clustering_pipeline(
        data_path=DATA_PATH,
        stage1_artifact=STAGE1_ARTIFACT,
        output_dir=OUTPUT_DIR,
        maps_dir=MAPS_DIR,
        reference_quantile=50,
        taxa_transform="octave",
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

    print("\nRunning threshold grid search to refresh Ward combined outputs ...")
    combined_table, class_count, best_th, _ = refresh_combined_robustness_outputs(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        env_variables=ENV_VARIABLES,
        verbose=True,
    )

    print(f"\n{result.summary()}")
    print(f"\nCluster distribution (reference sites):")
    print(result.cluster_distribution())
    print("\nBest thresholds:")
    print(best_th)
    print("\nUpdated 2x2 class counts:")
    print(class_count.to_string())
    print("\nUpdated TaxaEnv_Class distribution:")
    print(combined_table["TaxaEnv_Class"].value_counts().to_string())
    print(f"\nStatus distribution:")
    print(result.status_distribution())
    print(f"\nMean silhouette: {result.mean_silhouette():.4f}")
    print(f"\nRobustness table preview:")
    print(result.robustness_table.to_string())
