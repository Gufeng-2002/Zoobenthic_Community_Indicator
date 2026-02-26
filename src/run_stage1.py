#!/usr/bin/env python
"""
Run Stage 1: Pollution PCA Pipeline
====================================

Usage (from project root):
    python src/run_stage1.py

Reads  : data/processed/complete_env_taxa_chemical_Feb_3.xlsx
Writes : results/01_pollution_assessment/tables/  (pc_loadings, site_scores)
         results/01_pollution_assessment/figures/ (variance_explained, ridge_loadings,
                                                    corridor_bifurcation)
         results/01_pollution_assessment/artifacts/01_updated_data.xlsx
"""

from pathlib import Path

# Resolve project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "complete_env_taxa_chemical_Feb_3.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "results" / "01_pollution_assessment"
MAPS_DIR   = PROJECT_ROOT / "data" / "maps"

# Import the pipeline
from zci.pipeline.pollution_assessment import pollution_pca_pipeline


if __name__ == "__main__":
    result = pollution_pca_pipeline(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        n_components=5,
        standardise_scores="min-max",
        selected_pcs=None,               # None → all 5 PCs
        composite_transform="min-max",    # 'min-max' or 'z-score'
        composite_weights=None,           # None → equal weights
        maps_dir=MAPS_DIR,               # shapefile folder
        threshold_quantile=0.20,          # bottom 20 % bifurcation
        save_plots=True,
    )

    print(f"\nLoadings shape : {result.loadings.shape}")
    print(f"Scores shape   : {result.scores.shape}")
    print(f"\nLoadings preview:\n{result.loadings.round(4)}")
