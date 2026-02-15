#!/usr/bin/env python
"""
Run Stage 1: Pollution PCA Pipeline
====================================

Usage (from project root):
    python src2/run_stage1.py

Reads  : data/processed/complete_env_taxa_chemical_Feb_3.xlsx
Writes : results2/01_pollution_assessment/tables/  (pc_loadings, site_scores)
         results2/01_pollution_assessment/figures/ (variance_explained, ridge_loadings)
"""

from pathlib import Path

# Resolve project root (parent of src2/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "complete_env_taxa_chemical_Feb_3.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "results2" / "01_pollution_assessment"

# Import the pipeline
from zci.pipeline.pollution_assessment import pollution_pca_pipeline


if __name__ == "__main__":
    result = pollution_pca_pipeline(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        n_components=5,
        standardise_scores="min-max",
        save_plots=True,
    )

    print(f"\nLoadings shape : {result.loadings.shape}")
    print(f"Scores shape   : {result.scores.shape}")
    print(f"\nLoadings preview:\n{result.loadings.round(4)}")
