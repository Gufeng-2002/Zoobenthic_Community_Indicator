#!/usr/bin/env python
"""
Run Stage 1: Pollution PCA Pipeline (MaxRel + SumRel)
=====================================================

Usage (from project root):
    python src/run_stage1.py

Reads  : data/processed/complete_env_taxa_chemical_Feb_3.xlsx
Writes : results/01_pollution_assessment/tables/
             pc_loadings, site_scores,
             SumRel_site_rankings, MaxRel_site_rankings
         results/01_pollution_assessment/figures/
             variance_explained, ridge_loadings,
             SumRel_corridor_bifurcation, MaxRel_corridor_bifurcation
         results/01_pollution_assessment/artifacts/
             SumRel_01_updated_data.xlsx
             MaxRel_01_updated_data.xlsx
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
        composite_transform="min-max",    # rescaling before aggregation
        maps_dir=MAPS_DIR,               # shapefile folder
        threshold_quantile=0.20,          # bottom 20 % bifurcation
        save_plots=True,
    )

    pca = result.pca_result
    print(f"\nLoadings shape : {pca.loadings.shape}")
    print(f"Scores shape   : {pca.scores.shape}")
    print(f"\nLoadings preview:\n{pca.loadings.round(4)}")
    print(f"\nSumRel range: [{result.sumrel_score.min():.4f}, {result.sumrel_score.max():.4f}]")
    print(f"MaxRel range: [{result.maxrel_score.min():.4f}, {result.maxrel_score.max():.4f}]")
