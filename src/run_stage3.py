#!/usr/bin/env python
"""
Run Stage LDA: Linear Discriminant Analysis Classification
============================================================

Usage (from project root):
    python src/run_stage3.py

Fits LDA on the reference sites identified by Stage 1, using cluster
labels assigned by Stage 2.  Then predicts clusters for all non-reference
sites and produces a 4-panel comparison figure.

Reads  : data/processed/complete_env_taxa_chemical_Feb_3.xlsx
         results/01_pollution_assessment/artifacts/01_updated_data.xlsx
         results/02_taxa_assemblage/artifacts/02_hindsight_updated_data.xlsx
Writes : results/03_LDA_Classification/tables/  (6 tables)
         results/03_LDA_Classification/figures/ (lda_triplot, cluster_comparison)
         results/03_LDA_Classification/artifacts/03_updated_data.xlsx
"""

from pathlib import Path

# Resolve project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_PATH       = PROJECT_ROOT / "data" / "processed" / "complete_env_taxa_chemical_Feb_3.xlsx"
STAGE1_ARTIFACT = PROJECT_ROOT / "results" / "01_pollution_assessment" / "artifacts" / "01_updated_data.xlsx"
STAGE2_ARTIFACT = PROJECT_ROOT / "results" / "02_taxa_assemblage" / "artifacts" / "02_hindsight_updated_data.xlsx"
OUTPUT_DIR      = PROJECT_ROOT / "results" / "03_LDA_Classification"

# Import the pipeline
from zci.pipeline.lda_classification import lda_pipeline


if __name__ == "__main__":
    result = lda_pipeline(
        data_path=DATA_PATH,
        stage1_artifact=STAGE1_ARTIFACT,
        stage2_artifact=STAGE2_ARTIFACT,
        output_dir=OUTPUT_DIR,
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

    print(f"\n{result.summary()}")
    print("\n── LDA Axes Summary ──")
    print(result.wilks.axes_summary.to_string())
    print("\n── Variable Importance ──")
    print(result.wilks.variable_importance[["Delta Wilks' Lambda", "F-statistic", "p-value", "Significance"]].to_string())
