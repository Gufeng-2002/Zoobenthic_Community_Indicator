#!/usr/bin/env python
"""
Run Stage RDA: Redundancy Analysis on Reference Sites
======================================================

Usage (from project root):
    python src/run_stage_rda.py

This stage is **independent** of the clustering work.  It fits an RDA
with taxa as the response (Y) and environmental variables as the
predictors (X), using the reference sites identified by Stage 1.

Reads  : data/processed/complete_env_taxa_chemical_Feb_3.xlsx
         results/01_pollution_assessment/artifacts/01_updated_data.xlsx
Writes : results/RDA_analysis/tables/  (rda_axes_summary, rda_terms_summary)
         results/RDA_analysis/figures/ (rda_triplot)
"""

from pathlib import Path

# Resolve project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_PATH       = PROJECT_ROOT / "data" / "processed" / "complete_env_taxa_chemical_Feb_3.xlsx"
STAGE1_ARTIFACT = PROJECT_ROOT / "results" / "01_pollution_assessment" / "artifacts" / "01_updated_data.xlsx"
OUTPUT_DIR      = PROJECT_ROOT / "results" / "RDA_analysis"

# Import the pipeline
from zci.pipeline.rda_analysis import rda_pipeline


if __name__ == "__main__":
    result = rda_pipeline(
        data_path=DATA_PATH,
        stage1_artifact=STAGE1_ARTIFACT,
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
        log_transform_env=False,
        taxa_transform="octave",
        n_permutations=999,
        random_state=42,
        cluster_column="Cluster",        # colour triplot by Stage-2 clusters
        save_plots=True,
    )

    print(f"\n{result.summary()}")
    print("\n── Axes Summary ──")
    print(result.axes_table.to_string(index=False))
    print("\n── Terms Summary ──")
    print(result.terms_table.to_string(index=False))
