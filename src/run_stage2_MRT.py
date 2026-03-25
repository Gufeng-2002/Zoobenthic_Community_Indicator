#!/usr/bin/env python
"""
Run Stage 2 (MRT): Taxa Assemblage -- Multivariate Regression Tree
===================================================================

Usage (from project root):
    python src/run_stage2_MRT.py

Reads  : data/processed/complete_env_taxa_chemical_Feb_3.xlsx
         results/01_pollution_assessment/contamination_stressors/artifacts/SumRel_01_updated_data.xlsx
Writes : results/02_taxa_assemblage/MRT_Method/cluster_classifier/{figures,tables,artifacts}
         results/02_taxa_assemblage/MRT_Method/classifier_prediction/{tables,artifacts}

Phase 1 -- Cluster Classifier (reference sites only):
  Grow full MRT (10-fold CV x 100 perms) -> prune -> ANOVA -> cluster panel.

Phase 2 -- Classifier Prediction (non-reference sites):
  R predict on non-ref sites -> nearest-centroid assignment -> tables.
"""

from pathlib import Path

from zci.pipeline.mrt import mrt_pipeline

# Resolve project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "complete_env_taxa_chemical_Feb_3.xlsx"
STAGE1_ARTIFACT = (
    PROJECT_ROOT / "results" / "01_pollution_assessment"
    / "contamination_stressors" / "artifacts" / "SumRel_01_updated_data.xlsx"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "02_taxa_assemblage" / "MRT_Method"
MAPS_DIR   = PROJECT_ROOT / "data" / "maps"


if __name__ == "__main__":
    result = mrt_pipeline(
        data_path=DATA_PATH,
        stage1_artifact=STAGE1_ARTIFACT,
        output_dir=OUTPUT_DIR,
        maps_dir=MAPS_DIR,
        output_prefix="",
        response_transform="chord",
        reference_quantile=0.25,
        k_folds=10,
        cv_perms=100,
        minsplit=5,
        minbucket=2,
        verbose=True,
    )
    print(f"\n{result.summary()}")
