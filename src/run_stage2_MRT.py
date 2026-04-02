#!/usr/bin/env python
"""
Run Stage 2 (MRT): Taxa Assemblage -- Ward-Targeted Classifier Tree
===================================================================

Usage (from project root):
    python src/run_stage2_MRT.py

Reads  : data/processed/complete_env_taxa_chemical_Feb_3.xlsx
         results/01_pollution_assessment/PCA_Stressors/artifacts/SumRel_01_updated_data.xlsx
Writes : results/02_taxa_assemblage/MRT_Method/cluster_classifier/{figures,tables,artifacts}
         results/02_taxa_assemblage/MRT_Method/classifier_prediction/{tables,artifacts}

Phase 1 -- Cluster Classifier (reference sites only):
  Ward clustering on transformed taxa -> classifier tree fit/prune by CVRE
  -> ANOVA -> cluster panel.

Phase 2 -- Classifier Prediction (non-reference sites):
  Tree predict on non-ref sites -> tables.
"""

from pathlib import Path

from zci.pipeline.mrt import mrt_pipeline

# Resolve project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "complete_env_taxa_chemical_Feb_3.xlsx"
STAGE1_ARTIFACT = (
    PROJECT_ROOT / "results" / "01_pollution_assessment"
    / "PCA_Stressors" / "artifacts" / "SumRel_01_updated_data.xlsx"
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
        reference_quantile=0.23,
        n_clusters=3,
        k_folds=5,
        cv_perms=10,
        minsplit=3,
        minbucket=2,
        random_state=42,
        verbose=True,
    )
    print(f"\n{result.summary()}")
