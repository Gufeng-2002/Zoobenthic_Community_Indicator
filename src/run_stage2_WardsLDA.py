#!/usr/bin/env python
"""
Run Stage 2: Taxa Assemblage -- Ward's LDA Combined Pipeline
=============================================================

Usage (from project root):
    python src/run_stage2.py

Reads  : data/processed/complete_env_taxa_chemical_Feb_3.xlsx
         results/01_pollution_assessment/artifacts/SumRel_01_updated_data.xlsx
Writes : results/02_taxa_assemblage/Wards_LDA/cluster_classifier/{figures,tables,artifacts}
         results/02_taxa_assemblage/Wards_LDA/classifier_prediction/{figures,tables,artifacts}

Phase 1 -- Cluster Classifier (reference sites only):
  Ward clustering -> relabel -> ANOVA -> cluster panel -> LDA fit -> tables.

Phase 2 -- Classifier Prediction (non-reference sites):
  LDA predict -> probabilities -> cluster comparison figure.
"""

from pathlib import Path

from zci.pipeline.taxa_assemblage import taxa_assemblage_pipeline

# Resolve project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "complete_env_taxa_chemical_Feb_3.xlsx"
STAGE1_ARTIFACT = (
    PROJECT_ROOT / "results" / "01_pollution_assessment"
    / "PCA_Stressors" / "artifacts" / "SumRel_01_updated_data.xlsx"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "02_taxa_assemblage" / "Wards_LDA"
MAPS_DIR   = PROJECT_ROOT / "data" / "maps"


if __name__ == "__main__":
    results = taxa_assemblage_pipeline(
        data_path=DATA_PATH,
        stage1_artifact=STAGE1_ARTIFACT,
        output_dir=OUTPUT_DIR,
        maps_dir=MAPS_DIR,
        reference_quantile=40,
        taxa_transform="chord",
        n_clusters=2,
        label_map={1: 2,
                   2: 1},
        env_variables=[
            "Measured Depth (m)",
            "Water DO Bottom (mg/L)",
            "Temperature (oC)",
            "MPS (Phi)",
            "LOI (%)",
        ],
        standardize_env=True,
        n_mccv_iterations=1000,
        mccv_test_size=0.2,
        random_state=42,
        save_plots=True,
    )

    clustering = results["clustering"]
    lda = results["lda"]
    print(f"\n{clustering.summary()}")
    print(f"{lda.summary()}")
    print(f"\nCluster distribution (reference sites only):")
    print(results["labels_ref"].value_counts().sort_index())
    print(f"\nTotal reference sites: {results['ref_mask'].sum()}")

    print("\n-- LDA Axes Summary --")
    print(lda.wilks.axes_summary.to_string())
    print("\n-- Variable Importance --")
    print(lda.wilks.variable_importance[
        ["Delta Wilks' Lambda", "F-statistic", "p-value", "Significance"]
    ].to_string())
