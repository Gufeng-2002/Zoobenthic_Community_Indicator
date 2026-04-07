#!/usr/bin/env python
"""
Run Stage 2 -- MRT: Confidence-Aware Classification Pipeline
=============================================================

Usage (from project root):
    python src/run_stage2_MRTMethod.py

Requires : site_robustness.xlsx from run_stage2_WardsClustering.py
Reads    : data/processed/complete_env_taxa_chemical_Feb_3.xlsx
           results/01_pollution_assessment/PCA_Stressors/artifacts/SumRel_01_updated_data.xlsx
           results/02_taxa_assemblage/WardsClustering/tables/site_robustness.xlsx
Writes   : results/02_taxa_assemblage/MRT_Method/
               ModelA_CoreOnly/       {tables, figures, artifacts}
               ModelB_CorePeripheral/ {tables, figures, artifacts}
               ModelC_Weighted/       {tables, figures, artifacts}
               ModelD_AllSites/       {tables, figures, artifacts}
               ModelCompar/           {tables, figures}

Trains 4 MRT classifiers on different reference-site subsets, then
produces 3 comparison confusion-matrix tables:
  1) Training-data confusion matrices
  2) All-ref-sites confusion matrices
  3) Non-core (Peripheral+Uncertain) confusion matrices
"""

import pandas as pd
from pathlib import Path

from zci.pipeline.confidence_mrt import confidence_mrt_pipeline

# Resolve project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "complete_env_taxa_chemical_Feb_3.xlsx"
STAGE1_ARTIFACT = (
    PROJECT_ROOT / "results" / "01_pollution_assessment"
    / "PCA_Stressors" / "artifacts" / "SumRel_01_updated_data.xlsx"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "02_taxa_assemblage" / "MRT_Method"
WARDS_DIR  = PROJECT_ROOT / "results" / "02_taxa_assemblage" / "WardsClustering"


if __name__ == "__main__":
    # Read the Ward's clustering artifact
    print("Reading Ward's clustering site robustness artifact ...")
    robustness = pd.read_excel(
        WARDS_DIR / "tables" / "site_robustness.xlsx", index_col=0
    )
    print(f"  {len(robustness)} reference sites, "
          f"{robustness['Original_Cluster'].nunique()} clusters")
    print(f"  Status distribution: "
          f"{robustness['Status'].value_counts().to_dict()}")

    result = confidence_mrt_pipeline(
        data_path=DATA_PATH,
        stage1_artifact=STAGE1_ARTIFACT,
        output_dir=OUTPUT_DIR,
        site_robustness=robustness,
        response_transform="chord",
        k_folds=5,
        cv_perms=10,
        minsplit=3,
        minbucket=2,
        random_state=42,
        save_plots=True,
    )

    print(f"\nModel comparison summary:")
    print(result["summary_table"].to_string())
