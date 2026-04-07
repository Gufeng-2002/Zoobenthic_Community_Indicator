#!/usr/bin/env python
"""
Run Stage 2 -- LDA: Confidence-Aware Classification Pipeline
=============================================================

Usage (from project root):
    python src/run_stage2_LDAMethod.py

Requires : site_robustness.xlsx from run_stage2_WardsClustering.py
Reads    : data/processed/complete_env_taxa_chemical_Feb_3.xlsx
           results/01_pollution_assessment/PCA_Stressors/artifacts/SumRel_01_updated_data.xlsx
           results/02_taxa_assemblage/WardsClustering/tables/site_robustness.xlsx
Writes   : results/02_taxa_assemblage/LDA_Method/
               ModelA_CoreOnly/       {tables, figures, artifacts}
               ModelB_CorePeripheral/ {tables, figures, artifacts}
               ModelC_Weighted/       {tables, figures, artifacts}
               ModelCompar/           {tables, figures}

Step 1: Build soft training weights  w_i = max(0,s_i)*A_i*max(0,M_i)
Step 2: Define Core / Core+Peripheral / Uncertain subsets
Step 3: Train Model A (Core), Model B (Core+Periph), Model C (Weighted)
Step 4: Evaluate with CV accuracy, confusion matrix, posteriors (p_max, Δ)
Step 5: Predict held-out Uncertain sites
Step 6: Cross-model comparison
"""

import pandas as pd
from pathlib import Path

from zci.pipeline.confidence_lda import confidence_lda_pipeline
from zci.pipeline.finalized_lda import finalized_lda_pipeline

# Resolve project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "complete_env_taxa_chemical_Feb_3.xlsx"
STAGE1_ARTIFACT = (
    PROJECT_ROOT / "results" / "01_pollution_assessment"
    / "PCA_Stressors" / "artifacts" / "SumRel_01_updated_data.xlsx"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "02_taxa_assemblage" / "LDA_Method"
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

    result = confidence_lda_pipeline(
        data_path=DATA_PATH,
        stage1_artifact=STAGE1_ARTIFACT,
        output_dir=OUTPUT_DIR,
        site_robustness=robustness,
        env_variables=[
            "Measured Depth (m)",
            "Water DO Bottom (mg/L)",
            "Temperature (oC)",
            "MPS (Phi)",
            "LOI (%)",
        ],
        standardize_env=True,
        cv_folds=5,
        cv_repeats=10,
        random_state=42,
        save_plots=True,
    )

    print(f"\n{result.summary()}")
    print(f"\nModel comparison summary:")
    print(result.summary_table.to_string())

    # ────────────────────────────────────────────────────────────────
    #  Finalized Model — full-site PCA ordination with predictions
    # ────────────────────────────────────────────────────────────────
    # Pick Model B (Core+Peripheral) as the finalized classifier:
    # best trade-off between training coverage and holdout validation.
    chosen_key = "ModelB_CorePeripheral"
    chosen_model = result.models[chosen_key]
    print(f"\n>>> Finalized model: {chosen_model.model_name}")

    ENV_VARS = [
        "Measured Depth (m)",
        "Water DO Bottom (mg/L)",
        "Temperature (oC)",
        "MPS (Phi)",
        "LOI (%)",
    ]

    finalized_lda_pipeline(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        chosen_model=chosen_model,
        site_robustness=robustness,
        env_variables=ENV_VARS,
        grid_resolution=200,
        save_plots=True,
    )
