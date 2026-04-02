#!/usr/bin/env python
"""
Run Stage 3: Bray–Curtis NMDS + ZCI
====================================

Usage (from project root):
    python src/run_stage3.py

Performs 2-D non-metric MDS on Bray–Curtis dissimilarities computed
from the relative-abundance taxa matrix.  Constructs a ZCI (community-
condition index) per cluster with configurable method and endpoint size.

Produces an NMDS biplot per cluster (sites, endpoints, pollution bins,
waterbody shapes, species arrows), a ZCI distribution figure, and a
ZCI vs Pollution Score scatter with regression.

Reads  : data/processed/complete_env_taxa_chemical_Feb_3.xlsx
         results/01_pollution_assessment/PCA_Stressors/artifacts/SumRel_01_updated_data.xlsx
         results/02_taxa_assemblage/Wards_LDA/classifier_prediction/artifacts/02_predicted_data.xlsx
Writes : results/03_bray_curtis_NMDS/tables/  (nmds_summary, zci_summary)
         results/03_bray_curtis_NMDS/figures/ (nmds_biplot, zci_distribution,
                                                zci_vs_pollution)
         results/03_bray_curtis_NMDS/artifacts/03_updated_data.xlsx
"""

from pathlib import Path

# Resolve project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_PATH       = PROJECT_ROOT / "data" / "processed" / "complete_env_taxa_chemical_Feb_3.xlsx"
STAGE1_ARTIFACT = PROJECT_ROOT / "results" / "01_pollution_assessment" / "PCA_Stressors" / "artifacts" / "SumRel_01_updated_data.xlsx"
STAGE2_ARTIFACT = PROJECT_ROOT / "results" / "02_taxa_assemblage" / "Wards_LDA" / "classifier_prediction" / "artifacts" / "02_predicted_data.xlsx"
OUTPUT_DIR      = PROJECT_ROOT / "results" / "03_bray_curtis_NMDS"

# Import the pipeline
from zci.pipeline.bray_curtis_nmds import nmds_pipeline


if __name__ == "__main__":
    result = nmds_pipeline(
        data_path=DATA_PATH,
        stage1_artifact=STAGE1_ARTIFACT,
        stage2_artifact=STAGE2_ARTIFACT,
        output_dir=OUTPUT_DIR,

        # ── NMDS biplot endpoints ────────────────────────────────────
        nmds_n_ep=5,            # sites per endpoint for NMDS ordination

        # ── ZCI configuration per cluster ────────────────────────────
        #    Available methods: "BC-Direct", "Distance", "Projection",
        #                       "Centroid-Proj"
        zci_config={
            1: {"N_EP": 15, "Method": "BC-Direct"},
            2: {"N_EP":  3, "Method": "BC-Direct"},
        },

        # ── NMDS parameters ─────────────────────────────────────────
        n_components=2,
        n_nmds_iterations=3,       # iterative refinement passes
        max_iter_per_run=1000,
        n_init_first=10,
        n_init_subsequent=4,

        # ── Misc ─────────────────────────────────────────────────────
        random_state=42,
        save_plots=True,
        figure_formats=("png",),
        table_formats=("xlsx",),
    )

    print(f"\n{result.summary()}")
