#!/usr/bin/env python
"""
Run Stage 2: Taxa Assemblage Clustering + ANOVA + Cluster Panel
================================================================

Usage (from project root):
    python src2/run_stage2.py

Reads  : data/processed/complete_env_taxa_chemical_Feb_3.xlsx
         results2/01_pollution_assessment/artifacts/01_updated_data.xlsx
Writes : results2/02_taxa_assemblage/tables/  (reference_taxa_clusters,
                                                anova_env, anova_taxa)
         results2/02_taxa_assemblage/figures/ (ward_dendrogram, cluster_panel)
         results2/02_taxa_assemblage/artifacts/02_updated_data.xlsx
"""

from pathlib import Path

# Resolve project root (parent of src2/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_PATH       = PROJECT_ROOT / "data" / "processed" / "complete_env_taxa_chemical_Feb_3.xlsx"
STAGE1_ARTIFACT = PROJECT_ROOT / "results2" / "01_pollution_assessment" / "artifacts" / "01_updated_data.xlsx"
OUTPUT_DIR      = PROJECT_ROOT / "results2" / "02_taxa_assemblage"
MAPS_DIR        = PROJECT_ROOT / "data" / "maps"

# Import the pipeline
from zci.pipeline.taxa_assemblage import taxa_assemblage_pipeline


if __name__ == "__main__":
    results = taxa_assemblage_pipeline(
        data_path=DATA_PATH,
        stage1_artifact=STAGE1_ARTIFACT,
        output_dir=OUTPUT_DIR,
        reference_quantile=0.20,         # bottom 20 % → reference sites
        taxa_transform="octave",         # 'octave' (identity) or 'relative_abundance'
        n_clusters=3,                    # 3 groups
        env_variables=[                  # habitat variables for ANOVA / bar plot
            "Measured Depth (m)",
            "Water DO Bottom (mg/L)",
            "Temperature (oC)",
            "MPS (Phi)",
            "LOI (%)",
        ],
        anova_transform="none",          # 'none', 'log', or 'boxcox'
        maps_dir=MAPS_DIR,              # shapefile folder → enables cluster panel
        save_plots=True,
    )

    print(f"\n{results.summary()}")
    print(f"\nCluster distribution (reference sites only):")
    print(results.cluster_distribution())
    print(f"\nTotal reference sites: {results.ref_mask.sum()}")
