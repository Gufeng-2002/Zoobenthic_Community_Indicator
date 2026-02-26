#!/usr/bin/env python
"""
Run Stage 2: Taxa Assemblage Clustering + ANOVA + Cluster Panel
================================================================

Usage (from project root):
    python src/run_stage2.py

Reads  : data/processed/complete_env_taxa_chemical_Feb_3.xlsx
         results/01_pollution_assessment/artifacts/01_updated_data.xlsx
Writes : results/02_taxa_assemblage/tables/  (reference_taxa_clusters)
         results/02_taxa_assemblage/figures/ (ward_dendrogram)
         results/02_taxa_assemblage/artifacts/02_updated_data.xlsx

Note: ANOVA tests and the cluster panel figure are produced by
      run_hindsight_relabel.py after cluster labels are finalised.
"""

from pathlib import Path

# Resolve project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_PATH       = PROJECT_ROOT / "data" / "processed" / "complete_env_taxa_chemical_Feb_3.xlsx"
STAGE1_ARTIFACT = PROJECT_ROOT / "results" / "01_pollution_assessment" / "artifacts" / "01_updated_data.xlsx"
OUTPUT_DIR      = PROJECT_ROOT / "results" / "02_taxa_assemblage"

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
        save_plots=True,
    )

    print(f"\n{results.summary()}")
    print(f"\nCluster distribution (reference sites only):")
    print(results.cluster_distribution())
    print(f"\nTotal reference sites: {results.ref_mask.sum()}")
