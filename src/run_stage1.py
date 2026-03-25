#!/usr/bin/env python
"""
Run Stage 1: Pollution Assessment + Threshold Sensitivity
=========================================================

Usage (from project root):
    python src/run_stage1.py

Phase 1 — Contamination Stressors (PCA + SumRel / MaxRel scoring)
Phase 2 — Cutoff Reference (threshold sensitivity sweep + RDA at chosen cutoff)

Reads  : data/processed/complete_env_taxa_chemical_Feb_3.xlsx
Writes : results/01_pollution_assessment/
             contamination_stressors/
                 tables/ — pc_loadings, site_scores, site_rankings
                 figures/ — variance_explained, ridge_loadings, corridor maps
                 artifacts/ — SumRel_01_updated_data.xlsx, MaxRel_01_updated_data.xlsx
             cutoff_reference/
                 tables/ — threshold_metrics, comparison_summary, rda summaries
                 figures/ — threshold_sensitivity, threshold_comparison, rda triplots
"""

from pathlib import Path

import numpy as np

# Resolve project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_PATH  = PROJECT_ROOT / "data" / "processed" / "complete_env_taxa_chemical_Feb_3.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "results" / "01_pollution_assessment"
MAPS_DIR   = PROJECT_ROOT / "data" / "maps"

# Imports
from zci.pipeline.pollution_assessment import pollution_pca_pipeline
from zci.pipeline.threshold_sensitivity import threshold_sensitivity_pipeline

ENV_VARIABLES = [
    "Measured Depth (m)",
    "Water DO Bottom (mg/L)",
    "Temperature (oC)",
    "MPS (Phi)",
    "LOI (%)",
]


if __name__ == "__main__":
    # ==================================================================
    #  PHASE 1 — Contamination Stressors (PCA + SumRel / MaxRel)
    # ==================================================================
    print("=" * 60)
    print("  PHASE 1: Contamination Stressors (Pollution PCA)")
    print("=" * 60)

    result = pollution_pca_pipeline(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR / "contamination_stressors",
        pollution_standardize=True,
        n_components=5,
        selected_pcs=None,
        composite_transform="min-max",
        maps_dir=MAPS_DIR,
        threshold_quantile=0.25,
        save_plots=True,
    )

    pca = result.pca_result
    print(f"\nLoadings shape : {pca.loadings.shape}")
    print(f"Scores shape   : {pca.scores.shape}")
    print(f"\nLoadings preview:\n{pca.loadings.round(4)}")
    print(f"\nSumRel range: [{result.sumrel_score.min():.4f}, {result.sumrel_score.max():.4f}]")
    print(f"MaxRel range: [{result.maxrel_score.min():.4f}, {result.maxrel_score.max():.4f}]")

    # ==================================================================
    #  PHASE 2 — Cutoff Reference (Threshold Sensitivity)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  PHASE 2: Cutoff Reference (Threshold Sensitivity)")
    print("=" * 60)

    results = threshold_sensitivity_pipeline(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR / "cutoff_reference",
        scores={
            "SumRel": result.sumrel_score,
            "MaxRel": result.maxrel_score,
        },
        thresholds=np.arange(0.05, 1.01, 0.02).round(2),
        env_variables=ENV_VARIABLES,
        standardize_env=False,
        log_transform_env=False,
        taxa_transform="log_chord",
        n_permutations=999,
        random_state=42,
        shade_range=(0.22, 0.28),
        rda_threshold=0.25,
        save_plots=True,
    )

    # ── Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  THRESHOLD SENSITIVITY SUMMARY")
    print("=" * 60)
    for label, res in results.items():
        print(f"\n  {label}:")
        print(f"    Thresholds tested: "
              f"{res.metrics['threshold'].min():.0%} – "
              f"{res.metrics['threshold'].max():.0%}")
        print(f"    Best adj-R²: {res.metrics['r2_adj'].max():.4f} "
              f"at {res.metrics.loc[res.metrics['r2_adj'].idxmax(), 'threshold']:.0%}")
        best_p_row = res.metrics.loc[res.metrics['global_p'].idxmin()]
        print(f"    Best p-value: {best_p_row['global_p']:.4f} "
              f"at {best_p_row['threshold']:.0%}")
