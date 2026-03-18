#!/usr/bin/env python
"""
Run Threshold Sensitivity: Reference-subset cutoff sensitivity analysis
=======================================================================

Usage (from project root):
    python src/run_stage_threshold.py

This script:
1. Runs Stage 1 (PCA + SumRel / MaxRel scoring).
2. Sweeps reference-threshold proportions from 10 % to 30 % (step 2 %).
3. At each threshold, fits RDA and records performance metrics.
4. Detects stable threshold ranges where RDA is consistently significant.
5. Saves per-score and comparative plots + tables.

Reads  : data/processed/complete_env_taxa_chemical_Feb_3.xlsx
Writes : results/ref_threshold_sensitivity/
             tables/
                 SumRel_threshold_metrics.xlsx
                 MaxRel_threshold_metrics.xlsx
                 threshold_comparison_summary.xlsx
             figures/
                 SumRel_threshold_sensitivity.png
                 MaxRel_threshold_sensitivity.png
                 threshold_comparison.png
"""

from pathlib import Path

# Resolve project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_PATH   = PROJECT_ROOT / "data" / "processed" / "complete_env_taxa_chemical_Feb_3.xlsx"
STAGE1_DIR  = PROJECT_ROOT / "results" / "01_pollution_assessment"
OUTPUT_DIR  = PROJECT_ROOT / "results" / "ref_threshold_sensitivity"
MAPS_DIR    = PROJECT_ROOT / "data" / "maps"

# Imports
from zci.pipeline.pollution_assessment import pollution_pca_pipeline
from zci.pipeline.threshold_sensitivity import threshold_sensitivity_pipeline
import numpy as np

ENV_VARIABLES = [
    "Measured Depth (m)",
    "Water DO Bottom (mg/L)",
    "Temperature (oC)",
    "MPS (Phi)",
    "LOI (%)",
]


if __name__ == "__main__":
    # ── Stage 1: PCA + SumRel / MaxRel scores ────────────────────────
    print("=" * 60)
    print("  STAGE 1: Pollution PCA (SumRel + MaxRel)")
    print("=" * 60)

    stage1 = pollution_pca_pipeline(
        data_path=DATA_PATH,
        output_dir=STAGE1_DIR,
        n_components=5,
        selected_pcs=None,
        composite_transform="min-max",
        maps_dir=MAPS_DIR,
        threshold_quantile=0.25,
        save_plots=True,
    )

    # ── Threshold Sensitivity ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 60)

    results = threshold_sensitivity_pipeline(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        scores={
            "SumRel": stage1.sumrel_score,
            "MaxRel": stage1.maxrel_score,
        },
        thresholds=np.arange(0.05, 1.01, 0.02).round(2),  # default 10 %–100 %, step 2 %
        env_variables=ENV_VARIABLES,
        standardize_env=True,
        log_transform_env=False,
        taxa_transform="octave",
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
