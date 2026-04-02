#!/usr/bin/env python
"""
Run Stage 4: Piecewise Quantile Regression with Wild-Bootstrap CIs
===================================================================

Usage (from project root):
    python src/run_stage4.py

Fits piecewise (segmented) quantile regressions between ZCI and
Pollution Score per cluster.  For each of 17 quantile levels
(τ = 0.10, 0.15, …, 0.90), the module:

  1. Detects the optimal breakpoint via profile grid-search.
  2. Constructs 90 % wild-bootstrap confidence intervals for all
     parameters (intercept, slopes, breakpoint).
  3. Visualises error-bar plots of CIs across τ levels.
  4. Produces three-panel scatter + fit plots for τ = 0.20, 0.50, 0.80.
  5. Runs a sample-size sensitivity analysis (15 %–100 %, step 5 %,
     30 repeats each) and plots coverage diagnostics.

Reads  : results/01_pollution_assessment/PCA_Stressors/artifacts/01_updated_data.xlsx
         results/02_taxa_assemblage/Wards_LDA/classifier_prediction/artifacts/02_predicted_data.xlsx
         results/03_bray_curtis_NMDS/artifacts/03_updated_data.xlsx
Writes : results/04_piecewise_qr/tables/  (qr_coefficients, sensitivity)
         results/04_piecewise_qr/figures/ (ci_errorbars, three_quantiles,
                                            sensitivity, coverage)
         results/04_piecewise_qr/artifacts/04_updated_data.xlsx
"""

from pathlib import Path
import numpy as np

# Resolve project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
STAGE1_ARTIFACT = PROJECT_ROOT / "results" / "01_pollution_assessment" / "PCA_Stressors" / "artifacts" / "SumRel_01_updated_data.xlsx"
STAGE2_ARTIFACT = PROJECT_ROOT / "results" / "02_taxa_assemblage" / "Wards_LDA" / "classifier_prediction" / "artifacts" / "02_predicted_data.xlsx"
STAGE3_ARTIFACT = PROJECT_ROOT / "results" / "03_bray_curtis_NMDS" / "artifacts" / "03_updated_data.xlsx"
OUTPUT_DIR      = PROJECT_ROOT / "results" / "04_piecewise_qr"

# Import the pipeline
from zci.pipeline.piecewise_qr_pipeline import pqr_pipeline


if __name__ == "__main__":
    result = pqr_pipeline(
        stage1_artifact=STAGE1_ARTIFACT,
        stage2_artifact=STAGE2_ARTIFACT,
        stage3_artifact=STAGE3_ARTIFACT,
        output_dir=OUTPUT_DIR,

        # ── Piecewise QR settings ────────────────────────────────────
        n_breakpoints=1,
        taus= list(np.round(np.arange(0.1, 0.91, 0.5), 2)),                   # default 0.10 … 0.90 step 0.05
        highlight_taus=(0.20, 0.50, 0.80),
        n_boot=10,                     # wild-bootstrap replicates
        confidence=0.90,                # 90 % CI
        grid_size=30,                   # breakpoint grid density
        search_range=(0.4, 0.6),      # quantile range for BP search

        # ── Sensitivity settings ─────────────────────────────────────
        run_sensitivity=True,
        sensitivity_tau=0.50,           # median regression for sensitivity
        sensitivity_fracs=list(np.round(np.arange(0.15, 1.01, 0.3), 2)),         # default 15 % … 100 % step 5 %
        sensitivity_repeats=5,
        sensitivity_boot=10,

        # ── Misc ─────────────────────────────────────────────────────
        min_cluster_size=15,
        random_state=42,
        save_plots=True,
        figure_formats=("png",),
        table_formats=("xlsx",),
    )

    print(f"\n{result.summary()}")
