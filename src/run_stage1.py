#!/usr/bin/env python
"""
Run Stage 1: Pollution Assessment + Score-Focus Analyses
========================================================

Usage (from project root):
    python src/run_stage1.py

Phase 1 — PCA Stressors (PCA + SumRel / MaxRel scoring)
Phase 2 — HZD Toxicity (McPhedran hazard score)
Phase 3 — SumRel Focus (Env vs Stressor comparison + variance partitioning)
Phase 4 — MaxRel Focus (Env vs Stressor comparison + variance partitioning)
Phase 5 — HZD Focus   (Env vs Stressor comparison + variance partitioning)

Reads  : data/processed/complete_env_taxa_chemical_Feb_3.xlsx
         data/TEC_PEC_consensus/10_chemicals_TEC_PEC.xlsx
Writes : results/01_pollution_assessment/
             PCA_Stressors/
                 tables/ — pc_loadings, site_scores, site_rankings
                 figures/ — variance_explained, ridge_loadings, corridor maps
                 artifacts/ — SumRel_01_updated_data.xlsx, MaxRel_01_updated_data.xlsx
             HZD_Toxicity/
                 tables/ — hzd_site_scores, hzd_chemical_effects,
                           hzd_chemical_quotients, hzd_benchmark_summary
                 figures/ — HZD_corridor_bifurcation
             SumRel_Focus/
                 tables/ — env_cutoff_metrics, stressor_cutoff_metrics, varpart_summary
                 figures/ — r2 / pseudoF / pvalue / vif comparison
             MaxRel_Focus/
                 tables/ — env_cutoff_metrics, stressor_cutoff_metrics, varpart_summary
                 figures/ — r2 / pseudoF / pvalue / vif comparison
             HZD_Focus/
                 tables/ — env_cutoff_metrics, stressor_cutoff_metrics, varpart_summary
                 figures/ — r2 / pseudoF / pvalue / vif comparison
"""

from pathlib import Path

import numpy as np

# Resolve project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_PATH      = PROJECT_ROOT / "data" / "processed" / "complete_env_taxa_chemical_Apr17.xlsx"
BENCHMARK_PATH = PROJECT_ROOT / "data" / "TEC_PEC_consensus" / "10_chemicals_TEC_PEC.xlsx"
OUTPUT_DIR     = PROJECT_ROOT / "results" / "01_pollution_assessment"
MAPS_DIR       = PROJECT_ROOT / "data" / "maps"

# Imports
from zci.pipeline.pollution_assessment import pollution_pca_pipeline
from zci.pipeline.hzd_scoring import hzd_scoring_pipeline
from zci.pipeline.threshold_sensitivity import score_focus_pipeline

ENV_VARIABLES = [
    "Measured Depth (m)",
    "Water DO Bottom (mg/L)",
    "Temperature (oC)",
    "MPS (Phi)",
    "LOI (%)",
]


if __name__ == "__main__":
    # ==================================================================
    #  PHASE 1 — PCA Stressors (PCA + SumRel / MaxRel)
    # ==================================================================
    print("=" * 60)
    print("  PHASE 1: PCA Stressors (Pollution PCA)")
    print("=" * 60)

    result = pollution_pca_pipeline(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR / "PCA_Stressors",
        n_components=5,
        varimax = True,
        selected_pcs=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], # PC3 holds loadings on natural elements
        composite_transform="min-max",
        maps_dir=MAPS_DIR,
        threshold_quantile=0.20,
        save_plots=True,
    )

    pca = result.pca_result
    print(f"\nLoadings shape : {pca.loadings.shape}")
    print(f"Scores shape   : {pca.scores.shape}")
    print(f"\nLoadings preview:\n{pca.loadings.round(4)}")
    print(f"\nSumRel range: [{result.sumrel_score.min():.4f}, {result.sumrel_score.max():.4f}]")
    print(f"MaxRel range: [{result.maxrel_score.min():.4f}, {result.maxrel_score.max():.4f}]")

    # PCA site scores serve as stressor predictors for all focus pipelines
    stressor_pcs = pca.scores  # sites × PCs (e.g. PC1–PC5)

    # ==================================================================
    #  PHASE 2 — HZD Toxicity (McPhedran hazard score)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  PHASE 2: HZD Toxicity (McPhedran hazard score)")
    print("=" * 60)

    hzd_result = hzd_scoring_pipeline(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR / "HZD_Toxicity",
        benchmark_path=BENCHMARK_PATH,
        maps_dir=MAPS_DIR,
        threshold_quantile=0.20,
        save_plots=True,
    )

    print(
        f"\nHZD toxicity (%) range: [{hzd_result.hzd_score.min():.4f}, {hzd_result.hzd_score.max():.4f}]"
    )
    print(f"Categories:\n{hzd_result.hzd_category.value_counts().to_string()}")

    # ==================================================================
    #  PHASE 3 — SumRel Focus (Env vs Stressor comparison + varpart)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  PHASE 3: SumRel Focus (Env vs Stressor Comparison)")
    print("=" * 60)

    sumrel_varpart = score_focus_pipeline(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR / "SumRel_Focus",
        score=result.sumrel_score,
        score_label="SumRel",
        env_variables=ENV_VARIABLES,
        stressor_predictors=stressor_pcs,
        single_score_predictor=True,
        thresholds=np.arange(0.05, 1.01, 0.02).round(2),
        rda_threshold=0.22,
        taxa_transform="octave",
        shade_range=(0.20, 0.26),
        standardize_env=False,
        log_transform_env=False,
        save_plots=True,
    )

    # ==================================================================
    #  PHASE 4 — MaxRel Focus (Env vs Stressor comparison + varpart)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  PHASE 4: MaxRel Focus (Env vs Stressor Comparison)")
    print("=" * 60)

    maxrel_varpart = score_focus_pipeline(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR / "MaxRel_Focus",
        score=result.maxrel_score,
        score_label="MaxRel",
        env_variables=ENV_VARIABLES,
        stressor_predictors=stressor_pcs,
        single_score_predictor=True,
        thresholds=np.arange(0.05, 1.01, 0.02).round(2),
        rda_threshold=0.22,
        taxa_transform="octave",
        shade_range=(0.20, 0.26),
        standardize_env=False,
        log_transform_env=False,
        save_plots=True,
    )

    # ==================================================================
    #  PHASE 5 — HZD Focus (Env vs Stressor comparison + varpart)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  PHASE 5: HZD Focus (Env vs Stressor Comparison)")
    print("=" * 60)

    hzd_varpart = score_focus_pipeline(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR / "HZD_Focus",
        score=hzd_result.hzd_score,
        score_label="HZD",
        env_variables=ENV_VARIABLES,
        stressor_predictors=stressor_pcs,
        single_score_predictor=True,
        thresholds=np.arange(0.05, 1.01, 0.02).round(2),
        rda_threshold=0.22,
        taxa_transform="octave",
        shade_range=(0.20, 0.26),
        standardize_env=False,
        log_transform_env=False,
        save_plots=True,
    )
