#!/usr/bin/env python
"""
Run Stage 2 -- Full Pipeline
==============================
  Ward's Clustering → Cross-Support Evaluation (Model S)
  → Finalized Model → Non-Reference Site Prediction & PCA Ordination

Usage (from project root):
    python src/run_stage2.py

Runs three sub-pipelines in sequence:
  1. Ward's Clustering with robustness assessment (pvclust + co-assignment)
  2. Cross-Support Evaluation — Model S (LDA + MRT trained on
     EnvStrong_TaxaStrong sites, evaluated on all four 2×2 classes)
  3. Finalized LDA (Model S) → predict all sites (ref + non-ref)
     + full-site environmental PCA ordination with decision-region visualization

Global parameters
-----------------
TAXA_TRANSFORM : str
    Transformation applied to raw taxa counts before clustering *and*
    MRT classification.  One of "octave", "chord", "hellinger",
    "log_chord", "relative_abundance".
N_REFERENCE_SITES : int
    Number of least-polluted sites to enter Ward's clustering.  The
    resulting cluster labels and robustness status table then feed
    into the downstream LDA and MRT pipelines.
"""

import pandas as pd
from pathlib import Path

from zci.pipeline.wards_clustering import (
    refresh_combined_robustness_outputs,
    wards_clustering_pipeline,
)
from zci.pipeline.finalized_lda import finalized_lda_pipeline
from zci.pipeline.cross_support_eval import cross_support_eval_pipeline

# ═══════════════════════════════════════════════════════════════════════
#  GLOBAL PARAMETERS — change these as needed
# ═══════════════════════════════════════════════════════════════════════
TAXA_TRANSFORM: str = "octave"
"""Taxa transformation for Ward's clustering and MRT classification.
One of: "octave", "chord", "hellinger", "log_chord", "relative_abundance".
"""

N_REFERENCE_SITES: int = 42
"""Number of least-polluted reference sites entering Ward's clustering."""

ENV_STRENGTH_METHOD: str = "percentile"
"""Environmental strength classification method.
``"threshold"`` — original absolute-threshold rule (sil > esil & margin > emarg).
``"percentile"`` — per-cluster top-pct rule (top 70 % of combined score → Strong).
"""

ENV_STRENGTH_TOP_PCT: float = 0.60
"""Fraction of sites per cluster classified as Strong (used only when
ENV_STRENGTH_METHOD = "percentile")."""

# ═══════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "complete_env_taxa_chemical_Apr17.xlsx"
STAGE1_ARTIFACT = (
    PROJECT_ROOT / "results" / "01_pollution_assessment"
    / "PCA_Stressors" / "artifacts" / "SumRel_01_updated_data.xlsx"
)
MAPS_DIR = PROJECT_ROOT / "data" / "maps"

WARDS_OUTPUT = PROJECT_ROOT / "results" / "02_taxa_assemblage" / "WardsClustering"
LDA_OUTPUT   = PROJECT_ROOT / "results" / "02_taxa_assemblage" / "LDA_Method"
MRT_OUTPUT   = PROJECT_ROOT / "results" / "02_taxa_assemblage" / "MRT_Method"

# Shared environmental variables
ENV_VARIABLES = [
    "Measured Depth (m)",
    "Water DO Bottom (mg/L)",
    "Temperature (oC)",
    "MPS (Phi)",
    "LOI (%)",
]


if __name__ == "__main__":
    print("=" * 70)
    print("  Stage 2 Full Pipeline")
    print("=" * 70)
    print(f"  Taxa transform      : {TAXA_TRANSFORM}")
    print(f"  Reference sites     : {N_REFERENCE_SITES}")
    print(f"  Env strength method : {ENV_STRENGTH_METHOD}")
    if ENV_STRENGTH_METHOD == "percentile":
        print(f"  Env strength top %  : {ENV_STRENGTH_TOP_PCT:.0%}")
    print("=" * 70)

    # ── 1. Ward's Clustering + Robustness Testing ────────────────────
    print("\n" + "=" * 70)
    print("  [1/3] Ward's Clustering + Robustness (pvclust, co-assignment)")
    print("=" * 70)
    ward_result = wards_clustering_pipeline(
        data_path=DATA_PATH,
        stage1_artifact=STAGE1_ARTIFACT,
        output_dir=WARDS_OUTPUT,
        maps_dir=MAPS_DIR,
        reference_quantile=N_REFERENCE_SITES,
        taxa_transform=TAXA_TRANSFORM,
        n_clusters = 3,
        label_map={1: 1, 2: 2, 3: 3},
        env_variables=ENV_VARIABLES,
        # Robustness parameters
        n_boot_coassign=1000,
        coassign_sample_frac=0.8,
        n_boot_pvclust=1000,
        run_pvclust_bootstrap=True,
        sil_threshold=None,
        margin_threshold=None,
        env_coassign_sample_frac=0.8,
        env_sil_threshold=None,
        env_margin_threshold=None,
        env_strength_method=ENV_STRENGTH_METHOD,
        env_strength_top_pct=ENV_STRENGTH_TOP_PCT,
        random_state=42,
        save_plots=True,
    )

    print(f"\n{ward_result.summary()}")
    print(f"\nCluster distribution (reference sites):")
    print(ward_result.cluster_distribution())
    print(f"\nStatus distribution:")
    print(ward_result.status_distribution())
    print(f"\nMean silhouette: {ward_result.mean_silhouette():.4f}")

    # Refresh the saved Ward artifact with the optimized thresholds used
    # by the later 2x2 env/taxa classification.
    print("\n  Running threshold grid search ...")
    combined_table, class_count, best_th, _ = refresh_combined_robustness_outputs(
        data_path=DATA_PATH,
        output_dir=WARDS_OUTPUT,
        env_variables=ENV_VARIABLES,
        env_strength_method=ENV_STRENGTH_METHOD,
        env_strength_top_pct=ENV_STRENGTH_TOP_PCT,
        verbose=True,
    )
    print(f"  Best thresholds: tsil={best_th['tsil']}, tmarg={best_th['tmarg']}, "
          f"esil={best_th['esil']}, emarg={best_th['emarg']}")
    print(f"  n_train={int(best_th['n_train'])}, "
          f"LDA_C1={best_th['LDA_diag_C1_acc']:.1%}, "
          f"LDA_C3={best_th['LDA_diag_C3_acc']:.1%}")

    from zci.io.readers import read_study_data, extract_block
    data_full = read_study_data(DATA_PATH)
    env_block = extract_block(data_full, "environmental", "raw")
    env_ref_raw = env_block.loc[combined_table.index, ENV_VARIABLES].dropna()
    labels_for_xs = combined_table["Original_Cluster"]

    # ── 2. Cross-Support Evaluation (Model S: LDA + MRT) ────────────
    print("\n" + "=" * 70)
    print("  [2/3] Cross-Support Evaluation (EnvStrong_TaxaStrong training)")
    print("=" * 70)
    xs_result = cross_support_eval_pipeline(
        combined_table=combined_table,
        env_ref_complete=env_ref_raw,
        labels_ref_complete=labels_for_xs,
        lda_output_dir=LDA_OUTPUT / "ModelS_XSupport",
        mrt_output_dir=MRT_OUTPUT / "ModelS_XSupport",
        env_variables=ENV_VARIABLES,
        random_state=42,
        verbose=True,
    )

    # ── 3. Finalized LDA (All Env-Strong) → Non-Ref Prediction & PCA ───
    print("\n" + "=" * 70)
    print("  [3/3] Finalized LDA (EnvStrong_TaxaStrong Sites)")
    print("         → Non-Ref Prediction & Env PCA Ordination")
    print("=" * 70)

    n_double_strong = int((combined_table["TaxaEnv_Class"] == "EnvStrong_TaxaStrong").sum())
    print(f"  EnvStrong_TaxaStrong training set: {n_double_strong} sites")

    finalized = finalized_lda_pipeline(
        data_path=DATA_PATH,
        output_dir=LDA_OUTPUT,
        model_name="Finalized_EnvStrong",
        site_robustness=combined_table,  # has Original_Cluster + Env_Strength
        env_variables=ENV_VARIABLES,
        taxa_transform=TAXA_TRANSFORM,
        grid_resolution=200,
        save_plots=True,
    )

    # ── Done ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Stage 2 Complete")
    print("=" * 70)
    print(f"  Taxa transform      : {TAXA_TRANSFORM}")
    print(f"  Reference sites     : {N_REFERENCE_SITES}")
    print(f"  Finalized model     : Finalized_EnvStrong (EnvStrong_TaxaStrong sites)")
    print(f"  Ward's output       : {WARDS_OUTPUT}")
    print(f"  LDA output          : {LDA_OUTPUT}")
    print(f"  MRT output          : {MRT_OUTPUT}")
    print("=" * 70)
