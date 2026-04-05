#!/usr/bin/env python
"""
Run Detroit River Stage 1 + Stage 2 Pipelines
=============================================

Usage (from project root):
        python src/run_DR_full_pipeline.py

This runner is limited to the first three script-equivalent steps for the
Detroit River subset only:

    1. Pre-filter DR sites from the full workbook.
    2. Run Stage 1 outputs into ``results/DR_results/01_pollution_assessment``.
    3. Run Stage 2 MRT outputs into
         ``results/DR_results/02_taxa_assemblage/MRT_Method``.
    4. Run Stage 2 WardsLDA outputs into
         ``results/DR_results/02_taxa_assemblage/Wards_LDA``.

No downstream RDA, hindsight relabel, LDA-only, NMDS, or PQR stages are run
from this entrypoint.
"""

import time
from pathlib import Path

import numpy as np

# ── project root ─────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── shared paths ─────────────────────────────────────────────────────

ORIGINAL_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "complete_env_taxa_chemical_Feb_3.xlsx"
)
DR_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "DR_env_taxa_chemical.xlsx"
)
MAPS_DIR = PROJECT_ROOT / "data" / "maps"
BENCHMARK_PATH = (
    PROJECT_ROOT / "data" / "TEC_PEC_consensus" / "10_chemicals_TEC_PEC.xlsx"
)

# ── DR result folder (mirrors baseline results/ structure) ─────────────

DR_RESULTS = PROJECT_ROOT / "results" / "DR_results"

STAGE1_DIR = DR_RESULTS / "01_pollution_assessment"
STAGE2_DIR = DR_RESULTS / "02_taxa_assemblage"

# Artifact paths
SUMREL_STAGE1_ARTIFACT = (
    STAGE1_DIR / "PCA_Stressors" / "artifacts" / "SumRel_01_updated_data.xlsx"
)
HZD_STAGE1_ARTIFACT = (
    STAGE1_DIR / "HZD_Toxicity" / "artifacts" / "HZD_01_updated_data.xlsx"
)

# ── environmental variables ──────────────────────────────────────────

# Base 5 env variables (same as SCDRS, used in Stage 1 / PCA)
ENV_VARIABLES_BASE = [
    "Measured Depth (m)",
    "Water DO Bottom (mg/L)",
    "Temperature (oC)",
    "MPS (Phi)",
    "LOI (%)",
]

# DR has velocity available for Stage 2 habitat-based classification
ENV_VARIABLES_DR = ENV_VARIABLES_BASE + [
    "Velocity  at bottom (m/sec)",
]
ENV_VARIABLES_DR_SHORT = ["Depth", "DO", "Temp", "MPS", "LOI", "Velocity"]


# ── imports ──────────────────────────────────────────────────────────

from zci.io.readers import read_study_data, extract_block
from zci.pipeline.pollution_assessment import pollution_pca_pipeline
from zci.pipeline.hzd_scoring import hzd_scoring_pipeline
from zci.pipeline.threshold_sensitivity import score_focus_pipeline
from zci.pipeline.mrt import mrt_pipeline
from zci.pipeline.taxa_assemblage import taxa_assemblage_pipeline
from zci.viz.map_plots import plot_dr_bifurcation, plot_dr_map


# ── helpers ──────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print("\n")
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _filter_dr_sites(src_path: Path, dst_path: Path) -> int:
    """Read the full SCDRS workbook, keep only DR sites, save to *dst_path*.

    Returns the number of DR sites found.
    """
    _banner("PRE-FILTER — Extract Detroit River (DR) Sites")

    print(f"[1/3] Reading full SCDRS dataset:\n      {src_path}")
    df = read_study_data(src_path)
    print(f"      {len(df)} total sites")

    # Locate the Waterbody column in sample_info block
    sample_info = extract_block(df, "sample_info", "raw")
    if "Waterbody" not in sample_info.columns:
        raise ValueError(
            "Cannot find 'Waterbody' column in sample_info block. "
            f"Available columns: {list(sample_info.columns)}"
        )

    wb = sample_info["Waterbody"]
    dr_mask = wb == "DR"
    n_dr = dr_mask.sum()

    print(f"[2/3] Filtering to Waterbody == 'DR' …")
    print(f"      {n_dr} DR sites out of {len(df)} total")
    print(f"      Other waterbodies: {dict(wb.value_counts())}")

    df_dr = df.loc[dr_mask]

    print(f"[3/3] Saving DR subset:\n      {dst_path}")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df_dr.to_excel(dst_path)

    print(f"\n✓ DR pre-filter complete.  {n_dr} sites saved.")
    return n_dr


def _run_stage1() -> None:
    _banner("STAGE 1 — Pollution Assessment  (DR sites)")

    result = pollution_pca_pipeline(
        data_path=DR_DATA_PATH,
        output_dir=STAGE1_DIR / "PCA_Stressors",
        pollution_standardize=True,
        n_components=5,
        selected_pcs=None,
        composite_transform="min-max",
        maps_dir=MAPS_DIR,
        threshold_quantile=0.20,
        bifurcation_plot_func=plot_dr_bifurcation,
        save_plots=True,
    )

    stressor_pcs = result.pca_result.scores

    hzd_result = hzd_scoring_pipeline(
        data_path=DR_DATA_PATH,
        output_dir=STAGE1_DIR / "HZD_Toxicity",
        benchmark_path=BENCHMARK_PATH,
        maps_dir=MAPS_DIR,
        threshold_quantile=0.10,
        bifurcation_plot_func=plot_dr_bifurcation,
        save_plots=True,
    )

    focus_thresholds = np.arange(0.05, 1.01, 0.02).round(2)
    for score, label in (
        (result.sumrel_score, "SumRel"),
        (result.maxrel_score, "MaxRel"),
        (hzd_result.hzd_score, "HZD"),
    ):
        score_focus_pipeline(
            data_path=DR_DATA_PATH,
            output_dir=STAGE1_DIR / f"{label}_Focus",
            score=score,
            score_label=label,
            env_variables=ENV_VARIABLES_BASE,
            stressor_predictors=stressor_pcs,
            thresholds=focus_thresholds,
            rda_threshold=0.20,
            taxa_transform="octave",
            shade_range=(0.20, 0.26),
            standardize_env=False,
            log_transform_env=False,
            save_plots=True,
        )


def _run_stage2_mrt() -> None:
    _banner("STAGE 2 — MRT  (DR sites)")
    result = mrt_pipeline(
        data_path=DR_DATA_PATH,
        stage1_artifact=SUMREL_STAGE1_ARTIFACT,
        output_dir=STAGE2_DIR / "MRT_Method",
        maps_dir=MAPS_DIR,
        output_prefix="",
        env_variables=ENV_VARIABLES_DR,
        env_short=ENV_VARIABLES_DR_SHORT,
        response_transform="octave",
        reference_quantile=0.20,
        n_clusters=3,
        k_folds=5,
        cv_perms=5,
        minsplit=3,
        minbucket=2,
        random_state=42,
        map_func=plot_dr_map,
        verbose=True,
    )
    print(f"\n{result.summary()}")


def _run_stage2_wards_lda() -> None:
    _banner("STAGE 2 — WardsLDA  (DR sites)")
    results = taxa_assemblage_pipeline(
        data_path=DR_DATA_PATH,
        stage1_artifact=SUMREL_STAGE1_ARTIFACT,
        output_dir=STAGE2_DIR / "Wards_LDA",
        maps_dir=MAPS_DIR,
        reference_quantile=0.20,
        taxa_transform="octave",
        n_clusters=3,
        label_map={1: 2, 2: 1},
        env_variables=ENV_VARIABLES_DR,
        standardize_env=True,
        n_mccv_iterations=1000,
        mccv_test_size=0.2,
        random_state=42,
        map_func=plot_dr_map,
        save_plots=True,
    )

    clustering = results["clustering"]
    lda = results["lda"]
    print(f"\n{clustering.summary()}")
    print(f"{lda.summary()}")
    print("\nCluster distribution (reference sites only):")
    print(results["labels_ref"].value_counts().sort_index())
    print(f"\nTotal reference sites: {results['ref_mask'].sum()}")

    print("\n-- LDA Axes Summary --")
    print(lda.wilks.axes_summary.to_string())
    print("\n-- Variable Importance --")
    print(
        lda.wilks.variable_importance[
            ["Delta Wilks' Lambda", "F-statistic", "p-value", "Significance"]
        ].to_string()
    )


# ── main pipeline ────────────────────────────────────────────────────

def run_dr_full_pipeline() -> None:
    t0 = time.time()

    # ==================================================================
    #  0. PRE-FILTER — Extract Detroit River sites
    # ==================================================================
    n_dr = _filter_dr_sites(ORIGINAL_DATA_PATH, DR_DATA_PATH)

    # ==================================================================
    #  1. STAGE 1 — Pollution Assessment
    # ==================================================================
    _run_stage1()

    # ==================================================================
    #  2. STAGE 2 — MRT
    # ==================================================================
    _run_stage2_mrt()

    # ==================================================================
    #  3. STAGE 2 — WardsLDA
    # ==================================================================
    _run_stage2_wards_lda()

    # ==================================================================
    elapsed = time.time() - t0
    _banner(f"DR STAGE 1 + STAGE 2 PIPELINES COMPLETE — {elapsed:.1f} s")


if __name__ == "__main__":
    run_dr_full_pipeline()
