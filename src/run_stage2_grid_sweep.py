#!/usr/bin/env python
"""
Stage 2 Grid Sweep: Taxa Transform × Reference Site Count
==========================================================

Sweeps over:
    Transforms  : octave, chord, hellinger, relative_abundance
    N_Ref Sites : 40 .. 60  (step 1)

For each (transform, n_ref) combination, runs:
    Ward's Clustering → Confidence-Aware LDA → Confidence-Aware MRT

and collects model-level summary metrics.  All per-run figures /
tables are written to temp directories (cleaned up automatically);
only the aggregated summary is kept.

Output:
    results/02_taxa_assemblage/GridSweep/
        tables/   ward_sweep.xlsx, lda_sweep.xlsx, mrt_sweep.xlsx
        figures/  line-plots and heatmaps

Usage (from project root):
    python src/run_stage2_grid_sweep.py
"""

from __future__ import annotations

import sys
import time
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from zci.pipeline.wards_clustering import wards_clustering_pipeline
from zci.pipeline.confidence_lda import confidence_lda_pipeline
from zci.pipeline.confidence_mrt import confidence_mrt_pipeline

# ═══════════════════════════════════════════════════════════════════════
#  SWEEP GRID  — edit these to change the search space
# ═══════════════════════════════════════════════════════════════════════
TRANSFORMS  = ["octave", "chord", "hellinger", "relative_abundance"]
N_REF_RANGE = range(40, 61)          # 40 .. 60 inclusive

# Reduced bootstrap counts for speed (production uses 1000)
N_BOOT_COASSIGN = 200
N_BOOT_PVCLUST  = 200
RUN_PVCLUST     = True               # False → skip pvclust (faster, affects Model C weights)

# ═══════════════════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "complete_env_taxa_chemical_Feb_3.xlsx"
STAGE1_ARTIFACT = (
    PROJECT_ROOT / "results" / "01_pollution_assessment"
    / "PCA_Stressors" / "artifacts" / "SumRel_01_updated_data.xlsx"
)
MAPS_DIR   = PROJECT_ROOT / "data" / "maps"
OUTPUT_DIR = PROJECT_ROOT / "results" / "02_taxa_assemblage" / "GridSweep"

ENV_VARIABLES = [
    "Measured Depth (m)",
    "Water DO Bottom (mg/L)",
    "Temperature (oC)",
    "MPS (Phi)",
    "LOI (%)",
]

# ═══════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ═══════════════════════════════════════════════════════════════════════
COLORS = {
    "octave": "#1f77b4",
    "chord": "#ff7f0e",
    "hellinger": "#2ca02c",
    "relative_abundance": "#d62728",
}


def _plot_model_lines(df, metric, ylabel, title, savepath, models=None):
    """2×2 line-plot grid — one panel per model."""
    if models is None:
        models = sorted(df["Model"].unique())[:4]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    for ax, model in zip(axes.flat, models):
        for t in TRANSFORMS:
            sub = df[(df["Model"] == model) & (df["Transform"] == t)].sort_values("N_Ref")
            if sub.empty:
                continue
            ax.plot(sub["N_Ref"], sub[metric], label=t,
                    color=COLORS[t], marker="o", markersize=3, linewidth=1.2)
        ax.set_title(model, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("N Reference Sites")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ward_lines(df, savepath):
    """2×2 line-plot for Ward's metrics."""
    metrics = [
        ("Mean_Silhouette", "Mean Silhouette"),
        ("N_Core", "Core Sites"),
        ("N_Peripheral", "Peripheral Sites"),
        ("N_Uncertain", "Uncertain Sites"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    for ax, (col, label) in zip(axes.flat, metrics):
        for t in TRANSFORMS:
            sub = df[df["Transform"] == t].sort_values("N_Ref")
            ax.plot(sub["N_Ref"], sub[col], label=t,
                    color=COLORS[t], marker="o", markersize=3, linewidth=1.2)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("N Reference Sites")
    fig.suptitle("Ward's Clustering Sensitivity", fontsize=13)
    plt.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_model_heatmaps(df, metric, title, savepath, cmap="YlOrRd"):
    """2×2 heatmap grid — one panel per model."""
    models = sorted(df["Model"].unique())[:4]
    fig, axes = plt.subplots(2, 2, figsize=(18, 8))
    for ax, model in zip(axes.flat, models):
        sub = df[df["Model"] == model]
        pivot = sub.pivot(index="Transform", columns="N_Ref", values=metric)
        order = [t for t in TRANSFORMS if t in pivot.index]
        pivot = pivot.loc[order]
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=6, rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(model, fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_results(ward_rows, lda_rows, mrt_rows, errors):
    """Write current sweep results to disk (safe for incremental saves)."""
    tables_dir = OUTPUT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if ward_rows:
        pd.DataFrame(ward_rows).to_excel(tables_dir / "ward_sweep.xlsx", index=False)
    if lda_rows:
        pd.DataFrame(lda_rows).to_excel(tables_dir / "lda_sweep.xlsx", index=False)
    if mrt_rows:
        pd.DataFrame(mrt_rows).to_excel(tables_dir / "mrt_sweep.xlsx", index=False)
    if errors:
        pd.DataFrame(errors).to_excel(tables_dir / "sweep_errors.xlsx", index=False)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    total = len(TRANSFORMS) * len(N_REF_RANGE)
    print("=" * 70)
    print(f"  Stage 2 Grid Sweep")
    print(f"  {len(TRANSFORMS)} transforms × {len(N_REF_RANGE)} n_ref = {total} combos")
    print(f"  Bootstrap: coassign={N_BOOT_COASSIGN}  pvclust={N_BOOT_PVCLUST}"
          f"  run_pvclust={RUN_PVCLUST}")
    print("=" * 70)

    ward_rows: list[dict] = []
    lda_rows:  list[dict] = []
    mrt_rows:  list[dict] = []
    errors:    list[dict] = []

    t0 = time.time()
    idx = 0

    for transform in TRANSFORMS:
        for n_ref in N_REF_RANGE:
            idx += 1
            elapsed = time.time() - t0
            eta = (elapsed / idx) * (total - idx) if idx > 1 else 0
            print(f"\n[{idx:3d}/{total}]  transform={transform:22s}  n_ref={n_ref}"
                  f"   elapsed={elapsed / 60:.1f}m   eta≈{eta / 60:.1f}m")

            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp = Path(tmpdir)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        # ── Ward's Clustering ────────────────────────
                        ward_result = wards_clustering_pipeline(
                            data_path=DATA_PATH,
                            stage1_artifact=STAGE1_ARTIFACT,
                            output_dir=tmp / "wards",
                            maps_dir=MAPS_DIR,
                            reference_quantile=n_ref,
                            taxa_transform=transform,
                            n_clusters=3,
                            label_map={1: 1, 2: 2, 3: 3},
                            env_variables=ENV_VARIABLES,
                            n_boot_coassign=N_BOOT_COASSIGN,
                            coassign_sample_frac=0.8,
                            n_boot_pvclust=N_BOOT_PVCLUST,
                            run_pvclust_bootstrap=RUN_PVCLUST,
                            sil_threshold=0.25,
                            margin_threshold=0.25,
                            random_state=42,
                            save_plots=False,
                            verbose=False,
                        )

                        rob = ward_result.robustness_table
                        status = rob["Status"].value_counts()
                        clust = rob["Original_Cluster"].value_counts().sort_index()

                        ward_rows.append({
                            "Transform": transform,
                            "N_Ref": n_ref,
                            "Mean_Silhouette": ward_result.mean_silhouette(),
                            "N_Core": int(status.get("Core", 0)),
                            "N_Peripheral": int(status.get("Peripheral", 0)),
                            "N_Uncertain": int(status.get("Uncertain", 0)),
                            "Cluster_1": int(clust.get(1, 0)),
                            "Cluster_2": int(clust.get(2, 0)),
                            "Cluster_3": int(clust.get(3, 0)),
                        })

                        # ── Confidence-Aware LDA ─────────────────────
                        lda_res = confidence_lda_pipeline(
                            data_path=DATA_PATH,
                            stage1_artifact=STAGE1_ARTIFACT,
                            output_dir=tmp / "lda",
                            site_robustness=rob,
                            env_variables=ENV_VARIABLES,
                            standardize_env=True,
                            cv_folds=5,
                            cv_repeats=10,
                            random_state=42,
                            save_plots=False,
                            verbose=False,
                        )

                        for model_name, row in lda_res.summary_table.iterrows():
                            lda_rows.append({
                                "Transform": transform,
                                "N_Ref": n_ref,
                                "Model": model_name,
                                **row.to_dict(),
                            })

                        # ── Confidence-Aware MRT ─────────────────────
                        mrt_res = confidence_mrt_pipeline(
                            data_path=DATA_PATH,
                            stage1_artifact=STAGE1_ARTIFACT,
                            output_dir=tmp / "mrt",
                            site_robustness=rob,
                            env_variables=ENV_VARIABLES,
                            response_transform=transform,
                            k_folds=5,
                            cv_perms=10,
                            minsplit=3,
                            minbucket=2,
                            random_state=42,
                            save_plots=False,
                            verbose=False,
                        )

                        for model_name, row in mrt_res["summary_table"].iterrows():
                            mrt_rows.append({
                                "Transform": transform,
                                "N_Ref": n_ref,
                                "Model": model_name,
                                **row.to_dict(),
                            })

                n_core = int(status.get("Core", 0))
                n_per  = int(status.get("Peripheral", 0))
                n_unc  = int(status.get("Uncertain", 0))
                print(f"       ✓  Core={n_core}  Periph={n_per}  Uncert={n_unc}")

            except Exception as exc:
                print(f"       ✗  ERROR: {exc}")
                errors.append({
                    "Transform": transform,
                    "N_Ref": n_ref,
                    "Error": str(exc),
                })

            # Incremental save every 10 iterations
            if idx % 10 == 0:
                _save_results(ward_rows, lda_rows, mrt_rows, errors)

    # ── Final save ───────────────────────────────────────────────────
    elapsed_total = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  Grid sweep done in {elapsed_total / 60:.1f} minutes")
    print(f"  {len(errors)} error(s) out of {total} runs")
    print(f"{'=' * 70}")

    _save_results(ward_rows, lda_rows, mrt_rows, errors)

    ward_df = pd.DataFrame(ward_rows)
    lda_df  = pd.DataFrame(lda_rows)
    mrt_df  = pd.DataFrame(mrt_rows)

    # ── Figures ──────────────────────────────────────────────────────
    figs_dir = OUTPUT_DIR / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Ward's summary
    if not ward_df.empty:
        _plot_ward_lines(ward_df, figs_dir / "ward_summary_lines.png")
        print(f"  > {figs_dir / 'ward_summary_lines.png'}")

    # LDA
    if not lda_df.empty and "CV Accuracy (mean)" in lda_df.columns:
        _plot_model_lines(
            lda_df, "CV Accuracy (mean)", "CV Accuracy",
            "LDA Cross-Validated Accuracy",
            figs_dir / "lda_cv_accuracy_lines.png",
        )
        _plot_model_heatmaps(
            lda_df, "CV Accuracy (mean)",
            "LDA CV Accuracy (Transform × N_Ref)",
            figs_dir / "lda_cv_accuracy_heatmaps.png",
            cmap="YlGn",
        )
        _plot_model_lines(
            lda_df, "Train Accuracy", "Train Accuracy",
            "LDA Training Accuracy",
            figs_dir / "lda_train_accuracy_lines.png",
        )
        print(f"  > LDA figures saved")

    # MRT
    if not mrt_df.empty:
        _plot_model_lines(
            mrt_df, "Train Accuracy", "Train Accuracy",
            "MRT Training Accuracy",
            figs_dir / "mrt_train_accuracy_lines.png",
        )
        _plot_model_heatmaps(
            mrt_df, "Train Accuracy",
            "MRT Training Accuracy (Transform × N_Ref)",
            figs_dir / "mrt_train_accuracy_heatmaps.png",
            cmap="YlGn",
        )
        if "Min CVRE" in mrt_df.columns:
            _plot_model_lines(
                mrt_df, "Min CVRE", "CVRE",
                "MRT Cross-Validated Relative Error",
                figs_dir / "mrt_cvre_lines.png",
            )
            _plot_model_heatmaps(
                mrt_df, "Min CVRE",
                "MRT CVRE (Transform × N_Ref)",
                figs_dir / "mrt_cvre_heatmaps.png",
                cmap="YlOrRd",
            )
        print(f"  > MRT figures saved")

    print(f"\n  All outputs → {OUTPUT_DIR}")
