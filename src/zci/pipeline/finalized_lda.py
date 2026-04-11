"""Stage 2 — Finalized LDA Model: full-site PCA ordination with predictions.

Picks a specified model from the confidence-aware LDA comparison,
predicts *all* non-reference sites, and produces layered PCA visualizations
showing reference clusters, predicted labels, decision regions, and
environmental support (hulls / ellipses).

Output structure under ``output_dir / ModelFinalized``:
    tables/   site_pca_scores.xlsx, site_plot_data.xlsx, prediction_grid.xlsx
    figures/  ordination_hull.png, ordination_ellipse.png, ordination_combined.png
    artifacts/ pca_model.pkl, note.txt
"""

from __future__ import annotations

import pickle
from pathlib import Path as _Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.confidence_lda import evaluate_sites
from ..core.finalized_env_pca import (
    FullSitePCAResult,
    build_pc2_lda_grid,
    build_prediction_grid,
    build_site_plot_data,
    compute_allref_ellipse,
    compute_cluster_ellipses,
    compute_cluster_hulls,
    fit_pc2_lda,
    run_fullsite_env_pca,
)
from ..core.lda import predict_sites
from ..io.readers import extract_block, read_study_data
from ..io.writers import save_figure, save_table
from ..models.confidence_lda import ConfidenceLDAModelResult
from ..viz.finalized_env_plots import (
    plot_allref_ellipse,
    plot_ellipse_panel_2x2,
)


def finalized_lda_pipeline(
    data_path: str | _Path,
    output_dir: str | _Path,
    *,
    chosen_model: ConfidenceLDAModelResult,
    site_robustness: pd.DataFrame,
    env_variables: Sequence[str] | None = None,
    grid_resolution: int = 200,
    save_plots: bool = True,
    figure_formats: Sequence[str] = ("png",),
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> dict:
    """Run the finalized-model PCA ordination pipeline.

    Parameters
    ----------
    chosen_model : ConfidenceLDAModelResult
        One of the models from the confidence-aware comparison (e.g. Model B).
    site_robustness : pd.DataFrame
        Ward clustering artifact with Original_Cluster, Status columns.
    """
    output_dir = _Path(output_dir) / "ModelFinalized"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    artifacts_dir = output_dir / "artifacts"
    for d in [tables_dir, figures_dir, artifacts_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if env_variables is None:
        env_variables = [
            "Measured Depth (m)",
            "Water DO Bottom (mg/L)",
            "Temperature (oC)",
            "MPS (Phi)",
            "LOI (%)",
        ]
    env_short = [n.split("(")[0].strip() if "(" in n else n for n in env_variables]

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    _log("=" * 60)
    _log("  Finalized LDA Model — Full-Site PCA Ordination")
    _log("=" * 60)

    lda_fit = chosen_model.lda_fit

    # ─── Read data ───────────────────────────────────────────────────
    _log("[1] Reading data ...")
    data = read_study_data(data_path)
    env_block = extract_block(data, "environmental", "raw")

    env_vars_present = [v for v in env_variables if v in env_block.columns]
    env_all = env_block[env_vars_present].dropna()
    _log(f"    All sites with complete env data: {len(env_all)}")

    ref_sites = site_robustness.index
    ref_labels = site_robustness["Original_Cluster"]
    nonref_sites = env_all.index.difference(ref_sites)
    _log(f"    Reference: {len(env_all.index.intersection(ref_sites))}, "
         f"Non-reference: {len(nonref_sites)}")

    # ─── Predict all sites with the chosen model ─────────────────────
    _log(f"[2] Predicting all sites with {chosen_model.model_name} ...")
    pred_all, prob_all = predict_sites(lda_fit, env_all)
    _log(f"    Predictions: {pred_all.value_counts().to_dict()}")

    # Save predictions
    pred_df = pd.DataFrame({
        "Predicted_Cluster": pred_all,
        "Is_Reference": env_all.index.isin(ref_sites),
    })
    for j, cname in enumerate(lda_fit.cluster_names):
        pred_df[f"Prob_{cname}"] = prob_all.iloc[:, j]
    save_table(pred_df, tables_dir / "site_predictions",
               formats=table_formats, verbose=verbose)

    # ─── Step 1: PCA on all sites ────────────────────────────────────
    _log("[3] Running PCA on all sites ...")
    pca_result = run_fullsite_env_pca(env_all, standardize=True)
    _log(f"    Variance explained: "
         f"PC1={pca_result.variance_explained[0]:.1f}%, "
         f"PC2={pca_result.variance_explained[1]:.1f}%")

    # Save PCA scores
    save_table(pca_result.scores, tables_dir / "site_pca_scores",
               formats=table_formats, verbose=verbose)
    # Save PCA loadings (with variance summary rows, matching Stage 1 format)
    pc_names = list(pca_result.loadings.columns)
    variance_info = pd.DataFrame(
        {
            pc: [
                pca_result.pca.explained_variance_[i],
                pca_result.pca.explained_variance_ratio_[i],
                pca_result.pca.explained_variance_ratio_[: i + 1].sum(),
            ]
            for i, pc in enumerate(pc_names)
        },
        index=["Explained Variance", "Proportion of Variance",
               "Cumulative Proportion"],
    )
    sep = pd.DataFrame(
        {col: [""] for col in pca_result.loadings.columns},
        index=[""],
    ).astype(object)
    loadings_full = pd.concat([
        pca_result.loadings.astype(object),
        sep,
        variance_info.astype(object),
    ])
    save_table(loadings_full, tables_dir / "pca_loadings",
               formats=table_formats, verbose=verbose)

    # ─── Step 2: Build plotting dataset ──────────────────────────────
    _log("[4] Building site plotting dataset ...")
    site_data_12 = build_site_plot_data(pca_result, ref_labels, pred_all,
                                        pc_x="PC1", pc_y="PC2")
    save_table(site_data_12, tables_dir / "site_plot_data",
               formats=table_formats, verbose=verbose)

    site_data_13 = build_site_plot_data(pca_result, ref_labels, pred_all,
                                        pc_x="PC1", pc_y="PC3")

    site_data_23 = build_site_plot_data(pca_result, ref_labels, pred_all,
                                        pc_x="PC2", pc_y="PC3")

    # Classification summary for reference sites
    ref_data = site_data_12[site_data_12["is_reference"]]
    n_ref = len(ref_data)
    n_correct = int(ref_data["ref_correct"].sum())
    _log(f"    Reference accuracy: {n_correct}/{n_ref} "
         f"({100 * n_correct / n_ref:.1f}%)")

    # ─── Step 3: Reference-cluster support ───────────────────────────
    _log("[5] Computing reference-cluster support in PC space ...")
    ref_true = site_data_12.loc[site_data_12["is_reference"], "true_cluster"].values.astype(int)

    # PC1-PC2
    ref_pca_12 = site_data_12.loc[site_data_12["is_reference"], ["PC1", "PC2"]].values
    hulls_12 = compute_cluster_hulls(ref_pca_12, ref_true)
    ellipses_12 = compute_cluster_ellipses(ref_pca_12, ref_true, confidence=0.95)
    allref_ellipse_12 = compute_allref_ellipse(ref_pca_12, confidence=0.95)
    _log(f"    PC1-PC2 ellipses for clusters: {list(ellipses_12.keys())}")

    # PC1-PC3
    ref_pca_13 = site_data_13.loc[site_data_13["is_reference"], ["PC1", "PC3"]].values
    hulls_13 = compute_cluster_hulls(ref_pca_13, ref_true)
    ellipses_13 = compute_cluster_ellipses(ref_pca_13, ref_true, confidence=0.95)
    allref_ellipse_13 = compute_allref_ellipse(ref_pca_13, confidence=0.95)
    _log(f"    PC1-PC3 ellipses for clusters: {list(ellipses_13.keys())}")

    # PC2-PC3
    ref_pca_23 = site_data_23.loc[site_data_23["is_reference"], ["PC2", "PC3"]].values
    hulls_23 = compute_cluster_hulls(ref_pca_23, ref_true)
    ellipses_23 = compute_cluster_ellipses(ref_pca_23, ref_true, confidence=0.95)
    allref_ellipse_23 = compute_allref_ellipse(ref_pca_23, confidence=0.95)
    _log(f"    PC2-PC3 ellipses for clusters: {list(ellipses_23.keys())}")

    # ─── Step 4: Prediction grids ────────────────────────────────────
    _log("[6] Building prediction grids ...")

    # PC1-PC2 full-model grid
    xx_12, yy_12, gl_12 = build_prediction_grid(
        pca_result,
        classifier_model=lda_fit.model,
        classifier_scaler=lda_fit.scaler,
        pc_indices=(0, 1),
        grid_resolution=grid_resolution,
    )
    _log(f"    PC1-PC2 grid shape: {xx_12.shape}")

    # PC1-PC3 full-model grid
    xx_13, yy_13, gl_13 = build_prediction_grid(
        pca_result,
        classifier_model=lda_fit.model,
        classifier_scaler=lda_fit.scaler,
        pc_indices=(0, 2),
        grid_resolution=grid_resolution,
    )
    _log(f"    PC1-PC3 grid shape: {xx_13.shape}")

    # PC2-PC3 full-model grid
    xx_23, yy_23, gl_23 = build_prediction_grid(
        pca_result,
        classifier_model=lda_fit.model,
        classifier_scaler=lda_fit.scaler,
        pc_indices=(1, 2),
        grid_resolution=grid_resolution,
    )
    _log(f"    PC2-PC3 grid shape: {xx_23.shape}")

    # Save prediction grid (PC1-PC2)
    grid_df = pd.DataFrame({
        "PC1": xx_12.ravel(),
        "PC2": yy_12.ravel(),
        "Predicted_Cluster": gl_12.ravel().astype(int),
    })
    save_table(grid_df, tables_dir / "prediction_grid",
               formats=table_formats, verbose=verbose)

    # ─── Step 4b: 2-PC LDA (native PC1–PC2 and PC1–PC3) ─────────────
    _log("[6b] Fitting 2-PC LDA on reference sites ...")

    # PC1-PC2
    lda_pc12 = fit_pc2_lda(ref_pca_12, ref_true)
    pc12_acc = (lda_pc12.predict(ref_pca_12) == ref_true).mean()
    _log(f"    2-PC LDA (PC1-PC2) ref accuracy: {pc12_acc:.1%}")
    xx2_12, yy2_12, gl2_12 = build_pc2_lda_grid(
        pca_result, lda_pc12, pc_indices=(0, 1),
        grid_resolution=grid_resolution,
    )

    # PC1-PC3
    lda_pc13 = fit_pc2_lda(ref_pca_13, ref_true)
    pc13_acc = (lda_pc13.predict(ref_pca_13) == ref_true).mean()
    _log(f"    2-PC LDA (PC1-PC3) ref accuracy: {pc13_acc:.1%}")
    xx2_13, yy2_13, gl2_13 = build_pc2_lda_grid(
        pca_result, lda_pc13, pc_indices=(0, 2),
        grid_resolution=grid_resolution,
    )

    # PC2-PC3
    lda_pc23 = fit_pc2_lda(ref_pca_23, ref_true)
    pc23_acc = (lda_pc23.predict(ref_pca_23) == ref_true).mean()
    _log(f"    2-PC LDA (PC2-PC3) ref accuracy: {pc23_acc:.1%}")
    xx2_23, yy2_23, gl2_23 = build_pc2_lda_grid(
        pca_result, lda_pc23, pc_indices=(1, 2),
        grid_resolution=grid_resolution,
    )

    # ─── Step 5–6: Visualizations ────────────────────────────────────
    if save_plots:
        _log("[7] Creating visualizations ...")

        # === PC1 × PC2 figures ===
        plot_kw_12 = dict(
            xx=xx_12, yy=yy_12, grid_labels=gl_12,
            loadings=pca_result.loadings,
            variance_explained=pca_result.variance_explained,
            env_short_names=env_short,
            pc_x_col="PC1", pc_y_col="PC2",
            loading_indices=(0, 1),
        )

        _log("  Creating PC1-PC2 all-ref ellipse (full-model) ...")
        fig_full_12, _ = plot_allref_ellipse(
            site_data_12, allref_ellipse_12,
            title=(f"{chosen_model.model_name} — "
                   f"All Reference Sites 95% Ellipse (Full-Model)"),
            **plot_kw_12,
        )
        save_figure(fig_full_12, figures_dir / "pc12_allref_ellipse_fullmodel",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_full_12)

        _log("  Creating PC1-PC2 2×2 ellipse panel (full-model) ...")
        fig_panel_12 = plot_ellipse_panel_2x2(
            site_data_12, ellipses_12, allref_ellipse_12,
            suptitle=(f"{chosen_model.model_name} — "
                      f"Full-Model Decision Regions (PC1–PC2)"),
            **plot_kw_12,
        )
        save_figure(fig_panel_12, figures_dir / "pc12_ellipse_panel_fullmodel",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_panel_12)

        plot_kw2_12 = dict(
            xx=xx2_12, yy=yy2_12, grid_labels=gl2_12,
            loadings=pca_result.loadings,
            variance_explained=pca_result.variance_explained,
            env_short_names=env_short,
            pc_x_col="PC1", pc_y_col="PC2",
            loading_indices=(0, 1),
        )

        _log("  Creating PC1-PC2 all-ref ellipse (2-PC LDA) ...")
        fig_pc2_12, _ = plot_allref_ellipse(
            site_data_12, allref_ellipse_12,
            title="2-PC LDA — All Reference Sites 95% Ellipse (PC1–PC2)",
            **plot_kw2_12,
        )
        save_figure(fig_pc2_12, figures_dir / "pc12_allref_ellipse_pc2lda",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_pc2_12)

        _log("  Creating PC1-PC2 2×2 ellipse panel (2-PC LDA) ...")
        fig_panel2_12 = plot_ellipse_panel_2x2(
            site_data_12, ellipses_12, allref_ellipse_12,
            suptitle="2-PC LDA — Decision Regions in PC1–PC2",
            **plot_kw2_12,
        )
        save_figure(fig_panel2_12, figures_dir / "pc12_ellipse_panel_pc2lda",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_panel2_12)

        # === PC1 × PC3 figures ===
        plot_kw_13 = dict(
            xx=xx_13, yy=yy_13, grid_labels=gl_13,
            loadings=pca_result.loadings,
            variance_explained=pca_result.variance_explained,
            env_short_names=env_short,
            pc_x_col="PC1", pc_y_col="PC3",
            loading_indices=(0, 2),
        )

        _log("  Creating PC1-PC3 all-ref ellipse (full-model) ...")
        fig_full_13, _ = plot_allref_ellipse(
            site_data_13, allref_ellipse_13,
            title=(f"{chosen_model.model_name} — "
                   f"All Reference Sites 95% Ellipse (Full-Model, PC1–PC3)"),
            **plot_kw_13,
        )
        save_figure(fig_full_13, figures_dir / "pc13_allref_ellipse_fullmodel",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_full_13)

        _log("  Creating PC1-PC3 2×2 ellipse panel (full-model) ...")
        fig_panel_13 = plot_ellipse_panel_2x2(
            site_data_13, ellipses_13, allref_ellipse_13,
            suptitle=(f"{chosen_model.model_name} — "
                      f"Full-Model Decision Regions (PC1–PC3)"),
            **plot_kw_13,
        )
        save_figure(fig_panel_13, figures_dir / "pc13_ellipse_panel_fullmodel",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_panel_13)

        plot_kw2_13 = dict(
            xx=xx2_13, yy=yy2_13, grid_labels=gl2_13,
            loadings=pca_result.loadings,
            variance_explained=pca_result.variance_explained,
            env_short_names=env_short,
            pc_x_col="PC1", pc_y_col="PC3",
            loading_indices=(0, 2),
        )

        _log("  Creating PC1-PC3 all-ref ellipse (2-PC LDA) ...")
        fig_pc2_13, _ = plot_allref_ellipse(
            site_data_13, allref_ellipse_13,
            title="2-PC LDA — All Reference Sites 95% Ellipse (PC1–PC3)",
            **plot_kw2_13,
        )
        save_figure(fig_pc2_13, figures_dir / "pc13_allref_ellipse_pc2lda",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_pc2_13)

        _log("  Creating PC1-PC3 2×2 ellipse panel (2-PC LDA) ...")
        fig_panel2_13 = plot_ellipse_panel_2x2(
            site_data_13, ellipses_13, allref_ellipse_13,
            suptitle="2-PC LDA — Decision Regions in PC1–PC3",
            **plot_kw2_13,
        )
        save_figure(fig_panel2_13, figures_dir / "pc13_ellipse_panel_pc2lda",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_panel2_13)

        # === PC2 × PC3 figures ===
        plot_kw_23 = dict(
            xx=xx_23, yy=yy_23, grid_labels=gl_23,
            loadings=pca_result.loadings,
            variance_explained=pca_result.variance_explained,
            env_short_names=env_short,
            pc_x_col="PC2", pc_y_col="PC3",
            loading_indices=(1, 2),
        )

        _log("  Creating PC2-PC3 all-ref ellipse (full-model) ...")
        fig_full_23, _ = plot_allref_ellipse(
            site_data_23, allref_ellipse_23,
            title=(f"{chosen_model.model_name} — "
                   f"All Reference Sites 95% Ellipse (Full-Model, PC2–PC3)"),
            **plot_kw_23,
        )
        save_figure(fig_full_23, figures_dir / "pc23_allref_ellipse_fullmodel",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_full_23)

        _log("  Creating PC2-PC3 2×2 ellipse panel (full-model) ...")
        fig_panel_23 = plot_ellipse_panel_2x2(
            site_data_23, ellipses_23, allref_ellipse_23,
            suptitle=(f"{chosen_model.model_name} — "
                      f"Full-Model Decision Regions (PC2–PC3)"),
            **plot_kw_23,
        )
        save_figure(fig_panel_23, figures_dir / "pc23_ellipse_panel_fullmodel",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_panel_23)

        plot_kw2_23 = dict(
            xx=xx2_23, yy=yy2_23, grid_labels=gl2_23,
            loadings=pca_result.loadings,
            variance_explained=pca_result.variance_explained,
            env_short_names=env_short,
            pc_x_col="PC2", pc_y_col="PC3",
            loading_indices=(1, 2),
        )

        _log("  Creating PC2-PC3 all-ref ellipse (2-PC LDA) ...")
        fig_pc2_23, _ = plot_allref_ellipse(
            site_data_23, allref_ellipse_23,
            title="2-PC LDA — All Reference Sites 95% Ellipse (PC2–PC3)",
            **plot_kw2_23,
        )
        save_figure(fig_pc2_23, figures_dir / "pc23_allref_ellipse_pc2lda",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_pc2_23)

        _log("  Creating PC2-PC3 2×2 ellipse panel (2-PC LDA) ...")
        fig_panel2_23 = plot_ellipse_panel_2x2(
            site_data_23, ellipses_23, allref_ellipse_23,
            suptitle="2-PC LDA — Decision Regions in PC2–PC3",
            **plot_kw2_23,
        )
        save_figure(fig_panel2_23, figures_dir / "pc23_ellipse_panel_pc2lda",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_panel2_23)

    # ─── Step 7: Save artifacts ──────────────────────────────────────
    _log("[8] Saving artifacts ...")

    with open(artifacts_dir / "pca_model.pkl", "wb") as f:
        pickle.dump({
            "pca": pca_result.pca,
            "scaler": pca_result.scaler,
            "env_columns": pca_result.env_columns,
        }, f)
    _log(f"  ✓ Saved: {artifacts_dir / 'pca_model.pkl'}")

    note = (
        "Decision-region visualization note\n"
        "===================================\n\n"
        f"Model: {chosen_model.model_name}\n"
        f"Training subset: {chosen_model.training_subset}\n"
        f"N training sites: {chosen_model.n_train}\n"
        f"Training accuracy: {chosen_model.train_accuracy():.2%}\n"
        f"CV accuracy: {chosen_model.cv_accuracy():.2%} "
        f"± {chosen_model.cv_accuracy_std():.2%}\n\n"
        "The decision regions shown in the PC1–PC2 plane are a *conditional*\n"
        "visualization of the full classifier.  The classifier was trained\n"
        "on the full environmental descriptor space (not on PCA scores).\n"
        "For every grid point in the PC1–PC2 plane, all omitted PCA\n"
        f"coordinates (PC3 … PC{pca_result.scores.shape[1]}) are fixed at\n"
        "their overall-site mean values (computed from all sites, both\n"
        "reference and non-reference).  The grid PCA-score vectors are\n"
        "back-transformed into the original environmental space, then\n"
        "standardised with the classifier's training scaler, and finally\n"
        "passed to the fitted model for prediction.\n\n"
        "Because the omitted PCs are fixed at their means, the map shows\n"
        "an approximate slice through the full decision surface and may\n"
        "not capture classification behaviour that is driven primarily by\n"
        "variability along higher-order PCs.\n"
    )
    (artifacts_dir / "note.txt").write_text(note, encoding="utf-8")
    _log(f"  ✓ Saved: {artifacts_dir / 'note.txt'}")

    _log("\n" + "=" * 60)
    _log("  Finalized Model Pipeline Complete")
    _log("=" * 60)
    _log(f"  {chosen_model.model_name}: "
         f"ref accuracy {n_correct}/{n_ref} ({100 * n_correct / n_ref:.1f}%)")
    _log(f"  Non-ref sites predicted: {len(nonref_sites)}")

    return {
        "pca_result": pca_result,
        "site_data_12": site_data_12,
        "site_data_13": site_data_13,
        "site_data_23": site_data_23,
        "predictions": pred_df,
        "hulls_12": hulls_12,
        "ellipses_12": ellipses_12,
        "grid_12": (xx_12, yy_12, gl_12),
        "grid_13": (xx_13, yy_13, gl_13),
        "grid_23": (xx_23, yy_23, gl_23),
    }
