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
    site_data = build_site_plot_data(pca_result, ref_labels, pred_all)
    save_table(site_data, tables_dir / "site_plot_data",
               formats=table_formats, verbose=verbose)

    # Classification summary for reference sites
    ref_data = site_data[site_data["is_reference"]]
    n_ref = len(ref_data)
    n_correct = int(ref_data["ref_correct"].sum())
    _log(f"    Reference accuracy: {n_correct}/{n_ref} "
         f"({100 * n_correct / n_ref:.1f}%)")

    # ─── Step 3: Reference-cluster support ───────────────────────────
    _log("[5] Computing reference-cluster support in PC space ...")
    ref_pca = site_data.loc[site_data["is_reference"], ["PC1", "PC2"]].values
    ref_true = site_data.loc[site_data["is_reference"], "true_cluster"].values.astype(int)

    hulls = compute_cluster_hulls(ref_pca, ref_true)
    ellipses = compute_cluster_ellipses(ref_pca, ref_true, confidence=0.95)
    _log(f"    Hulls for clusters: {list(hulls.keys())}")
    _log(f"    Ellipses for clusters: {list(ellipses.keys())}")

    # All-ref ellipse (pooled)
    allref_ellipse = compute_allref_ellipse(ref_pca, confidence=0.95)
    _log(f"    All-ref ellipse center: ({allref_ellipse['center'][0]:.2f}, "
         f"{allref_ellipse['center'][1]:.2f})")

    # ─── Step 4: Prediction grid ─────────────────────────────────────
    _log("[6] Building prediction grid (conditional on PC1–PC2) ...")
    xx, yy, grid_labels = build_prediction_grid(
        pca_result,
        classifier_model=lda_fit.model,
        classifier_scaler=lda_fit.scaler,
        grid_resolution=grid_resolution,
    )
    _log(f"    Grid shape: {xx.shape}, unique labels: {sorted(np.unique(grid_labels))}")

    # Save prediction grid (down-sampled summary)
    grid_df = pd.DataFrame({
        "PC1": xx.ravel(),
        "PC2": yy.ravel(),
        "Predicted_Cluster": grid_labels.ravel().astype(int),
    })
    save_table(grid_df, tables_dir / "prediction_grid",
               formats=table_formats, verbose=verbose)

    # ─── Step 4b: 2-PC LDA (native PC1–PC2) ─────────────────────────
    _log("[6b] Fitting 2-PC LDA on reference sites in PC1–PC2 ...")
    lda_pc2 = fit_pc2_lda(ref_pca, ref_true)
    pc2_preds_ref = lda_pc2.predict(ref_pca)
    pc2_acc = (pc2_preds_ref == ref_true).mean()
    _log(f"    2-PC LDA ref accuracy: {pc2_acc:.1%}")

    xx2, yy2, grid_labels_pc2 = build_pc2_lda_grid(
        pca_result, lda_pc2, grid_resolution=grid_resolution,
    )
    _log(f"    2-PC LDA grid unique labels: {sorted(np.unique(grid_labels_pc2))}")

    # NOTE: site colours / shapes always reflect the *finalized* full-model
    # predictions (pred_all) and Ward clustering labels.  The 2-PC LDA is
    # used ONLY for the decision-region background, not for site labelling.

    # ─── Step 5–6: Visualizations ────────────────────────────────────
    if save_plots:
        _log("[7] Creating visualizations ...")
        plot_kwargs = dict(
            xx=xx, yy=yy, grid_labels=grid_labels,
            loadings=pca_result.loadings,
            variance_explained=pca_result.variance_explained,
            env_short_names=env_short,
        )

        # ── Single all-ref ellipse: full-model decision regions ─────
        _log("  Creating all-ref ellipse (full-model) ...")
        fig_full, _ = plot_allref_ellipse(
            site_data, allref_ellipse,
            title=(f"{chosen_model.model_name} — "
                   f"All Reference Sites 95% Ellipse (Full-Model)"),
            **plot_kwargs,
        )
        save_figure(fig_full, figures_dir / "allref_ellipse_fullmodel",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_full)

        # ── 2×2 ellipse panel: full-model decision regions ───────────
        _log("  Creating 2×2 ellipse panel (full-model) ...")
        fig_panel_full = plot_ellipse_panel_2x2(
            site_data, ellipses, allref_ellipse,
            suptitle=(f"{chosen_model.model_name} — "
                      f"Full-Model Decision Regions"),
            **plot_kwargs,
        )
        save_figure(fig_panel_full,
                    figures_dir / "ellipse_panel_fullmodel",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_panel_full)

        # ── Single all-ref ellipse: 2-PC LDA decision regions ──────
        _log("  Creating all-ref ellipse (2-PC LDA) ...")
        plot_kwargs_pc2 = dict(
            xx=xx2, yy=yy2, grid_labels=grid_labels_pc2,
            loadings=pca_result.loadings,
            variance_explained=pca_result.variance_explained,
            env_short_names=env_short,
        )
        fig_pc2, _ = plot_allref_ellipse(
            site_data, allref_ellipse,
            title="2-PC LDA — All Reference Sites 95% Ellipse",
            **plot_kwargs_pc2,
        )
        save_figure(fig_pc2, figures_dir / "allref_ellipse_pc2lda",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_pc2)

        # ── 2×2 ellipse panel: 2-PC LDA decision regions ────────────
        _log("  Creating 2×2 ellipse panel (2-PC LDA) ...")
        fig_panel_pc2 = plot_ellipse_panel_2x2(
            site_data, ellipses, allref_ellipse,
            suptitle="2-PC LDA — Decision Regions in PC1–PC2",
            **plot_kwargs_pc2,
        )
        save_figure(fig_panel_pc2,
                    figures_dir / "ellipse_panel_pc2lda",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_panel_pc2)

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
        "site_data": site_data,
        "predictions": pred_df,
        "hulls": hulls,
        "ellipses": ellipses,
        "grid": (xx, yy, grid_labels),
    }
