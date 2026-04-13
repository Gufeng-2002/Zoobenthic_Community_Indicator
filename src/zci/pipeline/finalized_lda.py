"""Stage 2 — Finalized LDA Model: env PCA ordination with posterior heatmaps.

Produces for each PC pair (1-2, 1-3, 2-3):
  - Reference-site ordination (heatmap background + ref sites)
  - Non-reference projection (heatmap background + non-ref sites)

Output structure under ``output_dir / ModelFinalized``:
    tables/env_ord_pca/   pca_loadings.xlsx, site_pca_scores.xlsx
    tables/anova/         anova_taxa.xlsx, anova_env.xlsx
    figures/              pc12_ref_ordination.png, pc12_nonref_projection.png, ...
    artifacts/            pca_model.pkl, all_sites_classified.xlsx, note.txt
"""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path as _Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..core.anova import anova_table
from ..core.finalized_env_pca import (
    FullSitePCAResult,
    build_prediction_grid_proba,
    run_fullsite_env_pca,
)
from ..core.lda import fit_lda, predict_sites
from ..io.readers import extract_block, read_study_data
from ..io.writers import save_figure, save_table
from ..models.lda import LDAFit
from ..viz.env_ord_heatmap import plot_nonref_projection, plot_ref_ordination


def finalized_lda_pipeline(
    data_path: str | _Path,
    output_dir: str | _Path,
    *,
    model_name: str = "Finalized_EnvStrong",
    site_robustness: pd.DataFrame,
    env_variables: Sequence[str] | None = None,
    taxa_transform: str = "octave",
    grid_resolution: int = 200,
    save_plots: bool = True,
    figure_formats: Sequence[str] = ("png",),
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> dict:
    """Run the finalized-model PCA ordination pipeline.

    Retrains LDA on **EnvStrong_TaxaStrong** reference sites (double-strong),
    then predicts all sites (ref + non-ref) and builds PCA ordination figures.

    Parameters
    ----------
    site_robustness : pd.DataFrame
        Ward's combined_robustness table with Original_Cluster,
        Env_Strength, TaxaEnv_Class columns.
    taxa_transform : str
        Same transform used in Ward's clustering (for ANOVA on taxa).
    """
    output_dir = _Path(output_dir) / "ModelFinalized"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    artifacts_dir = output_dir / "artifacts"

    # Clean previous tables, figures, and old table_latex (keep artifacts dir)
    for d in [tables_dir, figures_dir, output_dir / "table_latex"]:
        if d.exists():
            shutil.rmtree(d)
    for d in [tables_dir, figures_dir, artifacts_dir]:
        d.mkdir(parents=True, exist_ok=True)

    env_ord_dir = tables_dir / "env_ord_pca"
    anova_dir = tables_dir / "anova"
    env_ord_dir.mkdir(parents=True, exist_ok=True)
    anova_dir.mkdir(parents=True, exist_ok=True)

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
    _log("  Finalized LDA — Env PCA Ordination (Posterior Heatmap)")
    _log("=" * 60)

    # ─── 1. Read data ────────────────────────────────────────────────
    _log("[1] Reading data ...")
    data = read_study_data(data_path)
    env_block = extract_block(data, "environmental", "raw")
    env_vars_present = [v for v in env_variables if v in env_block.columns]
    env_all = env_block[env_vars_present].dropna()
    _log(f"    All sites with complete env data: {len(env_all)}")

    ref_sites = site_robustness.index
    ref_labels = site_robustness["Original_Cluster"]
    ref_env_strength = site_robustness["Env_Strength"]
    nonref_sites = env_all.index.difference(ref_sites)
    _log(f"    Reference: {len(env_all.index.intersection(ref_sites))}, "
         f"Non-reference: {len(nonref_sites)}")

    # ─── 2. Train LDA on EnvStrong_TaxaStrong reference sites ────────
    double_strong_mask = site_robustness["TaxaEnv_Class"] == "EnvStrong_TaxaStrong"
    train_sites = site_robustness.index[double_strong_mask]
    train_in_env = env_all.index.intersection(train_sites)
    env_train = env_all.loc[train_in_env, env_vars_present]
    labels_train = ref_labels.loc[train_in_env]
    n_train = len(train_in_env)
    _log(f"[2] Training LDA on {n_train} EnvStrong_TaxaStrong reference sites ...")
    _log(f"    Clusters in training set: {sorted(labels_train.unique())}")

    lda_fit = fit_lda(env_train, labels_train.values, standardize=True)
    _log(f"    Training accuracy: {lda_fit.accuracy:.2%}")

    # Predict all sites
    _log(f"    Predicting all {len(env_all)} sites ...")
    pred_all, prob_all = predict_sites(lda_fit, env_all)
    _log(f"    Predictions: {pred_all.value_counts().to_dict()}")

    # ─── 3. PCA on all sites ────────────────────────────────────────
    _log("[3] Running PCA on all sites ...")
    pca_result = run_fullsite_env_pca(env_all, standardize=True)
    _log(f"    Variance explained: "
         f"PC1={pca_result.variance_explained[0]:.1f}%, "
         f"PC2={pca_result.variance_explained[1]:.1f}%, "
         f"PC3={pca_result.variance_explained[2]:.1f}%")

    # Save PCA scores and loadings
    save_table(pca_result.scores, env_ord_dir / "site_pca_scores",
               formats=table_formats, verbose=verbose)

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
    save_table(loadings_full, env_ord_dir / "pca_loadings",
               formats=table_formats, verbose=verbose)

    # ─── 4. Build all_sites_classified → artifacts ───────────────────
    _log("[4] Building all_sites_classified table ...")
    classified = pd.DataFrame(index=env_all.index)
    classified["Is_Reference"] = env_all.index.isin(ref_sites)

    # Defined_Cluster: ref → Ward's label, non-ref → LDA prediction
    defined = pred_all.copy()
    defined.name = "Defined_Cluster"
    ref_overlap = env_all.index.intersection(ref_labels.index)
    defined.loc[ref_overlap] = ref_labels.loc[ref_overlap].astype(int)
    classified["Defined_Cluster"] = defined
    classified["Predicted_Cluster"] = pred_all
    for j, cname in enumerate(lda_fit.cluster_names):
        classified[f"Prob_{cname}"] = prob_all.iloc[:, j]

    save_table(classified, artifacts_dir / "all_sites_classified",
               formats=table_formats, verbose=verbose)

    # Reference accuracy
    ref_in_env = env_all.index.intersection(ref_sites)
    n_ref = len(ref_in_env)
    n_correct = int((pred_all.loc[ref_in_env] == ref_labels.loc[ref_in_env]).sum())
    _log(f"    Reference accuracy: {n_correct}/{n_ref} "
         f"({100 * n_correct / n_ref:.1f}%)")

    # ─── 5. Probability grids for 3 PC pairs ────────────────────────
    _log("[5] Building probability grids ...")
    cluster_ids = sorted(pred_all.unique())

    pc_pairs = [
        ((0, 1), "PC1", "PC2", "pc12"),
        ((0, 2), "PC1", "PC3", "pc13"),
        ((1, 2), "PC2", "PC3", "pc23"),
    ]

    grids = {}
    for (ix, iy), pcx, pcy, tag in pc_pairs:
        xx, yy, gl, gp = build_prediction_grid_proba(
            pca_result,
            classifier_model=lda_fit.model,
            classifier_scaler=lda_fit.scaler,
            pc_indices=(ix, iy),
            grid_resolution=grid_resolution,
        )
        grids[tag] = (xx, yy, gl, gp)
        _log(f"    {pcx}-{pcy} grid: {xx.shape}")

    # ─── 6. Figures ──────────────────────────────────────────────────
    if save_plots:
        _log("[6] Creating heatmap ordination figures ...")

        scores = pca_result.scores
        ref_mask_scores = scores.index.isin(ref_sites)
        nonref_mask_scores = ~ref_mask_scores

        for (ix, iy), pcx, pcy, tag in pc_pairs:
            xx, yy, gl, gp = grids[tag]

            plot_kw = dict(
                pc_x_col=pcx,
                pc_y_col=pcy,
                variance_explained=pca_result.variance_explained,
                loadings=pca_result.loadings,
                env_short_names=env_short,
                loading_indices=(ix, iy),
            )

            # Ref ordination
            _log(f"  Creating {pcx}-{pcy} ref ordination ...")
            ref_pc = scores.loc[ref_mask_scores, [pcx, pcy]]
            ref_cls = ref_labels.reindex(ref_pc.index).astype(int)
            ref_envs = ref_env_strength.reindex(ref_pc.index)

            fig_ref, _ = plot_ref_ordination(
                xx, yy, gl, gp, cluster_ids,
                ref_pc=ref_pc,
                ref_clusters=ref_cls,
                ref_env_strength=ref_envs,
                title=f"{model_name} — Ref Sites ({pcx}\u2013{pcy})",
                **plot_kw,
            )
            save_figure(fig_ref, figures_dir / f"{tag}_ref_ordination",
                        formats=figure_formats, verbose=verbose)
            plt.close(fig_ref)

            # Non-ref projection
            _log(f"  Creating {pcx}-{pcy} non-ref projection ...")
            nonref_pc = scores.loc[nonref_mask_scores, [pcx, pcy]]
            nonref_preds = pred_all.loc[nonref_mask_scores]

            fig_nr, _ = plot_nonref_projection(
                xx, yy, gl, gp, cluster_ids,
                nonref_pc=nonref_pc,
                nonref_pred_clusters=nonref_preds,
                title=f"{model_name} — Non-Ref Sites ({pcx}\u2013{pcy})",
                **plot_kw,
            )
            save_figure(fig_nr, figures_dir / f"{tag}_nonref_projection",
                        formats=figure_formats, verbose=verbose)
            plt.close(fig_nr)

    # ─── 7. ANOVA on augmented clusters ──────────────────────────────
    _log("[7] Running ANOVA on augmented clusters ...")
    aug_labels = classified["Defined_Cluster"]

    # Env ANOVA
    env_anova = anova_table(
        env_all, aug_labels,
        variables=env_vars_present,
        transform="none",
        label_col="Variable",
    )
    save_table(env_anova, anova_dir / "anova_env",
               formats=table_formats, verbose=verbose)
    _log(f"    Env ANOVA: {len(env_vars_present)} variables")

    # Taxa ANOVA (same transform as Ward's)
    from ..models.clustering import TAXA_COLUMNS
    from ..core.transforms import (
        octave_transform,
        octave_to_chord,
        octave_to_hellinger,
        octave_to_log_chord,
        octave_to_relative_abundance,
    )

    transform_funcs = {
        "octave": octave_transform,
        "chord": octave_to_chord,
        "hellinger": octave_to_hellinger,
        "log_chord": octave_to_log_chord,
        "relative_abundance": octave_to_relative_abundance,
    }

    taxa_block = extract_block(data, "taxa", "raw")
    taxa_cols_present = [c for c in TAXA_COLUMNS if c in taxa_block.columns]
    taxa_allsites = taxa_block.loc[env_all.index, taxa_cols_present]

    if taxa_transform in transform_funcs:
        taxa_transformed = transform_funcs[taxa_transform](taxa_allsites)
    else:
        taxa_transformed = taxa_allsites

    taxa_anova = anova_table(
        taxa_transformed,
        aug_labels,
        variables=list(taxa_transformed.columns),
        transform="none",
        label_col="Taxon",
    )
    save_table(taxa_anova, anova_dir / "anova_taxa",
               formats=table_formats, verbose=verbose)
    _log(f"    Taxa ANOVA: {len(taxa_cols_present)} taxa ({taxa_transform} transform)")

    # ─── 8. Artifacts ────────────────────────────────────────────────
    _log("[8] Saving artifacts ...")
    with open(artifacts_dir / "pca_model.pkl", "wb") as f:
        pickle.dump({
            "pca": pca_result.pca,
            "scaler": pca_result.scaler,
            "env_columns": pca_result.env_columns,
        }, f)
    _log(f"  ✓ Saved: {artifacts_dir / 'pca_model.pkl'}")

    note = (
        "Finalized LDA — Posterior Heatmap Ordination\n"
        "=============================================\n\n"
        f"Model: {model_name}\n"
        f"Training subset: EnvStrong_TaxaStrong reference sites\n"
        f"N training sites: {n_train}\n"
        f"Training accuracy: {lda_fit.accuracy:.2%}\n"
        f"Reference accuracy: {n_correct}/{n_ref} ({100*n_correct/n_ref:.1f}%)\n\n"
        "Figures use LDA posterior probabilities as heatmap background.\n"
        "Higher posterior probability → darker colour for that cluster.\n"
        f"Grid resolution: {grid_resolution}×{grid_resolution}\n"
        "Non-varied PCs fixed at overall-site means.\n"
    )
    (artifacts_dir / "note.txt").write_text(note, encoding="utf-8")
    _log(f"  ✓ Saved: {artifacts_dir / 'note.txt'}")

    _log("\n" + "=" * 60)
    _log("  Finalized Model Pipeline Complete")
    _log("=" * 60)
    _log(f"  {model_name}: ref accuracy {n_correct}/{n_ref} "
         f"({100 * n_correct / n_ref:.1f}%)")
    _log(f"  Non-ref sites predicted: {len(nonref_sites)}")

    return {
        "pca_result": pca_result,
        "classified": classified,
        "grids": grids,
    }
