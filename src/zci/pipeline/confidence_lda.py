"""Stage 2 -- Confidence-Aware LDA Classification Pipeline.

Trains three LDA models with different reference-site subsets/weighting,
evaluates each on training data and held-out Uncertain sites, and produces
a cross-model comparison.

Output structure under ``output_dir``:
    ModelA_CoreOnly/        {tables, figures, artifacts}
    ModelB_CorePeripheral/  {tables, figures, artifacts}
    ModelC_Weighted/        {tables, figures, artifacts}
    ModelCompar/            {tables, figures}
"""

from __future__ import annotations

from pathlib import Path as _Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as _sklearn_cm

from ..core.confidence_lda import (
    compute_confidence_weights,
    cross_validate_subset,
    evaluate_sites,
    fit_lda_on_subset,
    fit_weighted_lda,
)
from ..core.lda import (
    build_allref_comparison_table,
    build_confusion_matrix_table,
    build_cv_comparison_table,
    _model_site_desc,
    wilks_lambda_importance,
)
from ..io.readers import extract_block, read_study_data
from ..io.writers import save_figure, save_table
from ..models.confidence_lda import (
    ConfidenceLDAComparison,
    ConfidenceLDAModelResult,
)
from ..viz.ordination_plots import save_env_pca_ordination


# ─── Unified 3-part confusion matrix builder ────────────────────────


def _build_unified_cm(
    cm_train: np.ndarray,
    cm_cv: np.ndarray,
    cm_heldout: np.ndarray | None,
    cluster_names: list[str],
    train_label: str,
    cv_label: str,
    heldout_label: str | None,
) -> pd.DataFrame:
    """Build a single table with three stacked confusion-matrix sections.

    Each section has its own column-header row followed by the CM body,
    so the layout is easy to read when opened in a spreadsheet.
    """
    sections: list[tuple[str, np.ndarray]] = [
        (train_label, cm_train),
        (cv_label, cm_cv),
    ]
    if cm_heldout is not None and heldout_label is not None:
        sections.append((heldout_label, cm_heldout))

    # Column names used for every CM section
    col_pct = "% Correct"
    col_clusters = [f"Cluster C{int(n.split()[-1])}" for n in cluster_names]
    all_cols = [col_pct] + col_clusters

    parts: list[pd.DataFrame] = []
    for idx, (label, cm) in enumerate(sections):
        cm_df = build_confusion_matrix_table(cm, cluster_names, note="")

        # Section header row: label sits in the index, data cells blank
        header_row = pd.DataFrame(
            [[""] * len(all_cols)], columns=all_cols, index=[label],
        )

        # Column-name echo row: column names repeated as data values
        colname_row = pd.DataFrame(
            [all_cols], columns=all_cols, index=[""],
        )

        if idx > 0:
            # blank separator between sections
            blank = pd.DataFrame(
                [[""] * len(all_cols)], columns=all_cols, index=[""],
            )
            parts.append(blank)

        parts.append(header_row)
        parts.append(colname_row)
        parts.append(cm_df)

    return pd.concat(parts)


# ─── Per-model training helper ───────────────────────────────────────


def _train_single_model(
    model_name: str,
    training_subset: str,
    env_train: pd.DataFrame,
    labels_train: pd.Series,
    status_train: pd.Series,
    env_heldout: pd.DataFrame | None,
    labels_heldout: pd.Series | None,
    status_heldout: pd.Series | None,
    *,
    train_site_desc: str = "",
    heldout_site_desc: str = "",
    sample_weights: pd.Series | None = None,
    standardize: bool = True,
    cv_folds: int = 5,
    cv_repeats: int = 10,
    random_state: int | None = 42,
    use_weighted_lr: bool = False,
    model_dir: _Path,
    env_short: list[str] | None = None,
    save_plots: bool = True,
    figure_formats: Sequence[str] = ("png",),
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> ConfidenceLDAModelResult:
    """Train, evaluate, and save outputs for a single model."""

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    tables = model_dir / "tables"
    figures = model_dir / "figures"
    artifacts = model_dir / "artifacts"

    # -- Fit ---------------------------------------------------------------
    _log(f"  [{model_name}] Fitting on {len(env_train)} sites ...")
    if use_weighted_lr and sample_weights is not None:
        lda_fit = fit_weighted_lda(
            env_train, labels_train, sample_weights,
            standardize=standardize,
        )
    else:
        lda_fit = fit_lda_on_subset(
            env_train, labels_train, standardize=standardize,
        )
    _log(f"  [{model_name}] Training accuracy = {lda_fit.accuracy:.2%}")

    # -- Wilks Lambda importance -------------------------------------------
    wilks = wilks_lambda_importance(lda_fit)
    _log(f"  [{model_name}] Wilks Λ = {wilks.wilks_lambda_full:.4f}")

    # -- Cross-validation --------------------------------------------------
    _log(f"  [{model_name}] Cross-validation ({cv_folds}-fold x {cv_repeats} repeats) ...")
    cv_acc, cv_std, cv_cm, cluster_names, fold_cms = cross_validate_subset(
        env_train, labels_train,
        sample_weights=sample_weights,
        standardize=standardize,
        n_folds=cv_folds,
        n_repeats=cv_repeats,
        random_state=random_state,
        use_weighted_lr=use_weighted_lr,
    )
    _log(f"  [{model_name}] CV accuracy = {cv_acc:.2%} ± {cv_std:.2%}")

    # Build an MCCVResult-like object for the model result
    from ..models.lda import MCCVResult
    mccv = MCCVResult(
        mean_accuracy=cv_acc,
        std_accuracy=cv_std,
        median_accuracy=cv_acc,  # approximate
        min_accuracy=cv_acc - 2 * cv_std,
        max_accuracy=cv_acc + 2 * cv_std,
        aggregate_confusion_matrix=cv_cm,
        avg_classification_report={},
        cluster_names=cluster_names,
        n_iterations=cv_folds * cv_repeats,
        test_size=1.0 / cv_folds,
        all_true_labels=[],
        all_predictions=[],
        per_fold_cms=fold_cms,
    )

    # -- Evaluate training sites -------------------------------------------
    train_eval = evaluate_sites(
        lda_fit, env_train, labels_train, status_train, role="Train",
    )

    # -- Evaluate held-out sites -------------------------------------------
    heldout_eval = None
    if env_heldout is not None and len(env_heldout) > 0:
        _log(f"  [{model_name}] Predicting {len(env_heldout)} held-out sites ...")
        heldout_eval = evaluate_sites(
            lda_fit, env_heldout, labels_heldout, status_heldout,
            role="Held-out",
        )

    # -- Build held-out confusion matrix -----------------------------------
    cm_heldout = None
    if heldout_eval is not None and len(heldout_eval) > 0:
        preds_ho = heldout_eval["Predicted_Cluster"].values.astype(int)
        trues_ho = heldout_eval["Original_Cluster"].values.astype(int)
        # Use the same label set as training for consistent CM shape
        lbl_ints = [int(n.split()[-1]) for n in lda_fit.cluster_names]
        cm_heldout = _sklearn_cm(trues_ho, preds_ho, labels=lbl_ints)

    # -- Save tables -------------------------------------------------------
    _log(f"  [{model_name}] Saving tables ...")

    # Unified 3-part confusion matrix
    n_train = len(env_train)
    train_desc = train_site_desc or training_subset
    ho_desc = heldout_site_desc or "Held-out"
    n_ho = len(env_heldout) if env_heldout is not None else 0

    unified_cm = _build_unified_cm(
        cm_train=lda_fit.confusion_matrix,
        cm_cv=cv_cm,
        cm_heldout=cm_heldout,
        cluster_names=lda_fit.cluster_names,
        train_label=(
            f"Part 1: Training Resubstitution — {train_desc}"
        ),
        cv_label=(
            f"Part 2: {cv_folds}-Fold x {cv_repeats} Cross-Validation — {train_desc}"
        ),
        heldout_label=(
            f"Part 3: Held-out Prediction — {ho_desc}"
        ) if cm_heldout is not None else None,
    )
    save_table(unified_cm, tables / "confusion_matrix_unified",
               formats=table_formats, verbose=verbose, header=False)

    # Per-site evaluation (training)
    save_table(train_eval, tables / "site_eval_train",
               formats=table_formats, verbose=verbose)

    # Per-site evaluation (held-out)
    if heldout_eval is not None and len(heldout_eval) > 0:
        save_table(heldout_eval, tables / "site_eval_heldout",
                   formats=table_formats, verbose=verbose)

    # Wilks variable importance
    save_table(wilks.variable_importance, tables / "wilks_importance",
               formats=table_formats, verbose=verbose)
    save_table(wilks.axes_summary, tables / "lda_axes_summary",
               formats=table_formats, verbose=verbose)

    # -- Save figures ------------------------------------------------------
    if save_plots:
        _log(f"  [{model_name}] Saving figures ...")
        env_names = env_short if env_short else list(env_train.columns)

        pred_train = pd.Series(
            lda_fit.predictions, index=env_train.index, name="Predicted",
        )
        save_env_pca_ordination(
            env_ref=env_train,
            true_labels=labels_train,
            predicted_labels=pred_train,
            output_path=figures / "env_pca_ordination.png",
            env_feature_names=env_names,
            title=f"{model_name}: PCA Ordination ({len(env_train)} sites)",
        )
        if verbose:
            print(f"  > Saved figure: {figures / 'env_pca_ordination.png'}")

    # -- Save artifacts ----------------------------------------------------
    artifacts.mkdir(parents=True, exist_ok=True)

    # Augmented site robustness (train + held-out)
    parts = [train_eval]
    if heldout_eval is not None and len(heldout_eval) > 0:
        parts.append(heldout_eval)
    combined_eval = pd.concat(parts)
    combined_eval.to_excel(artifacts / "site_robustness.xlsx")
    if verbose:
        print(f"  > Saved artifact: {artifacts / 'site_robustness.xlsx'}")

    # Weight vector (for Model C, but save for all to be uniform)
    if sample_weights is not None:
        w_df = sample_weights.loc[env_train.index].to_frame("Weight")
        w_df.to_excel(artifacts / "training_weights.xlsx")
        if verbose:
            print(f"  > Saved artifact: {artifacts / 'training_weights.xlsx'}")

    return ConfidenceLDAModelResult(
        model_name=model_name,
        training_subset=training_subset,
        n_train=len(env_train),
        lda_fit=lda_fit,
        wilks=wilks,
        mccv=mccv,
        train_eval=train_eval,
        heldout_eval=heldout_eval,
    )


# ─── Main pipeline entry point ──────────────────────────────────────


def confidence_lda_pipeline(
    data_path: str | _Path,
    stage1_artifact: str | _Path,
    output_dir: str | _Path,
    *,
    site_robustness: pd.DataFrame,
    env_variables: Sequence[str] | None = None,
    standardize_env: bool = True,
    cv_folds: int = 5,
    cv_repeats: int = 10,
    random_state: int | None = 42,
    save_plots: bool = True,
    figure_formats: Sequence[str] = ("png",),
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> ConfidenceLDAComparison:
    """Run the confidence-aware LDA comparison pipeline.

    Parameters
    ----------
    site_robustness : pd.DataFrame
        From WardsClustering: index=Site, columns include
        Original_Cluster, Silhouette, Own_Coassign, Margin, Status.
    """
    output_dir = _Path(output_dir)

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

    # ============================================================
    #  Step 0. Read data
    # ============================================================
    _log("=" * 60)
    _log("  Confidence-Aware LDA Pipeline")
    _log("=" * 60)

    _log("[0] Reading data ...")
    data = read_study_data(data_path)
    env_block = extract_block(data, "environmental", "raw")

    # ============================================================
    #  Step 1. Compute confidence weights
    # ============================================================
    _log("[1] Computing confidence weights w_i ...")
    weights = compute_confidence_weights(site_robustness, normalize=True)
    _log(f"    weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    _log(f"    zero-weight sites: {(weights == 0).sum()}")

    # ============================================================
    #  Step 2. Define training subsets
    # ============================================================
    _log("[2] Defining training subsets ...")
    labels_all = site_robustness["Original_Cluster"]
    status_all = site_robustness["Status"]

    core_mask = status_all == "Core"
    periph_mask = status_all == "Peripheral"
    uncertain_mask = status_all == "Uncertain"

    core_sites = labels_all.index[core_mask]
    cp_sites = labels_all.index[core_mask | periph_mask]
    uncertain_sites = labels_all.index[uncertain_mask]

    _log(f"    Core: {len(core_sites)}  |  Core+Peripheral: {len(cp_sites)}  |  Uncertain: {len(uncertain_sites)}")

    # Prepare env subsets (drop NaN rows)
    env_vars_present = [v for v in env_variables if v in env_block.columns]

    def _prepare_env(sites: pd.Index) -> pd.DataFrame:
        return env_block.loc[sites, env_vars_present].dropna()

    env_core = _prepare_env(core_sites)
    env_cp = _prepare_env(cp_sites)
    env_uncertain = _prepare_env(uncertain_sites) if len(uncertain_sites) > 0 else None

    # Held-out for Model A = Peripheral + Uncertain (everything non-Core)
    noncore_sites = labels_all.index[periph_mask | uncertain_mask]
    env_noncore = _prepare_env(noncore_sites) if len(noncore_sites) > 0 else None

    # Align labels/status to complete-case env
    labels_core = labels_all.loc[env_core.index]
    status_core = status_all.loc[env_core.index]
    labels_cp = labels_all.loc[env_cp.index]
    status_cp = status_all.loc[env_cp.index]
    labels_uncertain = labels_all.loc[env_uncertain.index] if env_uncertain is not None else None
    status_uncertain = status_all.loc[env_uncertain.index] if env_uncertain is not None else None
    labels_noncore = labels_all.loc[env_noncore.index] if env_noncore is not None else None
    status_noncore = status_all.loc[env_noncore.index] if env_noncore is not None else None
    weights_cp = weights.loc[env_cp.index]

    # All sites (Core + Peripheral + Uncertain)
    all_sites = labels_all.index
    env_all = _prepare_env(all_sites)
    labels_allsites = labels_all.loc[env_all.index]
    status_allsites = status_all.loc[env_all.index]

    # ============================================================
    #  Step 3. Train models
    # ============================================================
    _log("[3] Training models ...")
    models: Dict[str, ConfidenceLDAModelResult] = {}

    def _site_desc(status_series: pd.Series | None) -> str:
        """Build e.g. 'Core (n=35) + Peripheral (n=10) + Uncertain (n=5)'."""
        if status_series is None or len(status_series) == 0:
            return ""
        counts = status_series.value_counts()
        parts = []
        for s in ["Core", "Peripheral", "Uncertain"]:
            if s in counts.index:
                parts.append(f"{s} (n={counts[s]})")
        return " + ".join(parts) if parts else f"(n={len(status_series)})"

    # --- Model A: Core only ---
    _log("\n--- Model A: LDA on Core only ---")
    models["ModelA_CoreOnly"] = _train_single_model(
        model_name="Model A (Core Only)",
        training_subset="Core",
        env_train=env_core,
        labels_train=labels_core,
        status_train=status_core,
        env_heldout=env_noncore,
        labels_heldout=labels_noncore,
        status_heldout=status_noncore,
        train_site_desc=_site_desc(status_core),
        heldout_site_desc=_site_desc(status_noncore),
        standardize=standardize_env,
        cv_folds=cv_folds,
        cv_repeats=cv_repeats,
        random_state=random_state,
        model_dir=output_dir / "ModelA_CoreOnly",
        env_short=env_short,
        save_plots=save_plots,
        figure_formats=figure_formats,
        table_formats=table_formats,
        verbose=verbose,
    )

    # --- Model B: Core + Peripheral ---
    _log("\n--- Model B: LDA on Core + Peripheral ---")
    models["ModelB_CorePeripheral"] = _train_single_model(
        model_name="Model B (Core+Peripheral)",
        training_subset="Core+Peripheral",
        env_train=env_cp,
        labels_train=labels_cp,
        status_train=status_cp,
        env_heldout=env_uncertain,
        labels_heldout=labels_uncertain,
        status_heldout=status_uncertain,
        train_site_desc=_site_desc(status_cp),
        heldout_site_desc=_site_desc(status_uncertain),
        standardize=standardize_env,
        cv_folds=cv_folds,
        cv_repeats=cv_repeats,
        random_state=random_state,
        model_dir=output_dir / "ModelB_CorePeripheral",
        env_short=env_short,
        save_plots=save_plots,
        figure_formats=figure_formats,
        table_formats=table_formats,
        verbose=verbose,
    )

    # --- Model C: Weighted (Core + Peripheral) ---
    _log("\n--- Model C: Weighted LDA on Core + Peripheral ---")
    models["ModelC_Weighted"] = _train_single_model(
        model_name="Model C (Weighted)",
        training_subset="Core+Peripheral (weighted)",
        env_train=env_cp,
        labels_train=labels_cp,
        status_train=status_cp,
        env_heldout=env_uncertain,
        labels_heldout=labels_uncertain,
        status_heldout=status_uncertain,
        train_site_desc=_site_desc(status_cp),
        heldout_site_desc=_site_desc(status_uncertain),
        sample_weights=weights_cp,
        standardize=standardize_env,
        cv_folds=cv_folds,
        cv_repeats=cv_repeats,
        random_state=random_state,
        use_weighted_lr=True,
        model_dir=output_dir / "ModelC_Weighted",
        env_short=env_short,
        save_plots=save_plots,
        figure_formats=figure_formats,
        table_formats=table_formats,
        verbose=verbose,
    )

    # --- Model D: All sites (benchmark, no holdout) ---
    _log("\n--- Model D: LDA on All Sites (benchmark) ---")
    models["ModelD_AllSites"] = _train_single_model(
        model_name="Model D (All Sites)",
        training_subset="All (no holdout)",
        env_train=env_all,
        labels_train=labels_allsites,
        status_train=status_allsites,
        env_heldout=None,
        labels_heldout=None,
        status_heldout=None,
        train_site_desc=_site_desc(status_allsites),
        standardize=standardize_env,
        cv_folds=cv_folds,
        cv_repeats=cv_repeats,
        random_state=random_state,
        model_dir=output_dir / "ModelD_AllSites",
        env_short=env_short,
        save_plots=save_plots,
        figure_formats=figure_formats,
        table_formats=table_formats,
        verbose=verbose,
    )

    # ============================================================
    #  Step 6. Model comparison
    # ============================================================
    _log("\n" + "=" * 60)
    _log("  Model Comparison")
    _log("=" * 60)

    # -- Summary table (one row per model) ----------------------------
    summary_rows = []
    for key, m in models.items():
        row = {
            "Model": m.model_name,
            "Training Subset": m.training_subset,
            "N_train": m.n_train,
            "Train Accuracy": m.train_accuracy(),
            "CV Accuracy (mean)": m.cv_accuracy(),
            "CV Accuracy (std)": m.cv_accuracy_std(),
        }
        # Posterior sharpness on training data
        row["Train p_max (mean)"] = m.train_eval["p_max"].mean()
        row["Train p_max (std)"] = m.train_eval["p_max"].std()
        row["Train delta_p (mean)"] = m.train_eval["delta_p"].mean()
        row["Train delta_p (std)"] = m.train_eval["delta_p"].std()
        # Held-out sites
        if m.heldout_eval is not None and len(m.heldout_eval) > 0:
            row["Heldout p_max (mean)"] = m.heldout_eval["p_max"].mean()
            row["Heldout delta_p (mean)"] = m.heldout_eval["delta_p"].mean()
            n_ambig = (
                (m.heldout_eval["p_max"] < 0.6)
                & (m.heldout_eval["delta_p"] < 0.3)
            ).sum()
            row["Heldout N_ambiguous"] = int(n_ambig)
        summary_rows.append(row)

    summary_table = pd.DataFrame(summary_rows).set_index("Model")
    _log(summary_table.to_string())

    # -- Full comparison table (one row per site per model) -----------
    comp_parts = []
    for key, m in models.items():
        for eval_df in [m.train_eval, m.heldout_eval]:
            if eval_df is None or len(eval_df) == 0:
                continue
            part = eval_df.copy()
            part["Model"] = m.model_name
            comp_parts.append(part)
    comparison_table = pd.concat(comp_parts)
    comparison_table.index.name = "Site"

    # -- Save comparison outputs --------------------------------------
    _log("\n[7] Saving comparison outputs ...")
    compar_dir = output_dir / "ModelCompar"
    compar_tables = compar_dir / "tables"
    compar_figures = compar_dir / "figures"

    save_table(summary_table, compar_tables / "model_summary",
               formats=table_formats, verbose=verbose)
    save_table(comparison_table, compar_tables / "full_comparison",
               formats=table_formats, verbose=verbose)

    # Weight vector
    w_table = weights.to_frame("Weight")
    w_table["Status"] = status_all
    save_table(w_table, compar_tables / "confidence_weights",
               formats=table_formats, verbose=verbose)

    # -- Per-model CV confusion matrices with median % correct ----------
    _log("  Building CV confusion matrix comparison ...")
    _weighted_keys = {"ModelC_Weighted"}
    cv_sections = []
    for key, m in models.items():
        is_w = key in _weighted_keys
        header = _model_site_desc(m.train_eval, m.model_name, is_weighted=is_w)
        header += f"; {cv_folds}-fold x {cv_repeats} cross-validation"
        cv_sections.append({
            "header": header,
            "agg_cm": m.mccv.aggregate_confusion_matrix,
            "fold_cms": m.mccv.per_fold_cms,
            "cnames": m.mccv.cluster_names,
        })
    cv_combined = build_cv_comparison_table(cv_sections)
    save_table(cv_combined, compar_tables / "cv_confusion_matrices_combined",
               formats=table_formats, verbose=verbose)

    # -- Prediction on ALL reference sites -> confusion_matrix_all_refsites
    _log("  Predicting all reference sites with each model ...")
    allref_sections = []
    for key, m in models.items():
        ev = evaluate_sites(
            m.lda_fit, env_all, labels_allsites, status_allsites,
            role="AllRef",
        )
        preds = ev["Predicted_Cluster"].values.astype(int)
        trues = ev["Original_Cluster"].values.astype(int)
        unique_labels = sorted(set(trues) | set(preds))
        cnames = [f"Cluster {int(i)}" for i in unique_labels]
        cm_arr = _sklearn_cm(trues, preds, labels=unique_labels)

        is_w = key in _weighted_keys
        header = _model_site_desc(m.train_eval, m.model_name, is_weighted=is_w)

        if m.heldout_eval is not None and len(m.heldout_eval) > 0:
            ho = m.heldout_eval
            ho_acc = float(
                (ho["Predicted_Cluster"].astype(int)
                 == ho["Original_Cluster"].astype(int)).mean()
            )
            stats = {
                "n_heldout": len(ho),
                "accuracy": ho_acc,
                "pmax_mean": float(ho["p_max"].mean()),
                "deltap_mean": float(ho["delta_p"].mean()),
            }
        else:
            stats = {"n_heldout": 0, "accuracy": 0, "pmax_mean": 0, "deltap_mean": 0}

        allref_sections.append({
            "header": header,
            "cm": cm_arr,
            "cnames": cnames,
            "stats": stats,
        })
    allref_combined = build_allref_comparison_table(allref_sections)
    save_table(allref_combined, compar_tables / "confusion_matrix_all_refsites",
               formats=table_formats, verbose=verbose)

    # -- Comparison figure: posterior sharpness -------------------------
    if save_plots:
        _log("  Creating comparison figures ...")
        compar_figures.mkdir(parents=True, exist_ok=True)

        # Bar chart: CV accuracy comparison
        fig_acc, ax = plt.subplots(figsize=(8, 5))
        model_names = [m.model_name for m in models.values()]
        cv_means = [m.cv_accuracy() for m in models.values()]
        cv_stds = [m.cv_accuracy_std() for m in models.values()]
        x_pos = np.arange(len(model_names))
        bars = ax.bar(x_pos, cv_means, yerr=cv_stds, capsize=5,
                      color=["#1f77b4", "#ff7f0e", "#2ca02c"], alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, fontsize=9)
        ax.set_ylabel("CV Accuracy")
        ax.set_title("Cross-Validated Accuracy Comparison")
        ax.set_ylim(0, 1)
        for bar, m, s in zip(bars, cv_means, cv_stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.02,
                    f"{m:.1%}", ha="center", fontsize=9)
        fig_acc.tight_layout()
        save_figure(fig_acc, compar_figures / "cv_accuracy_comparison",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_acc)

        # Scatter: p_max vs delta_p per model (train sites)
        fig_post, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5),
                                      sharey=True, squeeze=False)
        for idx, (key, m) in enumerate(models.items()):
            ax = axes[0, idx]
            ev = m.train_eval
            for cid in sorted(ev["Original_Cluster"].unique()):
                mask = ev["Original_Cluster"] == cid
                correct = ev.loc[mask, "Predicted_Cluster"] == cid
                ax.scatter(
                    ev.loc[mask & correct, "delta_p"],
                    ev.loc[mask & correct, "p_max"],
                    label=f"C{int(cid)} correct", alpha=0.7, s=40,
                )
                ax.scatter(
                    ev.loc[mask & ~correct, "delta_p"],
                    ev.loc[mask & ~correct, "p_max"],
                    marker="x", s=50, linewidths=2,
                    label=f"C{int(cid)} misclass",
                )
            if m.heldout_eval is not None and len(m.heldout_eval) > 0:
                ue = m.heldout_eval
                ax.scatter(ue["delta_p"], ue["p_max"],
                           marker="D", s=40, c="grey", alpha=0.6,
                           label="Held-out")
            ax.set_xlabel("Posterior gap Δ")
            ax.set_ylabel("p_max" if idx == 0 else "")
            ax.set_title(m.model_name, fontsize=10)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(0.25, 1.05)
            ax.legend(fontsize=7, loc="lower right")
        fig_post.suptitle("Posterior Sharpness: p_max vs Δ", fontsize=12, y=1.02)
        fig_post.tight_layout()
        save_figure(fig_post, compar_figures / "posterior_sharpness",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig_post)

    # ============================================================
    _log("\n" + "=" * 60)
    _log("  Confidence-Aware LDA Pipeline Complete")
    _log("=" * 60)
    for key, m in models.items():
        _log(f"  {m.summary()}")

    return ConfidenceLDAComparison(
        models=models,
        weights=weights,
        comparison_table=comparison_table,
        summary_table=summary_table,
    )
