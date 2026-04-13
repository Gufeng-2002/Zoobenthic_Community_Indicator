"""Cross-support evaluation pipeline.

Trains LDA and MRT classifiers on EnvStrong_TaxaStrong reference sites,
evaluates predictions within each of the four TaxaEnv_Class cells, and
saves structured outputs into classifier-specific subfolders.

Outputs per classifier (LDA / MRT):
- confusion_allref.xlsx    — Model S vs Baseline all-ref confusion matrix
- cv_confusion.xlsx        — CV confusion matrix comparison
- nested_confusion_22combined.xlsx  — 2×2 per-class confusion (Model S)
- baseline_nested_confusion_22combined.xlsx — same layout for baseline model
- cross_support_site_predictions.xlsx

Public API
----------
cross_support_eval_pipeline
    End-to-end: train LDA + MRT on anchor sites, evaluate, save.
"""

from __future__ import annotations

from pathlib import Path as _Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix as sk_confusion_matrix

from ..core.lda import fit_lda, predict_sites
from ..core.cross_support import (
    evaluate_classifier_per_class,
)
from ..io.writers import save_table


# ------------------------------------------------------------------
# MRT helpers (lightweight, project-consistent)
# ------------------------------------------------------------------

def _fit_pruned_tree(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    minsplit: int = 3,
    minbucket: int = 2,
    k_folds: int = 5,
    cv_perms: int = 10,
    random_state: int | None = 42,
) -> DecisionTreeClassifier:
    """Fit a cost-complexity-pruned decision tree via repeated stratified CV."""
    rng = np.random.default_rng(random_state)

    # Full tree
    full = DecisionTreeClassifier(
        criterion="gini",
        min_samples_split=minsplit,
        min_samples_leaf=minbucket,
        ccp_alpha=0.0,
        random_state=random_state,
    )
    full.fit(X, y)
    path = full.cost_complexity_pruning_path(X, y)
    alphas = np.unique(np.r_[path.ccp_alphas, 0.0])

    # Evaluate each alpha via repeated stratified k-fold
    min_class = int(y.value_counts().min())
    eff_k = min(k_folds, min_class)
    if eff_k < 2:
        return full  # Can't cross-validate

    seeds = rng.integers(0, np.iinfo(np.int32).max, size=cv_perms)
    Xv = X.values
    yv = y.values.astype(int)

    best_alpha = 0.0
    best_cvre = np.inf
    for alpha in alphas:
        errors = []
        for seed in seeds:
            skf = StratifiedKFold(n_splits=eff_k, shuffle=True, random_state=int(seed))
            fold_err = 0
            for tr, te in skf.split(Xv, yv):
                m = DecisionTreeClassifier(
                    criterion="gini",
                    min_samples_split=minsplit,
                    min_samples_leaf=minbucket,
                    ccp_alpha=float(alpha),
                    random_state=random_state,
                )
                m.fit(Xv[tr], yv[tr])
                fold_err += int(np.count_nonzero(yv[te] != m.predict(Xv[te])))
            errors.append(fold_err / len(yv))
        mean_cvre = float(np.mean(errors))
        if mean_cvre < best_cvre:
            best_cvre = mean_cvre
            best_alpha = float(alpha)

    pruned = DecisionTreeClassifier(
        criterion="gini",
        min_samples_split=minsplit,
        min_samples_leaf=minbucket,
        ccp_alpha=best_alpha,
        random_state=random_state,
    )
    pruned.fit(X, y)
    return pruned


def _leaf_to_cluster_map(
    model: DecisionTreeClassifier,
    X: pd.DataFrame,
    true_labels: pd.Series,
) -> dict:
    leaves = model.apply(X.values)
    mapping = {}
    for leaf_id in np.unique(leaves):
        mask = leaves == leaf_id
        majority = int(true_labels.iloc[mask.nonzero()[0]].mode().iloc[0])
        mapping[int(leaf_id)] = majority
    return mapping


def _mrt_predict(
    model: DecisionTreeClassifier,
    X: pd.DataFrame,
    leaf_map: dict,
) -> tuple:
    """Return (predictions, probabilities) for MRT."""
    leaves = model.apply(X.values)
    preds = pd.Series(
        [leaf_map.get(int(l), -1) for l in leaves],
        index=X.index, name="Predicted_Cluster",
    )
    probs_raw = model.predict_proba(X.values)
    tree_classes = list(model.classes_)
    prob_df = pd.DataFrame(
        probs_raw, index=X.index,
        columns=[f"Cluster {int(c)}" for c in tree_classes],
    )
    return preds, prob_df


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

def cross_support_eval_pipeline(
    combined_table: pd.DataFrame,
    env_ref_complete: pd.DataFrame,
    labels_ref_complete: pd.Series,
    *,
    lda_output_dir: _Path | str,
    mrt_output_dir: _Path | str,
    env_variables: Sequence[str],
    mrt_minsplit: int = 3,
    mrt_minbucket: int = 2,
    mrt_k_folds: int = 5,
    mrt_cv_perms: int = 10,
    cv_folds: int = 5,
    cv_repeats: int = 10,
    random_state: int | None = 42,
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> Dict:
    """Train LDA + MRT on EnvStrong_TaxaStrong, evaluate on all 4 classes.

    Produces three output tables per classifier:
    - confusion_allref: Model S vs Baseline on all ref sites (Image 1)
    - cv_confusion: CV performance comparison (Image 2)
    - nested_confusion_22combined: 2×2 per-class confusion with accuracy rows
    - baseline_nested_confusion_22combined: same layout for baseline model
    """
    lda_dir = _Path(lda_output_dir) / "tables"
    mrt_dir = _Path(mrt_output_dir) / "tables"

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    all_clusters = sorted(labels_ref_complete.unique())
    env_all = env_ref_complete.loc[combined_table.index, list(env_variables)]

    # ── Identify training set (EnvStrong_TaxaStrong) ─────────────────
    train_mask = combined_table["TaxaEnv_Class"] == "EnvStrong_TaxaStrong"
    train_sites = combined_table.index[train_mask]
    n_train = len(train_sites)
    _log(f"  Cross-support training set: {n_train} sites (EnvStrong_TaxaStrong)")

    if n_train < 3:
        _log("  WARNING: Fewer than 3 training sites. Skipping classifier training.")
        return {"model_comparison": pd.DataFrame(), "lda_results": {},
                "mrt_results": {}, "lda_fit": None}

    env_train = env_ref_complete.loc[train_sites, list(env_variables)]
    labels_train = labels_ref_complete.loc[train_sites]

    n_unique_clusters_train = labels_train.nunique()
    _log(f"  Training clusters: {sorted(labels_train.unique())} "
         f"({n_unique_clusters_train} unique)")

    # ── LDA: Model S (filtered) + Baseline (all sites) ──────────────
    _log("\n  --- LDA ---")
    lda_fit = fit_lda(env_train, labels_train.values, standardize=True)
    _log(f"  Model S: trained on {n_train} EnvStrong_TaxaStrong sites, "
         f"train_acc={lda_fit.accuracy:.2%}")

    lda_baseline = fit_lda(env_all, labels_ref_complete.values, standardize=True)
    _log(f"  Baseline: trained on {len(env_all)} all ref sites, "
         f"train_acc={lda_baseline.accuracy:.2%}")

    # Predict all ref sites
    lda_preds_s, lda_probs_s = predict_sites(lda_fit, env_all)
    lda_preds_b, lda_probs_b = predict_sites(lda_baseline, env_all)

    lda_acc_s = accuracy_score(labels_ref_complete.loc[combined_table.index], lda_preds_s)
    lda_acc_b = accuracy_score(labels_ref_complete.loc[combined_table.index], lda_preds_b)
    _log(f"  Model S all-ref acc: {lda_acc_s:.2%}")
    _log(f"  Baseline all-ref acc: {lda_acc_b:.2%}")

    # Per-class evaluation for Model S
    lda_class_results = evaluate_classifier_per_class(
        combined_table, lda_preds_s, lda_probs_s,
    )
    for cls, info in lda_class_results.items():
        if info["n_sites"] > 0:
            _log(f"    {cls}: n={info['n_sites']}, acc={info['accuracy']:.2%}")

    # Build all-ref confusion table (Image 1 format)
    lda_allref = _build_allref_confusion(
        labels_ref_complete.loc[combined_table.index],
        lda_preds_s, lda_preds_b,
        all_clusters, n_train, len(env_all),
    )
    save_table(lda_allref, lda_dir / "confusion_allref",
               formats=table_formats, verbose=verbose)

    # Build CV confusion table (Image 2 format)
    lda_cv = _build_cv_confusion_lda(
        env_train, labels_train, env_all, labels_ref_complete.loc[combined_table.index],
        all_clusters, n_train, len(env_all),
        cv_folds=cv_folds, cv_repeats=cv_repeats, random_state=random_state,
    )
    save_table(lda_cv, lda_dir / "cv_confusion",
               formats=table_formats, verbose=verbose)

    # Build nested 2×2 confusion (Model S)
    lda_combined = _build_improved_confusion_combined(lda_class_results)
    if not lda_combined.empty:
        save_table(lda_combined, lda_dir / "nested_confusion_22combined",
                   formats=table_formats, verbose=verbose)

    # Build nested 2×2 confusion (Baseline — trained on all sites)
    lda_baseline_class_results = evaluate_classifier_per_class(
        combined_table, lda_preds_b, lda_probs_b,
    )
    lda_baseline_combined = _build_improved_confusion_combined(lda_baseline_class_results)
    if not lda_baseline_combined.empty:
        save_table(lda_baseline_combined, lda_dir / "baseline_nested_confusion_22combined",
                   formats=table_formats, verbose=verbose)

    # Save site-level predictions
    lda_site_df = combined_table[["Original_Cluster", "TaxaEnv_Class"]].copy()
    lda_site_df["LDA_Predicted"] = lda_preds_s
    lda_site_df["LDA_Correct"] = (
        lda_site_df["Original_Cluster"] == lda_site_df["LDA_Predicted"]
    )
    for col in lda_probs_s.columns:
        lda_site_df[f"LDA_{col}"] = lda_probs_s[col]
    save_table(lda_site_df, lda_dir / "cross_support_site_predictions",
               formats=table_formats, verbose=verbose)

    # ── MRT: Model S (filtered) + Baseline (all sites) ──────────────
    _log("\n  --- MRT ---")
    mrt_model = _fit_pruned_tree(
        env_train, labels_train,
        minsplit=mrt_minsplit, minbucket=mrt_minbucket,
        k_folds=mrt_k_folds, cv_perms=mrt_cv_perms,
        random_state=random_state,
    )
    _log(f"  Model S: {mrt_model.get_n_leaves()} leaves, "
         f"trained on {n_train} EnvStrong_TaxaStrong sites")

    mrt_baseline = _fit_pruned_tree(
        env_all, labels_ref_complete.loc[combined_table.index],
        minsplit=mrt_minsplit, minbucket=mrt_minbucket,
        k_folds=mrt_k_folds, cv_perms=mrt_cv_perms,
        random_state=random_state,
    )
    _log(f"  Baseline: {mrt_baseline.get_n_leaves()} leaves, "
         f"trained on {len(env_all)} all ref sites")

    leaf_map_s = _leaf_to_cluster_map(mrt_model, env_train, labels_train)
    mrt_preds_s, mrt_probs_s = _mrt_predict(mrt_model, env_all, leaf_map_s)

    leaf_map_b = _leaf_to_cluster_map(
        mrt_baseline, env_all, labels_ref_complete.loc[combined_table.index],
    )
    mrt_preds_b, mrt_probs_b = _mrt_predict(mrt_baseline, env_all, leaf_map_b)

    mrt_acc_s = accuracy_score(labels_ref_complete.loc[combined_table.index], mrt_preds_s)
    mrt_acc_b = accuracy_score(labels_ref_complete.loc[combined_table.index], mrt_preds_b)
    _log(f"  Model S all-ref acc: {mrt_acc_s:.2%}")
    _log(f"  Baseline all-ref acc: {mrt_acc_b:.2%}")

    # Per-class evaluation for Model S
    mrt_class_results = evaluate_classifier_per_class(
        combined_table, mrt_preds_s, mrt_probs_s,
    )
    for cls, info in mrt_class_results.items():
        if info["n_sites"] > 0:
            _log(f"    {cls}: n={info['n_sites']}, acc={info['accuracy']:.2%}")

    # Build all-ref confusion table (Image 1 format)
    mrt_allref = _build_allref_confusion(
        labels_ref_complete.loc[combined_table.index],
        mrt_preds_s, mrt_preds_b,
        all_clusters, n_train, len(env_all),
    )
    save_table(mrt_allref, mrt_dir / "confusion_allref",
               formats=table_formats, verbose=verbose)

    # Build CV confusion table (Image 2 format)
    mrt_cv = _build_cv_confusion_mrt(
        env_train, labels_train, env_all, labels_ref_complete.loc[combined_table.index],
        all_clusters, n_train, len(env_all),
        minsplit=mrt_minsplit, minbucket=mrt_minbucket,
        k_folds=mrt_k_folds, cv_perms=mrt_cv_perms,
        random_state=random_state,
    )
    save_table(mrt_cv, mrt_dir / "cv_confusion",
               formats=table_formats, verbose=verbose)

    # Build nested 2×2 confusion (Model S)
    mrt_combined = _build_improved_confusion_combined(mrt_class_results)
    if not mrt_combined.empty:
        save_table(mrt_combined, mrt_dir / "nested_confusion_22combined",
                   formats=table_formats, verbose=verbose)

    # Build nested 2×2 confusion (Baseline — trained on all sites)
    mrt_baseline_class_results = evaluate_classifier_per_class(
        combined_table, mrt_preds_b, mrt_probs_b,
    )
    mrt_baseline_combined = _build_improved_confusion_combined(mrt_baseline_class_results)
    if not mrt_baseline_combined.empty:
        save_table(mrt_baseline_combined, mrt_dir / "baseline_nested_confusion_22combined",
                   formats=table_formats, verbose=verbose)

    # Save site-level predictions
    mrt_site_df = combined_table[["Original_Cluster", "TaxaEnv_Class"]].copy()
    mrt_site_df["MRT_Predicted"] = mrt_preds_s
    mrt_site_df["MRT_Correct"] = (
        mrt_site_df["Original_Cluster"] == mrt_site_df["MRT_Predicted"]
    )
    for col in mrt_probs_s.columns:
        mrt_site_df[f"MRT_{col}"] = mrt_probs_s[col]
    save_table(mrt_site_df, mrt_dir / "cross_support_site_predictions",
               formats=table_formats, verbose=verbose)

    _log("\n  --- Summary ---")
    _log(f"  LDA Model S: {lda_acc_s:.2%} | Baseline: {lda_acc_b:.2%}")
    _log(f"  MRT Model S: {mrt_acc_s:.2%} | Baseline: {mrt_acc_b:.2%}")

    return {
        "lda_results": lda_class_results,
        "mrt_results": mrt_class_results,
        "lda_fit": lda_fit,
    }


# ------------------------------------------------------------------
# Table builders
# ------------------------------------------------------------------

def _build_allref_confusion(
    y_true: pd.Series,
    preds_model_s: pd.Series,
    preds_baseline: pd.Series,
    all_clusters: list,
    n_train_s: int,
    n_train_b: int,
) -> pd.DataFrame:
    """Build Image-1-style all-ref confusion table.

    Format:
      Model S (trained on N sites)
        Cluster C1   | %correct | pred_C1 | pred_C2 | pred_C3
        ...
        Total         | %correct | sum     | sum     | sum
      Baseline (trained on M sites)
        ...
    """
    cluster_cols = [f"Cluster C{c}" for c in all_clusters]
    rows = []

    for label, preds, n_tr in [
        (f"Model S (trained on {n_train_s} EnvStrong_TaxaStrong sites)", preds_model_s, n_train_s),
        (f"Baseline (trained on all {n_train_b} ref sites)", preds_baseline, n_train_b),
    ]:
        # Header row
        rows.append({"": label, "% Correct": "", **{c: "" for c in cluster_cols}})

        # Per-cluster rows
        total_correct = 0
        total_n = 0
        for c in all_clusters:
            mask = y_true == c
            n_c = mask.sum()
            if n_c == 0:
                rows.append({"": f"Cluster C{c}", "% Correct": 0,
                             **{f"Cluster C{cc}": 0 for cc in all_clusters}})
                continue
            c_preds = preds.loc[mask]
            correct = int((c_preds == c).sum())
            pct = round(100 * correct / n_c)
            total_correct += correct
            total_n += n_c
            cm_row = {}
            for cc in all_clusters:
                cm_row[f"Cluster C{cc}"] = int((c_preds == cc).sum())
            rows.append({"": f"Cluster C{c}", "% Correct": pct, **cm_row})

        # Total row
        total_pct = round(100 * total_correct / total_n) if total_n > 0 else 0
        total_cm = {}
        for cc in all_clusters:
            total_cm[f"Cluster C{cc}"] = int((preds == cc).sum())
        rows.append({"": "Total", "% Correct": total_pct, **total_cm})

        # Blank separator
        rows.append({"": "", "% Correct": "", **{c: "" for c in cluster_cols}})

    df = pd.DataFrame(rows).set_index("")
    # Remove trailing blank row
    if df.index[-1] == "":
        df = df.iloc[:-1]
    return df


def _cv_lda(X_train, y_train, X_test, standardize=True):
    """Fit LDA on train, predict test — returns predictions."""
    fit = fit_lda(X_train, y_train.values, standardize=standardize)
    preds, _ = predict_sites(fit, X_test)
    return preds


def _build_cv_confusion_lda(
    env_train_s, labels_train_s,
    env_all, labels_all,
    all_clusters, n_train_s, n_train_b,
    cv_folds=5, cv_repeats=10, random_state=42,
) -> pd.DataFrame:
    """Build Image-2-style CV confusion table for LDA.

    Runs repeated stratified k-fold CV for both Model S and Baseline,
    accumulates predictions, and presents totals.
    """
    rng = np.random.default_rng(random_state)
    cluster_cols = [f"Cluster C{c}" for c in all_clusters]

    # -- Model S CV (on its training data: EnvStrong_TaxaStrong) --
    s_true_all, s_pred_all = [], []
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=cv_repeats)
    min_class_s = int(labels_train_s.value_counts().min())
    eff_k_s = min(cv_folds, min_class_s)

    if eff_k_s >= 2:
        for seed in seeds:
            skf = StratifiedKFold(n_splits=eff_k_s, shuffle=True, random_state=int(seed))
            for tr_idx, te_idx in skf.split(env_train_s.values, labels_train_s.values):
                X_tr = env_train_s.iloc[tr_idx]
                y_tr = labels_train_s.iloc[tr_idx]
                X_te = env_train_s.iloc[te_idx]
                y_te = labels_train_s.iloc[te_idx]
                preds = _cv_lda(X_tr, y_tr, X_te)
                s_true_all.extend(y_te.values)
                s_pred_all.extend(preds.values)

    # -- Baseline CV (on all ref sites) --
    b_true_all, b_pred_all = [], []
    seeds_b = rng.integers(0, np.iinfo(np.int32).max, size=cv_repeats)
    min_class_b = int(labels_all.value_counts().min())
    eff_k_b = min(cv_folds, min_class_b)

    if eff_k_b >= 2:
        for seed in seeds_b:
            skf = StratifiedKFold(n_splits=eff_k_b, shuffle=True, random_state=int(seed))
            for tr_idx, te_idx in skf.split(env_all.values, labels_all.values):
                X_tr = env_all.iloc[tr_idx]
                y_tr = labels_all.iloc[tr_idx]
                X_te = env_all.iloc[te_idx]
                y_te = labels_all.iloc[te_idx]
                preds = _cv_lda(X_tr, y_tr, X_te)
                b_true_all.extend(y_te.values)
                b_pred_all.extend(preds.values)

    return _format_cv_table(
        s_true_all, s_pred_all, b_true_all, b_pred_all,
        all_clusters, cluster_cols, n_train_s, n_train_b,
    )


def _build_cv_confusion_mrt(
    env_train_s, labels_train_s,
    env_all, labels_all,
    all_clusters, n_train_s, n_train_b,
    minsplit=3, minbucket=2, k_folds=5, cv_perms=10,
    random_state=42,
) -> pd.DataFrame:
    """Build Image-2-style CV confusion table for MRT."""
    rng = np.random.default_rng(random_state)
    cluster_cols = [f"Cluster C{c}" for c in all_clusters]

    # -- Model S CV --
    s_true_all, s_pred_all = [], []
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=cv_perms)
    min_class_s = int(labels_train_s.value_counts().min())
    eff_k_s = min(k_folds, min_class_s)

    if eff_k_s >= 2:
        for seed in seeds:
            skf = StratifiedKFold(n_splits=eff_k_s, shuffle=True, random_state=int(seed))
            for tr_idx, te_idx in skf.split(env_train_s.values, labels_train_s.values):
                X_tr = env_train_s.iloc[tr_idx]
                y_tr = labels_train_s.iloc[tr_idx]
                X_te = env_train_s.iloc[te_idx]
                y_te = labels_train_s.iloc[te_idx]
                tree = _fit_pruned_tree(X_tr, y_tr, minsplit=minsplit,
                                        minbucket=minbucket, k_folds=max(2, eff_k_s - 1),
                                        cv_perms=3, random_state=int(seed))
                lmap = _leaf_to_cluster_map(tree, X_tr, y_tr)
                preds, _ = _mrt_predict(tree, X_te, lmap)
                s_true_all.extend(y_te.values)
                s_pred_all.extend(preds.values)

    # -- Baseline CV --
    b_true_all, b_pred_all = [], []
    seeds_b = rng.integers(0, np.iinfo(np.int32).max, size=cv_perms)
    min_class_b = int(labels_all.value_counts().min())
    eff_k_b = min(k_folds, min_class_b)

    if eff_k_b >= 2:
        for seed in seeds_b:
            skf = StratifiedKFold(n_splits=eff_k_b, shuffle=True, random_state=int(seed))
            for tr_idx, te_idx in skf.split(env_all.values, labels_all.values):
                X_tr = env_all.iloc[tr_idx]
                y_tr = labels_all.iloc[tr_idx]
                X_te = env_all.iloc[te_idx]
                y_te = labels_all.iloc[te_idx]
                tree = _fit_pruned_tree(X_tr, y_tr, minsplit=minsplit,
                                        minbucket=minbucket, k_folds=max(2, eff_k_b - 1),
                                        cv_perms=3, random_state=int(seed))
                lmap = _leaf_to_cluster_map(tree, X_tr, y_tr)
                preds, _ = _mrt_predict(tree, X_te, lmap)
                b_true_all.extend(y_te.values)
                b_pred_all.extend(preds.values)

    return _format_cv_table(
        s_true_all, s_pred_all, b_true_all, b_pred_all,
        all_clusters, cluster_cols, n_train_s, n_train_b,
    )


def _format_cv_table(
    s_true, s_pred, b_true, b_pred,
    all_clusters, cluster_cols, n_train_s, n_train_b,
) -> pd.DataFrame:
    """Shared formatter for CV confusion tables (Image 2 style)."""
    rows = []
    s_true = np.array(s_true)
    s_pred = np.array(s_pred)
    b_true = np.array(b_true)
    b_pred = np.array(b_pred)

    for label, y_t, y_p, n_tr in [
        (f"Model S (CV on {n_train_s} EnvStrong_TaxaStrong sites)", s_true, s_pred, n_train_s),
        (f"Baseline (CV on all {n_train_b} ref sites)", b_true, b_pred, n_train_b),
    ]:
        rows.append({"": label, "% Correct": "", **{c: "" for c in cluster_cols}})

        total_correct = 0
        total_n = 0
        for c in all_clusters:
            mask = y_t == c
            n_c = mask.sum()
            if n_c == 0:
                rows.append({"": f"Cluster C{c}", "% Correct": 0,
                             **{f"Cluster C{cc}": 0 for cc in all_clusters}})
                continue
            c_pred = y_p[mask]
            correct = int((c_pred == c).sum())
            pct = round(100 * correct / n_c)
            total_correct += correct
            total_n += n_c
            cm_row = {}
            for cc in all_clusters:
                cm_row[f"Cluster C{cc}"] = int((c_pred == cc).sum())
            rows.append({"": f"Cluster C{c}", "% Correct": pct, **cm_row})

        total_pct = round(100 * total_correct / total_n) if total_n > 0 else 0
        total_cm = {}
        for cc in all_clusters:
            total_cm[f"Cluster C{cc}"] = int((y_p == cc).sum())
        rows.append({"": "Total", "% Correct": total_pct, **total_cm})

        rows.append({"": "", "% Correct": "", **{c: "" for c in cluster_cols}})

    df = pd.DataFrame(rows).set_index("")
    if df.index[-1] == "":
        df = df.iloc[:-1]
    return df


def _build_improved_confusion_combined(
    class_results: Dict[str, Dict],
) -> pd.DataFrame:
    """Build a compact 2×2 confusion combined table with accuracy rows.

    Layout: single row index 'Env_Strength' spans across two column groups
    'Taxa Strong' and 'Taxa Weak', with no blank columns between them.
    Each sub-confusion-matrix has an accuracy summary row appended.
    """
    env_levels = ["Strong", "Weak"]
    taxa_levels = ["Strong", "Weak"]
    all_clusters_set = set()
    for info in class_results.values():
        cm = info.get("confusion_matrix", pd.DataFrame())
        if not cm.empty:
            all_clusters_set.update(cm.index)
    if not all_clusters_set:
        return pd.DataFrame()

    # Determine the inner labels (True_1, True_2, ... / Pred_1, Pred_2, ...)
    sample_cm = None
    for info in class_results.values():
        cm = info.get("confusion_matrix", pd.DataFrame())
        if not cm.empty:
            sample_cm = cm
            break
    if sample_cm is None:
        return pd.DataFrame()

    inner_rows = list(sample_cm.index)   # e.g. ["True_1", "True_2", "True_3"]
    inner_cols = list(sample_cm.columns)  # e.g. ["Pred_1", "Pred_2", "Pred_3"]

    # Build column MultiIndex: (Taxa Strong, Pred_1), ... (Taxa Weak, Pred_1), ...
    col_tuples = []
    for taxa_lbl in taxa_levels:
        for pc in inner_cols:
            col_tuples.append((f"Taxa {taxa_lbl}", pc))
    col_mi = pd.MultiIndex.from_tuples(col_tuples, names=["Taxa_Strength", "Pred_Label"])

    all_rows = []
    row_index = []

    for env_lbl in env_levels:
        # For each env level, build rows: one per true cluster + accuracy row
        # Each row spans both taxa columns
        for ir in inner_rows:
            row_data = []
            for taxa_lbl in taxa_levels:
                cls_key = f"Env{env_lbl}_Taxa{taxa_lbl}"
                info = class_results.get(cls_key, {})
                cm = info.get("confusion_matrix", pd.DataFrame())
                if cm.empty or ir not in cm.index:
                    row_data.extend([0] * len(inner_cols))
                else:
                    for pc in inner_cols:
                        row_data.append(int(cm.loc[ir, pc]) if pc in cm.columns else 0)
            all_rows.append(row_data)
            row_index.append((f"Env {env_lbl}", ir))

        # Accuracy row for this env level
        acc_row = []
        for taxa_lbl in taxa_levels:
            cls_key = f"Env{env_lbl}_Taxa{taxa_lbl}"
            info = class_results.get(cls_key, {})
            acc = info.get("accuracy", np.nan)
            n = info.get("n_sites", 0)
            if n > 0 and not np.isnan(acc):
                acc_str = f"{acc:.0%} (n={n})"
            else:
                acc_str = "n=0"
            # Put the accuracy string in the first pred column, blanks in rest
            acc_row.append(acc_str)
            acc_row.extend([""] * (len(inner_cols) - 1))
        all_rows.append(acc_row)
        row_index.append((f"Env {env_lbl}", "Accuracy"))

    row_mi = pd.MultiIndex.from_tuples(row_index, names=["Env_Strength", "True_Label"])
    df = pd.DataFrame(all_rows, index=row_mi, columns=col_mi)
    return df
