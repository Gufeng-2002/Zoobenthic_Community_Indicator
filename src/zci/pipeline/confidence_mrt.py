"""Stage 2 -- Confidence-Aware MRT Classification Pipeline.

Trains four MRT (classification-tree) models with different reference-site
subsets / weighting, evaluates each via CVRE and prediction on held-out
sites, and produces cross-model comparison confusion matrices.

Output structure under ``output_dir``:
    ModelA_CoreOnly/        {tables, figures, artifacts}
    ModelB_CorePeripheral/  {tables, figures, artifacts}
    ModelC_Weighted/        {tables, figures, artifacts}
    ModelD_AllSites/        {tables, figures, artifacts}
    ModelCompar/            {tables, figures}
"""

from __future__ import annotations

from pathlib import Path as _Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as _sklearn_cm
from sklearn.tree import DecisionTreeClassifier

from ..core.confidence_lda import compute_confidence_weights
from ..core.lda import (
    build_allref_comparison_table,
    build_confusion_matrix_table,
    build_cv_comparison_table,
    _model_site_desc,
)
from ..core.mrt import fit_mrt
from ..core.transforms import (
    octave_to_chord,
    octave_to_hellinger,
    octave_to_log_chord,
    octave_to_relative_abundance,
    octave_transform,
)
from ..io.readers import extract_block, read_study_data
from ..io.writers import save_figure, save_table
from ..models.clustering import TAXA_COLUMNS
from ..models.mrt import MRTResult
from ..viz.mrt_plots import save_mrt_cp_tree_figure
from ..viz.ordination_plots import save_env_pca_ordination


_TRANSFORMS = {
    "octave": octave_transform,
    "relative_abundance": octave_to_relative_abundance,
    "chord": octave_to_chord,
    "hellinger": octave_to_hellinger,
    "log_chord": octave_to_log_chord,
}


# ─── Weighted tree helpers ───────────────────────────────────────────

def _fit_weighted_tree(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    ccp_alpha: float,
    minsplit: int,
    minbucket: int,
    random_state: int | None,
) -> DecisionTreeClassifier:
    """Fit a classification tree with sample weights."""
    model = DecisionTreeClassifier(
        criterion="gini",
        random_state=random_state,
        min_samples_split=minsplit,
        min_samples_leaf=minbucket,
        ccp_alpha=float(ccp_alpha),
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def _weighted_cvre(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    ccp_alpha: float,
    root_error: float,
    k_folds: int,
    cv_perms: int,
    minsplit: int,
    minbucket: int,
    random_state: int | None,
) -> np.ndarray:
    """Repeated stratified k-fold CVRE with sample weights."""
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(random_state)
    seeds = (
        rng.integers(0, np.iinfo(np.int32).max, size=cv_perms)
        if random_state is not None else np.repeat(None, cv_perms)
    )
    cv_errors: list[float] = []
    for seed in seeds:
        splitter = StratifiedKFold(
            n_splits=k_folds, shuffle=True,
            random_state=None if seed is None else int(seed),
        )
        fold_errors = 0
        for train_idx, test_idx in splitter.split(X, y):
            model = _fit_weighted_tree(
                X[train_idx], y[train_idx], sample_weight[train_idx],
                ccp_alpha=ccp_alpha, minsplit=minsplit,
                minbucket=minbucket, random_state=random_state,
            )
            preds = model.predict(X[test_idx])
            fold_errors += int(np.count_nonzero(y[test_idx] != preds))
        cv_errors.append(fold_errors / root_error)
    return np.asarray(cv_errors, dtype=float)


def _fit_weighted_mrt(
    env_df: pd.DataFrame,
    cluster_labels: pd.Series,
    sample_weights: pd.Series,
    *,
    k_folds: int,
    cv_perms: int,
    minsplit: int,
    minbucket: int,
    random_state: int | None,
) -> tuple[DecisionTreeClassifier, pd.DataFrame, float, float, float, int, int]:
    """Fit a weighted pruned tree, returning (model, cp_table, best_cp, min_cvre, se, nsplits, nleaves)."""
    from ..core.mrt import _root_node_error

    aligned = cluster_labels.loc[env_df.index].astype(int)
    X = env_df.astype(float).values
    y = aligned.values
    w = sample_weights.loc[env_df.index].values

    min_class_size = int(pd.Series(y).value_counts().min())
    effective_k = min(k_folds, min_class_size)
    if effective_k < 2:
        raise ValueError("At least 2 samples per cluster for CV")

    root_error = _root_node_error(y)

    # Full tree to get pruning path
    full = _fit_weighted_tree(X, y, w, ccp_alpha=0.0,
                              minsplit=minsplit, minbucket=minbucket,
                              random_state=random_state)
    path = full.cost_complexity_pruning_path(X, y)
    alphas = np.unique(np.r_[path.ccp_alphas, 0.0])

    candidates: dict[int, dict] = {}
    models: dict[int, DecisionTreeClassifier] = {}
    for alpha in alphas:
        m = _fit_weighted_tree(X, y, w, ccp_alpha=float(alpha),
                               minsplit=minsplit, minbucket=minbucket,
                               random_state=random_state)
        nsplit = int(m.get_n_leaves() - 1)
        if nsplit in candidates:
            continue
        train_err = int(np.count_nonzero(y != m.predict(X))) / root_error
        cv_errs = _weighted_cvre(
            X, y, w, ccp_alpha=float(alpha), root_error=root_error,
            k_folds=effective_k, cv_perms=cv_perms,
            minsplit=minsplit, minbucket=minbucket, random_state=random_state,
        )
        candidates[nsplit] = {
            "CP": float(alpha), "nsplit": nsplit,
            "rel error": float(train_err),
            "CV error": float(cv_errs.mean()),
            "CV std": float(cv_errs.std(ddof=1) / np.sqrt(len(cv_errs))) if len(cv_errs) > 1 else 0.0,
        }
        models[nsplit] = m

    cp_table = pd.DataFrame(candidates.values()).sort_values("nsplit").reset_index(drop=True)
    cp_table.index = np.arange(1, len(cp_table) + 1)

    best_pos = int(cp_table["CV error"].to_numpy(float).argmin())
    best_row = cp_table.iloc[best_pos]
    best_nsplit = int(best_row["nsplit"])
    best_tree = models[best_nsplit]

    return (
        best_tree, cp_table,
        float(best_row["CP"]),
        float(best_row["CV error"]),
        float(best_row["CV std"]),
        best_nsplit,
        int(best_tree.get_n_leaves()),
    )


# ─── Majority-vote leaf → cluster mapping ────────────────────────────

def _leaf_to_cluster_map(
    model: DecisionTreeClassifier,
    X: pd.DataFrame,
    true_labels: pd.Series,
) -> dict[int, int]:
    """Map each leaf node to the majority Ward cluster label."""
    leaves = model.apply(X.values)
    mapping: dict[int, int] = {}
    for leaf_id in np.unique(leaves):
        mask = leaves == leaf_id
        majority = int(true_labels.iloc[mask.nonzero()[0]].mode().iloc[0])
        mapping[int(leaf_id)] = majority
    return mapping


def _predict_with_mapping(
    model: DecisionTreeClassifier,
    X: pd.DataFrame,
    leaf_cluster_map: dict[int, int],
) -> pd.Series:
    """Predict cluster label for arbitrary sites via leaf→cluster mapping."""
    leaves = model.apply(X.values)
    preds = pd.Series(
        [leaf_cluster_map.get(int(l), -1) for l in leaves],
        index=X.index, name="Predicted_Cluster",
    )
    return preds


# ─── CV confusion matrix for MRT ─────────────────────────────────────

def _mrt_cv_confusion_matrix(
    env_df: pd.DataFrame,
    labels: pd.Series,
    sample_weights: pd.Series | None,
    *,
    ccp_alpha: float,
    k_folds: int,
    cv_perms: int,
    minsplit: int,
    minbucket: int,
    random_state: int | None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Repeated stratified k-fold CV → (aggregate confusion matrix, per-fold CMs)."""
    from sklearn.model_selection import StratifiedKFold

    X = env_df.values
    y = labels.loc[env_df.index].values.astype(int)
    w = sample_weights.loc[env_df.index].values if sample_weights is not None else None

    unique_labels = sorted(np.unique(y))
    agg_cm = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
    fold_cms: list[np.ndarray] = []

    min_class = int(pd.Series(y).value_counts().min())
    eff_k = min(k_folds, min_class)
    if eff_k < 2:
        return agg_cm, fold_cms

    rng = np.random.default_rng(random_state)
    seeds = (
        rng.integers(0, np.iinfo(np.int32).max, size=cv_perms)
        if random_state is not None else [None] * cv_perms
    )

    for seed in seeds:
        splitter = StratifiedKFold(
            n_splits=eff_k, shuffle=True,
            random_state=None if seed is None else int(seed),
        )
        for train_idx, test_idx in splitter.split(X, y):
            tree = DecisionTreeClassifier(
                criterion="gini", random_state=random_state,
                min_samples_split=minsplit, min_samples_leaf=minbucket,
                ccp_alpha=float(ccp_alpha),
            )
            if w is not None:
                tree.fit(X[train_idx], y[train_idx], sample_weight=w[train_idx])
            else:
                tree.fit(X[train_idx], y[train_idx])
            preds = tree.predict(X[test_idx])
            fold_cm = _sklearn_cm(y[test_idx], preds, labels=unique_labels)
            agg_cm += fold_cm
            fold_cms.append(fold_cm)

    return agg_cm, fold_cms


# ─── MRT site evaluation helper ─────────────────────────────────────

def _evaluate_mrt_sites(
    classifier: DecisionTreeClassifier,
    leaf_cluster_map: dict[int, int],
    env_data: pd.DataFrame,
    true_labels: pd.Series,
    status: pd.Series,
    unique_labels: list[int],
    role: str,
) -> pd.DataFrame:
    """Predict sites with tree and compute class probabilities."""
    preds = _predict_with_mapping(classifier, env_data, leaf_cluster_map)

    # Class probabilities from tree leaves
    probs = classifier.predict_proba(env_data.values)
    tree_classes = list(classifier.classes_)

    df = env_data.copy()
    df.insert(0, "Original_Cluster", true_labels.loc[env_data.index])
    df.insert(1, "Status", status.loc[env_data.index])
    df.insert(2, "Predicted_Cluster", preds)

    for cid in unique_labels:
        col_name = f"Prob_Cluster {int(cid)}"
        if cid in tree_classes:
            prob_idx = tree_classes.index(cid)
            df[col_name] = probs[:, prob_idx]
        else:
            df[col_name] = 0.0

    # p_max and delta_p
    prob_cols = [f"Prob_Cluster {int(c)}" for c in unique_labels]
    prob_arr = df[prob_cols].values
    sorted_probs = np.sort(prob_arr, axis=1)[:, ::-1]
    df["p_max"] = sorted_probs[:, 0]
    if prob_arr.shape[1] > 1:
        df["delta_p"] = sorted_probs[:, 0] - sorted_probs[:, 1]
    else:
        df["delta_p"] = sorted_probs[:, 0]
    df["Role"] = role

    return df


# ─── Per-model training helper ───────────────────────────────────────

def _train_single_mrt(
    model_name: str,
    training_subset: str,
    env_train: pd.DataFrame,
    labels_train: pd.Series,
    status_train: pd.Series,
    taxa_response_train: pd.DataFrame,
    taxa_ref_octave_train: pd.DataFrame,
    *,
    env_heldout: pd.DataFrame | None = None,
    labels_heldout: pd.Series | None = None,
    status_heldout: pd.Series | None = None,
    train_site_desc: str = "",
    heldout_site_desc: str = "",
    sample_weights: pd.Series | None = None,
    response_transform: str,
    env_variables: list[str],
    taxa_columns: list[str],
    k_folds: int,
    cv_perms: int,
    minsplit: int,
    minbucket: int,
    random_state: int | None,
    model_dir: _Path,
    env_short: list[str] | None = None,
    save_plots: bool = True,
    figure_formats: Sequence[str] = ("png",),
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> dict:
    """Train, evaluate, and save outputs for a single MRT model.

    Produces per-model: confusion_matrix_unified, site_eval_train,
    site_eval_heldout, mrt_cp_table.
    """
    from .confidence_lda import _build_unified_cm
    from sklearn.metrics import accuracy_score

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    tables = model_dir / "tables"
    figures = model_dir / "figures"
    artifacts = model_dir / "artifacts"

    n_clusters = int(labels_train.nunique())
    n_train = len(env_train)
    ref_mask = pd.Series(True, index=env_train.index, name="Is_Reference")

    # --- Fit ---
    mrt_result = None
    if sample_weights is not None:
        _log(f"  [{model_name}] Fitting weighted tree on {n_train} sites ...")
        (classifier_model, cp_table, best_cp, min_cvre, se,
         nsplits, nleaves) = _fit_weighted_mrt(
            env_train, labels_train, sample_weights,
            k_folds=k_folds, cv_perms=cv_perms,
            minsplit=minsplit, minbucket=minbucket,
            random_state=random_state,
        )
    else:
        _log(f"  [{model_name}] Fitting unweighted tree on {n_train} sites ...")
        mrt_result = fit_mrt(
            taxa_response_train, env_train,
            cluster_labels=labels_train,
            ward_linkage=None,
            ref_mask=ref_mask,
            ref_stations=env_train.index,
            taxa_ref_octave=taxa_ref_octave_train,
            reference_quantile=1.0,
            response_transform=response_transform,
            env_variables=env_variables,
            taxa_columns=taxa_columns,
            n_clusters=n_clusters,
            k_folds=k_folds, cv_perms=cv_perms,
            minsplit=minsplit, minbucket=minbucket,
            random_state=random_state,
        )
        classifier_model = mrt_result.classifier_model
        cp_table = mrt_result.cp_table
        best_cp = mrt_result.best_cp
        min_cvre = mrt_result.min_cv_error
        se = mrt_result.min_cv_se
        nsplits = mrt_result.pruned_nsplits
        nleaves = mrt_result.pruned_leaves

    _log(f"  [{model_name}] Pruned tree: {nleaves} leaves ({nsplits} splits), "
         f"CVRE={min_cvre:.4f} ± {se:.4f}")

    # Leaf → cluster majority-vote mapping
    leaf_cluster_map = _leaf_to_cluster_map(classifier_model, env_train, labels_train)

    # Training predictions & accuracy
    train_preds = _predict_with_mapping(classifier_model, env_train, leaf_cluster_map)
    train_acc = accuracy_score(labels_train.values, train_preds.values)
    _log(f"  [{model_name}] Training accuracy = {train_acc:.2%}")

    # --- Cluster names ---
    unique_clusters = sorted(labels_train.unique())
    cluster_names = [f"Cluster {int(c)}" for c in unique_clusters]

    # --- Training confusion matrix (resubstitution) ---
    cm_train = _sklearn_cm(labels_train.values, train_preds.values,
                           labels=unique_clusters)

    # --- Cross-validation confusion matrix ---
    _log(f"  [{model_name}] Cross-validation ({k_folds}-fold x {cv_perms} repeats) ...")
    cv_cm, fold_cms = _mrt_cv_confusion_matrix(
        env_train, labels_train, sample_weights,
        ccp_alpha=best_cp, k_folds=k_folds, cv_perms=cv_perms,
        minsplit=minsplit, minbucket=minbucket,
        random_state=random_state,
    )

    # --- Held-out confusion matrix ---
    cm_heldout = None
    heldout_eval = None
    if env_heldout is not None and len(env_heldout) > 0:
        _log(f"  [{model_name}] Predicting {len(env_heldout)} held-out sites ...")
        heldout_eval = _evaluate_mrt_sites(
            classifier_model, leaf_cluster_map,
            env_heldout, labels_heldout, status_heldout,
            unique_labels=unique_clusters, role="Held-out",
        )
        ho_preds = heldout_eval["Predicted_Cluster"].values.astype(int)
        ho_trues = heldout_eval["Original_Cluster"].values.astype(int)
        cm_heldout = _sklearn_cm(ho_trues, ho_preds, labels=unique_clusters)

    # --- Site evaluations ---
    train_eval = _evaluate_mrt_sites(
        classifier_model, leaf_cluster_map,
        env_train, labels_train, status_train,
        unique_labels=unique_clusters, role="Train",
    )

    # --- Build unified 3-part confusion matrix ---
    train_desc = train_site_desc or training_subset
    ho_desc = heldout_site_desc or "Held-out"
    n_ho = len(env_heldout) if env_heldout is not None else 0

    unified_cm = _build_unified_cm(
        cm_train=cm_train,
        cm_cv=cv_cm,
        cm_heldout=cm_heldout,
        cluster_names=cluster_names,
        train_label=f"Part 1: Training Resubstitution — {train_desc}",
        cv_label=f"Part 2: {k_folds}-Fold x {cv_perms} Cross-Validation — {train_desc}",
        heldout_label=(
            f"Part 3: Held-out Prediction — {ho_desc}"
        ) if cm_heldout is not None else None,
    )

    # --- Save tables ---
    _log(f"  [{model_name}] Saving tables ...")

    # CP table
    save_table(cp_table, tables / "mrt_cp_table",
               formats=table_formats, verbose=verbose)

    # Unified confusion matrix (no header row)
    save_table(unified_cm, tables / "confusion_matrix_unified",
               formats=table_formats, verbose=verbose, header=False)

    # Per-site evaluation (training)
    save_table(train_eval, tables / "site_eval_train",
               formats=table_formats, verbose=verbose)

    # Per-site evaluation (held-out)
    if heldout_eval is not None and len(heldout_eval) > 0:
        save_table(heldout_eval, tables / "site_eval_heldout",
                   formats=table_formats, verbose=verbose)

    # --- Save figures ---
    if save_plots:
        _log(f"  [{model_name}] Saving figures ...")

        # MRT cp+tree figure (only for unweighted models with full MRTResult)
        if mrt_result is not None:
            fig_path = save_mrt_cp_tree_figure(
                mrt_result, figures / "mrt_cp_tree.png",
            )
            if verbose:
                print(f"  > Saved figure: {fig_path}")

        # PCA ordination
        env_names = env_short if env_short else env_variables
        save_env_pca_ordination(
            env_ref=env_train,
            true_labels=labels_train,
            predicted_labels=train_preds.loc[labels_train.index],
            output_path=figures / "env_pca_ordination.png",
            env_feature_names=env_names,
            title=f"{model_name}: PCA Ordination ({n_train} sites)",
        )
        if verbose:
            print(f"  > Saved figure: {figures / 'env_pca_ordination.png'}")

    # --- Save artifacts ---
    artifacts.mkdir(parents=True, exist_ok=True)
    if sample_weights is not None:
        w_df = sample_weights.loc[env_train.index].to_frame("Weight")
        w_df.to_excel(artifacts / "training_weights.xlsx")
        if verbose:
            print(f"  > Saved artifact: {artifacts / 'training_weights.xlsx'}")

    return {
        "model_name": model_name,
        "training_subset": training_subset,
        "n_train": n_train,
        "classifier_model": classifier_model,
        "leaf_cluster_map": leaf_cluster_map,
        "mrt_result": mrt_result,
        "cp_table": cp_table,
        "best_cp": best_cp,
        "min_cvre": min_cvre,
        "se": se,
        "nsplits": nsplits,
        "nleaves": nleaves,
        "train_accuracy": train_acc,
        "train_eval": train_eval,
        "heldout_eval": heldout_eval,
        "cv_cm": cv_cm,
        "fold_cms": fold_cms,
        "cluster_names": cluster_names,
    }


# ─── Main pipeline entry point ──────────────────────────────────────

def confidence_mrt_pipeline(
    data_path: str | _Path,
    stage1_artifact: str | _Path,
    output_dir: str | _Path,
    *,
    site_robustness: pd.DataFrame,
    taxa_columns: Sequence[str] = TAXA_COLUMNS,
    env_variables: Sequence[str] | None = None,
    env_short: Sequence[str] | None = None,
    response_transform: str = "chord",
    k_folds: int = 5,
    cv_perms: int = 10,
    minsplit: int = 3,
    minbucket: int = 2,
    random_state: int | None = 42,
    save_plots: bool = True,
    figure_formats: Sequence[str] = ("png",),
    table_formats: Sequence[str] = ("xlsx",),
    verbose: bool = True,
) -> dict:
    """Run the confidence-aware MRT comparison pipeline.

    Trains 4 MRT models (Core, Core+Periph, Weighted, All) and produces
    cross-model comparison confusion matrices.
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
    if env_short is None:
        env_short = ["Depth", "DO", "Temp", "MPS", "LOI"]

    if response_transform not in _TRANSFORMS:
        raise ValueError(f"Unknown transform {response_transform!r}")
    transform_fn = _TRANSFORMS[response_transform]

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # ============================================================
    #  Step 0. Read data
    # ============================================================
    _log("=" * 60)
    _log("  Confidence-Aware MRT Pipeline")
    _log("=" * 60)

    _log("[0] Reading data ...")
    data = read_study_data(data_path)
    env_block = extract_block(data, "environmental", "raw")
    taxa_block = extract_block(data, "taxa", "raw")[list(taxa_columns)]

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
    all_sites = labels_all.index

    _log(f"    Core: {len(core_sites)}  |  Core+Peripheral: {len(cp_sites)}  "
         f"|  Uncertain: {len(uncertain_sites)}  |  All: {len(all_sites)}")

    # Prepare env and taxa subsets (drop NaN rows in env)
    env_vars_present = [v for v in env_variables if v in env_block.columns]

    def _prepare_subset(sites: pd.Index):
        """Return (env_short_columns, labels, taxa_oct, taxa_transformed)."""
        env_raw = env_block.loc[sites, env_vars_present].dropna()
        env_short_df = env_raw.copy()
        env_short_df.columns = list(env_short)
        labs = labels_all.loc[env_raw.index]
        taxa_oct = taxa_block.loc[env_raw.index]
        taxa_trans = transform_fn(taxa_oct)
        taxa_trans.index = env_raw.index
        return env_short_df, labs, taxa_oct, taxa_trans

    env_core, labels_core, taxa_oct_core, taxa_trans_core = _prepare_subset(core_sites)
    env_cp, labels_cp, taxa_oct_cp, taxa_trans_cp = _prepare_subset(cp_sites)
    env_all, labels_allsites, taxa_oct_all, taxa_trans_all = _prepare_subset(all_sites)
    weights_cp = weights.loc[env_cp.index]

    # Status aligned to env subsets
    status_core = status_all.loc[env_core.index]
    status_cp = status_all.loc[env_cp.index]
    status_allsites = status_all.loc[env_all.index]

    # Held-out for Model A = Peripheral + Uncertain
    noncore_sites = labels_all.index[(status_all == "Peripheral") | (status_all == "Uncertain")]
    env_noncore, labels_noncore, _, _ = _prepare_subset(noncore_sites)
    status_noncore = status_all.loc[env_noncore.index] if len(noncore_sites) > 0 else None

    # Held-out for Models B/C = Uncertain only
    env_uncertain, labels_uncertain = None, None
    status_uncertain = None
    if len(uncertain_sites) > 0:
        env_uncertain, labels_uncertain, _, _ = _prepare_subset(uncertain_sites)
        status_uncertain = status_all.loc[env_uncertain.index]

    def _site_desc(status_series: pd.Series | None) -> str:
        """Build e.g. 'Core (n=35) + Peripheral (n=10)'."""
        if status_series is None or len(status_series) == 0:
            return ""
        counts = status_series.value_counts()
        parts = []
        for s in ["Core", "Peripheral", "Uncertain"]:
            if s in counts.index:
                parts.append(f"{s} (n={counts[s]})")
        return " + ".join(parts) if parts else f"(n={len(status_series)})"

    # Common kwargs for _train_single_mrt
    common = dict(
        response_transform=response_transform,
        env_variables=list(env_short),
        taxa_columns=list(taxa_columns),
        k_folds=k_folds, cv_perms=cv_perms,
        minsplit=minsplit, minbucket=minbucket,
        random_state=random_state,
        env_short=list(env_short),
        save_plots=save_plots,
        figure_formats=figure_formats,
        table_formats=table_formats,
        verbose=verbose,
    )

    # ============================================================
    #  Step 3. Train models
    # ============================================================
    _log("[3] Training models ...")
    models: Dict[str, dict] = {}

    _log("\n--- Model A: MRT on Core only ---")
    models["ModelA_CoreOnly"] = _train_single_mrt(
        "Model A (Core Only)", "Core",
        env_core, labels_core, status_core,
        taxa_trans_core, taxa_oct_core,
        env_heldout=env_noncore,
        labels_heldout=labels_noncore,
        status_heldout=status_noncore,
        train_site_desc=_site_desc(status_core),
        heldout_site_desc=_site_desc(status_noncore),
        model_dir=output_dir / "ModelA_CoreOnly", **common,
    )

    _log("\n--- Model B: MRT on Core + Peripheral ---")
    models["ModelB_CorePeripheral"] = _train_single_mrt(
        "Model B (Core+Peripheral)", "Core+Peripheral",
        env_cp, labels_cp, status_cp,
        taxa_trans_cp, taxa_oct_cp,
        env_heldout=env_uncertain,
        labels_heldout=labels_uncertain,
        status_heldout=status_uncertain,
        train_site_desc=_site_desc(status_cp),
        heldout_site_desc=_site_desc(status_uncertain),
        model_dir=output_dir / "ModelB_CorePeripheral", **common,
    )

    _log("\n--- Model C: Weighted MRT on Core + Peripheral ---")
    models["ModelC_Weighted"] = _train_single_mrt(
        "Model C (Weighted)", "Core+Peripheral (weighted)",
        env_cp, labels_cp, status_cp,
        taxa_trans_cp, taxa_oct_cp,
        env_heldout=env_uncertain,
        labels_heldout=labels_uncertain,
        status_heldout=status_uncertain,
        train_site_desc=_site_desc(status_cp),
        heldout_site_desc=_site_desc(status_uncertain),
        sample_weights=weights_cp,
        model_dir=output_dir / "ModelC_Weighted", **common,
    )

    _log("\n--- Model D: MRT on All Sites (benchmark) ---")
    models["ModelD_AllSites"] = _train_single_mrt(
        "Model D (All Sites)", "All (no holdout)",
        env_all, labels_allsites, status_allsites,
        taxa_trans_all, taxa_oct_all,
        train_site_desc=_site_desc(status_allsites),
        model_dir=output_dir / "ModelD_AllSites", **common,
    )

    # ============================================================
    #  Step 4. Model comparison
    # ============================================================
    _log("\n" + "=" * 60)
    _log("  Model Comparison")
    _log("=" * 60)

    # -- Summary table ------------------------------------------------
    summary_rows = []
    for key, m in models.items():
        row = {
            "Model": m["model_name"],
            "Training Subset": m["training_subset"],
            "N_train": m["n_train"],
            "Train Accuracy": m["train_accuracy"],
            "Tree Leaves": m["nleaves"],
            "Tree Splits": m["nsplits"],
            "Best CP": m["best_cp"],
            "Min CVRE": m["min_cvre"],
            "CVRE SE": m["se"],
        }
        if m["heldout_eval"] is not None and len(m["heldout_eval"]) > 0:
            row["Heldout p_max (mean)"] = m["heldout_eval"]["p_max"].mean()
            row["Heldout delta_p (mean)"] = m["heldout_eval"]["delta_p"].mean()
        summary_rows.append(row)
    summary_table = pd.DataFrame(summary_rows).set_index("Model")
    _log(summary_table.to_string())

    # -- Save comparison outputs --------------------------------------
    _log("\n[5] Saving comparison outputs ...")
    compar_dir = output_dir / "ModelCompar"
    compar_tables = compar_dir / "tables"
    compar_figures = compar_dir / "figures"

    save_table(summary_table, compar_tables / "model_summary",
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
        header = _model_site_desc(m["train_eval"], m["model_name"], is_weighted=is_w)
        header += f"; {k_folds}-fold x {cv_perms} cross-validation"
        cv_sections.append({
            "header": header,
            "agg_cm": m["cv_cm"],
            "fold_cms": m["fold_cms"],
            "cnames": m["cluster_names"],
        })
    cv_combined = build_cv_comparison_table(cv_sections)
    save_table(cv_combined, compar_tables / "cv_confusion_matrices_combined",
               formats=table_formats, verbose=verbose)

    # -- Prediction on ALL reference sites -> confusion_matrix_all_refsites
    _log("  Predicting all reference sites with each model ...")
    unique_clusters = sorted(labels_allsites.unique())
    cnames_all = [f"Cluster {int(c)}" for c in unique_clusters]

    allref_sections = []
    for key, m in models.items():
        ev = _evaluate_mrt_sites(
            m["classifier_model"], m["leaf_cluster_map"],
            env_all, labels_allsites, status_allsites,
            unique_labels=unique_clusters, role="AllRef",
        )
        preds = ev["Predicted_Cluster"].values.astype(int)
        trues = ev["Original_Cluster"].values.astype(int)
        cm_arr = _sklearn_cm(trues, preds, labels=unique_clusters)

        is_w = key in _weighted_keys
        header = _model_site_desc(m["train_eval"], m["model_name"], is_weighted=is_w)

        if m["heldout_eval"] is not None and len(m["heldout_eval"]) > 0:
            ho = m["heldout_eval"]
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
            "cnames": cnames_all,
            "stats": stats,
        })
    allref_combined = build_allref_comparison_table(allref_sections)
    save_table(allref_combined, compar_tables / "confusion_matrix_all_refsites",
               formats=table_formats, verbose=verbose)

    # -- Comparison figure: CVRE bar chart ----------------------------
    if save_plots:
        _log("  Creating comparison figures ...")
        compar_figures.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        model_names = [m["model_name"] for m in models.values()]
        cvres = [m["min_cvre"] for m in models.values()]
        ses = [m["se"] for m in models.values()]
        x_pos = np.arange(len(model_names))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        bars = ax.bar(x_pos, cvres, yerr=ses, capsize=5,
                      color=colors, alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, fontsize=9)
        ax.set_ylabel("Min CVRE")
        ax.set_title("Cross-Validated Relative Error Comparison (MRT)")
        for bar, v, s in zip(bars, cvres, ses):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.02,
                    f"{v:.3f}", ha="center", fontsize=9)
        fig.tight_layout()
        save_figure(fig, compar_figures / "cvre_comparison",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig)

        # Train accuracy bar chart
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        train_accs = [m["train_accuracy"] for m in models.values()]
        bars2 = ax2.bar(x_pos, train_accs, color=colors, alpha=0.8)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(model_names, fontsize=9)
        ax2.set_ylabel("Training Accuracy")
        ax2.set_title("Training Accuracy Comparison (MRT)")
        ax2.set_ylim(0, 1)
        for bar, v in zip(bars2, train_accs):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{v:.1%}", ha="center", fontsize=9)
        fig2.tight_layout()
        save_figure(fig2, compar_figures / "train_accuracy_comparison",
                    formats=figure_formats, verbose=verbose)
        plt.close(fig2)

    # ============================================================
    _log("\n" + "=" * 60)
    _log("  Confidence-Aware MRT Pipeline Complete")
    _log("=" * 60)
    for key, m in models.items():
        _log(f"  {m['model_name']}(n_train={m['n_train']}, "
             f"train_acc={m['train_accuracy']:.2%}, "
             f"leaves={m['nleaves']}, CVRE={m['min_cvre']:.4f})")

    return {
        "models": models,
        "weights": weights,
        "summary_table": summary_table,
    }
