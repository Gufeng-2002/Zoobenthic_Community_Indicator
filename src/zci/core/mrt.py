"""Ward-targeted MRT classifier fitting via scikit-learn."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from ..models.mrt import MRTResult


def _fit_tree(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    ccp_alpha: float,
    minsplit: int,
    minbucket: int,
    random_state: int | None,
) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(
        criterion="gini",
        random_state=random_state,
        min_samples_split=minsplit,
        min_samples_leaf=minbucket,
        ccp_alpha=float(ccp_alpha),
    )
    model.fit(X, y)
    return model


def _node_feature_counts(model: DecisionTreeClassifier, feature_names: list[str]) -> pd.Series:
    feature_ids = model.tree_.feature
    used_ids = feature_ids[feature_ids >= 0]
    counts = Counter(feature_names[i] for i in used_ids)
    return pd.Series(counts, dtype=int).sort_values(ascending=False)


def _misclassified(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    return int(np.count_nonzero(y_true != y_pred))


def _root_node_error(y: np.ndarray) -> float:
    counts = pd.Series(y).value_counts()
    if counts.empty:
        return 1.0
    err = float(len(y) - counts.max())
    return err if err > 0 else 1.0


def _cross_validated_relative_errors(
    X: np.ndarray,
    y: np.ndarray,
    *,
    ccp_alpha: float,
    root_error: float,
    k_folds: int,
    cv_perms: int,
    minsplit: int,
    minbucket: int,
    random_state: int | None,
) -> np.ndarray:
    if cv_perms < 1:
        raise ValueError("cv_perms must be >= 1")

    rng = np.random.default_rng(random_state)
    seeds = (
        rng.integers(0, np.iinfo(np.int32).max, size=cv_perms)
        if random_state is not None else np.repeat(None, cv_perms)
    )

    cv_errors: list[float] = []
    for seed in seeds:
        splitter = StratifiedKFold(
            n_splits=k_folds,
            shuffle=True,
            random_state=None if seed is None else int(seed),
        )
        fold_errors = 0
        for train_idx, test_idx in splitter.split(X, y):
            model = _fit_tree(
                X[train_idx],
                y[train_idx],
                ccp_alpha=ccp_alpha,
                minsplit=minsplit,
                minbucket=minbucket,
                random_state=random_state,
            )
            preds = model.predict(X[test_idx])
            fold_errors += _misclassified(y[test_idx], preds)
        cv_errors.append(fold_errors / root_error)

    return np.asarray(cv_errors, dtype=float)


def fit_mrt(
    taxa_response: pd.DataFrame,
    env_df: pd.DataFrame,
    *,
    cluster_labels: pd.Series,
    ward_linkage: np.ndarray | None,
    ref_mask: pd.Series,
    ref_stations: pd.Index,
    taxa_ref_octave: pd.DataFrame | None = None,
    reference_quantile: float,
    response_transform: str,
    env_variables: list[str],
    taxa_columns: list[str],
    n_clusters: int,
    k_folds: int = 10,
    cv_perms: int = 100,
    minsplit: int = 5,
    minbucket: int = 2,
    random_state: int | None = 42,
) -> MRTResult:
    """Fit a pruned tree classifier to Ward-defined reference clusters."""
    aligned_clusters = cluster_labels.loc[env_df.index].astype(int)
    X = env_df.astype(float)
    y = aligned_clusters.astype(int)

    min_class_size = int(aligned_clusters.value_counts().min())
    effective_k_folds = min(k_folds, min_class_size)
    if effective_k_folds < 2:
        raise ValueError(
            "At least two samples per Ward cluster are required for cross-validation"
        )

    root_error = _root_node_error(y.to_numpy(dtype=int))

    full_tree = _fit_tree(
        X,
        y,
        ccp_alpha=0.0,
        minsplit=minsplit,
        minbucket=minbucket,
        random_state=random_state,
    )
    full_tree_splits = int(full_tree.get_n_leaves() - 1)
    full_tree_leaves = int(full_tree.get_n_leaves())
    variable_counts = _node_feature_counts(full_tree, env_df.columns.tolist())

    path = full_tree.cost_complexity_pruning_path(X, y)
    alphas = np.unique(np.r_[path.ccp_alphas, 0.0])

    candidate_rows: dict[int, dict[str, float]] = {}
    candidate_models: dict[int, DecisionTreeClassifier] = {}
    for alpha in alphas:
        model = _fit_tree(
            X,
            y,
            ccp_alpha=float(alpha),
            minsplit=minsplit,
            minbucket=minbucket,
            random_state=random_state,
        )
        nsplit = int(model.get_n_leaves() - 1)
        if nsplit in candidate_rows:
            continue

        train_error = _misclassified(y.to_numpy(dtype=int), model.predict(X)) / root_error
        cv_errors = _cross_validated_relative_errors(
            X.to_numpy(dtype=float),
            y.to_numpy(dtype=int),
            ccp_alpha=float(alpha),
            root_error=root_error,
            k_folds=effective_k_folds,
            cv_perms=cv_perms,
            minsplit=minsplit,
            minbucket=minbucket,
            random_state=random_state,
        )

        candidate_rows[nsplit] = {
            "CP": float(alpha),
            "nsplit": nsplit,
            "rel error": float(train_error),
            "xerror": float(cv_errors.mean()),
            "xstd": float(cv_errors.std(ddof=1) / np.sqrt(len(cv_errors))) if len(cv_errors) > 1 else 0.0,
            "xcv_lo": float(np.percentile(cv_errors, 2.5)) if len(cv_errors) > 1 else float(cv_errors.mean()),
            "xcv_hi": float(np.percentile(cv_errors, 97.5)) if len(cv_errors) > 1 else float(cv_errors.mean()),
        }
        candidate_models[nsplit] = model

    cp_table = pd.DataFrame(candidate_rows.values()).sort_values("nsplit").reset_index(drop=True)
    cp_table.index = np.arange(1, len(cp_table) + 1)

    best_pos = int(cp_table["xerror"].to_numpy(dtype=float).argmin())
    best_row = cp_table.iloc[best_pos]
    best_nsplit = int(best_row["nsplit"])
    best_tree = candidate_models[best_nsplit]

    leaf_membership = pd.DataFrame(
        {
            "StationID": list(env_df.index),
            "Leaf": best_tree.apply(X),
        }
    )

    return MRTResult(
        ref_mask=ref_mask,
        ref_stations=ref_stations,
        cluster_labels_ref=aligned_clusters,
        taxa_response=taxa_response.loc[env_df.index],
        taxa_ref_octave=(taxa_ref_octave if taxa_ref_octave is not None else taxa_response).loc[env_df.index],
        env_ref=env_df,
        cp_table=cp_table,
        leaf_membership=leaf_membership,
        variable_counts=variable_counts,
        reference_quantile=reference_quantile,
        response_transform=response_transform,
        env_variables=env_variables,
        taxa_columns=taxa_columns,
        n_clusters=n_clusters,
        k_folds=k_folds,
        cv_perms=cv_perms,
        effective_k_folds=effective_k_folds,
        minsplit=minsplit,
        minbucket=minbucket,
        random_state=random_state,
        best_cp=float(best_row["CP"]),
        min_cv_error=float(best_row["xerror"]),
        min_cv_se=float(best_row["xstd"]),
        root_node_error=float(root_error),
        pruned_nsplits=best_nsplit,
        pruned_leaves=int(best_tree.get_n_leaves()),
        full_tree_splits=full_tree_splits,
        full_tree_leaves=full_tree_leaves,
        classifier_model=best_tree,
        full_tree_model=full_tree,
        ward_linkage=ward_linkage,
    )