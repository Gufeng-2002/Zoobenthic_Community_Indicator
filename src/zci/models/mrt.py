"""Structured result containers for the combined Ward-targeted MRT pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


@dataclass
class MRTResult:
    """Container returned by the combined Ward-targeted MRT pipeline.

    The target classes are Ward clusters defined on the transformed taxa
    matrix of the selected reference sites. The fitted model is a pruned
    decision-tree classifier trained only on environmental covariates.
    """

    ref_mask: pd.Series
    ref_stations: pd.Index
    cluster_labels_ref: pd.Series
    taxa_response: pd.DataFrame
    taxa_ref_octave: pd.DataFrame
    env_ref: pd.DataFrame
    cp_table: pd.DataFrame
    leaf_membership: pd.DataFrame
    variable_counts: pd.Series
    reference_quantile: float
    response_transform: str
    env_variables: Sequence[str]
    taxa_columns: Sequence[str]
    n_clusters: int
    k_folds: int
    cv_perms: int
    effective_k_folds: int
    minsplit: int
    minbucket: int
    random_state: int | None
    best_cp: float
    min_cv_error: float
    min_cv_se: float
    root_node_error: float
    pruned_nsplits: int
    pruned_leaves: int
    full_tree_splits: int
    full_tree_leaves: int
    classifier_model: DecisionTreeClassifier
    full_tree_model: DecisionTreeClassifier | None = None
    ward_linkage: np.ndarray | None = None

    @property
    def cluster_labels(self) -> pd.Series:
        """Ward cluster labels for reference sites."""
        return self.cluster_labels_ref.rename("Cluster")

    def to_ref_table(self) -> pd.DataFrame:
        """Reference-site taxa matrix with a leading Cluster column."""
        tbl = self.taxa_ref_octave.copy()
        tbl.insert(0, "Cluster", self.cluster_labels_ref.values)
        return tbl

    def summary(self) -> str:
        dist = self.cluster_labels_ref.value_counts().sort_index()
        dist_txt = ", ".join(f"C{int(label)}: {int(count)}" for label, count in dist.items())
        return (
            f"MRTResult(n_ref={len(self.ref_stations)}, transform={self.response_transform!r}, "
            f"clusters={self.n_clusters}, cv={self.effective_k_folds}-fold x {self.cv_perms}, "
            f"final_tree={self.pruned_leaves} leaves, min_cv_error={self.min_cv_error:.4f}, "
            f"distribution=[{dist_txt}])"
        )

    def artifact_payload(self) -> Dict[str, object]:
        return {
            "ref_stations": list(self.ref_stations),
            "cluster_labels_ref": self.cluster_labels_ref,
            "cp_table": self.cp_table,
            "leaf_membership": self.leaf_membership,
            "variable_counts": self.variable_counts,
            "reference_quantile": self.reference_quantile,
            "response_transform": self.response_transform,
            "env_variables": list(self.env_variables),
            "taxa_columns": list(self.taxa_columns),
            "n_clusters": self.n_clusters,
            "k_folds": self.k_folds,
            "cv_perms": self.cv_perms,
            "effective_k_folds": self.effective_k_folds,
            "minsplit": self.minsplit,
            "minbucket": self.minbucket,
            "random_state": self.random_state,
            "best_cp": self.best_cp,
            "min_cv_error": self.min_cv_error,
            "min_cv_se": self.min_cv_se,
            "root_node_error": self.root_node_error,
            "pruned_nsplits": self.pruned_nsplits,
            "pruned_leaves": self.pruned_leaves,
            "full_tree_splits": self.full_tree_splits,
            "full_tree_leaves": self.full_tree_leaves,
        }