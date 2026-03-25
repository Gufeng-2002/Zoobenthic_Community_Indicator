"""Structured result containers for the MRT pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import pandas as pd


@dataclass
class MRTResult:
    """Container returned by the Python MRT pipeline.

    The fitted tree is produced through ``mvpart`` via ``rpy2`` so the
    Python pipeline reproduces the current R workflow exactly.
    """

    ref_mask: pd.Series
    ref_stations: pd.Index
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
    k_folds: int
    cv_perms: int
    minsplit: int
    minbucket: int
    best_cp: float
    min_cv_error: float
    min_cv_se: float
    root_node_error: float
    pruned_nsplits: int
    pruned_leaves: int
    full_tree_splits: int
    full_tree_leaves: int

    @property
    def cluster_labels(self) -> pd.Series:
        """Convert MRT leaf indices to sequential 1-indexed cluster labels."""
        leaf_col = self.leaf_membership["Leaf"]
        unique_leaves = sorted(leaf_col.unique())
        label_map = {leaf: i + 1 for i, leaf in enumerate(unique_leaves)}
        labels = leaf_col.map(label_map)
        return pd.Series(
            labels.values,
            index=self.leaf_membership["StationID"].values,
            name="Cluster",
        )

    def to_ref_table(self) -> pd.DataFrame:
        """Reference-site taxa matrix (octave) with a leading Cluster column."""
        tbl = self.taxa_ref_octave.copy()
        tbl.insert(0, "Cluster", self.cluster_labels.values)
        return tbl

    def summary(self) -> str:
        return (
            f"MRTResult(n_ref={len(self.ref_stations)}, transform={self.response_transform!r}, "
            f"cv={self.k_folds}-fold x {self.cv_perms}, final_tree={self.pruned_leaves} leaves, "
            f"min_cv_error={self.min_cv_error:.4f})"
        )

    def artifact_payload(self) -> Dict[str, object]:
        return {
            "ref_stations": list(self.ref_stations),
            "cp_table": self.cp_table,
            "leaf_membership": self.leaf_membership,
            "variable_counts": self.variable_counts,
            "reference_quantile": self.reference_quantile,
            "response_transform": self.response_transform,
            "env_variables": list(self.env_variables),
            "taxa_columns": list(self.taxa_columns),
            "k_folds": self.k_folds,
            "cv_perms": self.cv_perms,
            "minsplit": self.minsplit,
            "minbucket": self.minbucket,
            "best_cp": self.best_cp,
            "min_cv_error": self.min_cv_error,
            "min_cv_se": self.min_cv_se,
            "root_node_error": self.root_node_error,
            "pruned_nsplits": self.pruned_nsplits,
            "pruned_leaves": self.pruned_leaves,
            "full_tree_splits": self.full_tree_splits,
            "full_tree_leaves": self.full_tree_leaves,
        }