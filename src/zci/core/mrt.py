"""Multivariate Regression Tree fitting via mvpart through rpy2."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from ..models.mrt import MRTResult


@dataclass
class MRTRuntime:
    """Names of live R objects needed for plotting in the current process."""

    full_name: str
    pruned_name: str


def _assign_dataframe(name: str, df: pd.DataFrame) -> None:
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv[name] = ro.conversion.py2rpy(df)


def _r_dataframe(expr: str) -> pd.DataFrame:
    with localconverter(ro.default_converter + pandas2ri.converter):
        out = ro.conversion.rpy2py(ro.r(expr))
    if not isinstance(out, pd.DataFrame):
        out = pd.DataFrame(out)
    return out


def _r_vector(expr: str) -> np.ndarray:
    return np.asarray(ro.r(expr))


def fit_mrt(
    taxa_response: pd.DataFrame,
    env_df: pd.DataFrame,
    *,
    ref_mask: pd.Series,
    ref_stations: pd.Index,
    taxa_ref_octave: pd.DataFrame | None = None,
    reference_quantile: float,
    response_transform: str,
    env_variables: list[str],
    taxa_columns: list[str],
    k_folds: int = 10,
    cv_perms: int = 100,
    minsplit: int = 5,
    minbucket: int = 2,
) -> tuple[MRTResult, MRTRuntime]:
    """Fit the MRT model using R's mvpart backend."""
    importr("mvpart")
    importr("rpart")

    _assign_dataframe("zci_mrt_taxa_mat", taxa_response)
    _assign_dataframe("zci_mrt_env_df", env_df)
    ro.r("zci_mrt_taxa_mat <- as.matrix(zci_mrt_taxa_mat)")

    ro.r(
        f"zci_mrt_ctrl <- rpart.control(cp = 0, minsplit = {minsplit}, "
        f"minbucket = {minbucket}, xval = {k_folds})"
    )
    ro.r(
        "zci_mrt_full <- mvpart("
        "zci_mrt_taxa_mat ~ ., "
        "data = zci_mrt_env_df, "
        "minauto = FALSE, "
        "xv = \"none\", "
        f"xvmult = {cv_perms}, "
        "plot.add = FALSE, "
        "text.add = FALSE, "
        "control = zci_mrt_ctrl)"
    )
    ro.r("while (!is.null(dev.list())) dev.off()")

    cp_table = _r_dataframe("as.data.frame(zci_mrt_full$cptable)")
    cp_table["nsplit"] = cp_table["nsplit"].astype(int)
    cp_table.index = np.arange(1, len(cp_table) + 1)

    min_pos = int(cp_table["xerror"].to_numpy(dtype=float).argmin())
    min_row = cp_table.iloc[min_pos]
    best_cp = float(min_row["CP"])
    ro.globalenv["zci_mrt_best_cp"] = ro.FloatVector([best_cp])
    ro.r("zci_mrt_pruned <- prune(zci_mrt_full, cp = zci_mrt_best_cp[1])")

    frame_vars = [str(v) for v in _r_vector("as.character(zci_mrt_full$frame$var)")]
    variable_counts = pd.Series(
        Counter(v for v in frame_vars if v != "<leaf>"),
        dtype=int,
    ).sort_values(ascending=False)

    root_node_error = float(_r_vector("zci_mrt_full$frame$dev[1]")[0])
    pruned_nsplits = int(_r_vector("sum(zci_mrt_pruned$frame$var != \"<leaf>\")")[0])
    pruned_leaves = pruned_nsplits + 1
    full_tree_splits = int(cp_table["nsplit"].max())
    full_tree_leaves = full_tree_splits + 1

    selected_matches = np.flatnonzero(cp_table["nsplit"].to_numpy(dtype=int) == pruned_nsplits)
    if len(selected_matches) == 0:
        raise RuntimeError("Could not locate the selected tree size in the CP table")

    where = _r_vector("zci_mrt_pruned$where").astype(int)
    leaf_membership = pd.DataFrame(
        {"StationID": list(ref_stations), "Leaf": where},
    )

    result = MRTResult(
        ref_mask=ref_mask,
        ref_stations=ref_stations,
        taxa_response=taxa_response,
        taxa_ref_octave=taxa_ref_octave if taxa_ref_octave is not None else taxa_response,
        env_ref=env_df,
        cp_table=cp_table,
        leaf_membership=leaf_membership,
        variable_counts=variable_counts,
        reference_quantile=reference_quantile,
        response_transform=response_transform,
        env_variables=env_variables,
        taxa_columns=taxa_columns,
        k_folds=k_folds,
        cv_perms=cv_perms,
        minsplit=minsplit,
        minbucket=minbucket,
        best_cp=best_cp,
        min_cv_error=float(min_row["xerror"]),
        min_cv_se=float(min_row["xstd"]),
        root_node_error=root_node_error,
        pruned_nsplits=pruned_nsplits,
        pruned_leaves=pruned_leaves,
        full_tree_splits=full_tree_splits,
        full_tree_leaves=full_tree_leaves,
    )
    runtime = MRTRuntime(full_name="zci_mrt_full", pruned_name="zci_mrt_pruned")
    return result, runtime