"""Cut-off sensitivity analysis — pure computation, no plotting, no I/O.

Sweeps a grid of low-contamination cut-off proportions, fits RDA (or MRT)
at each cut-off, and collects performance metrics.  Also identifies stable
cut-off ranges where model performance is consistently strong.

Public API
----------
sweep_cutoffs
    Run RDA at each cut-off and return a tidy metrics DataFrame.
sweep_cutoffs_mrt
    Run MRT at each cut-off and return CVRE / SE / tree-size metrics.
detect_stable_ranges
    Identify contiguous cut-off ranges with consistently good performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..core.rda import RDA, compute_vif
from ..core.clustering import select_reference_sites
from ..core.transforms import (
    octave_transform,
    octave_to_relative_abundance,
    octave_to_chord,
    octave_to_hellinger,
    octave_to_log_chord,
)

_TRANSFORMS = {
    "octave": octave_transform,
    "relative_abundance": octave_to_relative_abundance,
    "chord": octave_to_chord,
    "hellinger": octave_to_hellinger,
    "log_chord": octave_to_log_chord,
}


# ─── result container ───────────────────────────────────────────────


@dataclass
class ThresholdSweepResult:
    """Container for cut-off sensitivity sweep outputs."""
    metrics: pd.DataFrame
    stable_ranges: List[Tuple[float, float]]
    score_name: str
    env_variables: List[str]
    taxa_columns: List[str]


# ─── single-threshold RDA fitter ────────────────────────────────────


def _fit_rda_at_threshold(
    pollution_score: pd.Series,
    env_all: pd.DataFrame,
    taxa_all: pd.DataFrame,
    threshold: float,
    *,
    standardize_env: bool = True,
    log_transform_env: bool = False,
    taxa_transform: str = "octave",
    n_permutations: int = 999,
    random_state: int | None = None,
) -> Dict[str, float]:
    """Fit RDA on the lowest-*threshold* fraction of sites and return metrics.

    Parameters
    ----------
    pollution_score : pd.Series
        Site-level contamination score (higher = more contaminated).
    env_all : pd.DataFrame
        Full environmental matrix (all sites, selected columns).
    taxa_all : pd.DataFrame
        Full taxa matrix (all sites).
    threshold : float
        Proportion (0–1) of least-contaminated sites to select.
    standardize_env : bool
        Whether to z-score environmental variables.
    log_transform_env : bool
        Whether to ln(1+x) env vars before z-scoring.
    taxa_transform : str
        One of ``"octave"``, ``"relative_abundance"``, ``"hellinger"``,
        ``"chord"``, or ``"log_chord"``.
    n_permutations : int
        Number of permutations for global test.
    random_state : int or None
        RNG seed.

    Returns
    -------
    dict
        Keys: threshold, n_sites, r2, r2_adj, global_F, global_p,
        constrained_inertia, total_inertia, rda1_eigenvalue.
    """
    ref_mask = select_reference_sites(pollution_score, quantile=threshold)
    n_ref = int(ref_mask.sum())

    env_ref = env_all.loc[ref_mask].copy()
    taxa_ref = taxa_all.loc[ref_mask].copy()

    # Env transforms
    if log_transform_env:
        for col in env_ref.columns:
            mn = env_ref[col].min()
            shift = abs(mn) + 1e-6 if mn <= 0 else 0.0
            env_ref[col] = np.log(env_ref[col] + shift)

    if standardize_env:
        scaler = StandardScaler()
        arr = scaler.fit_transform(env_ref)
        env_ref = pd.DataFrame(arr, index=env_ref.index, columns=env_ref.columns)

    # Taxa transforms
    if taxa_transform in _TRANSFORMS:
        taxa_ref = _TRANSFORMS[taxa_transform](taxa_ref)
    else:
        raise ValueError(
            f"Unknown taxa_transform={taxa_transform!r}. "
            f"Choose from {sorted(_TRANSFORMS)}."
        )

    # Drop NaN rows
    valid = env_ref.dropna().index.intersection(taxa_ref.dropna().index)
    env_ref = env_ref.loc[valid]
    taxa_ref = taxa_ref.loc[valid]
    n_valid = len(valid)

    # Need more sites than predictors + 1
    if n_valid <= env_ref.shape[1] + 1:
        return {
            "threshold": threshold,
            "n_sites": n_valid,
            "r2": float("nan"),
            "r2_adj": float("nan"),
            "global_F": float("nan"),
            "global_p": float("nan"),
            "constrained_inertia": float("nan"),
            "total_inertia": float("nan"),
            "rda1_eigenvalue": float("nan"),
            "max_vif": float("nan"),
        }

    # VIF on the predictor matrix
    vif_vals = compute_vif(env_ref)
    max_vif = float(vif_vals.max())

    # Fit RDA
    rda = RDA(center_X=True, center_Y=True, scale_X=False, ddof=1)
    rda.fit(env_ref, taxa_ref)
    fit = rda.fit_

    # Global permutation test
    global_test = rda.test_global(
        n_permutations=n_permutations, random_state=random_state,
    )

    rda1_eig = (
        float(fit.constrained_eigenvalues.iloc[0])
        if len(fit.constrained_eigenvalues) > 0
        else float("nan")
    )

    return {
        "threshold": threshold,
        "n_sites": n_valid,
        "r2": fit.r2,
        "r2_adj": fit.r2_adj,
        "global_F": global_test.statistic,
        "global_p": global_test.p_value,
        "constrained_inertia": fit.inertia_constrained,
        "total_inertia": fit.inertia_total,
        "rda1_eigenvalue": rda1_eig,
        "max_vif": max_vif,
    }


# ─── sweep ──────────────────────────────────────────────────────────


def sweep_thresholds(
    pollution_score: pd.Series,
    env_all: pd.DataFrame,
    taxa_all: pd.DataFrame,
    *,
    thresholds: Sequence[float] | None = None,
    standardize_env: bool = True,
    log_transform_env: bool = False,
    taxa_transform: str = "octave",
    n_permutations: int = 999,
    random_state: int | None = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run RDA at each threshold and return a tidy metrics table.

    Parameters
    ----------
    pollution_score : pd.Series
        Site-level contamination score.
    env_all : pd.DataFrame
        Full environmental matrix (all sites, selected columns only).
    taxa_all : pd.DataFrame
        Full taxa matrix (all sites).
    thresholds : sequence of float, optional
        Grid of proportions.  Default: 0.10, 0.12, …, 0.30.
    standardize_env, log_transform_env, taxa_transform : misc
        Passed through to each RDA fit.
    n_permutations : int
        Permutations for global test at each threshold.
    random_state : int or None
        Base seed; each threshold uses ``random_state + i``.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Columns: threshold, n_sites, r2, r2_adj, global_F, global_p,
        constrained_inertia, total_inertia, rda1_eigenvalue.
    """
    if thresholds is None:
        thresholds = list(np.arange(0.10, 0.31, 0.02).round(2))

    rows = []
    for i, thr in enumerate(thresholds):
        if verbose:
            print(f"  [{i+1}/{len(thresholds)}] threshold = {thr:.0%} …", end="")

        seed = (random_state + i) if random_state is not None else None
        row = _fit_rda_at_threshold(
            pollution_score, env_all, taxa_all, thr,
            standardize_env=standardize_env,
            log_transform_env=log_transform_env,
            taxa_transform=taxa_transform,
            n_permutations=n_permutations,
            random_state=seed,
        )
        rows.append(row)

        if verbose:
            print(f"  n={row['n_sites']}, R²={row['r2']:.4f}, "
                  f"adj-R²={row['r2_adj']:.4f}, p={row['global_p']:.4f}")

    return pd.DataFrame(rows)


# ─── stable-range detection ─────────────────────────────────────────


def detect_stable_ranges(
    metrics: pd.DataFrame,
    *,
    p_threshold: float = 0.05,
    r2_adj_min: float = 0.0,
    min_sites: int = 0,
) -> List[Tuple[float, float]]:
    """Identify contiguous threshold ranges with consistently good RDA.

    A threshold is considered 'good' if:
    - ``global_p <= p_threshold``
    - ``r2_adj >= r2_adj_min``
    - ``n_sites >= min_sites``

    Contiguous runs of good thresholds are grouped into (start, end) ranges.

    Parameters
    ----------
    metrics : pd.DataFrame
        Output of :func:`sweep_thresholds`.
    p_threshold : float
        Maximum acceptable global p-value.
    r2_adj_min : float
        Minimum acceptable adjusted R².
    min_sites : int
        Minimum acceptable number of reference sites.

    Returns
    -------
    list of (float, float)
        Each tuple is (start_threshold, end_threshold) of a contiguous
        stable range.
    """
    df = metrics.sort_values("threshold").reset_index(drop=True)
    good = (
        (df["global_p"] <= p_threshold)
        & (df["r2_adj"] >= r2_adj_min)
        & (df["n_sites"] >= min_sites)
    )

    ranges: List[Tuple[float, float]] = []
    start = None
    for i, is_good in enumerate(good):
        if is_good and start is None:
            start = df.loc[i, "threshold"]
        elif not is_good and start is not None:
            ranges.append((start, df.loc[i - 1, "threshold"]))
            start = None
    if start is not None:
        ranges.append((start, df.loc[len(df) - 1, "threshold"]))

    return ranges


# ─── backward-compatible aliases ─────────────────────────────────────

sweep_cutoffs = sweep_thresholds


# ─── MRT cut-off sweep ───────────────────────────────────────────────


def _fit_mrt_at_cutoff(
    pollution_score: pd.Series,
    env_all: pd.DataFrame,
    taxa_all: pd.DataFrame,
    cutoff: float,
    *,
    env_variables: List[str],
    env_short: List[str],
    taxa_columns: List[str],
    taxa_transform: str = "chord",
    k_folds: int = 10,
    cv_perms: int = 100,
    minsplit: int = 5,
    minbucket: int = 2,
) -> Dict[str, float]:
    """Fit MRT on the lowest-*cutoff* fraction of sites and return metrics.

    Returns
    -------
    dict
        Keys: cutoff, n_sites, min_cvre, cvre_se, best_tree_size,
        best_cp, root_node_error.
    """
    from ..core.mrt import fit_mrt

    ref_mask = select_reference_sites(pollution_score, quantile=cutoff)
    n_ref = int(ref_mask.sum())

    env_ref = env_all.loc[ref_mask].copy()
    taxa_ref = taxa_all.loc[ref_mask].copy()

    # Use only requested taxa and env columns
    taxa_cols_present = [c for c in taxa_columns if c in taxa_ref.columns]
    taxa_ref = taxa_ref[taxa_cols_present]
    env_vars_present = [v for v in env_variables if v in env_ref.columns]
    env_ref = env_ref[env_vars_present].copy()

    # Rename env columns to short names
    env_ref.columns = [env_short[env_variables.index(c)] for c in env_ref.columns]

    # Drop NaN
    valid = env_ref.dropna().index.intersection(taxa_ref.dropna().index)
    env_ref = env_ref.loc[valid]
    taxa_ref = taxa_ref.loc[valid]
    n_valid = len(valid)

    # Need enough sites for meaningful MRT
    if n_valid < minsplit + 2:
        return {
            "cutoff": cutoff,
            "n_sites": n_valid,
            "min_cvre": float("nan"),
            "cvre_se": float("nan"),
            "best_tree_size": float("nan"),
            "best_cp": float("nan"),
            "root_node_error": float("nan"),
        }

    # Apply taxa transform
    if taxa_transform in _TRANSFORMS:
        taxa_response = _TRANSFORMS[taxa_transform](taxa_ref)
    else:
        raise ValueError(f"Unknown taxa_transform={taxa_transform!r}")
    taxa_response.index = valid

    try:
        mrt_result, _runtime = fit_mrt(
            taxa_response,
            env_ref,
            ref_mask=ref_mask,
            ref_stations=valid,
            taxa_ref_octave=taxa_ref,
            reference_quantile=cutoff,
            response_transform=taxa_transform,
            env_variables=list(env_variables),
            taxa_columns=list(taxa_columns),
            k_folds=k_folds,
            cv_perms=cv_perms,
            minsplit=minsplit,
            minbucket=minbucket,
        )
        return {
            "cutoff": cutoff,
            "n_sites": n_valid,
            "min_cvre": mrt_result.min_cv_error,
            "cvre_se": mrt_result.min_cv_se,
            "best_tree_size": mrt_result.pruned_leaves,
            "best_cp": mrt_result.best_cp,
            "root_node_error": mrt_result.root_node_error,
        }
    except Exception as e:
        return {
            "cutoff": cutoff,
            "n_sites": n_valid,
            "min_cvre": float("nan"),
            "cvre_se": float("nan"),
            "best_tree_size": float("nan"),
            "best_cp": float("nan"),
            "root_node_error": float("nan"),
        }


def sweep_cutoffs_mrt(
    pollution_score: pd.Series,
    env_all: pd.DataFrame,
    taxa_all: pd.DataFrame,
    *,
    cutoffs: Sequence[float] | None = None,
    env_variables: List[str],
    env_short: List[str],
    taxa_columns: List[str],
    taxa_transform: str = "chord",
    k_folds: int = 10,
    cv_perms: int = 100,
    minsplit: int = 5,
    minbucket: int = 2,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run MRT at each cut-off and return a tidy metrics table.

    Returns
    -------
    pd.DataFrame
        Columns: cutoff, n_sites, min_cvre, cvre_se, best_tree_size,
        best_cp, root_node_error.
    """
    if cutoffs is None:
        cutoffs = list(np.arange(0.10, 1.01, 0.02).round(2))

    rows = []
    for i, co in enumerate(cutoffs):
        if verbose:
            print(f"  [{i+1}/{len(cutoffs)}] cut-off = {co:.0%} …", end="")

        row = _fit_mrt_at_cutoff(
            pollution_score, env_all, taxa_all, co,
            env_variables=env_variables,
            env_short=env_short,
            taxa_columns=taxa_columns,
            taxa_transform=taxa_transform,
            k_folds=k_folds,
            cv_perms=cv_perms,
            minsplit=minsplit,
            minbucket=minbucket,
        )
        rows.append(row)

        if verbose:
            print(f"  n={row['n_sites']}, CVRE={row['min_cvre']:.4f}, "
                  f"SE={row['cvre_se']:.4f}, tree={row['best_tree_size']}")

    return pd.DataFrame(rows)
