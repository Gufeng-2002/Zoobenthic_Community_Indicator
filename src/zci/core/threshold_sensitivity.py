"""Threshold-sensitivity analysis — pure computation, no plotting, no I/O.

Sweeps a grid of low-contamination cutoff proportions, fits RDA at each
threshold, and collects performance metrics.  Also identifies stable
threshold ranges where RDA performance is consistently strong.

Public API
----------
sweep_thresholds
    Run RDA at each threshold and return a tidy metrics DataFrame.
detect_stable_ranges
    Identify contiguous threshold ranges with consistently good performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..core.rda import RDA
from ..core.clustering import select_reference_sites


# ─── result container ───────────────────────────────────────────────


@dataclass
class ThresholdSweepResult:
    """Container for threshold-sensitivity sweep outputs."""
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
        ``"octave"`` or ``"hellinger"``.
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
    if taxa_transform == "hellinger":
        row_sums = taxa_ref.sum(axis=1)
        taxa_ref = taxa_ref.div(row_sums, axis=0).fillna(0).apply(np.sqrt)

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
        }

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
