"""SIMPROF — Similarity Profile permutation test (Clarke et al. 2008).

Pure computation only: no plotting, no file paths.

Reference
---------
Clarke, K.R., Somerfield, P.J. & Gorley, R.N. (2008). Testing of null
hypotheses in exploratory community analyses: similarity profiles and
biota-environment linkage. J. Exp. Mar. Biol. Ecol. 366(1-2), 56-69.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _sim_profile(X: np.ndarray, metric: str, max_dist: float | None = None) -> np.ndarray:
    """Return sorted-descending condensed pairwise similarities."""
    dists = pdist(X, metric=metric)
    if metric == "braycurtis":
        sims = 1.0 - dists
    else:
        # Euclidean: normalise by max to get [0, 1] similarities
        if max_dist is None:
            max_dist = dists.max()
        denom = max_dist if max_dist > 0 else 1.0
        sims = 1.0 - dists / denom
    return np.sort(sims)[::-1]


# ------------------------------------------------------------------
# Single-group SIMPROF test
# ------------------------------------------------------------------


def simprof_test(
    taxa: pd.DataFrame,
    n_perm: int = 999,
    alpha: float = 0.05,
    metric: str = "braycurtis",
    random_state: int | None = 42,
    verbose: bool = False,
) -> dict:
    """SIMPROF permutation test for a single group of samples.

    Tests H0: the group is internally homogeneous (no significant
    clustering structure).  A significant result (p < alpha) means the
    group has structure and the clustering split is meaningful.

    Parameters
    ----------
    taxa : pd.DataFrame
        Taxa matrix, shape (n_sites, n_taxa).  Values should already be
        transformed (octave, chord, etc.) to match the upstream Ward step.
    n_perm : int
        Number of permutation replicates (default 999).
    alpha : float
        Significance threshold (default 0.05).
    metric : str
        Pairwise distance metric passed to ``scipy.spatial.distance.pdist``.
        Use ``"braycurtis"`` (default) for community ecology data or
        ``"euclidean"`` to match the Ward clustering metric.
    random_state : int or None
        RNG seed for reproducibility.
    verbose : bool
        Print progress during permutations.

    Returns
    -------
    dict
        Keys: T_obs, T_perm, p_value, significant, pi_obs, pi_mean_perm,
        n_sites, n_perm, alpha, metric.
    """
    rng = np.random.default_rng(random_state)
    X = taxa.values.copy().astype(float)
    n_sites, n_taxa = X.shape

    if n_sites < 3:
        empty = np.array([])
        return {
            "T_obs": np.nan,
            "T_perm": empty,
            "p_value": np.nan,
            "significant": False,
            "pi_obs": empty,
            "pi_mean_perm": empty,
            "n_sites": n_sites,
            "n_perm": n_perm,
            "alpha": alpha,
            "metric": metric,
        }

    # Euclidean needs a fixed scale reference computed from observed data
    if metric != "braycurtis":
        max_dist = float(pdist(X, metric=metric).max())
    else:
        max_dist = None

    # Observed similarity profile
    pi_obs = _sim_profile(X, metric, max_dist)
    n_pairs = len(pi_obs)

    # Null profiles via column-wise permutation (preserves species marginals)
    null_profiles = np.empty((n_perm, n_pairs), dtype=float)
    for k in range(n_perm):
        if verbose and (k + 1) % 100 == 0:
            print(f"    SIMPROF permutation {k + 1}/{n_perm}", flush=True)
        X_perm = X.copy()
        for col in range(n_taxa):
            rng.shuffle(X_perm[:, col])
        null_profiles[k] = _sim_profile(X_perm, metric, max_dist)

    pi_mean_null = null_profiles.mean(axis=0)

    # Test statistic: sum of absolute deviations from mean null profile
    T_obs = float(np.sum(np.abs(pi_obs - pi_mean_null)))
    T_perm = np.array(
        [float(np.sum(np.abs(null_profiles[k] - pi_mean_null))) for k in range(n_perm)]
    )

    p_value = float((T_perm >= T_obs).sum()) / n_perm

    return {
        "T_obs": T_obs,
        "T_perm": T_perm,
        "p_value": p_value,
        "significant": p_value < alpha,
        "pi_obs": pi_obs,
        "pi_mean_perm": pi_mean_null,
        "n_sites": n_sites,
        "n_perm": n_perm,
        "alpha": alpha,
        "metric": metric,
    }


# ------------------------------------------------------------------
# Multi-group runner (full set + per-cluster)
# ------------------------------------------------------------------


def run_simprof_analysis(
    taxa_ref_transformed: pd.DataFrame,
    cluster_labels: pd.Series,
    n_perm: int = 999,
    alpha: float = 0.05,
    metric: str = "braycurtis",
    random_state: int | None = 42,
    verbose: bool = True,
) -> list[dict]:
    """Run SIMPROF on the full reference set and on each cluster separately.

    Parameters
    ----------
    taxa_ref_transformed : pd.DataFrame
        Transformed taxa matrix for reference sites (sites × taxa).
    cluster_labels : pd.Series
        1-indexed cluster assignments, same index as *taxa_ref_transformed*.
    n_perm : int
        Permutation replicates (default 999).
    alpha : float
        Significance threshold (default 0.05).
    metric : str
        Distance metric (default ``"braycurtis"``).
    random_state : int or None
        RNG seed.
    verbose : bool
        Print progress.

    Returns
    -------
    list of dict
        One entry per group tested.  Each dict is the output of
        :func:`simprof_test` augmented with ``"group_label"``.
    """
    results: list[dict] = []

    # 1. Full reference set
    if verbose:
        n_total = len(taxa_ref_transformed)
        print(f"  SIMPROF — full reference set  (N={n_total}, n_perm={n_perm})")
    r_all = simprof_test(
        taxa_ref_transformed,
        n_perm=n_perm,
        alpha=alpha,
        metric=metric,
        random_state=random_state,
        verbose=verbose,
    )
    r_all["group_label"] = "All reference sites"
    results.append(r_all)

    # 2. Per-cluster
    cluster_ids = sorted(cluster_labels.unique())
    for cid in cluster_ids:
        mask = cluster_labels == cid
        taxa_cluster = taxa_ref_transformed.loc[mask]
        n_c = len(taxa_cluster)
        if verbose:
            print(f"  SIMPROF — Cluster {cid}  (N={n_c}, n_perm={n_perm})")
        # Use a reproducible but distinct seed per cluster
        seed = None if random_state is None else (random_state + int(cid) * 7)
        r_c = simprof_test(
            taxa_cluster,
            n_perm=n_perm,
            alpha=alpha,
            metric=metric,
            random_state=seed,
            verbose=verbose,
        )
        r_c["group_label"] = f"Cluster {cid}"
        results.append(r_c)

    return results


# ------------------------------------------------------------------
# Summary table helper
# ------------------------------------------------------------------


def simprof_summary_table(results: list[dict]) -> pd.DataFrame:
    """Convert a list of SIMPROF result dicts into a summary DataFrame."""
    rows = []
    for r in results:
        sig = r["significant"]
        rows.append(
            {
                "Group": r["group_label"],
                "N_Sites": r["n_sites"],
                "T_obs": round(r["T_obs"], 4) if not np.isnan(r["T_obs"]) else np.nan,
                "p_value": round(r["p_value"], 4) if not np.isnan(r["p_value"]) else np.nan,
                "Significant": sig,
                "alpha": r["alpha"],
                "Metric": r["metric"],
                "n_perm": r["n_perm"],
                "Interpretation": (
                    "Heterogeneous — clustering is meaningful"
                    if sig
                    else "Homogeneous — no further split warranted"
                ),
            }
        )
    return pd.DataFrame(rows)
