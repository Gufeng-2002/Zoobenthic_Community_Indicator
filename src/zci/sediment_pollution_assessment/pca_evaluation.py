"""
PCA assessment evaluation utilities.

This module provides:
- Group construction from pollution scores or labels (reference/medium/degraded).
- A distance-based multivariate permutation test (PERMANOVA-like using Euclidean space).
- A compact wrapper that takes the master dataframe and runs the test on selected blocks.
- A simple loss function to optimize weight settings: loss = -log10(p) * R2.

Notes
-----
- The permutation test implemented here uses Euclidean geometry on the feature matrix X,
  computing SS_between/SS_total and an F-like statistic, then permuting group labels
  to obtain a p-value. This mirrors PERMANOVA for Euclidean distances.
- If you need alternative distance metrics (e.g., Bray-Curtis), this can be extended
  to operate on distance matrices directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict

import numpy as np
import pandas as pd

try:
    # Optional: if scikit-bio is available, we can add an adapter later
    import skbio  # type: ignore
    _HAVE_SKBIO = True
except Exception:
    _HAVE_SKBIO = False

# Public API
__all__ = [
    "build_groups_from_labels",
    "build_groups_from_quantiles",
    "prepare_feature_matrix",
    "permutation_manova_euclidean",
    "evaluate_pca_assessment",
    "loss_from_result",
    "directional_mean_permutation_test",
    "evaluate_directional_mean_test",
    "plot_permanova_null_distribution",
    "plot_directional_null_distribution",
    "run_assessment_suite",
]


@dataclass
class EvalResult:
    groups: pd.Series  # index aligned to samples, categorical labels
    F_stat: float
    p_value: float
    R2: float
    df_between: int
    df_within: int
    n_perm: int
    method: str = "permutation_euclidean"
    meta: Optional[Dict] = None


# --------------------------- Grouping helpers ---------------------------

def build_groups_from_labels(master: pd.DataFrame,
                             pollution_block: str = "pollution",
                             pollution_subblock: str = "sumreal_by_logz_chemical",
                             label_var: str = "Quality") -> pd.Series:
    """
    Use existing Quality labels stored under (pollution_block, pollution_subblock, label_var).

    Returns a Series of group labels indexed by master index.
    """
    col = (pollution_block, pollution_subblock, label_var)
    if col not in master.columns:
        raise KeyError(f"Label column {col} not found in master")
    labels = master[col]
    return labels.astype("category")


def build_groups_from_quantiles(master: pd.DataFrame,
                                pollution_block: str = "pollution",
                                pollution_subblock: str = "sumreal_by_logz_chemical",
                                score_var: str = "SumReal",
                                low_q: float = 0.2,
                                high_q: float = 0.8) -> pd.Series:
    """
    Build groups based on score quantiles into three bins: bottom/middle/top.
    """
    col = (pollution_block, pollution_subblock, score_var)
    if col not in master.columns:
        raise KeyError(f"Score column {col} not found in master")
    scores = master[col]
    lo = scores.quantile(low_q)
    hi = scores.quantile(high_q)
    def _lab(v: float) -> str:
        if v <= lo:
            return "bottom"
        elif v >= hi:
            return "top"
        else:
            return "middle"
    groups = scores.apply(_lab)
    return groups.astype("category")


# ---------------------- Feature matrix preparation ----------------------

def prepare_feature_matrix(master: pd.DataFrame,
                           block: str = "chemical",
                           subblock: str = "raw",
                           variables: Optional[Iterable[str]] = None,
                           standardize: bool = True) -> Tuple[np.ndarray, pd.Index]:
    """
    Extract an (n_samples x n_features) matrix from master[(block, subblock)].

    - variables: optional subset of columns to use (by variable name at level 2).
    - standardize: if True, z-score each feature to mean=0, std=1 (ignores all-NaN columns).

    Returns (X, index) where index is sample index aligned with master.
    """
    # Pull the sub-dataframe
    sub = master[(block, subblock)]
    if variables is not None:
        missing = [v for v in variables if v not in sub.columns]
        if missing:
            raise KeyError(f"Variables not found in ({block}, {subblock}): {missing}")
        sub = sub.loc[:, list(variables)]

    # Drop columns with all-NaN
    sub = sub.dropna(axis=1, how='all')
    # Simple impute: fill remaining NaN with column means to keep rows aligned
    if sub.isna().any().any():
        sub = sub.apply(lambda c: c.fillna(c.mean()), axis=0)

    X = sub.to_numpy(dtype=float)
    if standardize:
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=1)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd
    return X, sub.index


# ----------------- Euclidean permutation MANOVA (PERMANOVA-like) -----------------

def _ss_total(X: np.ndarray) -> float:
    mu = X.mean(axis=0, keepdims=True)
    diff = X - mu
    return float(np.sum(diff * diff))


def _ss_between(X: np.ndarray, groups: np.ndarray) -> float:
    mu = X.mean(axis=0, keepdims=True)
    ssb = 0.0
    for g in np.unique(groups):
        idx = groups == g
        if not np.any(idx):
            continue
        n_g = int(idx.sum())
        mu_g = X[idx].mean(axis=0, keepdims=True)
        d = mu_g - mu
        ssb += n_g * float(np.sum(d * d))
    return float(ssb)


def permutation_manova_euclidean(X: np.ndarray,
                                 group_labels: Iterable[str],
                                 permutations: int = 999,
                                 rng: Optional[np.random.Generator] = None) -> EvalResult:
    """
    Distance-based MANOVA for Euclidean space with permutation p-value.

    Returns EvalResult with F, p, R2.
    """
    if rng is None:
        rng = np.random.default_rng()

    groups = np.asarray(pd.Categorical(group_labels))
    n = X.shape[0]
    levels = np.unique(groups)
    k = len(levels)
    if k < 2:
        raise ValueError("Need at least 2 groups for the test")

    ssT = _ss_total(X)
    ssB = _ss_between(X, groups)
    ssW = ssT - ssB
    dfB = k - 1
    dfW = n - k
    msB = ssB / dfB if dfB > 0 else np.nan
    msW = ssW / dfW if dfW > 0 else np.nan
    F_obs = msB / msW if msW > 0 else np.inf
    R2 = ssB / ssT if ssT > 0 else 0.0

    # Permute labels
    count_ge = 1  # include observed
    for _ in range(permutations):
        perm = rng.permutation(groups)
        ssB_p = _ss_between(X, perm)
        ssW_p = ssT - ssB_p
        msB_p = ssB_p / dfB if dfB > 0 else np.nan
        msW_p = ssW_p / dfW if dfW > 0 else np.nan
        F_p = msB_p / msW_p if msW_p > 0 else np.inf
        if F_p >= F_obs:
            count_ge += 1
    p_val = count_ge / (permutations + 1)

    return EvalResult(
        groups=pd.Series(groups, index=np.arange(n), dtype="category"),
        F_stat=float(F_obs),
        p_value=float(p_val),
        R2=float(R2),
        df_between=int(dfB),
        df_within=int(dfW),
        n_perm=int(permutations),
        method="permutation_euclidean",
        meta={"levels": levels.tolist()},
    )


# ------------------------------ High-level API ------------------------------

def evaluate_pca_assessment(master: pd.DataFrame,
                            features_block: str = "chemical",
                            features_subblock: str = "raw",
                            pollution_block: str = "pollution",
                            pollution_subblock: str = "sumreal_by_logz_chemical",
                            group_mode: str = "labels",
                            variables: Optional[Iterable[str]] = None,
                            standardize: bool = True,
                            permutations: int = 999) -> EvalResult:
    """
    One-call evaluation:
    - Build X from master[(features_block, features_subblock)].
    - Build grouping from pollution labels or quantiles.
    - Run Euclidean permutation MANOVA.

    group_mode: 'labels' uses (pollution_block, pollution_subblock, 'Quality'),
                'quantiles' builds bottom/middle/top via SumReal quantiles (0.2, 0.8).
    """
    X, idx = prepare_feature_matrix(master, features_block, features_subblock, variables, standardize)
    if group_mode == "labels":
        groups = build_groups_from_labels(master, pollution_block, pollution_subblock)
    elif group_mode == "quantiles":
        groups = build_groups_from_quantiles(master, pollution_block, pollution_subblock)
    else:
        raise ValueError("group_mode must be 'labels' or 'quantiles'")

    # Align groups to X index
    if not groups.index.equals(idx):
        groups = groups.reindex(idx)
    if groups.isna().any():
        raise ValueError("Grouping labels contain NaN after alignment; check indexes")

    res = permutation_manova_euclidean(X, groups, permutations=permutations)
    # Attach names for clarity
    res.meta = (res.meta or {})
    res.meta.update({
        "features_block": features_block,
        "features_subblock": features_subblock,
        "pollution_block": pollution_block,
        "pollution_subblock": pollution_subblock,
        "group_mode": group_mode,
    })
    # Preserve original sample index on groups
    res.groups = pd.Series(pd.Categorical(groups), index=idx)
    return res


# ------------------------------- Loss function -------------------------------

def loss_from_result(result: EvalResult, *, min_p: float = 1e-12, weight_R2: float = 1.0) -> float:
    """
    Turn a test result into a scalar to maximize when tuning weights.

    loss = -log10(max(p, min_p)) * (R2 ** weight_R2)

    Larger is better (stronger separation and effect size).
    """
    p = max(result.p_value, min_p)
    return float(-np.log10(p) * (result.R2 ** weight_R2))


# -------------------- Directional one-sided permutation test --------------------

@dataclass
class DirMeanTestResult:
    statistic: float  # aggregated one-sided z-like score (higher => more evidence for degraded > reference)
    p_value: float
    per_variable: Dict[str, Dict[str, float]]  # var -> {mean_ref, mean_deg, diff, z}
    reference_label: str
    degraded_label: str
    mode: str  # 'average' or 'min'
    weights: Optional[Dict[str, float]]
    n_perm: int


def _dir_score(mean_ref: np.ndarray, mean_deg: np.ndarray, sd: np.ndarray,
               weights: Optional[np.ndarray], mode: str) -> float:
    # z-like per variable
    sd_safe = sd.copy()
    sd_safe[sd_safe == 0] = 1.0
    z = (mean_deg - mean_ref) / sd_safe
    if mode == "min":
        return float(np.min(z))
    # default average (weighted)
    if weights is None:
        return float(np.mean(z))
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size != z.size:
        raise ValueError("weights must be a vector aligned to variables")
    w_sum = np.sum(np.abs(w))
    if w_sum == 0:
        return float(np.mean(z))
    return float(np.dot(w, z) / w_sum)


def directional_mean_permutation_test(X: np.ndarray,
                                      groups: Iterable[str],
                                      reference_label: str,
                                      degraded_label: str,
                                      var_names: Optional[Iterable[str]] = None,
                                      *,
                                      standardize: bool = False,
                                      weights: Optional[Iterable[float]] = None,
                                      permutations: int = 999,
                                      rng: Optional[np.random.Generator] = None,
                                      mode: str = "average") -> DirMeanTestResult:
    """
    One-sided directional test for the hypothesis that mean(degraded) > mean(reference)
    across chemicals (features). Aggregates standardized per-variable differences using
    either the average (weighted) or the minimum (intersection-union style) and computes
    a permutation p-value by shuffling group labels between the two groups.

    - standardize: if True, z-score columns before computing means (optional; not required).
    - weights: optional variable weights (positive). If provided, must match number of features.
    - mode: 'average' (default) or 'min'.
    """
    if rng is None:
        rng = np.random.default_rng()
    g = pd.Series(groups)
    keep = g.isin([reference_label, degraded_label])
    X2 = X[keep.to_numpy()]
    g2 = g[keep].reset_index(drop=True)
    if X2.shape[0] == 0:
        raise ValueError("No samples after filtering groups")
    if (g2 == reference_label).sum() == 0 or (g2 == degraded_label).sum() == 0:
        raise ValueError("Both reference and degraded groups must have at least one sample")

    if standardize:
        mu = np.nanmean(X2, axis=0)
        sd = np.nanstd(X2, axis=0, ddof=1)
        sd[sd == 0] = 1.0
        X2 = (X2 - mu) / sd

    # Overall SD per variable for z-like scaling (using all samples in X2)
    sd_all = np.nanstd(X2, axis=0, ddof=1)
    sd_all[sd_all == 0] = 1.0

    X_ref = X2[(g2 == reference_label).to_numpy()]
    X_deg = X2[(g2 == degraded_label).to_numpy()]
    mean_ref = np.nanmean(X_ref, axis=0)
    mean_deg = np.nanmean(X_deg, axis=0)

    w_vec = None if weights is None else np.asarray(list(weights), dtype=float)
    stat_obs = _dir_score(mean_ref, mean_deg, sd_all, w_vec, mode)

    # Permutations: shuffle between the two groups
    labels = g2.to_numpy()
    count_ge = 1  # include observed
    for _ in range(permutations):
        perm = rng.permutation(labels)
        X_ref_p = X2[perm == reference_label]
        X_deg_p = X2[perm == degraded_label]
        if X_ref_p.size == 0 or X_deg_p.size == 0:
            # extremely unlikely with balanced labels; skip counting
            continue
        m_ref_p = np.nanmean(X_ref_p, axis=0)
        m_deg_p = np.nanmean(X_deg_p, axis=0)
        s_p = _dir_score(m_ref_p, m_deg_p, sd_all, w_vec, mode)
        if s_p >= stat_obs:
            count_ge += 1
    p_val = count_ge / (permutations + 1)

    # Per-variable details
    if var_names is None:
        var_names = [f"v{i}" for i in range(X.shape[1])]
    var_names = list(var_names)
    sd_safe = sd_all.copy()
    sd_safe[sd_safe == 0] = 1.0
    z = (mean_deg - mean_ref) / sd_safe
    per_var = {
        name: {
            "mean_ref": float(mean_ref[i]),
            "mean_deg": float(mean_deg[i]),
            "diff": float(mean_deg[i] - mean_ref[i]),
            "z": float(z[i]),
        }
        for i, name in enumerate(var_names)
    }

    w_map = None
    if w_vec is not None:
        w_map = {name: float(w_vec[i]) for i, name in enumerate(var_names)}

    return DirMeanTestResult(
        statistic=float(stat_obs),
        p_value=float(p_val),
        per_variable=per_var,
        reference_label=str(reference_label),
        degraded_label=str(degraded_label),
        mode=mode,
        weights=w_map,
        n_perm=int(permutations),
    )


def evaluate_directional_mean_test(master: pd.DataFrame,
                                   features_block: str = "chemical",
                                   features_subblock: str = "logz",
                                   *,
                                   group_mode: str = "labels",
                                   pollution_block: str = "pollution",
                                   pollution_subblock: str = "sumreal_by_logz_chemical",
                                   reference_label: Optional[str] = None,
                                   degraded_label: Optional[str] = None,
                                   variables: Optional[Iterable[str]] = None,
                                   standardize: bool = True,
                                   weights: Optional[Iterable[float]] = None,
                                   permutations: int = 999,
                                   mode: str = "average") -> DirMeanTestResult:
    """
    Convenience wrapper to run the directional one-sided test directly on master:
    - Extracts X from master[(features_block, features_subblock)]
    - Builds groups by labels or quantiles
    - Chooses reference/degraded labels automatically if not provided:
        labels: expects categories like {'reference','degraded'}
        quantiles: uses {'bottom','top'} where top is degraded
    - Runs directional_mean_permutation_test
    """
    X, idx = prepare_feature_matrix(master, features_block, features_subblock, variables, standardize)
    if group_mode == "labels":
        g = build_groups_from_labels(master, pollution_block, pollution_subblock)
        ref = reference_label or "reference"
        deg = degraded_label or "degraded"
    elif group_mode == "quantiles":
        g = build_groups_from_quantiles(master, pollution_block, pollution_subblock)
        ref = reference_label or "bottom"
        deg = degraded_label or "top"
    else:
        raise ValueError("group_mode must be 'labels' or 'quantiles'")
    if not g.index.equals(idx):
        g = g.reindex(idx)
    var_names = list(master[(features_block, features_subblock)].columns)
    return directional_mean_permutation_test(
        X, g, ref, deg, var_names=var_names,
        standardize=False,  # already standardized via prepare_feature_matrix if requested
        weights=weights,
        permutations=permutations,
        mode=mode,
    )


# ------------------------------ Plotting helpers ------------------------------

def plot_permanova_null_distribution(
    X: np.ndarray,
    groups: Iterable[str],
    *,
    permutations: int = 499,
    seed: Optional[int] = 0,
    ax=None,
    title: Optional[str] = None,
):
    """
    Plot the permutation null distribution for the PERMANOVA-like F statistic.

    Parameters
    ----------
    X : ndarray (n_samples x n_features)
    groups : iterable of labels, length n_samples
    permutations : int, number of permutations for the histogram (default 499)
    seed : int | None, RNG seed for reproducibility
    ax : matplotlib Axes or None; if None, a new fig/ax are created
    title : Optional title string

    Returns
    -------
    (fig, ax, info) where info is a dict with keys:
      F_obs, p_value, R2, df_between, df_within, n_perm, p_tail, group_counts
    """
    import matplotlib.pyplot as plt
    import numpy as _np
    from collections import Counter

    # Compute observed using our permutation test (also gives R2/df)
    res = permutation_manova_euclidean(X, groups, permutations=permutations)
    F_obs = res.F_stat
    p_obs = res.p_value
    R2_obs = res.R2
    dfB = res.df_between
    dfW = res.df_within
    n_perm = res.n_perm

    g_arr = _np.asarray(pd.Categorical(groups))
    rng = _np.random.default_rng(seed)
    ssT = _ss_total(X)
    k = len(_np.unique(g_arr))
    n = X.shape[0]
    dfB_local = k - 1
    dfW_local = n - k

    B = int(n_perm)
    F_null = _np.empty(B)
    for b in range(B):
        perm = rng.permutation(g_arr)
        ssB_p = _ss_between(X, perm)
        ssW_p = ssT - ssB_p
        msB_p = ssB_p / dfB_local if dfB_local > 0 else _np.nan
        msW_p = ssW_p / dfW_local if dfW_local > 0 else _np.nan
        F_null[b] = msB_p / msW_p if msW_p > 0 else _np.inf

    p_tail = (int(_np.sum(F_null >= F_obs)) + 1) / (B + 1)
    counts = Counter(g_arr)
    group_text = ", ".join([f"{k}:{v}" for k, v in counts.items()])

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        created_fig = True
    else:
        fig = ax.figure

    bins = max(10, int(_np.sqrt(B)))
    ax.hist(F_null, bins=bins, color="#cbd5e1", edgecolor="#94a3b8", alpha=0.9, label="Permutation null")
    ax.axvline(F_obs, color="#ef4444", lw=2, label=f"Observed F = {F_obs:.3f}")

    xmin, xmax = ax.get_xlim()
    xs = _np.linspace(F_obs, xmax, 200)
    ax.fill_between(xs, 0, ax.get_ylim()[1], color="#ef4444", alpha=0.15, label=f"Tail p ≈ {p_tail:.3f}")

    ttl = title or "PERMANOVA-like F null distribution"
    subtitle = f"p={p_obs:.3f}, null≈{p_tail:.3f}, R2={R2_obs:.3f}, df=({dfB},{dfW}), perms={B}"
    ax.set_title(ttl + "\n" + subtitle)
    ax.set_xlabel("F statistic under H0 (permuted labels)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right")
    ax.text(0.02, 0.95, f"Groups: {group_text}", transform=ax.transAxes, va="top", fontsize=9, color="#374151")

    if created_fig:
        fig.tight_layout()

    info = {
        "F_obs": float(F_obs),
        "p_value": float(p_obs),
        "R2": float(R2_obs),
        "df_between": int(dfB),
        "df_within": int(dfW),
        "n_perm": int(B),
        "p_tail": float(p_tail),
        "group_counts": dict(counts),
    }
    return fig, ax, info


def plot_directional_null_distribution(
    X: np.ndarray,
    labels: Iterable[str],
    reference_label: str,
    degraded_label: str,
    *,
    mode: str = "average",
    permutations: int = 499,
    standardize: bool = True,
    weights: Optional[Iterable[float]] = None,
    var_names: Optional[Iterable[str]] = None,
    seed: Optional[int] = 0,
    ax=None,
    title: Optional[str] = None,
):
    """
    Plot the directional one-sided permutation null (degraded > reference).

    Parameters
    ----------
    X : ndarray (n_samples x n_features)
    labels : iterable of group labels
    reference_label, degraded_label : str
    mode : 'average' or 'min'
    permutations : number of permutations for both observed test and null histogram
    standardize : if True, z-score columns before computing stat
    weights : optional per-variable weights
    var_names : optional variable names (for annotation in returned info)
    seed : RNG seed for null histogram
    ax : matplotlib Axes or None
    title : optional title

    Returns
    -------
    (fig, ax, info) where info contains statistic, p_value, p_tail, labels, top_vars
    """
    import matplotlib.pyplot as plt
    import numpy as _np

    # Compute observed statistic and p via the core API
    res = directional_mean_permutation_test(
        X, labels, reference_label, degraded_label,
        var_names=var_names, standardize=standardize,
        weights=weights, permutations=permutations, mode=mode,
    )

    # Build null by shuffling between ref/deg only
    g = pd.Series(labels)
    mask_keep = g.isin([reference_label, degraded_label]).to_numpy()
    X2 = X[mask_keep]
    labs2 = g[mask_keep].to_numpy()
    rng = _np.random.default_rng(seed)
    B = permutations

    # If standardize=True, standardize once using all X2 (same as core test approach)
    if standardize:
        mu = _np.nanmean(X2, axis=0)
        sd = _np.nanstd(X2, axis=0, ddof=1)
        sd[sd == 0] = 1.0
        X2s = (X2 - mu) / sd
    else:
        X2s = X2

    # Shared sd for z-like scaling
    sd_all = _np.nanstd(X2s, axis=0, ddof=1)
    sd_all[sd_all == 0] = 1.0

    def _dir_stat_avg(Xa, la):
        m_ref = _np.nanmean(Xa[la == reference_label], axis=0)
        m_deg = _np.nanmean(Xa[la == degraded_label], axis=0)
        z = (m_deg - m_ref) / sd_all
        if mode == "min":
            return float(_np.min(z))
        if weights is None:
            return float(_np.mean(z))
        w = _np.asarray(list(weights), dtype=float)
        w_sum = _np.sum(_np.abs(w))
        return float(_np.dot(w, z) / (w_sum if w_sum != 0 else len(w)))

    S_null = _np.empty(B)
    for b in range(B):
        perm = rng.permutation(labs2)
        S_null[b] = _dir_stat_avg(X2s, perm)

    p_tail = (int(_np.sum(S_null >= res.statistic)) + 1) / (B + 1)
    top_vars = sorted(res.per_variable, key=lambda k: res.per_variable[k].get('z', 0), reverse=True)[:5]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        created_fig = True
    else:
        fig = ax.figure

    bins = max(10, int(_np.sqrt(B)))
    ax.hist(S_null, bins=bins, color="#fde68a", edgecolor="#f59e0b", alpha=0.9, label="Permutation null (directional)")
    ax.axvline(res.statistic, color="#b91c1c", lw=2, label=f"Observed stat = {res.statistic:.3f}")
    xmin, xmax = ax.get_xlim()
    xs = _np.linspace(res.statistic, xmax, 200)
    ax.fill_between(xs, 0, ax.get_ylim()[1], color="#b91c1c", alpha=0.15, label=f"Tail p ≈ {p_tail:.3f}")
    ttl = title or "Directional one-sided null (degraded > reference)"
    subtitle = f"mode={res.mode}, p={res.p_value:.3f}, null≈{p_tail:.3f}, perms={B}, ref='{res.reference_label}', deg='{res.degraded_label}'"
    ax.set_title(ttl + "\n" + subtitle)
    ax.set_xlabel("Directional statistic under H0 (shuffled labels)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right")

    if created_fig:
        fig.tight_layout()

    info = {
        "statistic": float(res.statistic),
        "p_value": float(res.p_value),
        "p_tail": float(p_tail),
        "mode": res.mode,
        "reference_label": res.reference_label,
        "degraded_label": res.degraded_label,
        "n_perm": int(B),
        "top_vars": top_vars,
    }
    return fig, ax, info


def _infer_ref_deg_from_labels(labels: Iterable[str]) -> tuple[str, str]:
    cats = list(pd.Categorical(labels).categories)
    ref_candidates = [c for c in cats if str(c).lower().startswith("ref")]
    deg_candidates = [c for c in cats if str(c).lower().startswith("deg")]
    if ref_candidates and deg_candidates:
        return ref_candidates[0], deg_candidates[0]
    # quantile-like convention
    if "bottom" in cats and "top" in cats:
        return "bottom", "top"
    # fallback: first and last by sort
    return min(cats), max(cats)


def run_assessment_suite(
    X: np.ndarray,
    labels: Iterable[str],
    *,
    reference_label: Optional[str] = None,
    degraded_label: Optional[str] = None,
    var_names: Optional[Iterable[str]] = None,
    permutations: int = 499,
    seed: Optional[int] = 0,
    standardize_directional: bool = True,
) -> Dict:
    """
    High-level API: run 5 analyses on features and labels and return a summary dict.

    Tests included:
      1) PERMANOVA-like Euclidean (direct)
      2) PERMANOVA null-tail estimate for F (hist-based p_tail)
      3) Directional test (average mode)
      4) Directional test (min mode)
      5) loss = -log10(p) * R2 (from PERMANOVA result)

    Parameters align with plotting helpers for consistency.
    """
    # 1) Direct PERMANOVA with permutations
    perm_res = permutation_manova_euclidean(X, labels, permutations=permutations)

    # 2) Build null for F (tail p) using same B for comparability
    g_arr = np.asarray(pd.Categorical(labels))
    rng = np.random.default_rng(seed)
    ssT = _ss_total(X)
    k = len(np.unique(g_arr))
    n = X.shape[0]
    dfB_local = k - 1
    dfW_local = n - k
    B = int(perm_res.n_perm)
    F_null = np.empty(B)
    for b in range(B):
        perm = rng.permutation(g_arr)
        ssB_p = _ss_between(X, perm)
        ssW_p = ssT - ssB_p
        msB_p = ssB_p / dfB_local if dfB_local > 0 else np.nan
        msW_p = ssW_p / dfW_local if dfW_local > 0 else np.nan
        F_null[b] = msB_p / msW_p if msW_p > 0 else np.inf
    p_tail = (int(np.sum(F_null >= perm_res.F_stat)) + 1) / (B + 1)

    # 3 & 4) Directional tests (average and min)
    ref = reference_label
    deg = degraded_label
    if ref is None or deg is None:
        inf_ref, inf_deg = _infer_ref_deg_from_labels(labels)
        ref = ref or inf_ref
        deg = deg or inf_deg

    dir_avg = directional_mean_permutation_test(
        X, labels, reference_label=ref, degraded_label=deg,
        var_names=var_names, standardize=standardize_directional,
        permutations=permutations, mode="average",
    )
    dir_min = directional_mean_permutation_test(
        X, labels, reference_label=ref, degraded_label=deg,
        var_names=var_names, standardize=standardize_directional,
        permutations=permutations, mode="min",
    )

    # 5) Loss from PERMANOVA
    loss_val = loss_from_result(perm_res)

    # Pack summary (keep small, avoid storing large arrays)
    summary = {
        "perm_direct": {
            "F": perm_res.F_stat,
            "p": perm_res.p_value,
            "R2": perm_res.R2,
            "df_between": perm_res.df_between,
            "df_within": perm_res.df_within,
            "n_perm": perm_res.n_perm,
            "method": perm_res.method,
        },
        "perm_null_tail": {
            "p_tail": float(p_tail),
            "n_perm": int(B),
        },
        "loss": float(loss_val),
        "dir_avg": {
            "stat": dir_avg.statistic,
            "p": dir_avg.p_value,
            "n_perm": dir_avg.n_perm,
            "mode": dir_avg.mode,
            "reference_label": dir_avg.reference_label,
            "degraded_label": dir_avg.degraded_label,
            "top_vars": sorted(dir_avg.per_variable, key=lambda k: dir_avg.per_variable[k].get('z', 0), reverse=True)[:5],
        },
        "dir_min": {
            "stat": dir_min.statistic,
            "p": dir_min.p_value,
            "n_perm": dir_min.n_perm,
            "mode": dir_min.mode,
            "reference_label": dir_min.reference_label,
            "degraded_label": dir_min.degraded_label,
            "top_vars": sorted(dir_min.per_variable, key=lambda k: dir_min.per_variable[k].get('z', 0), reverse=True)[:5],
        },
    }
    return summary
