"""Environmental-coherence diagnostics for taxa-defined cluster memberships.

Evaluates whether the original taxa-defined cluster memberships are also
coherent in environmental space.  This is *not* a second clustering-
confidence measure of the taxa data; it is an environmental-coherence
validation layer.

Public API
----------
compute_env_silhouettes
    Standard silhouette formula using environmental distances and taxa
    cluster labels.
compute_env_coassignment
    Bootstrap co-assignment matrix from Ward clustering in standardized
    environmental space.
env_site_confidence
    Own-cluster, best-alternative, and margin derived from the env
    co-assignment matrix relative to the original taxa cluster labels.
assign_env_strength
    Binary environmental coherence label (Strong / Weak).
build_env_coherence_table
    One-row-per-site summary of all environmental diagnostics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform


# ------------------------------------------------------------------
# Environmental silhouette (standard formula, env distances, taxa labels)
# ------------------------------------------------------------------

def compute_env_silhouettes(
    env_std: pd.DataFrame,
    labels: pd.Series,
) -> pd.Series:
    """Per-site silhouette using environmental distances and taxa cluster labels.

    Parameters
    ----------
    env_std : pd.DataFrame
        Standardized environmental matrix (sites × variables).
    labels : pd.Series
        Original taxa-defined cluster labels, same index as *env_std*.

    Returns
    -------
    pd.Series
        Environmental silhouette for each site.
    """
    D = squareform(pdist(env_std.values, metric="euclidean"))
    site_ids = env_std.index
    clusters = sorted(labels.unique())
    sil = pd.Series(np.nan, index=site_ids, name="Env_Silhouette")

    for idx_i, site_i in enumerate(site_ids):
        gi = labels.loc[site_i]
        own_mask = np.array([labels.loc[s] == gi and s != site_i for s in site_ids])
        n_own = own_mask.sum()

        if n_own == 0:
            sil.loc[site_i] = 0.0
            continue

        a_i = D[idx_i, own_mask].mean()

        b_i = np.inf
        for h in clusters:
            if h == gi:
                continue
            alt_mask = np.array([labels.loc[s] == h for s in site_ids])
            n_alt = alt_mask.sum()
            if n_alt == 0:
                continue
            b_ih = D[idx_i, alt_mask].mean()
            b_i = min(b_i, b_ih)

        if b_i == np.inf:
            sil.loc[site_i] = 0.0
            continue

        denom = max(a_i, b_i)
        sil.loc[site_i] = (b_i - a_i) / denom if denom > 0 else 0.0

    return sil


# ------------------------------------------------------------------
# Environmental co-assignment via bootstrap Ward clustering
# ------------------------------------------------------------------

def compute_env_coassignment(
    env_std: pd.DataFrame,
    n_clusters: int,
    n_boot: int = 1000,
    sample_frac: float = 0.8,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Bootstrap co-assignment matrix from Ward clustering in env space.

    Repeatedly resamples reference sites, clusters them in standardized
    environmental space, and records pairwise co-assignment.

    Parameters
    ----------
    env_std : pd.DataFrame
        Standardized environmental matrix (sites × variables).
    n_clusters : int
        Number of clusters to cut at each replicate (same *k* as taxa).
    n_boot : int
        Number of bootstrap replicates.
    sample_frac : float
        Fraction of sites to sample per replicate.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Square co-assignment matrix (sites × sites), values in [0, 1].
    """
    rng = np.random.default_rng(random_state)
    n_sites = len(env_std)
    site_ids = env_std.index
    X = env_std.values

    n_sample = max(n_clusters + 1, int(n_sites * sample_frac))
    coassign_count = np.zeros((n_sites, n_sites), dtype=np.float64)
    copresent_count = np.zeros((n_sites, n_sites), dtype=np.float64)

    for _ in range(n_boot):
        idx = rng.choice(n_sites, size=n_sample, replace=False)
        idx_sorted = np.sort(idx)

        X_sub = X[idx_sorted]
        if len(X_sub) <= n_clusters:
            continue

        Z_sub = linkage(X_sub, method="ward", metric="euclidean")
        boot_labels = fcluster(Z_sub, t=n_clusters, criterion="maxclust")

        for a_pos in range(len(idx_sorted)):
            for b_pos in range(a_pos + 1, len(idx_sorted)):
                i, j = idx_sorted[a_pos], idx_sorted[b_pos]
                copresent_count[i, j] += 1
                copresent_count[j, i] += 1
                if boot_labels[a_pos] == boot_labels[b_pos]:
                    coassign_count[i, j] += 1
                    coassign_count[j, i] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(copresent_count > 0, coassign_count / copresent_count, 0.0)
    np.fill_diagonal(C, 1.0)

    return pd.DataFrame(C, index=site_ids, columns=site_ids)


# ------------------------------------------------------------------
# Environmental own-cluster / best-alt / margin (taxa labels as grouping)
# ------------------------------------------------------------------

def env_site_confidence(
    env_coassignment: pd.DataFrame,
    labels: pd.Series,
) -> pd.DataFrame:
    """Own-cluster, best-alt, and margin from env co-assignment vs taxa labels.

    Parameters
    ----------
    env_coassignment : pd.DataFrame
        Square environmental co-assignment matrix (sites × sites).
    labels : pd.Series
        Original taxa-defined cluster labels for each site.

    Returns
    -------
    pd.DataFrame
        Columns: Env_Own_Coassign, Env_BestAlt_Coassign, Env_Margin.
    """
    site_ids = labels.index
    clusters = sorted(labels.unique())

    A = pd.Series(np.nan, index=site_ids, name="Env_Own_Coassign")
    B = pd.Series(np.nan, index=site_ids, name="Env_BestAlt_Coassign")

    for i in site_ids:
        gi = labels.loc[i]
        own_members = labels.index[labels == gi].drop(i, errors="ignore")
        if len(own_members) > 0:
            A.loc[i] = env_coassignment.loc[i, own_members].mean()
        else:
            A.loc[i] = 1.0

        best_alt = -np.inf
        for h in clusters:
            if h == gi:
                continue
            alt_members = labels.index[labels == h]
            if len(alt_members) > 0:
                b_ih = env_coassignment.loc[i, alt_members].mean()
                best_alt = max(best_alt, b_ih)
        B.loc[i] = best_alt if best_alt > -np.inf else 0.0

    M = (A - B).rename("Env_Margin")
    return pd.DataFrame({
        "Env_Own_Coassign": A,
        "Env_BestAlt_Coassign": B,
        "Env_Margin": M,
    })


# ------------------------------------------------------------------
# Environmental strength labelling (binary: Strong / Weak)
# ------------------------------------------------------------------

def assign_env_strength(
    env_silhouette: float,
    env_margin: float,
    sil_threshold: float = 0.0,
    margin_threshold: float = 0.0,
) -> str:
    """Classify a site's environmental coherence as Strong or Weak.

    Parameters
    ----------
    env_silhouette : float
        Environmental silhouette for the site.
    env_margin : float
        Environmental margin (Env_Own_Coassign - Env_BestAlt_Coassign).
    sil_threshold : float
        Minimum env silhouette for Strong.
    margin_threshold : float
        Minimum env margin for Strong.

    Returns
    -------
    str
        ``"Strong"`` or ``"Weak"``.
    """
    if env_silhouette > sil_threshold and env_margin > margin_threshold:
        return "Strong"
    return "Weak"


# ------------------------------------------------------------------
# Build environmental coherence table
# ------------------------------------------------------------------

def build_env_coherence_table(
    labels: pd.Series,
    env_silhouettes: pd.Series,
    env_confidence: pd.DataFrame,
    sil_threshold: float = 0.0,
    margin_threshold: float = 0.0,
) -> pd.DataFrame:
    """One-row-per-site environmental coherence summary.

    Parameters
    ----------
    labels : pd.Series
        Original taxa-defined cluster labels.
    env_silhouettes : pd.Series
        Per-site environmental silhouette.
    env_confidence : pd.DataFrame
        From :func:`env_site_confidence` — Env_Own_Coassign,
        Env_BestAlt_Coassign, Env_Margin.
    sil_threshold : float
        Minimum env silhouette for Strong.
    margin_threshold : float
        Minimum env margin for Strong.

    Returns
    -------
    pd.DataFrame
        Columns: Env_Silhouette, Env_Own_Coassign,
        Env_BestAlt_Coassign, Env_Margin, Env_Strength.
    """
    rows = []
    for site in labels.index:
        sil = float(env_silhouettes.loc[site])
        own = float(env_confidence.loc[site, "Env_Own_Coassign"])
        alt = float(env_confidence.loc[site, "Env_BestAlt_Coassign"])
        margin = float(env_confidence.loc[site, "Env_Margin"])
        strength = assign_env_strength(
            sil, margin,
            sil_threshold=sil_threshold,
            margin_threshold=margin_threshold,
        )
        rows.append({
            "Site": site,
            "Env_Silhouette": round(sil, 4),
            "Env_Own_Coassign": round(own, 4),
            "Env_BestAlt_Coassign": round(alt, 4),
            "Env_Margin": round(margin, 4),
            "Env_Strength": strength,
        })
    return pd.DataFrame(rows).set_index("Site")
