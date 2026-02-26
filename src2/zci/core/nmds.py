"""Bray–Curtis NMDS — pure computation, no plotting, no file paths.

Provides:
- Iterative 2-D non-metric MDS on Bray–Curtis dissimilarity
- PCA-rotation of NMDS axes
- Axis-flip heuristic (negative correlation with pollution score)
- Weighted-average species scores
- Endpoint construction (reference / degraded centroids in RA space)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.manifold import MDS


# ------------------------------------------------------------------
# Endpoint construction
# ------------------------------------------------------------------


def build_endpoints(
    taxa_ra: pd.DataFrame,
    pollution_scores: pd.Series,
    n_ep: int = 5,
    cluster_label: str = "",
) -> Tuple[pd.Series, pd.Series, str, str, pd.Index, pd.Index]:
    """Build reference and degraded endpoint taxa vectors.

    Averages the relative-abundance vectors of the *n_ep* least-polluted
    (reference) and *n_ep* most-polluted (degraded) sites.

    Parameters
    ----------
    taxa_ra : pd.DataFrame
        Sites × taxa in relative-abundance scale.
    pollution_scores : pd.Series
        Pollution score for each site (aligned index).
    n_ep : int
        Number of extreme sites to average per endpoint.
    cluster_label : str
        Used to create unique row names for the endpoint rows.

    Returns
    -------
    ref_taxa, deg_taxa : pd.Series
        Mean RA vector for the reference / degraded endpoint.
    ref_label, deg_label : str
        Unique string identifiers for the endpoint rows.
    ref_ids, deg_ids : pd.Index
        Site IDs used to construct each endpoint.
    """
    ps_sorted = pollution_scores.sort_values()
    ref_ids = ps_sorted.index[:n_ep]
    deg_ids = ps_sorted.index[-n_ep:]

    ref_taxa = taxa_ra.loc[ref_ids].mean(axis=0)
    deg_taxa = taxa_ra.loc[deg_ids].mean(axis=0)

    ref_label = f"_REF_EP_{cluster_label}"
    deg_label = f"_DEG_EP_{cluster_label}"

    return ref_taxa, deg_taxa, ref_label, deg_label, ref_ids, deg_ids


# ------------------------------------------------------------------
# Iterative NMDS
# ------------------------------------------------------------------


def iterative_nmds(
    dissimilarity_matrix: np.ndarray,
    n_components: int = 2,
    n_iterations: int = 3,
    max_iter_per_run: int = 1000,
    n_init_first: int = 10,
    n_init_subsequent: int = 4,
    random_state: int = 42,
) -> Tuple[np.ndarray, float]:
    """Run non-metric MDS iteratively, seeding each run with the previous result.

    Parameters
    ----------
    dissimilarity_matrix : np.ndarray
        Square, symmetric Bray–Curtis (or other) dissimilarity matrix.
    n_components : int
        Dimensionality of the ordination (default 2).
    n_iterations : int
        Number of iterative refinement passes (default 3).
    max_iter_per_run : int
        SMACOF iterations per pass.
    n_init_first : int
        Number of random starts on the **first** pass.
    n_init_subsequent : int
        Number of random starts on subsequent passes.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    coords : np.ndarray, shape (n_sites, n_components)
        Final ordination coordinates.
    stress : float
        Normalised stress of the final solution.
    """
    coords = None
    stress = np.inf

    for it in range(n_iterations):
        n_init = n_init_first if it == 0 else n_init_subsequent
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mds = MDS(
                n_components=n_components,
                metric=False,
                dissimilarity="precomputed",
                max_iter=max_iter_per_run,
                n_init=n_init,
                random_state=random_state + it,
                normalized_stress="auto",
            )
            if coords is not None:
                # Seed with previous solution
                mds.n_init = 1
                coords = mds.fit_transform(dissimilarity_matrix, init=coords)
            else:
                coords = mds.fit_transform(dissimilarity_matrix)
            stress = mds.stress_

    return coords, stress


# ------------------------------------------------------------------
# PCA-rotation + axis flip
# ------------------------------------------------------------------


def pca_rotate(
    coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """PCA-rotate NMDS coordinates so Axis 1 has maximum variance.

    Parameters
    ----------
    coords : np.ndarray, shape (n, k)

    Returns
    -------
    rotated : np.ndarray, shape (n, k)
    var_explained : np.ndarray, shape (k,)
        Fraction of variance explained by each rotated axis.
    """
    pca = PCA(n_components=coords.shape[1]).fit(coords)
    rotated = pca.transform(coords)
    return rotated, pca.explained_variance_ratio_


def flip_axis(
    coords: np.ndarray,
    pollution_scores: np.ndarray,
    real_mask: np.ndarray,
    axis: int = 0,
) -> Tuple[np.ndarray, bool]:
    """Flip an axis so that it is negatively correlated with pollution scores.

    Parameters
    ----------
    coords : np.ndarray, shape (n, k)
    pollution_scores : np.ndarray, shape (n_real,)
        PS values for the **real** (non-endpoint) sites.
    real_mask : np.ndarray[bool], shape (n,)
        True for real sites, False for endpoint rows.
    axis : int
        Which axis to consider.

    Returns
    -------
    coords : np.ndarray
        Possibly sign-flipped.
    flipped : bool
        Whether the axis was flipped.
    """
    r, _ = pearsonr(coords[real_mask, axis], pollution_scores)
    if r > 0:
        coords[:, axis] *= -1
        return coords, True
    return coords, False


# ------------------------------------------------------------------
# Weighted-average species scores
# ------------------------------------------------------------------


def weighted_average_scores(
    taxa_ra: pd.DataFrame,
    site_coords: np.ndarray,
    taxa_columns: Sequence[str],
) -> pd.DataFrame:
    """Compute weighted-average species scores in ordination space.

    Parameters
    ----------
    taxa_ra : pd.DataFrame
        Sites × taxa relative-abundance matrix (rows aligned with *site_coords*).
    site_coords : np.ndarray, shape (n_sites, 2)
        Ordination coordinates of the real sites.
    taxa_columns : sequence of str
        Taxa names (column order in *taxa_ra*).

    Returns
    -------
    pd.DataFrame
        Taxa × (WA1, WA2) weighted-average positions.
    """
    abund = taxa_ra.values
    col_sums = abund.sum(axis=0)
    col_sums[col_sums == 0] = 1e-12
    wa = (abund.T @ site_coords) / col_sums[:, None]
    cols = [f"WA{i+1}" for i in range(site_coords.shape[1])]
    return pd.DataFrame(wa, index=taxa_columns, columns=cols)


# ------------------------------------------------------------------
# Full per-cluster NMDS pipeline
# ------------------------------------------------------------------


def run_nmds_cluster(
    taxa_ra_cluster: pd.DataFrame,
    pollution_scores_cluster: pd.Series,
    taxa_columns: Sequence[str],
    n_ep: int = 5,
    cluster_id: int | str = 1,
    n_components: int = 2,
    n_iterations: int = 3,
    max_iter_per_run: int = 1000,
    n_init_first: int = 10,
    n_init_subsequent: int = 4,
    random_state: int = 42,
) -> Dict:
    """Run the full NMDS pipeline for one cluster.

    Steps: build endpoints → augment taxa matrix → Bray–Curtis →
    iterative NMDS → PCA-rotate → flip axis → WA species scores.

    Returns
    -------
    dict with keys: coords_df, stress, wa_df, ref_label, deg_label,
    var_exp, taxa_aug, ref_ids, deg_ids, flipped
    """
    ref_taxa, deg_taxa, ref_label, deg_label, ref_ids, deg_ids = build_endpoints(
        taxa_ra_cluster, pollution_scores_cluster, n_ep=n_ep,
        cluster_label=str(cluster_id),
    )

    # Augment taxa matrix with endpoints
    taxa_aug = pd.concat([
        taxa_ra_cluster,
        pd.DataFrame([ref_taxa], index=[ref_label], columns=taxa_columns),
        pd.DataFrame([deg_taxa], index=[deg_label], columns=taxa_columns),
    ])

    # Bray–Curtis dissimilarity
    bc_sq = squareform(pdist(taxa_aug.values, metric="braycurtis"))

    # Iterative NMDS
    coords_arr, stress = iterative_nmds(
        bc_sq,
        n_components=n_components,
        n_iterations=n_iterations,
        max_iter_per_run=max_iter_per_run,
        n_init_first=n_init_first,
        n_init_subsequent=n_init_subsequent,
        random_state=random_state,
    )

    # PCA-rotation
    coords_arr, var_exp = pca_rotate(coords_arr)

    # Flip axis 1 so corr(NMDS1, PS) < 0
    real_mask = np.array([i not in (ref_label, deg_label)
                          for i in taxa_aug.index])
    ps_real = pollution_scores_cluster.values
    coords_arr, flipped = flip_axis(coords_arr, ps_real, real_mask, axis=0)

    # Build DataFrame
    coords_df = pd.DataFrame(
        coords_arr, index=taxa_aug.index,
        columns=[f"NMDS{i+1}" for i in range(n_components)],
    )

    # Weighted-average species scores (real sites only)
    real_sites = [s for s in coords_df.index if s not in (ref_label, deg_label)]
    wa_df = weighted_average_scores(
        taxa_ra_cluster.loc[real_sites],
        coords_df.loc[real_sites].values,
        taxa_columns,
    )

    return {
        "coords_df":  coords_df,
        "stress":     stress,
        "wa_df":      wa_df,
        "ref_label":  ref_label,
        "deg_label":  deg_label,
        "var_exp":    var_exp,
        "taxa_aug":   taxa_aug,
        "ref_ids":    ref_ids,
        "deg_ids":    deg_ids,
        "flipped":    flipped,
    }
