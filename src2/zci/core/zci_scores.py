"""ZCI score computation — pure functions, no plotting, no file paths.

Provides multiple methods for computing a single community-condition
index (ZCI) from NMDS coordinates or raw Bray–Curtis distances:

- **Distance**         : d_deg / (d_ref + d_deg) in ordination space
- **Projection**       : project onto DEG→REF line in ordination space
- **Centroid-Proj**    : project onto centroid-defined axis in existing NMDS
- **BC-Direct**        : d_deg / (d_ref + d_deg) using raw Bray–Curtis
                         distances (no ordination needed)
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr


# ------------------------------------------------------------------
# ZCI formulae
# ------------------------------------------------------------------


def zci_distance(
    site_coords: np.ndarray,
    ref_coords: np.ndarray,
    deg_coords: np.ndarray,
) -> np.ndarray:
    """ZCI = d_deg / (d_ref + d_deg) in ordination space.

    Parameters
    ----------
    site_coords : (n, k)  — real-site coordinates
    ref_coords  : (k,)    — reference endpoint coordinates
    deg_coords  : (k,)    — degraded endpoint coordinates

    Returns
    -------
    zci : (n,)
    """
    d_ref = np.linalg.norm(site_coords - ref_coords, axis=1)
    d_deg = np.linalg.norm(site_coords - deg_coords, axis=1)
    return d_deg / (d_ref + d_deg + 1e-15)


def zci_projection(
    site_coords: np.ndarray,
    ref_coords: np.ndarray,
    deg_coords: np.ndarray,
) -> np.ndarray:
    """ZCI = projection onto the DEG→REF axis, normalised.

    Parameters
    ----------
    site_coords : (n, k)
    ref_coords  : (k,)
    deg_coords  : (k,)

    Returns
    -------
    zci : (n,)
    """
    v = ref_coords - deg_coords
    return ((site_coords - deg_coords) @ v) / (v @ v + 1e-15)


def zci_centroid_projection(
    site_coords: np.ndarray,
    pollution_scores: np.ndarray,
    n_ep: int,
) -> np.ndarray:
    """ZCI = projection onto centroid-defined axis in NMDS space.

    The centroids are the mean coordinates of the *n_ep* least-polluted
    (reference) and *n_ep* most-polluted (degraded) **real** sites.

    Parameters
    ----------
    site_coords : (n, k)
    pollution_scores : (n,)
    n_ep : int

    Returns
    -------
    zci : (n,)
    """
    order = np.argsort(pollution_scores)
    ref_c = site_coords[order[:n_ep]].mean(0)
    deg_c = site_coords[order[-n_ep:]].mean(0)
    v = ref_c - deg_c
    return ((site_coords - deg_c) @ v) / (v @ v + 1e-15)


def zci_bc_direct(
    taxa_ra: pd.DataFrame,
    pollution_scores: pd.Series,
    n_ep: int,
) -> np.ndarray:
    """ZCI = d_deg / (d_ref + d_deg) using raw Bray–Curtis distances.

    No ordination needed — distances are computed directly from the
    relative-abundance taxa vectors.

    Parameters
    ----------
    taxa_ra : pd.DataFrame
        Sites × taxa in relative abundance.
    pollution_scores : pd.Series
        Aligned PS values.
    n_ep : int
        Number of extreme sites per endpoint.

    Returns
    -------
    zci : np.ndarray, shape (n_sites,)
    """
    ps_sorted = pollution_scores.sort_values()
    ref_ids = ps_sorted.index[:n_ep]
    deg_ids = ps_sorted.index[-n_ep:]
    ref_centroid = taxa_ra.loc[ref_ids].mean(0).values.reshape(1, -1)
    deg_centroid = taxa_ra.loc[deg_ids].mean(0).values.reshape(1, -1)
    bc_ref = cdist(ref_centroid, taxa_ra.values, metric="braycurtis")[0]
    bc_deg = cdist(deg_centroid, taxa_ra.values, metric="braycurtis")[0]
    return bc_deg / (bc_ref + bc_deg + 1e-15)


# ------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------

ZCI_METHODS = {
    "BC-Direct":     "zci_bc_direct",
    "Distance":      "zci_distance",
    "Projection":    "zci_projection",
    "Centroid-Proj": "zci_centroid_projection",
}


def compute_zci(
    method: str,
    *,
    taxa_ra: pd.DataFrame | None = None,
    pollution_scores: pd.Series | None = None,
    n_ep: int = 5,
    nmds_results: Dict | None = None,
) -> pd.Series:
    """Compute ZCI for one cluster using the specified method.

    Parameters
    ----------
    method : str
        One of ``"BC-Direct"``, ``"Distance"``, ``"Projection"``,
        ``"Centroid-Proj"``.
    taxa_ra : pd.DataFrame
        Sites × taxa relative-abundance matrix (required for BC-Direct).
    pollution_scores : pd.Series
        Pollution scores (aligned index).
    n_ep : int
        Number of extreme sites for endpoint construction.
    nmds_results : dict
        Output of ``run_nmds_cluster()`` — needed for Distance,
        Projection, and Centroid-Proj methods.

    Returns
    -------
    pd.Series
        Named ``"ZCI"`` with the same index as *taxa_ra*.
    """
    if method not in ZCI_METHODS:
        raise ValueError(
            f"Unknown ZCI method: {method!r}. "
            f"Choose from: {list(ZCI_METHODS)}"
        )

    idx = taxa_ra.index if taxa_ra is not None else None

    if method == "BC-Direct":
        if taxa_ra is None or pollution_scores is None:
            raise ValueError("BC-Direct requires taxa_ra and pollution_scores")
        zci_vals = zci_bc_direct(taxa_ra, pollution_scores, n_ep)
        return pd.Series(zci_vals, index=taxa_ra.index, name="ZCI")

    # All other methods need NMDS results
    if nmds_results is None:
        raise ValueError(f"Method {method!r} requires nmds_results dict")

    coords_df = nmds_results["coords_df"]
    ref_label = nmds_results["ref_label"]
    deg_label = nmds_results["deg_label"]
    real_sites = [s for s in coords_df.index
                  if s not in (ref_label, deg_label)]
    rc = coords_df.loc[real_sites].values

    if method == "Distance":
        ref_xy = coords_df.loc[ref_label].values
        deg_xy = coords_df.loc[deg_label].values
        zci_vals = zci_distance(rc, ref_xy, deg_xy)

    elif method == "Projection":
        ref_xy = coords_df.loc[ref_label].values
        deg_xy = coords_df.loc[deg_label].values
        zci_vals = zci_projection(rc, ref_xy, deg_xy)

    elif method == "Centroid-Proj":
        if pollution_scores is None:
            raise ValueError("Centroid-Proj requires pollution_scores")
        ps_real = pollution_scores.loc[real_sites].values
        zci_vals = zci_centroid_projection(rc, ps_real, n_ep)

    return pd.Series(zci_vals, index=real_sites, name="ZCI")


# ------------------------------------------------------------------
# Correlation summary
# ------------------------------------------------------------------


def zci_correlation(
    zci: pd.Series,
    pollution_scores: pd.Series,
) -> Dict[str, float]:
    """Compute Pearson and Spearman correlations between ZCI and PS.

    Returns
    -------
    dict with keys: r_pearson, p_pearson, r_spearman, p_spearman
    """
    common = zci.index.intersection(pollution_scores.index)
    z = zci.loc[common].values
    ps = pollution_scores.loc[common].values
    rp, pp = pearsonr(ps, z)
    rs, ps_ = spearmanr(ps, z)
    return {
        "r_pearson": rp, "p_pearson": pp,
        "r_spearman": rs, "p_spearman": ps_,
    }
