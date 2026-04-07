"""Ward clustering robustness — silhouette, co-assignment, and pvclust bootstrap.

Pure computation functions (except the pvclust call which shells out to R).
No plotting, no file paths in the public API.
"""

from __future__ import annotations

import csv
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_samples


# Path to the R bootstrap script bundled alongside this module
_R_SCRIPT = Path(__file__).with_name("pvclust_bootstrap.R")


# ------------------------------------------------------------------
# Silhouette widths
# ------------------------------------------------------------------


def compute_silhouettes(
    taxa_transformed: pd.DataFrame,
    labels: pd.Series,
) -> pd.Series:
    """Per-site silhouette width using Euclidean distance.

    Parameters
    ----------
    taxa_transformed : pd.DataFrame
        Transformed taxa matrix (sites × taxa).
    labels : pd.Series
        1-indexed cluster labels, same index as *taxa_transformed*.

    Returns
    -------
    pd.Series
        Silhouette width for each site, index = site IDs.
    """
    sil = silhouette_samples(
        taxa_transformed.values,
        labels.values,
        metric="euclidean",
    )
    return pd.Series(sil, index=taxa_transformed.index, name="Silhouette")


# ------------------------------------------------------------------
# Bootstrap co-assignment matrix
# ------------------------------------------------------------------


def compute_coassignment(
    taxa_transformed: pd.DataFrame,
    labels: pd.Series,
    n_clusters: int,
    n_boot: int = 1000,
    sample_frac: float = 0.8,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Build the pairwise co-assignment matrix via bootstrap resampling.

    For each bootstrap replicate, draw *sample_frac* of sites (without
    replacement), re-run Ward clustering at *n_clusters*, and record
    whether pairs of sampled sites land in the same cluster.

    Parameters
    ----------
    taxa_transformed : pd.DataFrame
        Transformed taxa matrix (sites × taxa).
    labels : pd.Series
        Original 1-indexed cluster labels (used only for alignment).
    n_clusters : int
        Number of clusters for each resample.
    n_boot : int
        Number of bootstrap replicates.
    sample_frac : float
        Fraction of sites to sample per replicate.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Square co-assignment matrix C_{ij} (sites × sites), values in [0, 1].
    """
    rng = np.random.default_rng(random_state)
    n_sites = len(taxa_transformed)
    site_ids = taxa_transformed.index
    X = taxa_transformed.values

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

        # Update co-presence and co-assignment
        for a_pos in range(len(idx_sorted)):
            for b_pos in range(a_pos + 1, len(idx_sorted)):
                i, j = idx_sorted[a_pos], idx_sorted[b_pos]
                copresent_count[i, j] += 1
                copresent_count[j, i] += 1
                if boot_labels[a_pos] == boot_labels[b_pos]:
                    coassign_count[i, j] += 1
                    coassign_count[j, i] += 1

    # Compute C_{ij}
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(copresent_count > 0, coassign_count / copresent_count, 0.0)
    np.fill_diagonal(C, 1.0)

    return pd.DataFrame(C, index=site_ids, columns=site_ids)


# ------------------------------------------------------------------
# Site-level confidence from co-assignment
# ------------------------------------------------------------------


def site_confidence(
    coassignment: pd.DataFrame,
    labels: pd.Series,
) -> pd.DataFrame:
    """Compute own-cluster co-assignment, best-alt co-assignment, and margin.

    Parameters
    ----------
    coassignment : pd.DataFrame
        Square co-assignment matrix (sites × sites).
    labels : pd.Series
        1-indexed cluster labels for each site.

    Returns
    -------
    pd.DataFrame
        Columns: Own_Coassign (A_i), Best_Alt_Coassign (B_i), Margin (M_i).
    """
    site_ids = labels.index
    clusters = sorted(labels.unique())

    A = pd.Series(np.nan, index=site_ids, name="Own_Coassign")
    B = pd.Series(np.nan, index=site_ids, name="Best_Alt_Coassign")

    for i in site_ids:
        gi = labels.loc[i]
        own_members = labels.index[labels == gi].drop(i, errors="ignore")
        if len(own_members) > 0:
            A.loc[i] = coassignment.loc[i, own_members].mean()
        else:
            A.loc[i] = 1.0

        best_alt = -np.inf
        for h in clusters:
            if h == gi:
                continue
            alt_members = labels.index[labels == h]
            if len(alt_members) > 0:
                b_ih = coassignment.loc[i, alt_members].mean()
                best_alt = max(best_alt, b_ih)
        B.loc[i] = best_alt if best_alt > -np.inf else 0.0

    M = (A - B).rename("Margin")
    return pd.DataFrame({"Own_Coassign": A, "Best_Alt_Coassign": B, "Margin": M})


# ------------------------------------------------------------------
# pvclust via R subprocess
# ------------------------------------------------------------------


def run_pvclust(
    taxa_transformed: pd.DataFrame,
    n_clusters: int,
    nboot: int = 1000,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run pvclust multiscale bootstrap via an R subprocess.

    Requires R and the ``pvclust`` R package (auto-installed if missing).

    Parameters
    ----------
    taxa_transformed : pd.DataFrame
        Transformed taxa matrix (sites × taxa).
    n_clusters : int
        Number of clusters.
    nboot : int
        Number of bootstrap replications.
    verbose : bool
        Print R stdout/stderr.

    Returns
    -------
    pd.DataFrame
        Columns: Cluster, AU, BP.

    Raises
    ------
    RuntimeError
        If the R script fails.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        taxa_csv = Path(tmpdir) / "taxa.csv"
        out_csv = Path(tmpdir) / "pvclust_results.csv"

        taxa_transformed.to_csv(taxa_csv)

        cmd = [
            "Rscript",
            str(_R_SCRIPT),
            str(taxa_csv),
            str(n_clusters),
            str(nboot),
            str(out_csv),
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if verbose and proc.stdout:
            print(proc.stdout)
        if proc.returncode != 0:
            msg = proc.stderr or proc.stdout or "Unknown error"
            raise RuntimeError(f"pvclust R script failed (rc={proc.returncode}):\n{msg}")

        return pd.read_csv(out_csv)


# ------------------------------------------------------------------
# Qualitative status labeling
# ------------------------------------------------------------------


def assign_status(
    silhouette: float,
    margin: float,
    sil_threshold: float = 0.25,
    margin_threshold: float = 0.25,
) -> str:
    """Classify a site as Core / Peripheral / Uncertain.

    Parameters
    ----------
    silhouette : float
        Silhouette width for the site.
    margin : float
        Membership margin (A_i - B_i) for the site.
    sil_threshold : float
        Minimum silhouette to be considered well-clustered.
    margin_threshold : float
        Minimum margin to be considered well-clustered.

    Returns
    -------
    str
        One of ``"Core"``, ``"Peripheral"``, or ``"Uncertain"``.
    """
    if silhouette >= sil_threshold and margin >= margin_threshold:
        return "Core"
    elif silhouette >= 0 and margin >= 0:
        return "Peripheral"
    else:
        return "Uncertain"


# ------------------------------------------------------------------
# Build the final site-level robustness table
# ------------------------------------------------------------------


def build_robustness_table(
    labels: pd.Series,
    pvclust_df: pd.DataFrame | None,
    silhouettes: pd.Series,
    confidence: pd.DataFrame,
    sil_threshold: float = 0.25,
    margin_threshold: float = 0.25,
) -> pd.DataFrame:
    """Assemble the one-row-per-site robustness summary.

    Parameters
    ----------
    labels : pd.Series
        1-indexed cluster labels for each site.
    pvclust_df : pd.DataFrame or None
        pvclust output with columns Cluster, AU, BP.
        If None (R unavailable), AU columns are filled with NaN.
    silhouettes : pd.Series
        Per-site silhouette widths.
    confidence : pd.DataFrame
        From :func:`site_confidence` — columns Own_Coassign, Best_Alt_Coassign, Margin.
    sil_threshold : float
        Threshold for Core status (silhouette).
    margin_threshold : float
        Threshold for Core status (margin).

    Returns
    -------
    pd.DataFrame
        One row per site, columns: Site, Original_Cluster, Branch_AU,
        Silhouette, Own_Coassign, Best_Alt_Coassign, Margin, Status.
    """
    site_ids = labels.index

    # Map cluster -> AU from pvclust
    au_map: dict[int, float] = {}
    if pvclust_df is not None:
        for _, row in pvclust_df.iterrows():
            au_map[int(row["Cluster"])] = float(row["AU"])

    rows = []
    for site in site_ids:
        g = int(labels.loc[site])
        sil = float(silhouettes.loc[site])
        own = float(confidence.loc[site, "Own_Coassign"])
        alt = float(confidence.loc[site, "Best_Alt_Coassign"])
        margin = float(confidence.loc[site, "Margin"])
        au = au_map.get(g, np.nan)
        status = assign_status(sil, margin, sil_threshold, margin_threshold)
        rows.append({
            "Site": site,
            "Original_Cluster": g,
            "Branch_AU": au,
            "Silhouette": round(sil, 4),
            "Own_Coassign": round(own, 4),
            "Best_Alt_Coassign": round(alt, 4),
            "Margin": round(margin, 4),
            "Status": status,
        })

    return pd.DataFrame(rows).set_index("Site")
