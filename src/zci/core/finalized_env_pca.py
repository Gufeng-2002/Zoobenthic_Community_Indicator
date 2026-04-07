"""Core functions for finalized-model PCA ordination and prediction grids.

Provides:
- ``run_fullsite_env_pca``: PCA on all sites (ref + non-ref) in env space.
- ``build_site_plot_data``: Combine PCA scores with cluster labels and roles.
- ``build_prediction_grid``: Conditional prediction map in PC1-PC2 plane.
- ``build_pc2_lda_grid``: Decision boundary from LDA fitted on PC1-PC2 only.
- ``compute_cluster_hulls``: Convex hulls per cluster.
- ``compute_cluster_ellipses``: 95 % confidence ellipses per cluster.
- ``compute_allref_ellipse``: Single ellipse for all reference sites combined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


# ─── dataclass ──────────────────────────────────────────────────────


@dataclass
class FullSitePCAResult:
    """Container for full-site environmental PCA artifacts."""

    pca: PCA
    scaler: StandardScaler
    scores: pd.DataFrame          # (n_sites × n_components)
    loadings: pd.DataFrame        # (n_vars × n_components)
    variance_explained: np.ndarray  # per-component %
    env_columns: List[str]


# ─── 1. PCA on ALL sites ────────────────────────────────────────────


def run_fullsite_env_pca(
    env_all: pd.DataFrame,
    *,
    standardize: bool = True,
) -> FullSitePCAResult:
    """Standardize + PCA on all sites (ref and non-ref).

    Returns all components so that inverse_transform is exact.
    """
    env = env_all.dropna()
    cols = list(env.columns)

    scaler = StandardScaler() if standardize else None
    X = scaler.fit_transform(env.values) if scaler else env.values.copy()

    pca = PCA()  # keep all components
    scores_arr = pca.fit_transform(X)

    pc_names = [f"PC{i+1}" for i in range(scores_arr.shape[1])]
    scores = pd.DataFrame(scores_arr, index=env.index, columns=pc_names)
    loadings = pd.DataFrame(
        pca.components_.T, index=cols, columns=pc_names,
    )
    var_pct = pca.explained_variance_ratio_ * 100

    return FullSitePCAResult(
        pca=pca,
        scaler=scaler,
        scores=scores,
        loadings=loadings,
        variance_explained=var_pct,
        env_columns=cols,
    )


# ─── 2. Site plotting dataset ───────────────────────────────────────


def build_site_plot_data(
    pca_result: FullSitePCAResult,
    ref_labels: pd.Series,
    pred_labels_all: pd.Series,
) -> pd.DataFrame:
    """Create a per-site plotting table.

    Parameters
    ----------
    ref_labels : Series
        Ward-defined true cluster labels (index = ref-site IDs).
    pred_labels_all : Series
        Classifier-predicted labels for ALL sites (ref + non-ref).
    """
    scores = pca_result.scores
    common = scores.index

    is_ref = pd.Series(False, index=common, name="is_reference")
    is_ref.loc[is_ref.index.isin(ref_labels.index)] = True

    true_label = pd.Series(np.nan, index=common, name="true_cluster", dtype="Int64")
    ref_overlap = common.intersection(ref_labels.index)
    true_label.loc[ref_overlap] = ref_labels.loc[ref_overlap].astype(int)

    pred_label = pred_labels_all.reindex(common).astype("Int64")
    pred_label.name = "pred_cluster"

    correct = pd.Series(np.nan, index=common, name="ref_correct", dtype="boolean")
    correct.loc[ref_overlap] = (
        true_label.loc[ref_overlap] == pred_label.loc[ref_overlap]
    )

    # Display cluster: true cluster for ref, predicted for non-ref
    display_cluster = pred_label.copy()
    display_cluster.name = "display_cluster"
    display_cluster.loc[ref_overlap] = true_label.loc[ref_overlap]

    df = pd.DataFrame({
        "is_reference": is_ref,
        "true_cluster": true_label,
        "pred_cluster": pred_label,
        "ref_correct": correct,
        "display_cluster": display_cluster,
        "PC1": scores["PC1"],
        "PC2": scores["PC2"],
    })
    return df


# ─── 3. Prediction grid ─────────────────────────────────────────────


def build_prediction_grid(
    pca_result: FullSitePCAResult,
    classifier_model: Any,
    classifier_scaler: Any,
    *,
    grid_resolution: int = 200,
    margin_pct: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a conditional full-model prediction map in the PC1-PC2 plane.

    Fix PC3 … PCn at their overall-site means; back-transform to raw
    env space and apply the *classifier's* scaler before predicting.

    Returns
    -------
    xx, yy : 2-D meshgrid arrays for PC1, PC2.
    grid_labels : 2-D array of predicted cluster labels.
    """
    scores = pca_result.scores.values  # (n, p)
    pc_means = scores.mean(axis=0)     # length p

    pc1 = scores[:, 0]
    pc2 = scores[:, 1]
    pad1 = (pc1.max() - pc1.min()) * margin_pct
    pad2 = (pc2.max() - pc2.min()) * margin_pct

    g1 = np.linspace(pc1.min() - pad1, pc1.max() + pad1, grid_resolution)
    g2 = np.linspace(pc2.min() - pad2, pc2.max() + pad2, grid_resolution)
    xx, yy = np.meshgrid(g1, g2)

    # Build full PCA-score vectors for every grid point
    n_pts = xx.size
    n_comp = scores.shape[1]
    grid_scores = np.tile(pc_means, (n_pts, 1))  # (n_pts, p)
    grid_scores[:, 0] = xx.ravel()
    grid_scores[:, 1] = yy.ravel()

    # Inverse PCA → standardized-env → inverse all-site scaler → raw env
    X_std_grid = pca_result.pca.inverse_transform(grid_scores)
    if pca_result.scaler is not None:
        X_raw_grid = pca_result.scaler.inverse_transform(X_std_grid)
    else:
        X_raw_grid = X_std_grid

    # Apply classifier's training scaler → predict
    if classifier_scaler is not None:
        X_clf = classifier_scaler.transform(X_raw_grid)
    else:
        X_clf = X_raw_grid

    preds = classifier_model.predict(X_clf)
    grid_labels = preds.reshape(xx.shape)

    return xx, yy, grid_labels


# ─── 4. Convex hulls per reference cluster ──────────────────────────


def compute_cluster_hulls(
    pca_scores_pc12: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Return convex-hull vertices in PC1-PC2 for each cluster.

    Parameters
    ----------
    pca_scores_pc12 : (n, 2) array of PC1, PC2 scores (reference sites only).
    labels : (n,) array of cluster labels.

    Returns
    -------
    dict mapping cluster_id → (m, 2) hull vertices (closed polygon).
    """
    hulls: Dict[int, np.ndarray] = {}
    for cl in sorted(np.unique(labels)):
        pts = pca_scores_pc12[labels == cl]
        if len(pts) < 3:
            continue
        hull = ConvexHull(pts)
        verts = pts[hull.vertices]
        # Close polygon
        verts = np.vstack([verts, verts[0:1]])
        hulls[cl] = verts
    return hulls


# ─── 5. Confidence ellipses per reference cluster ───────────────────


def compute_cluster_ellipses(
    pca_scores_pc12: np.ndarray,
    labels: np.ndarray,
    *,
    confidence: float = 0.95,
) -> Dict[int, Dict[str, Any]]:
    """Return 95 % confidence ellipse parameters for each cluster.

    Returns dict mapping cluster_id → {center, width, height, angle_deg}.
    """
    from scipy.stats import chi2

    chi2_val = chi2.ppf(confidence, df=2)

    ellipses: Dict[int, Dict[str, Any]] = {}
    for cl in sorted(np.unique(labels)):
        pts = pca_scores_pc12[labels == cl]
        if len(pts) < 3:
            continue
        center = pts.mean(axis=0)
        cov = np.cov(pts, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Sort descending
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width = 2 * np.sqrt(chi2_val * eigvals[0])
        height = 2 * np.sqrt(chi2_val * eigvals[1])
        ellipses[cl] = {
            "center": center,
            "width": width,
            "height": height,
            "angle_deg": angle,
        }
    return ellipses


# ─── 6. Single ellipse for ALL reference sites combined ─────────────


def compute_allref_ellipse(
    pca_scores_pc12: np.ndarray,
    *,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Return 95 % confidence ellipse for all reference sites pooled."""
    from scipy.stats import chi2

    chi2_val = chi2.ppf(confidence, df=2)
    center = pca_scores_pc12.mean(axis=0)
    cov = np.cov(pca_scores_pc12, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width = 2 * np.sqrt(chi2_val * eigvals[0])
    height = 2 * np.sqrt(chi2_val * eigvals[1])
    return {
        "center": center,
        "width": width,
        "height": height,
        "angle_deg": angle,
    }


# ─── 7. 2-PC LDA: fit on PC1-PC2 and build native decision grid ────


def fit_pc2_lda(
    ref_pc12: np.ndarray,
    ref_labels: np.ndarray,
) -> LinearDiscriminantAnalysis:
    """Fit an LDA directly on the first two PC scores of reference sites."""
    lda = LinearDiscriminantAnalysis()
    lda.fit(ref_pc12, ref_labels)
    return lda


def build_pc2_lda_grid(
    pca_result: FullSitePCAResult,
    lda_pc2: LinearDiscriminantAnalysis,
    *,
    grid_resolution: int = 200,
    margin_pct: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a decision grid from the 2-PC LDA in the PC1-PC2 plane.

    Unlike ``build_prediction_grid``, this LDA was trained directly on
    PC1-PC2 scores, so no inverse transform is needed.
    """
    scores = pca_result.scores.values
    pc1 = scores[:, 0]
    pc2 = scores[:, 1]
    pad1 = (pc1.max() - pc1.min()) * margin_pct
    pad2 = (pc2.max() - pc2.min()) * margin_pct

    g1 = np.linspace(pc1.min() - pad1, pc1.max() + pad1, grid_resolution)
    g2 = np.linspace(pc2.min() - pad2, pc2.max() + pad2, grid_resolution)
    xx, yy = np.meshgrid(g1, g2)

    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    preds = lda_pc2.predict(grid_pts)
    grid_labels = preds.reshape(xx.shape)

    return xx, yy, grid_labels
