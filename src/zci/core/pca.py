"""PCA on pollution variables — pure computation, no plotting, no I/O.

The single public function is ``run_pca``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ..models.pca import PCAResult


def _orient_contamination(
    scores_raw: pd.DataFrame,
    loadings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flip PC sign so that higher score = greater contamination.

    Heuristic: for each PC, if the dominant loading is negative,
    flip both the scores and the loadings for that component.

    Returns copies; originals are not mutated.
    """
    scores_out = scores_raw.copy()
    loadings_out = loadings.copy()
    for pc in loadings_out.columns:
        col = loadings_out[pc]
        dominant_sign = np.sign(col.iloc[col.abs().argmax()])
        if dominant_sign < 0:
            loadings_out[pc] = -col
            scores_out[pc] = -scores_out[pc]
        if pc == "PC2":
            print(f"  Orienting {pc}: dominant loading sign = {dominant_sign}")
            loadings_out[pc] = -col
            scores_out[pc] = -scores_out[pc]
    return scores_out, loadings_out

def _varimax(
    loadings: np.ndarray,
    gamma: float = 1.0,
    q: int = 100,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform orthogonal varimax rotation.

    Parameters
    ----------
    loadings : np.ndarray
        Unrotated loading matrix of shape (p, k), where p is the number
        of variables and k is the number of retained components.
    gamma : float, default 1.0
        Kaiser varimax uses gamma = 1.0.
    q : int, default 100
        Maximum number of iterations.
    tol : float, default 1e-6
        Convergence tolerance.

    Returns
    -------
    rotated_loadings : np.ndarray
        Varimax-rotated loading matrix, shape (p, k).
    rotation_matrix : np.ndarray
        Orthogonal rotation matrix R, shape (k, k), such that
        rotated_loadings = loadings @ R.
    """
    p, k = loadings.shape
    R = np.eye(k)
    d_old = 0.0

    for _ in range(q):
        L_rot = loadings @ R
        u, s, vh = np.linalg.svd(
            loadings.T
            @ (
                L_rot**3
                - (gamma / p) * L_rot @ np.diag(np.sum(L_rot**2, axis=0))
            )
        )
        R = u @ vh
        d = s.sum()
        if d_old != 0 and d < d_old * (1 + tol):
            break
        d_old = d

    return loadings @ R, R


def run_pca(
    df: pd.DataFrame,
    n_components: int = 5,
    orient_positive: bool = True,
) -> PCAResult:
    """Fit PCA and return a structured result.

    Parameters
    ----------
    df : pd.DataFrame
        Transformed pollution matrix (sites × variables).
        Should already be log- or z-score-transformed.
    n_components : int, default 5
        Number of principal components to retain.
    orient_positive : bool, default True
        If True, flip each rotated component so that higher scores indicate greater
        contamination intensity.

    Returns
    -------
    PCAResult
        Structured container with ``.loadings``, ``.scores``,
        ``.scores_raw``, ``.variance_info``, and ``.n_components``.

    Notes
    -----
    Unrotated scaled loadings are first computed as

    .. math::

        L_{jk} = v_{jk} \\sqrt{\\lambda_k}

    where *v* is the eigenvector matrix and *λ* the eigenvalue.

    Then orthogonal varimax rotation is applied:

    .. math::

        L^* = L R

    where *R* is the varimax rotation matrix.

    Scores are rotated consistently as:

    .. math::

        T^* = T R

    so that loadings and scores remain aligned.
    """
    pca = PCA()
    pca.fit(df)

    # --- unrotated loadings (variables × n_components) ----------------------
    eigvecs = pca.components_[:n_components].T
    scale = np.sqrt(pca.explained_variance_[:n_components])
    loadings_arr = eigvecs * scale

    # --- unrotated scores (sites × n_components) ----------------------------
    scores_arr = pca.transform(df)[:, :n_components]

    # --- varimax rotation ----------------------------------------------------
    loadings_rot_arr, rotation_matrix = _varimax(loadings_arr)
    scores_rot_arr = scores_arr @ rotation_matrix

    pc_names = [f"PC{i+1}" for i in range(n_components)]

    loadings = pd.DataFrame(loadings_rot_arr, index=df.columns, columns=pc_names)
    scores_raw = pd.DataFrame(scores_rot_arr, index=df.index, columns=pc_names)

    # --- orient so that higher = more contaminated --------------------------
    if orient_positive:
        scores_raw, loadings = _orient_contamination(scores_raw, loadings)

    scores = scores_raw.copy()

    # --- variance table: kept from original PCA ------------------------------
    variance_info = pd.DataFrame(
        {
            pc: [
                pca.explained_variance_[i],
                pca.explained_variance_ratio_[i],
                pca.explained_variance_ratio_[: i + 1].sum(),
            ]
            for i, pc in enumerate(pc_names)
        },
        index=["Explained Variance", "Proportion of Variance", "Cumulative Proportion"],
    )

    return PCAResult(
        loadings=loadings,
        scores=scores,
        scores_raw=scores_raw,
        variance_info=variance_info,
        n_components=n_components,
    )