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

    Heuristic: for each PC, if the majority of the highest-magnitude
    loadings are negative, flip both the scores and the loadings for
    that component.  This ensures larger score values always indicate
    higher contamination intensity.

    Returns copies; originals are not mutated.
    """
    scores_out = scores_raw.copy()
    loadings_out = loadings.copy()
    for pc in loadings_out.columns:
        col = loadings_out[pc]
        # Sign of the loading with the largest absolute value
        dominant_sign = np.sign(col.iloc[col.abs().argmax()])
        if dominant_sign < 0:
            loadings_out[pc] = -col
            scores_out[pc] = -scores_out[pc]
    return scores_out, loadings_out


def run_pca(
    df: pd.DataFrame,
    n_components: int = 5,
    standardise_scores: str = "min-max",
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
    standardise_scores : str or None
        How to rescale site scores after projection.
        ``"min-max"`` → [0, 1] per column.
        ``"z-score"`` → mean 0, std 1 per column.
        ``None``      → raw projection scores.
    orient_positive : bool, default True
        If True, flip each PC so that higher scores indicate greater
        contamination intensity.

    Returns
    -------
    PCAResult
        Structured container with ``.loadings``, ``.scores``,
        ``.scores_raw``, ``.variance_info``, and ``.n_components``.

    Notes
    -----
    *Scaled loadings* are computed as:

    .. math::

        L_{jk} = v_{jk} \\sqrt{\\lambda_k}

    where *v* is the eigenvector matrix and *λ* the eigenvalue.
    This matches the convention in the original ``pca_analysis.py``.
    """
    pca = PCA()
    pca.fit(df)

    # --- loadings (variables × n_components) --------------------------------
    eigvecs = pca.components_[:n_components].T          # (p, k)
    scale = np.sqrt(pca.explained_variance_[:n_components])  # (k,)
    loadings_arr = eigvecs * scale                       # broadcast → (p, k)

    pc_names = [f"PC{i+1}" for i in range(n_components)]
    loadings = pd.DataFrame(loadings_arr, index=df.columns, columns=pc_names)

    # --- scores (sites × n_components) --------------------------------------
    scores_raw = pd.DataFrame(
        pca.transform(df)[:, :n_components],
        index=df.index,
        columns=pc_names,
    )

    # --- orient so that higher = more contaminated --------------------------
    if orient_positive:
        scores_raw, loadings = _orient_contamination(scores_raw, loadings)

    if standardise_scores == "min-max":
        scores = (scores_raw - scores_raw.min()) / (scores_raw.max() - scores_raw.min())
    elif standardise_scores == "z-score":
        scores = (scores_raw - scores_raw.mean()) / scores_raw.std()
    else:
        scores = scores_raw.copy()

    # --- variance table ------------------------------------------------------
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
