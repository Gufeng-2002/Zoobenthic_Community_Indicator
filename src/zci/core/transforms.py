"""Data transformations — pure functions, no plotting, no file paths.

Every function: DataFrame in → DataFrame out.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def log2_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply *log₂(1 + x)* element-wise.

    This is the transform used by the production notebook (``transform_method='log2'``).

    Parameters
    ----------
    df : pd.DataFrame
        Raw pollution-variable matrix (sites × variables).

    Returns
    -------
    pd.DataFrame
        Transformed matrix, same shape and index.
    """
    return df.apply(lambda col: np.log2(col + 1))


# ---------------------------------------------------------------------------
# Composite pollution scoring
# ---------------------------------------------------------------------------


def _rescale_component_scores(
    scores: pd.DataFrame,
    selected_pcs: Sequence[str] | None = None,
    transform: str = "min-max",
) -> pd.DataFrame:
    """Rescale each selected PC column to a common relative scale.

    Parameters
    ----------
    scores : pd.DataFrame
        Site-score matrix (sites × PCs).
    selected_pcs : list of str, optional
        Which PCs to include.  ``None`` → all columns.
    transform : str
        ``"min-max"`` rescales each PC to [0, 1]; ``"z-score"`` centres
        to mean 0 / std 1.

    Returns
    -------
    pd.DataFrame
        Rescaled PC columns (sites × selected PCs).
    """
    if selected_pcs is None:
        selected_pcs = list(scores.columns)
    missing = set(selected_pcs) - set(scores.columns)
    if missing:
        raise KeyError(f"PCs not found in scores: {sorted(missing)}")
    sub = scores[list(selected_pcs)].copy()

    if transform == "min-max":
        sub = (sub - sub.min()) / (sub.max() - sub.min())
    elif transform == "z-score":
        sub = (sub - sub.mean()) / sub.std()
    else:
        raise ValueError(f"Unsupported transform: {transform!r}. Use 'min-max' or 'z-score'.")
    return sub


def composite_pollution_score(
    scores: pd.DataFrame,
    selected_pcs: Sequence[str] | None = None,
    transform: str = "min-max",
    weights: dict[str, float] | Sequence[float] | None = None,
) -> pd.Series:
    """Compute a single composite pollution score per site (weighted sum).

    This is the legacy API — equivalent to ``score_sumrel`` with weights.

    Parameters
    ----------
    scores : pd.DataFrame
        Site-score matrix (sites × PCs).  Already standardised or raw.
    selected_pcs : list of str, optional
        Which PCs to include (e.g. ``["PC1", "PC2", "PC3"]``).
        *None* → all columns.
    transform : str
        ``"min-max"`` rescales each PC to [0, 1]; ``"z-score"`` centres
        to mean 0 / std 1.
    weights : dict, list, or None
        Per-PC weights.  ``None`` → equal weights (all 1).
        A *dict* maps ``{"PC1": 1.0, "PC3": 2.0, …}``;
        a *list/array* must match *selected_pcs* length.

    Returns
    -------
    pd.Series
        Named ``"Pollution_Score"`` with the same row index as *scores*.
    """
    sub = _rescale_component_scores(scores, selected_pcs, transform)

    # Build weight array
    n = sub.shape[1]
    if weights is None:
        w = np.ones(n)
    elif isinstance(weights, dict):
        w = np.array([weights.get(pc, 0.0) for pc in sub.columns])
    else:
        w = np.array(weights)
        if len(w) != n:
            raise ValueError(
                f"Length of weights ({len(w)}) != number of selected PCs ({n})"
            )

    composite = sub.values @ w
    return pd.Series(composite, index=scores.index, name="Pollution_Score")


def score_sumrel(
    scores: pd.DataFrame,
    selected_pcs: Sequence[str] | None = None,
    transform: str = "min-max",
) -> pd.Series:
    """SumRel: sum of rescaled component scores.

    For each site, SumRel = sum of rescaled PC scores.  Equal weights
    are used (unweighted sum).

    Parameters
    ----------
    scores : pd.DataFrame
        Site-score matrix (sites × PCs).
    selected_pcs : list of str, optional
        Which PCs to include.  ``None`` → all columns.
    transform : str
        Rescaling method applied to each PC before summing.

    Returns
    -------
    pd.Series
        Named ``"SumRel_Score"``.
    """
    sub = _rescale_component_scores(scores, selected_pcs, transform)
    composite = sub.sum(axis=1)
    return pd.Series(composite.values, index=scores.index, name="SumRel_Score")


def score_maxrel(
    scores: pd.DataFrame,
    selected_pcs: Sequence[str] | None = None,
    transform: str = "min-max",
) -> pd.Series:
    """MaxRel: maximum of rescaled component scores.

    For each site, MaxRel = max over the rescaled PC scores.  This
    captures the *single worst* contamination syndrome per site.

    Parameters
    ----------
    scores : pd.DataFrame
        Site-score matrix (sites × PCs).
    selected_pcs : list of str, optional
        Which PCs to include.  ``None`` → all columns.
    transform : str
        Rescaling method applied to each PC before taking the max.

    Returns
    -------
    pd.Series
        Named ``"MaxRel_Score"``.
    """
    sub = _rescale_component_scores(scores, selected_pcs, transform)
    composite = sub.max(axis=1)
    return pd.Series(composite.values, index=scores.index, name="MaxRel_Score")


def log1p_zscore_transform(
    df: pd.DataFrame,
    skip_log_cols: Sequence[str] = ("As", "Bi"),
) -> pd.DataFrame:
    """Apply *ln(1 + x)* then z-score standardisation.

    This is the alternative transform (``transform_method='log_z_score'``).
    Columns listed in *skip_log_cols* are exempted from the log step
    but still z-scored.

    Parameters
    ----------
    df : pd.DataFrame
        Raw pollution-variable matrix.
    skip_log_cols : sequence of str
        Columns to skip during the log step.

    Returns
    -------
    pd.DataFrame
        Transformed and standardised matrix.
    """
    from sklearn.preprocessing import StandardScaler

    out = df.copy()
    for col in out.columns:
        if col not in skip_log_cols:
            out[col] = np.log1p(out[col])

    scaled = StandardScaler().fit_transform(out)
    return pd.DataFrame(scaled, index=df.index, columns=df.columns)


# ---------------------------------------------------------------------------
# Taxa (octave) transforms
# ---------------------------------------------------------------------------


def octave_to_relative_abundance(octave_df: pd.DataFrame) -> pd.DataFrame:
    """Convert octave-transformed values back to relative abundances.

    Inverse of  ``o_ij = log₂(100 · (p_ij + 0.01))``:
        ``p = (2^o − 0.0625)``, clipped to ≥ 0, then row-normalised.

    Parameters
    ----------
    octave_df : pd.DataFrame
        Sites × taxa matrix in octave scale.

    Returns
    -------
    pd.DataFrame
        Relative-abundance matrix (rows sum to 1).
    """
    p = np.power(2, octave_df) - 0.0625
    p = p.clip(lower=0)
    row_sum = p.sum(axis=1)
    p = p.div(row_sum, axis=0).fillna(0)
    return p


def octave_transform(octave_df: pd.DataFrame) -> pd.DataFrame:
    """Identity — the data are *already* in octave scale.

    This function exists so callers can use a uniform
    ``transform="octave"`` keyword without special-casing.
    """
    return octave_df.copy()
