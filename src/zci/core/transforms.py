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


def log10_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply *log₁₀(1 + x)* element-wise.

    This is the transform used by the production notebook (``transform_method='log10'``).

    Parameters
    ----------
    df : pd.DataFrame
        Raw pollution-variable matrix (sites × variables).

    Returns
    -------
    pd.DataFrame
        Transformed matrix, same shape and index.
    """
    return df.apply(lambda col: np.log10(col + 1))



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
        to mean 0 / std 1; ``"none"`` leaves the scores unchanged.

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
    elif transform == "none":
        pass  # no rescaling
    else:
        raise ValueError(f"Unsupported transform: {transform!r}. Use 'min-max', 'z-score', or 'none'.")
    return sub


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


def chord_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply chord standardisation row-wise.

    Each row is divided by its Euclidean norm. Euclidean distance on the
    resulting matrix is the chord distance on the original row profiles.
    Rows with zero norm remain zero.
    """
    values = df.to_numpy(dtype=float, copy=True)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
    return pd.DataFrame(values, index=df.index, columns=df.columns)


def hellinger_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Hellinger standardisation row-wise.

    Each element is replaced by sqrt(p_ij / row_sum). Equivalent to
    ``vegan::decostand(x, "hellinger")``.
    Rows with zero sum remain zero.
    """
    values = df.to_numpy(dtype=float, copy=True)
    row_sums = values.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        values = np.sqrt(np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums > 0))
    return pd.DataFrame(np.nan_to_num(values), index=df.index, columns=df.columns)


def octave_to_chord(octave_df: pd.DataFrame) -> pd.DataFrame:
    """Convert octave-scale taxa to chord-standardised abundances."""
    return chord_transform(octave_to_relative_abundance(octave_df))


def octave_to_hellinger(octave_df: pd.DataFrame) -> pd.DataFrame:
    """Convert octave-scale taxa to Hellinger-standardised abundances."""
    return hellinger_transform(octave_to_relative_abundance(octave_df))


def octave_to_log_chord(octave_df: pd.DataFrame) -> pd.DataFrame:
    """Convert octave-scale taxa to log-chord-standardised abundances.

    Pipeline: octave → relative abundance → log(1 + x) → chord.
    """
    rel = octave_to_relative_abundance(octave_df)
    logged = np.log1p(rel)
    return chord_transform(pd.DataFrame(logged, index=rel.index, columns=rel.columns))
