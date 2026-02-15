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
