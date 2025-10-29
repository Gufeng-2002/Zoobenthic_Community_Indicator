from __future__ import annotations # this is for postponed type annotations, PEP563, check it later
import numpy as np
import pandas as pd

def hellinger_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hellinger transform for nonnegative composition/count-like data.
    Returns a DataFrame with the same index/columns and values in [0, 1].
    """
    X = df.to_numpy(dtype=float, copy=True)
    if np.any(X < 0):
        raise ValueError("hellinger_transform requires nonnegative inputs.")
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid division by zero
    H = np.sqrt(X / row_sums)
    return pd.DataFrame(H, index=df.index, columns=df.columns)

def log1p_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log1p + z-score by column: robust for right-skewed chem variables.
    """
    X = np.log1p(df.to_numpy(dtype=float, copy=True))
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    return pd.DataFrame(Z, index=df.index, columns=df.columns)

def log10p_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log10p + z-score by column: robust for right-skewed chem variables.
    """
    X = np.log10(df.to_numpy(dtype=float, copy=True) + 1)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    return pd.DataFrame(Z, index=df.index, columns=df.columns)