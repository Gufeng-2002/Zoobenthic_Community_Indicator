"""Read study data from Excel and extract blocks from MultiIndex DataFrames.

No ecology math here — only parsing and slicing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd


def read_study_data(
    path: Union[str, Path],
    header_levels: int = 3,
    index_col: int = 0,
) -> pd.DataFrame:
    """Read the 3-level MultiIndex Excel workbook.

    Parameters
    ----------
    path : str or Path
        Path to the ``.xlsx`` file
        (e.g. ``data/processed/complete_env_taxa_chemical_Feb_3.xlsx``).
    header_levels : int, default 3
        Number of header rows that form the column MultiIndex.
    index_col : int, default 0
        Column to use as the row index (typically ``StationID``).

    Returns
    -------
    pd.DataFrame
        DataFrame with a ``pd.MultiIndex`` on columns
        (block, subblock, var) and ``StationID`` as the row index.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_excel(
        path,
        header=list(range(header_levels)),
        index_col=index_col,
    )
    return df


def extract_block(
    df: pd.DataFrame,
    level0: str,
    level1: Optional[str] = None,
) -> pd.DataFrame:
    """Slice a block from a MultiIndex DataFrame and flatten to plain columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns are a ``pd.MultiIndex`` (2 or 3 levels).
    level0 : str
        Value to match on the **first** column level (e.g. ``"chemical"``).
    level1 : str, optional
        Value to match on the **second** column level (e.g. ``"raw"``).
        If ``None``, all sub-levels under *level0* are included and the
        returned columns are the *last* level values.

    Returns
    -------
    pd.DataFrame
        A plain (non-MultiIndex) DataFrame whose columns are the
        lowest-level variable names.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        raise TypeError("DataFrame columns are not a MultiIndex")

    if level1 is not None:
        block = df.loc[:, (level0, level1, slice(None))].copy()
    else:
        block = df.loc[:, (level0, slice(None), slice(None))].copy()

    # Flatten to the leaf variable names
    block.columns = block.columns.get_level_values(-1)
    return block
