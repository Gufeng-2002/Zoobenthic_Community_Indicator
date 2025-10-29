from __future__ import annotations
import re
from typing import Iterable, Tuple, Optional, Union, Literal
import pandas as pd

# -------------------------------
# Column MultiIndex helpers
# -------------------------------

_BLOCK_NAMES_2 = ("block", "var")
_BLOCK_NAMES_3 = ("block", "subblock", "var")

def _ensure_index(df: pd.DataFrame, index: Optional[Union[str, Iterable[str]]] = None) -> pd.DataFrame:
    """Ensure df has an index set if index is provided; otherwise return as-is."""
    if index is None:
        return df
    if isinstance(index, str):
        index = [index]
    if any(c not in df.columns for c in index):
        missing = [c for c in index if c not in df.columns]
        raise KeyError(f"Index columns not found: {missing}. Available: {list(df.columns)[:20]}...")
    return df.set_index(list(index), drop=True)

def wrap_columns(
    df: pd.DataFrame,
    block: str,
    subblock: Optional[str] = None,
    *,
    index: Optional[Union[str, Iterable[str]]] = None,
) -> pd.DataFrame:
    """
    Wrap columns of df into a MultiIndex:
      - with subblock: (block, subblock, var)
      - without subblock: (block, var)
    Optionally set index first (e.g., index='site_id').
    """
    out = _ensure_index(df, index=index).copy()
    # sanitize level labels just in case
    b = str(block)
    if subblock is None:
        out.columns = pd.MultiIndex.from_product([[b], out.columns], names=_BLOCK_NAMES_2)
    else:
        sb = str(subblock)
        out.columns = pd.MultiIndex.from_product([[b], [sb], out.columns], names=_BLOCK_NAMES_3)
    return out

def add_site_block(
    master: pd.DataFrame,
    block_df: pd.DataFrame,
    block: str,
    subblock: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add a wrapped block into master via .join on the index.
    master must already be indexed by site_id (or desired key).
    """
    wrapped = wrap_columns(block_df, block, subblock)
    _assert_same_index(master, wrapped)
    # avoid duplicate columns (rare but possible)
    overlap = master.columns.intersection(wrapped.columns)
    if len(overlap) > 0:
        raise ValueError(f"Columns already exist in master: {list(overlap)[:10]} ...")
    return master.join(wrapped)

def concat_blocks(blocks: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate multiple already-wrapped blocks along columns.
    Requires identical indexes.
    """
    blocks = list(blocks)
    if not blocks:
        return pd.DataFrame()
    
    # Use the existing _assert_same_index helper for consistent error messaging
    for i, b in enumerate(blocks[1:], start=1):
        try:
            _assert_same_index(blocks[0], b)
        except ValueError as e:
            raise ValueError(f"Index mismatch between block 0 and block {i}: {e}")
    
    return pd.concat(blocks, axis=1).sort_index(axis=1)

def get_block(master: pd.DataFrame, block: str, subblock: Optional[str] = None) -> pd.DataFrame:
    """Return a single block (optionally subblock) as a plain DataFrame (drop column MI)."""
    if is_three_level(master):
        if subblock is None:
            df = master.loc[:, (block, slice(None), slice(None))] # type: ignore
            # collapse top level(s)
            df = df.copy()
            df.columns = df.columns.get_level_values("var")
            return df
        else:
            df = master.loc[:, (block, subblock, slice(None))] # type: ignore
            df = df.copy()
            df.columns = df.columns.get_level_values("var")
            return df
    else:
        df = master.loc[:, (block, slice(None))] # type: ignore
        df = df.copy()
        df.columns = df.columns.get_level_values("var")
        return df

def set_block(master: pd.DataFrame, block_df: pd.DataFrame, block: str, subblock: Optional[str] = None) -> pd.DataFrame:
    """
    Replace or insert a block (optionally subblock) in master.
    """
    wrapped = wrap_columns(block_df, block, subblock)
    _assert_same_index(master, wrapped)
    out = master.drop(columns=wrapped.columns.intersection(master.columns), errors="ignore")
    return out.join(wrapped).sort_index(axis=1)

def is_three_level(df: pd.DataFrame) -> bool:
    return isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 3

# -------------------------------
# Align in index
# -------------------------------

JoinHow = Literal["inner", "outer", "left"]

def _target_index(blocks: list[pd.DataFrame], how: JoinHow, anchor: Optional[int]) -> pd.Index:
    if how == "inner":
        idx = blocks[0].index
        for b in blocks[1:]:
            idx = idx.intersection(b.index)
        return idx
    elif how == "outer":
        idx = blocks[0].index
        for b in blocks[1:]:
            idx = idx.union(b.index)
        return idx
    elif how == "left":
        if anchor is None:
            raise ValueError("anchor must be provided for how='left' (0-based index of blocks)")
        return blocks[anchor].index
    else:
        raise ValueError(f"Unsupported how={how}")

def align_blocks_by_index(
    blocks: Iterable[pd.DataFrame],
    how: JoinHow = "inner",
    anchor: Optional[int] = None,
) -> list[pd.DataFrame]:
    """
    Reindex each block to a common index (inner/outer/left).
    Returns new DataFrames; originals untouched.
    """
    blocks = list(blocks)
    tgt = _target_index(blocks, how=how, anchor=anchor)
    return [b.reindex(tgt) for b in blocks]

def coverage_report(blocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Quick availability table: rows=union of sites, cols=block names, values=has row (True/False).
    """
    all_idx = pd.Index([])
    for b in blocks.values():
        all_idx = all_idx.union(b.index)
    rep = {}
    for name, b in blocks.items():
        rep[name] = all_idx.isin(b.index)
    out = pd.DataFrame(rep, index=all_idx).sort_index()
    out.index.name = "site_id"
    return out

# -------------------------------
# Flatten / Unflatten
# -------------------------------

def flatten_columns(df: pd.DataFrame, sep: str = "__") -> pd.DataFrame:
    """
    Flatten MultiIndex columns to strings like 'chem__logz__Pb' or 'taxa__EPT'.
    Leaves single-level columns unchanged.
    """
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            sep.join([str(x) for x in tup if x not in ("", None)])
            for tup in out.columns.to_flat_index()
        ]
    return out

def unflatten_columns(df: pd.DataFrame, sep: str = "__") -> pd.DataFrame:
    """
    Split flattened names back to MultiIndex.
    This assumes names were created by `flatten_columns` using the same separator.
    """
    out = df.copy()
    if not isinstance(out.columns, pd.MultiIndex):
        splits = [tuple(str(c).split(sep)) for c in out.columns]
        max_levels = max(len(t) for t in splits) if splits else 1
        # pad tuples to uniform length
        padded = [t + ("",) * (max_levels - len(t)) for t in splits]
        if max_levels == 2:
            out.columns = pd.MultiIndex.from_tuples(padded, names=_BLOCK_NAMES_2)
        elif max_levels == 3:
            out.columns = pd.MultiIndex.from_tuples(padded, names=_BLOCK_NAMES_3)
        else:
            # 1-level remains as-is
            pass
    return out

# -------------------------------
# Validation / Safety
# -------------------------------

def _assert_same_index(a: pd.DataFrame, b: pd.DataFrame) -> None:
    if not a.index.equals(b.index):
        # Helpful message for common causes
        raise ValueError(
            "Index mismatch between frames. "
            "Check that both are indexed by the same key(s) (e.g., 'site_id'), "
            "sorted, and have identical dtype/normalization."
        )

def assert_unique_index(df: pd.DataFrame, name: str = "frame") -> None:
    if not df.index.is_unique:
        dups = df.index[df.index.duplicated()].unique()
        raise ValueError(f"{name} has duplicate index values (first 10): {dups[:10].tolist()}")

def ensure_multiindex_names(df: pd.DataFrame) -> pd.DataFrame:
    """Set consistent column level names if missing."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        if out.columns.nlevels == 2:
            out.columns = pd.MultiIndex.from_tuples(out.columns, names=_BLOCK_NAMES_2)
        elif out.columns.nlevels == 3:
            out.columns = pd.MultiIndex.from_tuples(out.columns, names=_BLOCK_NAMES_3)
    return out


# -------------------------------
# Merge helpers
# -------------------------------

def merge_into_master_by_station(
    master_df: pd.DataFrame,
    data: Union[pd.Series, pd.DataFrame],
    *,
    key_col: str = "StationID",
    block_name: str = "pollution",
    subblock_name: Optional[str] = None,
    rename_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """
    General helper to merge any Series/DataFrame keyed by StationID into master.

    Parameters
    ----------
    master_df : pd.DataFrame
        Target master with MultiIndex columns (2- or 3-level). Indexed by StationID.
    data : pd.Series or pd.DataFrame
        Values to merge. If DataFrame and contains `key_col`, it's used as index.
        If already indexed by StationID, index is used. Series name becomes column name.
    key_col : str, default 'StationID'
        Column name in `data` to use as key if not already indexed by StationID.
    block_name : str
        Top-level block name to place merged columns under.
    subblock_name : str | None
        Optional subblock level name. If None and master has 3 levels, subblock is omitted
        and a 3rd level will still be 'var'. If provided, structure is (block, subblock, var).
    rename_map : dict[str, str] | None
        Optional column rename mapping for `data` prior to wrapping.

    Returns
    -------
    pd.DataFrame
        Updated master with merged columns appended.
    """
    if isinstance(data, pd.Series):
        df = data.to_frame()
    else:
        df = data.copy()

    # If key present as a column, set as index
    if key_col in df.columns:
        df = df.set_index(key_col)

    # Validate index alignment key
    assert_unique_index(df, name="data to merge")

    if rename_map:
        df = df.rename(columns=rename_map)

    # Wrap columns under (block [,subblock], var)
    wrapped = wrap_columns(df, block=block_name, subblock=subblock_name)

    # Align indices by union, then join
    # Prefer strict join on existing master index positions
    _assert_same_index(master_df, master_df)  # no-op to reuse messaging style
    # Use map to align values to master index ordering
    out = master_df.copy()
    for col in wrapped.columns:
        # Extract the leaf name to get a Series to map
        leaf = col[-1]
        series = df[leaf]
        mapped = master_df.index.to_series().map(series)
        out[col] = mapped

    return out


def merge_pollution_scores_into_master(
    master_df: pd.DataFrame,
    scores_df: Union[pd.Series, pd.DataFrame],
    *,
    block_name: str = "pollution",
    subblock_name: str = "sumreal",
    key_col: str = "StationID",
) -> pd.DataFrame:
    """
    Backward-compatible wrapper for merging pollution scores into master.

    This is a convenience alias over `merge_into_master_by_station` and can
    merge any DataFrame/Series keyed by StationID, not just pollution scores.

    Parameters are identical in spirit; see `merge_into_master_by_station`.
    """
    return merge_into_master_by_station(
        master_df,
        scores_df,
        key_col=key_col,
        block_name=block_name,
        subblock_name=subblock_name,
    )