from __future__ import annotations
import re
from typing import Iterable, Tuple, Optional, Union, Iterable, Literal
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
            df = master.loc[:, (block, slice(None), slice(None))]
            # collapse top level(s)
            df = df.copy()
            df.columns = df.columns.get_level_values("var")
            return df
        else:
            df = master.loc[:, (block, subblock, slice(None))]
            df = df.copy()
            df.columns = df.columns.get_level_values("var")
            return df
    else:
        df = master.loc[:, (block, slice(None))]
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