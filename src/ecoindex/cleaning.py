from __future__ import annotations
import pandas as pd
import numpy as np

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names by stripping whitespace, replacing spaces with underscores,
    and removing special characters.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with normalized column names
    """
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^0-9A-Za-z_]", "", regex=True)
        # .str.lower()
    )
    return df

def cast_types(df: pd.DataFrame, spec: dict[str, str]) -> pd.DataFrame:
    """
    Cast columns to specified data types based on a specification dictionary.
    
    Args:
        df: Input DataFrame
        spec: Dictionary mapping column names to target data types
        
    Returns:
        DataFrame with columns cast to specified types
    """
    df = df.copy()
    for col, t in spec.items():
        if col not in df.columns:
            continue
        if t.startswith("datetime"):
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            df[col] = df[col].astype(t)
    return df

def harmonize_ids(df: pd.DataFrame, id_col="site_id") -> pd.DataFrame:
    """
    Standardize ID column values by converting to uppercase strings and stripping whitespace.
    
    Args:
        df: Input DataFrame
        id_col: Name of the ID column to harmonize (default: "site_id")
        
    Returns:
        DataFrame with standardized ID column
    """
    if id_col in df.columns:
        df[id_col] = df[id_col].astype(str).str.strip().str.upper()
    return df

def drop_duplicates_on_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """
    Remove duplicate rows based on specified key columns.
    
    Args:
        df: Input DataFrame
        keys: List of column names to use for duplicate detection
        
    Returns:
        DataFrame with duplicates removed
    """
    # Handles single key or multiple keys
    return df.drop_duplicates(subset=keys[0] if len(keys) == 1 else keys)

def handle_missing(df: pd.DataFrame, strategy: str, cols_like: str | None = None) -> pd.DataFrame:
    """
    Handle missing values in numeric columns using specified strategy.
    
    Args:
        df: Input DataFrame
        strategy: Strategy for handling missing values ("zero_for_absent_taxa", 
                 "median_by_column", or "drop_rows_if_any")
        cols_like: Filter columns containing this substring (optional)
        
    Returns:
        DataFrame with missing values handled according to strategy
    """
    df = df.copy()
    cols = (
    df.filter(like=cols_like).select_dtypes(include="number").columns
    if cols_like else df.select_dtypes(include="number").columns
    )
    if strategy == "zero_for_absent_taxa":
        df[cols] = df[cols].fillna(0.0)
    elif strategy == "median_by_column":
        df[cols] = df[cols].apply(lambda s: s.fillna(s.median()))
    elif strategy == "drop_rows_if_any":
        df = df.dropna(subset=cols)
    else:
        raise ValueError(f"Unknown missing strategy: {strategy}")
    return df

def ensure_nonnegative(df: pd.DataFrame, cols_like: str) -> pd.DataFrame:
    """
    Validate that numeric columns contain only non-negative values.
    
    Args:
        df: Input DataFrame
        cols_like: Filter columns containing this substring
        
    Returns:
        Original DataFrame if validation passes
        
    Raises:
        ValueError: If negative values are found in specified columns
    """
    cols = (
    df.filter(like=cols_like).select_dtypes(include="number").columns
    if cols_like else df.select_dtypes(include="number").columns
    )
    if (df[cols] < 0).any().any():
        raise ValueError(f"Negative values found in {cols_like} columns.")
    return df