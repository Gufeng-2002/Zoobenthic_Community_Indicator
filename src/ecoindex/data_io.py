from __future__ import annotations
from pathlib import Path
import pandas as pd
from .config import INTERIM

def save_interim(df: pd.DataFrame, name: str) -> Path:
    """
    Save a DataFrame to the interim data directory as a Parquet file.
    
    Args:
        df: The DataFrame to save
        name: The filename (without path) for the saved file
        
    Returns:
        Path: The full path to the saved file
    """
    INTERIM.mkdir(parents=True, exist_ok=True)
    path = INTERIM / name
    df.to_parquet(path, index=False)
    return path

def load_interim(name: str) -> pd.DataFrame:
    """
    Load a DataFrame from the interim data directory.
    
    Args:
        name: The filename (without path) to load
        
    Returns:
        pd.DataFrame: The loaded DataFrame
    """
    return pd.read_parquet(INTERIM / name)