from __future__ import annotations
import pandas as pd
from .config import RAW_TAXA_XLSX, RAW_CHEM_XLSX, RAW_ENV_XLSX

def read_taxa_raw(path: str | None = None) -> pd.DataFrame:
    return pd.read_excel(path or RAW_TAXA_XLSX, engine="openpyxl")

def read_chem_raw(path: str | None = None) -> pd.DataFrame:
    return pd.read_excel(path or RAW_CHEM_XLSX, engine="openpyxl")

def read_env_raw(path: str | None = None) -> pd.DataFrame:
    return pd.read_excel(path or RAW_ENV_XLSX, engine="openpyxl")