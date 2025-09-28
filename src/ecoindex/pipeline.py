from __future__ import annotations
import pandas as pd
from .config import KEYS
from .ingest import read_taxa_raw, read_chem_raw, read_env_raw
from .cleaning import (
    normalize_columns, cast_types, harmonize_ids,
    drop_duplicates_on_keys, handle_missing, ensure_nonnegative
)
from .data_io import save_interim
# from .validators import assert_keys  # optional

def make_interim():
    # ---- Taxa ----
    taxa = read_taxa_raw()
    taxa = normalize_columns(taxa)
    taxa = cast_types(taxa, {"site_id": "string", "sample_datetime": "datetime64[ns]"})
    taxa = drop_duplicates_on_keys(taxa, KEYS)
    taxa = handle_missing(taxa, "zero_for_absent_taxa", cols_like="taxa_")
    taxa = ensure_nonnegative(taxa, "taxa_")
    # assert_keys(taxa)
    save_interim(taxa, "taxa_clean.parquet")

    # ---- Chemistry ----
    chem = read_chem_raw()
    chem = normalize_columns(chem)
    chem = cast_types(chem, {"site_id": "string", "sample_datetime": "datetime64[ns]"})
    chem = drop_duplicates_on_keys(chem, KEYS)
    chem = handle_missing(chem, "median_by_column", cols_like="chem_")
    chem = ensure_nonnegative(chem, "chem_")
    # assert_keys(chem)
    save_interim(chem, "chem_clean.parquet")

    # ---- Environment ----
    env = read_env_raw()
    env = normalize_columns(env)
    env = cast_types(env, {"site_id": "string", "sample_datetime": "datetime64[ns]"})
    env = drop_duplicates_on_keys(env, KEYS)
    # choose your policy: median/forward-fill/drop
    env = handle_missing(env, "median_by_column")
    # assert_keys(env)
    save_interim(env, "env_clean.parquet")