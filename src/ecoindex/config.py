from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw/from_proposal"
INTERIM = DATA / "interim"
PROC = DATA / "processed"

# raw Excel filenames (adjust to yours)
RAW_TAXA_XLSX = RAW / "taxa_data.xlsx"
RAW_CHEM_XLSX = RAW / "chemical_data.xlsx"
RAW_ENV_XLSX  = RAW / "environmental_data.xlsx"

# keys
KEYS = ["StationID"]  # shared keys for joins
