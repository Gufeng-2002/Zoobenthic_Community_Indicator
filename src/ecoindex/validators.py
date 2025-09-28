from __future__ import annotations
import pandera as pa
from pandera import Column, DataFrameSchema, Check

schema_keys = DataFrameSchema({
    "site_id": Column(str, nullable=False),
    "sample_datetime": Column(pa.DateTime, nullable=False),
})

# Tailor these to your columns/prefixes
def assert_keys(df):
    schema_keys.validate(df, lazy=True)