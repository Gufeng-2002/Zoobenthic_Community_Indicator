# tests/test_cleaning.py
import pandas as pd
from ecoindex.cleaning import drop_duplicates_on_keys, handle_missing, harmonize_ids

def test_drop_duplicates_on_keys():
    df = pd.DataFrame({"site_id":["A","A"], "sample_datetime":["2024-01-01","2024-01-01"], "x":[1,1]})
    out = drop_duplicates_on_keys(df, keys=["site_id","sample_datetime"])
    assert len(out) == 1

def test_handle_missing_zero_for_taxa():
    df = pd.DataFrame({"taxa_A":[1, None], "taxa_B":[None, 2.0]})
    out = handle_missing(df, "zero_for_absent_taxa", cols_like="taxa_")
    assert out.isna().sum().sum() == 0
    assert float(out.loc[0, "taxa_B"]) == 0.0

def test_harmonize_ids_upper_trim():
    df = pd.DataFrame({"site_id":[" abc ","X-1"], "v":[1,2]})
    out = harmonize_ids(df)
    assert list(out["site_id"]) == ["ABC","X-1"]