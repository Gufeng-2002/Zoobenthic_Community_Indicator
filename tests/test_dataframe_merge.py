import pandas as pd
import pytest

from ecoindex.dataframe_ops import (
    wrap_columns,
    add_site_block,
    concat_blocks,
    align_blocks_by_index,
    get_block,
)


def test_add_site_block_rejects_misaligned_index():
    # Create two frames with the same index values but different orders
    df1 = pd.DataFrame({"site": ["A", "B", "C"], "x": [1, 2, 3]}).set_index("site")
    df2 = pd.DataFrame({"site": ["C", "A", "B"], "y": [10, 20, 30]}).set_index("site")

    # Start a master from df1 (wrapped)
    master = wrap_columns(df1, block="taxa", subblock="raw")

    # add_site_block requires identical indexes; mis-ordered index should raise
    with pytest.raises(ValueError):
        _ = add_site_block(master, df2, block="chemical", subblock="raw")


def test_add_site_block_after_align_merges_by_index():
    # Same data, mismatched order
    df1 = pd.DataFrame({"site": ["A", "B", "C"], "x": [1, 2, 3]}).set_index("site")
    df2 = pd.DataFrame({"site": ["C", "A", "B"], "y": [10, 20, 30]}).set_index("site")

    # Align df2 to df1's index using left-anchor
    aligned1, aligned2 = align_blocks_by_index([df1, df2], how="left", anchor=0)

    # Build master from aligned1 and then add aligned2 via add_site_block
    master = wrap_columns(aligned1, block="taxa", subblock="raw")
    merged = add_site_block(master, aligned2, block="chemical", subblock="raw")

    taxa_raw = get_block(merged, "taxa", "raw")
    chem_raw = get_block(merged, "chemical", "raw")

    # Expect index-wise alignment:
    # A -> x=1, y=20; B -> x=2, y=30; C -> x=3, y=10
    assert taxa_raw.loc["A", "x"] == 1
    assert chem_raw.loc["A", "y"] == 20

    assert taxa_raw.loc["B", "x"] == 2
    assert chem_raw.loc["B", "y"] == 30

    assert taxa_raw.loc["C", "x"] == 3
    assert chem_raw.loc["C", "y"] == 10


def test_align_then_concat_preserves_index_alignment():
    # Same data, mismatched order
    df1 = pd.DataFrame({"site": ["A", "B", "C"], "x": [1, 2, 3]}).set_index("site")
    df2 = pd.DataFrame({"site": ["C", "A", "B"], "y": [10, 20, 30]}).set_index("site")

    # Align df2 to df1's index using left-anchor
    aligned1, aligned2 = align_blocks_by_index([df1, df2], how="left", anchor=0)

    # Sanity: aligned2's order now matches df1
    assert list(aligned1.index) == ["A", "B", "C"]
    assert list(aligned2.index) == ["A", "B", "C"]

    # Wrap and concatenate blocks
    w1 = wrap_columns(aligned1, block="taxa", subblock="raw")
    w2 = wrap_columns(aligned2, block="chemical", subblock="raw")
    merged = concat_blocks([w1, w2])

    taxa_raw = get_block(merged, "taxa", "raw")
    chem_raw = get_block(merged, "chemical", "raw")

    # Same expectations as above
    assert taxa_raw.loc["A", "x"] == 1
    assert chem_raw.loc["A", "y"] == 20

    assert taxa_raw.loc["B", "x"] == 2
    assert chem_raw.loc["B", "y"] == 30

    assert taxa_raw.loc["C", "x"] == 3
    assert chem_raw.loc["C", "y"] == 10
