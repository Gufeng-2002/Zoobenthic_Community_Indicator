import numpy as np
import pandas as pd
import pytest
from ecoindex.transform import hellinger_transform, log1p_standardize

def test_hellinger_shape_and_range():
    df = pd.DataFrame([[0, 2, 8], [4, 0, 1]], columns=list("ABC"))
    H = hellinger_transform(df)
    assert H.shape == df.shape
    assert (H.values >= -1e-12).all() and (H.values <= 1 + 1e-12).all()
    # identical rows produce identical transforms
    H2 = hellinger_transform(pd.concat([df.iloc[[0]], df.iloc[[0]]], ignore_index=True))
    assert np.allclose(H2.iloc[0], H2.iloc[1])

def test_hellinger_rejects_negative():
    df = pd.DataFrame([[-1, 0]])
    with pytest.raises(ValueError):
        hellinger_transform(df)

def test_log1p_standardize_basic():
    df = pd.DataFrame([[0, 1, 10], [3, 0, 7]], columns=list("ABC"))
    Z = log1p_standardize(df)
    assert Z.shape == df.shape
    # columnwise z-score: mean ~0, std ~1 (on log1p scale)
    col_means = Z.mean(0).to_numpy()
    col_stds = Z.std(0, ddof=1).to_numpy()
    assert np.allclose(col_means, 0, atol=1e-8)
    assert np.allclose(col_stds, 1, atol=1e-8)