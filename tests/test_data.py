import numpy as np
import pytest

from pymash.data import mash_set_data


def test_mash_set_data_basic_shapes():
    Bhat = np.array([[1.0, 2.0], [3.0, 4.0]])
    data = mash_set_data(Bhat, Shat=1.0)

    assert data.Bhat.shape == (2, 2)
    assert data.Shat.shape == (2, 2)
    assert data.Shat_alpha.shape == (2, 2)
    assert data.common_V
    assert data.V.shape == (2, 2)


def test_mash_set_data_missing_pattern_and_alpha():
    Bhat = np.array([[1.0, np.nan], [0.5, 1.5]])
    Shat = np.array([[0.2, np.nan], [0.4, 0.5]])
    data = mash_set_data(Bhat, Shat=Shat, alpha=1.0)

    assert data.alpha == 1.0
    assert np.isfinite(data.Bhat).all()
    assert np.isfinite(data.Shat).all()
    assert np.isfinite(data.Shat_alpha).all()
    assert data.Bhat[0, 1] == 0.0
    assert data.Shat[0, 1] == 1e6


def test_mash_set_data_negative_shat_has_swap_hint():
    Bhat = np.array([[1.0, 2.0], [3.0, 4.0]])
    Shat = np.array([[0.2, -0.1], [0.4, 0.5]])
    with pytest.raises(ValueError, match="swapped"):
        mash_set_data(Bhat=Bhat, Shat=Shat)
