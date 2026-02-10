import numpy as np
import pytest

from pymash.data import contrast_matrix, mash_set_data, mash_update_data


def test_contrast_matrix_and_update_data_ref_mean():
    Bhat = np.arange(20, dtype=float).reshape(5, 4)
    Shat = np.ones((5, 4), dtype=float)
    data = mash_set_data(Bhat, Shat)

    L = contrast_matrix(4, "mean")
    assert L.shape == (3, 4)

    data_c = mash_update_data(data, ref="mean")
    assert data_c.Bhat.shape == (5, 3)
    assert data_c.L is not None
    assert data_c.Shat_orig is not None


def test_effect_specific_v_shapes_supported():
    rng = np.random.default_rng(1)
    Bhat = rng.normal(size=(6, 3))
    Shat = np.exp(rng.normal(loc=-0.2, scale=0.1, size=(6, 3)))

    # Provide V in (R, R, J) layout.
    V_rrj = np.stack([np.eye(3) for _ in range(6)], axis=2)
    data_rrj = mash_set_data(Bhat, Shat, V=V_rrj)
    assert not data_rrj.common_V
    assert data_rrj.V.shape == (6, 3, 3)

    # Provide V in (J, R, R) layout.
    V_jrr = np.stack([np.eye(3) for _ in range(6)], axis=0)
    data_jrr = mash_set_data(Bhat, Shat, V=V_jrr)
    assert not data_jrr.common_V
    assert data_jrr.V.shape == (6, 3, 3)


def test_alpha_transformation_not_identity_for_alpha_one():
    Bhat = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    Shat = np.array([[2.0, 4.0], [2.0, 4.0]], dtype=float)
    data = mash_set_data(Bhat, Shat, alpha=1.0)

    expected_bhat = Bhat / Shat
    assert np.allclose(data.Bhat, expected_bhat)
    assert np.allclose(data.Shat, np.ones_like(Shat))
    assert np.allclose(data.Shat_alpha, Shat)


def test_contrast_matrix_integer_ref_is_zero_based():
    L = contrast_matrix(4, ref=0)
    assert L.shape == (3, 4)
    assert np.allclose(L[:, 0], -1.0)

    with pytest.raises(ValueError, match="between 0 and R-1"):
        contrast_matrix(4, ref=4)
