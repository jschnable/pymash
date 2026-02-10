import numpy as np
import pytest

from pymash.covariances import cov_canonical, cov_udi, expand_cov
from pymash.data import mash_set_data


def test_cov_canonical_and_expand():
    Bhat = np.zeros((3, 4))
    data = mash_set_data(Bhat, Shat=1.0)

    U = cov_canonical(data)
    assert "identity" in U
    assert "equal_effects" in U
    assert any(k.startswith("singleton_") for k in U)

    xU = expand_cov(U, grid=np.array([0.5, 1.0]), usepointmass=True)
    assert len(xU) == 1 + 2 * len(U)
    assert np.allclose(xU[0], np.zeros((4, 4)))


def test_cov_udi_model_validation_and_naming():
    rng = np.random.default_rng(1)
    bhat = rng.normal(size=(20, 5))
    z = rng.normal(size=(5, 5))
    cov = z @ z.T + np.eye(5)
    d = np.sqrt(np.diag(cov))
    V = cov / np.outer(d, d)
    data = mash_set_data(bhat, Shat=1.0, V=V)

    with pytest.raises(ValueError, match="elements U, D, I"):
        cov_udi(data, ["U", "U", "D", "D", "A"])
    with pytest.raises(ValueError, match="at least one direct association"):
        cov_udi(data, ["U", "U", "I", "I", "I"])

    out = cov_udi(data, ["U", "U", "I", "I", "D"])
    assert list(out.keys()) == ["cov_udi_UUIID"]
    mat = out["cov_udi_UUIID"]
    assert mat.shape == (5, 5)
    assert np.allclose(mat, mat.T, atol=1e-10)


def test_cov_udi_default_model_count():
    rng = np.random.default_rng(2)
    bhat = rng.normal(size=(8, 3))
    data = mash_set_data(bhat, Shat=1.0)
    out = cov_udi(data)
    assert len(out) == 3**3 - 2**3
