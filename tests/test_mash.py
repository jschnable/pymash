import numpy as np
import pytest
import importlib

from pymash.covariances import cov_canonical
from pymash.data import mash_set_data
from pymash.mash import mash, mash_compute_posterior_matrices
from pymash.simulations import simple_sims


def test_mash_end_to_end():
    sim = simple_sims(nsamp=20, ncond=4, err_sd=0.5, seed=1)
    data = mash_set_data(sim["Bhat"], sim["Shat"])
    U = cov_canonical(data)

    m = mash(data, Ulist=U, grid=np.array([0.5, 1.0]), outputlevel=2)

    assert m.posterior_mean is not None
    assert m.posterior_sd is not None
    assert m.lfsr is not None
    assert m.posterior_mean.shape == sim["Bhat"].shape
    assert m.posterior_sd.shape == sim["Bhat"].shape
    assert m.lfsr.shape == sim["Bhat"].shape
    assert np.isfinite(m.loglik)
    assert np.isclose(np.sum(m.fitted_g.pi), 1.0)


def test_mash_compute_posterior_matrices_reuse_model():
    sim = simple_sims(nsamp=10, ncond=3, err_sd=0.5, seed=2)
    data = mash_set_data(sim["Bhat"], sim["Shat"])
    U = cov_canonical(data)
    m = mash(data, Ulist=U, grid=np.array([1.0]), outputlevel=2)

    post = mash_compute_posterior_matrices(m, data)
    assert post.posterior_mean.shape == sim["Bhat"].shape
    assert post.posterior_sd.shape == sim["Bhat"].shape


def test_mash_fails_fast_when_cpp_backend_missing(monkeypatch):
    mash_mod = importlib.import_module("pymash.mash")
    sim = simple_sims(nsamp=8, ncond=3, err_sd=0.5, seed=5)
    data = mash_set_data(sim["Bhat"], sim["Shat"])
    U = cov_canonical(data)

    def _fail(context: str = "likelihood") -> None:
        _ = context
        raise RuntimeError("missing backend")

    monkeypatch.setattr(mash_mod, "require_cpp_backend", _fail)
    with pytest.raises(RuntimeError, match="missing backend"):
        mash(data, Ulist=U, grid=np.array([1.0]), outputlevel=1)
