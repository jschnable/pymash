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


def test_mash_fixg_chunked_matches_non_chunked():
    sim = simple_sims(nsamp=12, ncond=3, err_sd=0.6, seed=21)
    data = mash_set_data(sim["Bhat"], sim["Shat"])
    U = cov_canonical(data)
    trained = mash(data, Ulist=U, grid=np.array([1.0]), outputlevel=1)

    full = mash(
        data,
        g=trained.fitted_g,
        fixg=True,
        outputlevel=2,
        output_lfdr=True,
        chunk_size=None,
    )
    chunked = mash(
        data,
        g=trained.fitted_g,
        fixg=True,
        outputlevel=2,
        output_lfdr=True,
        chunk_size=7,
    )

    assert chunked.posterior_weights is None
    assert np.allclose(chunked.posterior_mean, full.posterior_mean, atol=1e-10, rtol=1e-10)
    assert np.allclose(chunked.posterior_sd, full.posterior_sd, atol=1e-10, rtol=1e-10)
    assert np.allclose(chunked.lfsr, full.lfsr, atol=1e-10, rtol=1e-10)
    assert np.allclose(chunked.lfdr, full.lfdr, atol=1e-10, rtol=1e-10)
    assert np.allclose(chunked.negative_prob, full.negative_prob, atol=1e-10, rtol=1e-10)
    assert np.allclose(chunked.vloglik, full.vloglik, atol=1e-10, rtol=1e-10)
    assert np.isclose(chunked.loglik, full.loglik, atol=1e-8, rtol=0.0)


def test_mash_compute_posterior_matrices_chunked_matches_non_chunked():
    sim = simple_sims(nsamp=12, ncond=4, err_sd=0.7, seed=22)
    data = mash_set_data(sim["Bhat"], sim["Shat"])
    U = cov_canonical(data)
    trained = mash(data, Ulist=U, grid=np.array([1.0]), outputlevel=1)

    full = mash_compute_posterior_matrices(trained, data, chunk_size=None)
    chunked = mash_compute_posterior_matrices(trained, data, chunk_size=6)

    assert np.allclose(chunked.posterior_mean, full.posterior_mean, atol=1e-10, rtol=1e-10)
    assert np.allclose(chunked.posterior_sd, full.posterior_sd, atol=1e-10, rtol=1e-10)
    assert np.allclose(chunked.negative_prob, full.negative_prob, atol=1e-10, rtol=1e-10)
    assert np.allclose(chunked.zero_prob, full.zero_prob, atol=1e-10, rtol=1e-10)
    assert np.allclose(chunked.lfsr, full.lfsr, atol=1e-10, rtol=1e-10)


def test_mash_large_auto_two_stage_runs_with_chunk_size():
    sim = simple_sims(nsamp=12, ncond=3, err_sd=0.7, seed=23)
    data = mash_set_data(sim["Bhat"], sim["Shat"])
    U = cov_canonical(data)

    out = mash(
        data,
        Ulist=U,
        grid=np.array([1.0]),
        outputlevel=2,
        chunk_size=7,
        output_lfdr=True,
    )

    assert out.posterior_mean is not None
    assert out.posterior_mean.shape == sim["Bhat"].shape
    assert out.posterior_weights is None
    assert np.isfinite(out.loglik)


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
