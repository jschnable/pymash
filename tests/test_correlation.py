import numpy as np

from pymash import estimate_null_correlation_simple as estimate_null_correlation_simple_top_level
from pymash.correlation import estimate_null_correlation_simple, mash_estimate_corr_em
from pymash.covariances import cov_canonical
from pymash.data import mash_set_data
from pymash.simulations import simple_sims


def test_estimate_null_correlation_simple_returns_correlation():
    sim = simple_sims(nsamp=40, ncond=4, err_sd=1.0, seed=100)
    data = mash_set_data(sim["Bhat"], sim["Shat"])

    Vhat = estimate_null_correlation_simple(data, z_thresh=2.5, est_cor=True)
    assert Vhat.shape == (4, 4)
    assert np.allclose(Vhat, Vhat.T, atol=1e-10)
    assert np.allclose(np.diag(Vhat), 1.0, atol=1e-8)


def test_estimate_null_correlation_simple_is_top_level_export():
    sim = simple_sims(nsamp=35, ncond=4, err_sd=1.0, seed=102)
    data = mash_set_data(sim["Bhat"], sim["Shat"])
    Vhat = estimate_null_correlation_simple_top_level(data, z_thresh=2.5, est_cor=True)
    assert Vhat.shape == (4, 4)


def test_mash_estimate_corr_em_runs_and_returns_details():
    sim = simple_sims(nsamp=12, ncond=4, err_sd=0.8, seed=101)
    data = mash_set_data(sim["Bhat"], sim["Shat"])
    U = cov_canonical(data)

    out = mash_estimate_corr_em(
        data,
        U,
        max_iter=3,
        tol=1e-3,
        est_cor=True,
        track_fit=True,
        prior="uniform",
        details=True,
        grid=np.array([0.5, 1.0]),
    )

    V = out["V"]
    assert V.shape == (4, 4)
    assert np.allclose(V, V.T, atol=1e-8)
    assert np.allclose(np.diag(V), 1.0, atol=1e-6)
    assert out["mash_model"].posterior_cov is not None
    assert out["niter"] >= 1
    assert out["loglik"].ndim == 1
    assert "trace" in out
