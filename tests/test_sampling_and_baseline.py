import numpy as np

import pymash as mash_api
from pymash.covariances import cov_canonical
from pymash.data import mash_set_data
from pymash.mash import mash, mash_1by1
from pymash.results import get_pairwise_sharing_from_samples
from pymash.simulations import simple_sims


def test_posterior_sampling_and_pairwise_sharing_from_samples():
    sim = simple_sims(nsamp=10, ncond=3, err_sd=0.6, seed=50)
    data = mash_set_data(sim["Bhat"], sim["Shat"])
    U = cov_canonical(data)

    m = mash(
        data,
        Ulist=U,
        grid=np.array([0.5, 1.0]),
        outputlevel=2,
        posterior_samples=8,
        seed=123,
    )

    assert m.posterior_samples is not None
    assert m.posterior_samples.shape == (sim["Bhat"].shape[0], sim["Bhat"].shape[1], 8)

    sharing = get_pairwise_sharing_from_samples(m, factor=0.5, lfsr_thresh=0.5)
    assert sharing.shape == (3, 3)


def test_mash_1by1_baseline():
    sim = simple_sims(nsamp=12, ncond=4, err_sd=0.7, seed=51)
    data = mash_set_data(sim["Bhat"], sim["Shat"])
    m = mash_1by1(data)

    assert m.posterior_mean is not None
    assert m.posterior_sd is not None
    assert m.lfsr is not None
    assert m.lfdr is not None
    assert m.posterior_mean.shape == sim["Bhat"].shape
    assert np.isfinite(m.loglik)


def test_mash_1by1_applies_univariate_shrinkage():
    rng = np.random.default_rng(60)
    J, R = 600, 3
    shat = np.exp(rng.normal(loc=0.0, scale=0.4, size=(J, R)))
    bhat = rng.normal(loc=0.0, scale=shat)
    data = mash_set_data(bhat, shat)
    m = mash_1by1(data)

    assert m.posterior_mean is not None
    assert np.mean(np.abs(m.posterior_mean)) < np.mean(np.abs(bhat))
    assert np.mean(m.lfdr) > 0.1


def test_mash_1by1_supports_alpha_one_scale():
    sim = simple_sims(nsamp=10, ncond=3, err_sd=0.9, seed=61)
    data = mash_set_data(sim["Bhat"], sim["Shat"], alpha=1.0)
    m = mash_1by1(data, alpha=1.0)

    assert m.posterior_mean is not None
    assert m.posterior_sd is not None
    assert np.mean(np.abs(m.posterior_mean)) <= np.mean(np.abs(sim["Bhat"]))
    assert np.mean(m.posterior_sd) <= np.mean(sim["Shat"])


def test_general_mode_posterior_sampling_runs():
    rng = np.random.default_rng(62)
    sim = simple_sims(nsamp=8, ncond=4, err_sd=0.7, seed=63)
    J, R = sim["Bhat"].shape

    V = np.empty((J, R, R), dtype=float)
    for j in range(J):
        z = rng.normal(size=(R, R))
        cov = z @ z.T + 0.2 * np.eye(R)
        d = np.sqrt(np.diag(cov))
        V[j] = cov / np.outer(d, d)

    data = mash_set_data(sim["Bhat"], sim["Shat"], V=V)
    U = cov_canonical(data)
    m = mash(
        data,
        Ulist=U,
        grid=np.array([0.5, 1.0]),
        outputlevel=2,
        posterior_samples=5,
        seed=64,
    )

    assert m.posterior_samples is not None
    assert m.posterior_samples.shape == (J, R, 5)
    assert np.all(np.isfinite(m.posterior_samples))


def test_top_level_exports_pairwise_sharing_from_samples():
    assert hasattr(mash_api, "get_pairwise_sharing_from_samples")
    assert hasattr(mash_api, "cov_udi")
