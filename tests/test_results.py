import numpy as np

from pymash.covariances import cov_canonical
from pymash.data import mash_set_data
from pymash.mash import mash, mash_compute_posterior_matrices
from pymash.results import (
    get_estimated_pi,
    get_n_significant_conditions,
    get_pairwise_sharing,
    get_pm,
    get_psd,
    get_significant_results,
)
from pymash.simulations import simple_sims


def test_result_helpers_work():
    sim = simple_sims(nsamp=15, ncond=3, err_sd=0.3, seed=3)
    data = mash_set_data(sim["Bhat"], sim["Shat"])
    U = cov_canonical(data)
    m = mash(data, Ulist=U, grid=np.array([0.5, 1.0]), outputlevel=2)

    pm = get_pm(m)
    psd = get_psd(m)
    sig = get_significant_results(m, thresh=0.2)
    nsig = get_n_significant_conditions(m, thresh=0.2)
    pi_cov = get_estimated_pi(m, dimension="cov")
    sharing = get_pairwise_sharing(m, factor=0.5, lfsr_thresh=0.2)

    assert pm.shape == sim["Bhat"].shape
    assert psd.shape == sim["Bhat"].shape
    assert sig.ndim == 1
    assert nsig.shape[0] == sim["Bhat"].shape[0]
    assert np.isclose(np.sum(pi_cov), 1.0)
    assert sharing.shape == (3, 3)


def test_result_helpers_accept_posterior_matrices():
    sim = simple_sims(nsamp=15, ncond=3, err_sd=0.3, seed=4)
    data = mash_set_data(sim["Bhat"], sim["Shat"])
    U = cov_canonical(data)
    m = mash(data, Ulist=U, grid=np.array([0.5, 1.0]), outputlevel=2)
    post = mash_compute_posterior_matrices(m, data)

    pm = get_pm(post)
    sig = get_significant_results(post, thresh=0.2)
    nsig = get_n_significant_conditions(post, thresh=0.2)

    assert pm.shape == sim["Bhat"].shape
    assert sig.ndim == 1
    assert nsig.shape[0] == sim["Bhat"].shape[0]
