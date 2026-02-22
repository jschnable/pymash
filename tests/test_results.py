import numpy as np

from pymash.covariances import cov_canonical
from pymash.data import mash_set_data
from pymash.mash import mash, mash_compute_posterior_matrices
from pymash.results import (
    format_credible_set_report,
    get_estimated_pi,
    get_lfdr,
    get_lfsr,
    get_log10bf,
    get_n_significant_conditions,
    get_pairwise_sharing,
    get_pm,
    get_psd,
    get_significant_results,
    renumber_credible_sets_by_logbf,
    renumber_credible_sets_from_result,
)
from pymash.simulations import simple_sims
from pymash.workflow import apply_mash_prior_chunked, mash_train_apply


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


def test_result_helpers_accept_train_apply_result():
    sim = simple_sims(nsamp=14, ncond=3, err_sd=0.4, seed=33)
    data = mash_set_data(sim["Bhat"], sim["Shat"])
    U = cov_canonical(data)
    workflow = mash_train_apply(
        data,
        U,
        n_train=20,
        select_method="topz_random",
        select_seed=7,
        train_mash_kwargs={"grid": np.array([1.0]), "optmethod": "auto"},
        apply_mash_kwargs={"outputlevel": 2, "output_lfdr": True},
    )

    pm = get_pm(workflow)
    lfsr = get_lfsr(workflow)
    lfdr = get_lfdr(workflow)
    pi_cov = get_estimated_pi(workflow, dimension="cov")
    bf = get_log10bf(workflow)

    assert pm.shape == sim["Bhat"].shape
    assert lfsr.shape == sim["Bhat"].shape
    assert lfdr.shape == sim["Bhat"].shape
    assert np.isclose(np.sum(pi_cov), 1.0)
    assert bf is not None and bf.shape[0] == sim["Bhat"].shape[0]


def test_result_helpers_accept_chunked_apply_result():
    sim = simple_sims(nsamp=14, ncond=3, err_sd=0.4, seed=34)
    data = mash_set_data(sim["Bhat"], sim["Shat"])
    U = cov_canonical(data)
    workflow = mash_train_apply(
        data,
        U,
        n_train=18,
        select_method="random",
        select_seed=11,
        train_mash_kwargs={"grid": np.array([1.0]), "optmethod": "auto"},
        apply_mash_kwargs={"outputlevel": 2, "output_lfdr": True},
    )

    chunked = apply_mash_prior_chunked(
        data,
        workflow.train_result.fitted_g,
        chunk_size=9,
        mash_kwargs={"outputlevel": 2, "output_lfdr": True},
    )

    pm = get_pm(chunked)
    psd = get_psd(chunked)
    lfsr = get_lfsr(chunked)
    lfdr = get_lfdr(chunked)
    pi_cov = get_estimated_pi(chunked, dimension="cov")
    bf = get_log10bf(chunked)

    assert pm.shape == sim["Bhat"].shape
    assert psd.shape == sim["Bhat"].shape
    assert lfsr.shape == sim["Bhat"].shape
    assert lfdr.shape == sim["Bhat"].shape
    assert np.isclose(np.sum(pi_cov), 1.0)
    assert bf is not None and bf.shape[0] == sim["Bhat"].shape[0]


def test_credible_set_renumbering_descends_by_logbf():
    cs = np.array([0, 4, 4, 2, 2, 3, 3, 3], dtype=int)
    log10bf = np.array([1.0, 0.7, 2.1, 3.2, np.nan, 1.1, 1.0, 0.4], dtype=float)

    renumbered, summary = renumber_credible_sets_by_logbf(cs, log10bf, unassigned_label=0)

    assert renumbered.tolist() == [0, 2, 2, 1, 1, 3, 3, 3]
    assert [row.old_credible_set for row in summary] == [2, 4, 3]
    assert [row.credible_set for row in summary] == [1, 2, 3]
    assert np.isclose(summary[0].lead_log10bf, 3.2)
    assert np.isclose(summary[1].lead_log10bf, 2.1)
    report = format_credible_set_report(summary, top_n=2)
    assert "CS1" in report
    assert "3.2000" in report


def test_credible_set_renumbering_from_result_uses_logbf():
    sim = simple_sims(nsamp=14, ncond=3, err_sd=0.4, seed=35)
    data = mash_set_data(sim["Bhat"], sim["Shat"])
    U = cov_canonical(data)
    m = mash(data, Ulist=U, grid=np.array([1.0]), outputlevel=2)

    J = sim["Bhat"].shape[0]
    cs = np.where(np.arange(J) < (J // 2), 10, 20)
    renumbered, summary = renumber_credible_sets_from_result(m, cs, unassigned_label=0)

    bf = get_log10bf(m)
    assert bf is not None
    lead10 = np.nanmax(bf[cs == 10])
    lead20 = np.nanmax(bf[cs == 20])
    expected_first_old = 10 if lead10 >= lead20 else 20

    assert summary[0].old_credible_set == expected_first_old
    assert len(np.unique(renumbered[cs == 10])) == 1
    assert len(np.unique(renumbered[cs == 20])) == 1
    assert set(np.unique(renumbered[cs != 0]).tolist()) == {1, 2}
