from __future__ import annotations

import numpy as np

from pymash.covariances import cov_canonical
from pymash.data import mash_set_data
from pymash.mash import mash
from pymash.simulations import simple_sims
from pymash.workflow import apply_mash_prior, fit_mash_prior, mash_train_apply, select_training_effects


def test_select_training_effects_random_deterministic():
    sim = simple_sims(nsamp=20, ncond=4, err_sd=0.8, seed=200)
    data = mash_set_data(sim["Bhat"], sim["Shat"])
    idx1 = select_training_effects(data, n_train=25, method="random", seed=11)
    idx2 = select_training_effects(data, n_train=25, method="random", seed=11)
    assert idx1.shape == (25,)
    assert np.array_equal(idx1, idx2)
    assert np.all(np.diff(idx1) >= 0)


def test_select_training_effects_topz_random_captures_top_hits():
    rng = np.random.default_rng(201)
    J, R = 200, 3
    shat = np.ones((J, R), dtype=float)
    bhat = rng.normal(scale=0.1, size=(J, R))
    bhat[5, 1] = 20.0
    bhat[40, 2] = -15.0
    data = mash_set_data(bhat, shat)

    idx = select_training_effects(
        data,
        n_train=30,
        method="topz_random",
        seed=7,
        background_fraction=0.3,
    )
    assert 5 in idx
    assert 40 in idx


def test_fit_and_apply_prior_matches_manual_two_stage():
    sim = simple_sims(nsamp=15, ncond=4, err_sd=0.7, seed=202)
    data = mash_set_data(sim["Bhat"], sim["Shat"])
    U = cov_canonical(data)
    train_idx = np.arange(0, data.n_effects, 3)

    kwargs = {"grid": np.array([1.0]), "prior": "uniform", "optmethod": "mixsqp"}

    g_fit, idx_fit, train_fit = fit_mash_prior(
        data,
        U,
        train_indices=train_idx,
        mash_kwargs=kwargs,
    )
    apply_fit = apply_mash_prior(data, g_fit, mash_kwargs={"outputlevel": 2, "output_lfdr": True})

    data_train = mash_set_data(sim["Bhat"][train_idx], sim["Shat"][train_idx])
    train_manual = mash(data_train, Ulist=U, outputlevel=1, **kwargs)
    apply_manual = mash(data, g=train_manual.fitted_g, fixg=True, outputlevel=2, output_lfdr=True)

    assert np.array_equal(idx_fit, np.sort(train_idx))
    assert np.allclose(g_fit.pi, train_manual.fitted_g.pi, atol=1e-10, rtol=1e-10)
    assert train_fit.posterior_mean is None

    assert apply_fit.posterior_mean is not None and apply_manual.posterior_mean is not None
    assert apply_fit.posterior_sd is not None and apply_manual.posterior_sd is not None
    assert apply_fit.lfsr is not None and apply_manual.lfsr is not None

    assert np.allclose(apply_fit.posterior_mean, apply_manual.posterior_mean, atol=1e-10, rtol=1e-10)
    assert np.allclose(apply_fit.posterior_sd, apply_manual.posterior_sd, atol=1e-10, rtol=1e-10)
    assert np.allclose(apply_fit.lfsr, apply_manual.lfsr, atol=1e-10, rtol=1e-10)


def test_mash_train_apply_effect_specific_v_runs():
    rng = np.random.default_rng(203)
    sim = simple_sims(nsamp=8, ncond=3, err_sd=0.6, seed=204)
    J, R = sim["Bhat"].shape
    V = np.empty((J, R, R), dtype=float)
    for j in range(J):
        z = rng.normal(size=(R, R))
        cov = z @ z.T + 0.3 * np.eye(R)
        d = np.sqrt(np.diag(cov))
        V[j] = cov / np.outer(d, d)

    data = mash_set_data(sim["Bhat"], sim["Shat"], V=V)
    U = cov_canonical(data)

    out = mash_train_apply(
        data,
        U,
        n_train=12,
        select_method="random",
        select_seed=5,
        train_mash_kwargs={"grid": np.array([1.0]), "optmethod": "auto"},
        apply_mash_kwargs={"outputlevel": 2},
    )

    assert out.train_indices.shape == (12,)
    assert out.apply_result.posterior_mean is not None
    assert out.apply_result.posterior_mean.shape == sim["Bhat"].shape
