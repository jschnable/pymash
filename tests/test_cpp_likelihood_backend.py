from __future__ import annotations

import numpy as np
import pytest

from pymash.data import mash_set_data
import pymash.likelihoods as lk


def _random_corr_stack(J: int, R: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((J, R, R), dtype=float)
    for j in range(J):
        m = rng.normal(size=(R, R))
        c = m @ m.T + 0.2 * np.eye(R)
        d = np.sqrt(np.diag(c))
        out[j] = c / np.outer(d, d)
    return out


@pytest.mark.skipif(not lk._use_cpp_general_lik(), reason="C++ likelihood backend unavailable")
def test_cpp_general_lik_matches_numpy() -> None:
    rng = np.random.default_rng(3)
    J, R = 25, 6
    bhat = rng.normal(size=(J, R))
    shat = np.exp(rng.normal(scale=0.2, size=(J, R)))
    V = _random_corr_stack(J, R, seed=8)
    data = mash_set_data(bhat, shat, V=V)

    v = rng.normal(size=R)
    u_rank1 = np.outer(v, v)
    u_full = np.eye(R)
    u_zero = np.zeros((R, R), dtype=float)
    Ulist = [u_zero, 0.5 * u_rank1, 1.3 * u_full]

    ll_np = np.empty((J, len(Ulist)), dtype=float)
    for j in range(J):
        ll_np[j] = lk.calc_lik_vector(data.Bhat[j], data.get_cov(j), Ulist, log=True)
    ll_cpp = lk._calc_lik_matrix_general_cpp(data, Ulist, log=True)

    np.testing.assert_allclose(ll_cpp, ll_np, atol=1e-8, rtol=1e-8)


@pytest.mark.skipif(not lk._use_cpp_general_lik(), reason="C++ likelihood backend unavailable")
def test_cpp_common_lik_matches_numpy_with_shared_cov() -> None:
    rng = np.random.default_rng(11)
    J, R = 30, 5
    bhat = rng.normal(size=(J, R))
    shat = np.ones((J, R), dtype=float)
    data = mash_set_data(bhat, shat, V=np.eye(R))

    v = rng.normal(size=R)
    u_rank1 = np.outer(v, v)
    u_full = np.eye(R)
    u_zero = np.zeros((R, R), dtype=float)
    Ulist = [u_zero, 0.4 * u_rank1, 1.1 * u_full]

    ll_np = np.empty((J, len(Ulist)), dtype=float)
    common_cov = data.get_cov(0)
    for j in range(J):
        ll_np[j] = lk.calc_lik_vector(data.Bhat[j], common_cov, Ulist, log=True)

    ll_cpp = lk._calc_lik_matrix_general_cpp(data, Ulist, log=True, cov_stack=common_cov)
    ll_dispatch = lk.calc_lik_matrix(data, Ulist, log=True)

    np.testing.assert_allclose(ll_cpp, ll_np, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(ll_dispatch, ll_np, atol=1e-8, rtol=1e-8)
