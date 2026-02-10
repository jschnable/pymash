from __future__ import annotations

import numpy as np
import pytest

from pymash.data import mash_set_data
from pymash.likelihoods import calc_relative_lik_matrix
from pymash.posterior import compute_posterior_general_from_loglik, compute_posterior_matrices, compute_posterior_weights
import pymash.posterior as posterior_mod
from pymash._numerics import compute_lfsr, pnorm


def _random_corr_stack(J: int, R: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((J, R, R), dtype=float)
    for j in range(J):
        m = rng.normal(size=(R, R))
        c = m @ m.T + 0.2 * np.eye(R)
        d = np.sqrt(np.diag(c))
        out[j] = c / np.outer(d, d)
    return out


def _posterior_general_numpy_ref(
    data,
    A: np.ndarray,
    Ulist: list[np.ndarray],
    posterior_weights: np.ndarray,
    output_posterior_cov: bool,
):
    J = data.n_effects
    Q = A.shape[0]
    P = len(Ulist)

    res_post_mean = np.zeros((J, Q), dtype=float)
    res_post_mean2 = np.zeros((J, Q), dtype=float)
    res_post_zero = np.zeros((J, Q), dtype=float)
    res_post_neg = np.zeros((J, Q), dtype=float)
    post_sec_w_sum = np.zeros((J, Q, Q), dtype=float) if output_posterior_cov else None

    R = data.n_conditions
    cov_stack = np.empty((J, R, R), dtype=float)
    for j in range(J):
        cov_stack[j] = data.get_cov(j)

    Vinv_stack = np.linalg.inv(cov_stack)
    shat_alpha = data.Shat_alpha
    AS = shat_alpha[:, None, :] * A[None, :, :]
    AS_t = np.transpose(AS, (0, 2, 1)) if output_posterior_cov else None

    Vinv_bhat = np.matmul(Vinv_stack, data.Bhat[..., None]).squeeze(-1)
    eye_r = np.tile(np.eye(R, dtype=float), (J, 1, 1))

    for p in range(P):
        U = Ulist[p]
        solved = np.linalg.solve(np.matmul(Vinv_stack, U) + eye_r, eye_r)
        U1 = np.matmul(U, solved)

        mu1 = np.matmul(U1, Vinv_bhat[..., None]).squeeze(-1)
        muA = np.where(np.isfinite((mu1 * shat_alpha) @ A.T), (mu1 * shat_alpha) @ A.T, 0.0)

        temp = np.matmul(AS, U1)
        post_var = np.maximum(0.0, np.where(np.isfinite(np.sum(temp * AS, axis=2)), np.sum(temp * AS, axis=2), 0.0))

        w = posterior_weights[:, p][:, None]
        res_post_mean += w * muA
        res_post_mean2 += w * (muA * muA + post_var)

        null_cond = post_var == 0.0
        res_post_zero += w * null_cond

        neg_prob = pnorm(0.0, mean=muA, sd=np.sqrt(np.maximum(post_var, np.finfo(float).tiny)))
        neg_prob = np.where(np.isfinite(neg_prob), neg_prob, 0.0)
        res_post_neg += w * np.where(null_cond, 0.0, neg_prob)

        if output_posterior_cov:
            assert post_sec_w_sum is not None
            pvar = np.where(np.isfinite(np.matmul(temp, AS_t)), np.matmul(temp, AS_t), 0.0)
            post_sec_w_sum += posterior_weights[:, p][:, None, None] * (
                pvar + np.einsum("jq,jk->jqk", muA, muA)
            )

    res_post_var = np.maximum(0.0, res_post_mean2 - res_post_mean * res_post_mean)
    res_post_sd = np.sqrt(res_post_var)
    res_lfsr = compute_lfsr(res_post_neg, res_post_zero)

    post_cov = None
    if output_posterior_cov:
        assert post_sec_w_sum is not None
        post_cov = np.transpose(
            post_sec_w_sum - np.einsum("jq,jk->jqk", res_post_mean, res_post_mean),
            (1, 2, 0),
        )

    return res_post_mean, res_post_sd, res_post_neg, res_post_zero, res_lfsr, post_cov


@pytest.mark.skipif(not posterior_mod._use_cpp_general_posterior(), reason="C++ posterior backend unavailable")
def test_cpp_general_posterior_matches_numpy_reference() -> None:
    rng = np.random.default_rng(41)
    J, R = 16, 5
    bhat = rng.normal(size=(J, R))
    shat = np.exp(rng.normal(scale=0.2, size=(J, R)))
    V = _random_corr_stack(J, R, seed=12)
    data = mash_set_data(bhat, shat, V=V)

    A = rng.normal(size=(3, R))
    v = rng.normal(size=R)
    Ulist = [
        np.zeros((R, R), dtype=float),
        np.outer(v, v),
        np.eye(R, dtype=float),
    ]

    W = rng.uniform(size=(J, len(Ulist)))
    W /= np.sum(W, axis=1, keepdims=True)

    pm_cpp = compute_posterior_matrices(data, Ulist, W, A=A, output_posterior_cov=True)
    pm_ref = _posterior_general_numpy_ref(data, A, Ulist, W, output_posterior_cov=True)

    np.testing.assert_allclose(pm_cpp.posterior_mean, pm_ref[0], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(pm_cpp.posterior_sd, pm_ref[1], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(pm_cpp.negative_prob, pm_ref[2], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(pm_cpp.zero_prob, pm_ref[3], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(pm_cpp.lfsr, pm_ref[4], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(pm_cpp.posterior_cov, pm_ref[5], atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(
    not posterior_mod._use_cpp_general_posterior_from_loglik(),
    reason="C++ posterior-from-loglik backend unavailable",
)
def test_cpp_general_posterior_from_loglik_matches_weighted_backend() -> None:
    rng = np.random.default_rng(314)
    J, R = 20, 6
    bhat = rng.normal(size=(J, R))
    shat = np.exp(rng.normal(scale=0.2, size=(J, R)))
    V = _random_corr_stack(J, R, seed=9)
    data = mash_set_data(bhat, shat, V=V)

    A = rng.normal(size=(4, R))
    x = rng.normal(size=R)
    Ulist = [
        np.zeros((R, R), dtype=float),
        np.outer(x, x),
        np.eye(R, dtype=float),
    ]
    pi = np.array([0.2, 0.5, 0.3], dtype=float)

    lm = calc_relative_lik_matrix(data, Ulist)
    W = compute_posterior_weights(pi, np.exp(lm.loglik_matrix))
    pm_w = compute_posterior_matrices(data, Ulist, W, A=A, output_posterior_cov=True)
    pm_l = compute_posterior_general_from_loglik(
        data,
        A=A,
        Ulist=Ulist,
        component_loglik=lm.loglik_matrix,
        component_pi=pi,
        output_posterior_cov=True,
    )

    np.testing.assert_allclose(pm_l.posterior_mean, pm_w.posterior_mean, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(pm_l.posterior_sd, pm_w.posterior_sd, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(pm_l.negative_prob, pm_w.negative_prob, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(pm_l.zero_prob, pm_w.zero_prob, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(pm_l.lfsr, pm_w.lfsr, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(pm_l.posterior_cov, pm_w.posterior_cov, atol=1e-7, rtol=1e-7)
