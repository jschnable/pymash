from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._numerics import compute_lfsr, pnorm
from .data import MashData, build_cov_stack

try:
    from . import _edcpp as _edcpp_backend
except Exception:  # pragma: no cover - import failure fallback
    _edcpp_backend = None


@dataclass
class PosteriorMatrices:
    posterior_mean: np.ndarray
    posterior_sd: np.ndarray
    negative_prob: np.ndarray
    zero_prob: np.ndarray
    lfsr: np.ndarray
    lfdr: np.ndarray
    posterior_cov: np.ndarray | None = None
    posterior_samples: np.ndarray | None = None


def posterior_cov(Vinv: np.ndarray, U: np.ndarray) -> np.ndarray:
    R = U.shape[0]
    with np.errstate(all="ignore"):
        system = Vinv @ U + np.eye(R)
    try:
        solved = np.linalg.solve(system, np.eye(R))
    except np.linalg.LinAlgError:
        return np.zeros_like(U, dtype=float)
    with np.errstate(all="ignore"):
        out = U @ solved
    return np.where(np.isfinite(out), out, 0.0)


def posterior_mean(bhat: np.ndarray, Vinv: np.ndarray, U1: np.ndarray) -> np.ndarray:
    return U1 @ (Vinv @ bhat)


def posterior_mean_matrix(Bhat: np.ndarray, Vinv: np.ndarray, U1: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        out = Bhat @ (Vinv @ U1)
    return np.where(np.isfinite(out), out, 0.0)


def compute_posterior_weights(pi: np.ndarray, lik_mat: np.ndarray) -> np.ndarray:
    pi = np.asarray(pi, dtype=float)
    lik_mat = np.asarray(lik_mat, dtype=float)
    d = lik_mat * pi[None, :]
    norm = np.sum(d, axis=1, keepdims=True)
    norm = np.maximum(norm, np.finfo(float).tiny)
    return d / norm


def compute_posterior_from_component_terms(
    component_mean: np.ndarray,
    component_var: np.ndarray,
    posterior_weights: np.ndarray,
) -> PosteriorMatrices:
    """Aggregate per-component posterior terms into final posterior summaries."""
    mu = np.asarray(component_mean, dtype=float)
    var = np.asarray(component_var, dtype=float)
    w = np.asarray(posterior_weights, dtype=float)

    if mu.ndim != 3 or var.ndim != 3:
        raise ValueError("component_mean and component_var must have shape (J, K, Q)")
    if mu.shape != var.shape:
        raise ValueError("component_mean and component_var must have the same shape")
    if w.ndim != 2 or w.shape != mu.shape[:2]:
        raise ValueError("posterior_weights must have shape (J, K)")

    J, K, Q = mu.shape
    res_post_mean = np.einsum("jk,jkq->jq", w, mu)
    res_post_mean2 = np.einsum("jk,jkq->jq", w, mu * mu + var)
    res_post_zero = np.einsum("jk,jkq->jq", w, (var == 0.0).astype(float))

    res_post_neg = np.zeros((J, Q), dtype=float)
    tiny = np.finfo(float).tiny
    for k in range(K):
        wk = w[:, k][:, None]
        if not np.any(wk > 0.0):
            continue
        var_k = var[:, k, :]
        mu_k = mu[:, k, :]
        null_k = var_k == 0.0
        if np.any(~null_k):
            with np.errstate(all="ignore"):
                neg_k = pnorm(
                    0.0,
                    mean=mu_k,
                    sd=np.sqrt(np.maximum(var_k, tiny)),
                )
            neg_k = np.where(np.isfinite(neg_k), neg_k, 0.0)
            res_post_neg += wk * np.where(null_k, 0.0, neg_k)

    res_post_var = np.maximum(0.0, res_post_mean2 - res_post_mean * res_post_mean)
    res_post_sd = np.sqrt(res_post_var)
    res_lfsr = compute_lfsr(res_post_neg, res_post_zero)

    return PosteriorMatrices(
        posterior_mean=res_post_mean,
        posterior_sd=res_post_sd,
        negative_prob=res_post_neg,
        zero_prob=res_post_zero,
        lfsr=res_lfsr,
        lfdr=res_post_zero,
        posterior_cov=None,
        posterior_samples=None,
    )


def _use_cpp_general_posterior() -> bool:
    return _edcpp_backend is not None and hasattr(_edcpp_backend, "compute_posterior_general_stats")


def _use_cpp_general_posterior_from_loglik() -> bool:
    return _edcpp_backend is not None and hasattr(_edcpp_backend, "compute_posterior_general_stats_from_loglik")


def _prepare_u_factors(Ulist: list[np.ndarray], rank_tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    P = len(Ulist)
    R = Ulist[0].shape[0]
    low_rank_max_rank = max(1, R // 2)
    ranks = np.full(P, -1, dtype=np.int32)
    factors = np.zeros((P, R, low_rank_max_rank), dtype=float)

    for p, U in enumerate(Ulist):
        evals, evecs = np.linalg.eigh(U)
        pos = evals > rank_tol
        rank = int(np.sum(pos))
        if rank == 0:
            ranks[p] = 0
            continue
        if rank > low_rank_max_rank:
            continue
        idx = np.where(pos)[0]
        L = evecs[:, idx] * np.sqrt(np.maximum(evals[idx], 0.0))[None, :]
        factors[p, :, :rank] = L
        ranks[p] = rank

    return factors, ranks


def _posterior_matrices_from_backend_stats(
    stats: dict,
    output_posterior_cov: bool,
) -> PosteriorMatrices:
    res_post_mean = np.asarray(stats["mean"], dtype=float)
    res_post_mean2 = np.asarray(stats["mean2"], dtype=float)
    res_post_zero = np.asarray(stats["zero"], dtype=float)
    res_post_neg = np.asarray(stats["neg"], dtype=float)
    post_sec_w_sum = np.asarray(stats["post_sec_w_sum"], dtype=float) if output_posterior_cov else None

    res_post_var = np.maximum(0.0, res_post_mean2 - res_post_mean * res_post_mean)
    res_post_sd = np.sqrt(res_post_var)
    res_lfsr = compute_lfsr(res_post_neg, res_post_zero)

    res_post_cov = None
    if output_posterior_cov:
        assert post_sec_w_sum is not None
        res_post_cov = np.transpose(post_sec_w_sum - np.einsum("jq,jk->jqk", res_post_mean, res_post_mean), (1, 2, 0))

    return PosteriorMatrices(
        posterior_mean=res_post_mean,
        posterior_sd=res_post_sd,
        negative_prob=res_post_neg,
        zero_prob=res_post_zero,
        lfsr=res_lfsr,
        lfdr=res_post_zero,
        posterior_cov=res_post_cov,
        posterior_samples=None,
    )


def _compute_posterior_common_cov(
    data: MashData,
    A: np.ndarray,
    Ulist: list[np.ndarray],
    posterior_weights: np.ndarray,
    output_posterior_cov: bool,
    posterior_samples: int,
    seed: int,
) -> PosteriorMatrices:
    J = data.n_effects
    Q = A.shape[0]
    P = len(Ulist)

    res_post_mean = np.zeros((J, Q), dtype=float)
    res_post_mean2 = np.zeros((J, Q), dtype=float)
    res_post_zero = np.zeros((J, Q), dtype=float)
    res_post_neg = np.zeros((J, Q), dtype=float)

    post_sec_w_sum = np.zeros((J, Q, Q), dtype=float) if output_posterior_cov else None

    cov_list: list[np.ndarray] = []
    mean_list: list[np.ndarray] = []

    V = data.get_cov(0)
    Vinv = np.linalg.inv(V)
    sa = data.Shat_alpha[0]
    AS = A * sa[None, :]
    need_full_pvar = output_posterior_cov or posterior_samples > 0

    if (not data.is_common_cov_shat()) and (not data.is_common_cov_shat_alpha()):
        raise ValueError("Common-covariance posterior called with non-common covariance data")

    for p in range(P):
        U1 = posterior_cov(Vinv, Ulist[p])
        with np.errstate(all="ignore"):
            transform = ((Vinv @ U1) * sa[None, :]) @ A.T
            muA = data.Bhat @ transform
        muA = np.where(np.isfinite(muA), muA, 0.0)

        with np.errstate(all="ignore"):
            temp = AS @ U1
            post_var = np.sum(temp * AS, axis=1)
        post_var = np.maximum(0.0, np.where(np.isfinite(post_var), post_var, 0.0))

        pvar = None
        if need_full_pvar:
            with np.errstate(all="ignore"):
                pvar = temp @ AS.T
            pvar = np.where(np.isfinite(pvar), pvar, 0.0)

        w = posterior_weights[:, p][:, None]
        res_post_mean += w * muA
        res_post_mean2 += w * (muA * muA + post_var[None, :])

        null_cond = post_var == 0.0
        if np.any(null_cond):
            res_post_zero[:, null_cond] += posterior_weights[:, p][:, None]

        nonnull = ~null_cond
        if np.any(nonnull):
            res_post_neg[:, nonnull] += w * pnorm(
                0.0,
                mean=muA[:, nonnull],
                sd=np.sqrt(post_var[nonnull])[None, :],
            )

        if output_posterior_cov:
            assert post_sec_w_sum is not None
            assert pvar is not None
            post_sec_w_sum += posterior_weights[:, p][:, None, None] * (
                pvar[None, :, :] + np.einsum("jq,jk->jqk", muA, muA)
            )

        if posterior_samples > 0:
            assert pvar is not None
            cov_list.append(pvar)
            mean_list.append(muA)

    res_post_var = np.maximum(0.0, res_post_mean2 - res_post_mean * res_post_mean)
    res_post_sd = np.sqrt(res_post_var)
    res_lfsr = compute_lfsr(res_post_neg, res_post_zero)

    res_post_cov = None
    if output_posterior_cov:
        assert post_sec_w_sum is not None
        res_post_cov = np.transpose(post_sec_w_sum - np.einsum("jq,jk->jqk", res_post_mean, res_post_mean), (1, 2, 0))

    res_post_samples = None
    if posterior_samples > 0:
        rng = np.random.default_rng(seed)
        res_post_samples = np.zeros((J, Q, posterior_samples), dtype=float)
        for j in range(J):
            counts = rng.multinomial(posterior_samples, posterior_weights[j])
            draws: list[np.ndarray] = []
            for p in range(P):
                c = int(counts[p])
                if c == 0:
                    continue
                mu = mean_list[p][j]
                cov = cov_list[p]
                sample = rng.multivariate_normal(mu, cov, size=c, check_valid="ignore", method="eigh")
                draws.append(np.atleast_2d(sample))
            merged = np.vstack(draws)
            perm = rng.permutation(posterior_samples)
            res_post_samples[j] = merged[perm].T

    return PosteriorMatrices(
        posterior_mean=res_post_mean,
        posterior_sd=res_post_sd,
        negative_prob=res_post_neg,
        zero_prob=res_post_zero,
        lfsr=res_lfsr,
        lfdr=res_post_zero,
        posterior_cov=res_post_cov,
        posterior_samples=res_post_samples,
    )


def _compute_posterior_general(
    data: MashData,
    A: np.ndarray,
    Ulist: list[np.ndarray],
    posterior_weights: np.ndarray,
    output_posterior_cov: bool,
    cov_stack: np.ndarray | None = None,
    u_factors: np.ndarray | None = None,
    u_ranks: np.ndarray | None = None,
) -> PosteriorMatrices:
    if not _use_cpp_general_posterior():
        raise RuntimeError(
            "C++ posterior backend is required for effect-specific covariance mode. "
            "Reinstall pymash with extension build support."
        )

    cov_stack = _resolve_general_cov_stack(data, cov_stack)

    u_stack = np.asarray(Ulist, dtype=float)
    if u_factors is None or u_ranks is None:
        u_factors, u_ranks = _prepare_u_factors(Ulist)
    else:
        u_factors = np.asarray(u_factors, dtype=float)
        u_ranks = np.asarray(u_ranks, dtype=np.int32)
    stats = _edcpp_backend.compute_posterior_general_stats(
        data.Bhat,
        data.Shat_alpha,
        cov_stack,
        u_stack,
        u_factors,
        u_ranks,
        posterior_weights,
        A,
        output_posterior_cov,
    )
    return _posterior_matrices_from_backend_stats(stats, output_posterior_cov)


def _resolve_general_cov_stack(data: MashData, cov_stack: np.ndarray | None) -> np.ndarray:
    if cov_stack is None:
        return build_cov_stack(data)
    cov_stack = np.asarray(cov_stack, dtype=float)
    expected = (data.n_effects, data.n_conditions, data.n_conditions)
    if cov_stack.shape != expected:
        raise ValueError(f"cov_stack must have shape {expected}")
    return cov_stack


def _compute_posterior_general_with_sampling(
    data: MashData,
    A: np.ndarray,
    Ulist: list[np.ndarray],
    posterior_weights: np.ndarray,
    output_posterior_cov: bool,
    posterior_samples: int,
    seed: int,
    cov_stack: np.ndarray | None = None,
) -> PosteriorMatrices:
    J = data.n_effects
    Q = A.shape[0]
    P = len(Ulist)

    cov_stack_arr = _resolve_general_cov_stack(data, cov_stack)
    Vinv_stack = np.linalg.inv(cov_stack_arr)
    tiny = np.finfo(float).tiny

    res_post_mean = np.zeros((J, Q), dtype=float)
    res_post_mean2 = np.zeros((J, Q), dtype=float)
    res_post_zero = np.zeros((J, Q), dtype=float)
    res_post_neg = np.zeros((J, Q), dtype=float)
    post_sec_w_sum = np.zeros((J, Q, Q), dtype=float) if output_posterior_cov else None

    rng = np.random.default_rng(seed)
    res_post_samples = np.zeros((J, Q, posterior_samples), dtype=float)

    for j in range(J):
        bhat_j = data.Bhat[j]
        shat_alpha_j = data.Shat_alpha[j]
        AS_j = A * shat_alpha_j[None, :]
        Vinv_j = Vinv_stack[j]

        mean_list: list[np.ndarray] = []
        cov_list: list[np.ndarray] = []

        for p in range(P):
            U1 = posterior_cov(Vinv_j, Ulist[p])
            with np.errstate(all="ignore"):
                mu1 = posterior_mean(bhat_j, Vinv_j, U1) * shat_alpha_j
                muA = A @ mu1
            muA = np.where(np.isfinite(muA), muA, 0.0)

            with np.errstate(all="ignore"):
                temp = AS_j @ U1
                post_var = np.sum(temp * AS_j, axis=1)
                pvar = temp @ AS_j.T
            post_var = np.maximum(0.0, np.where(np.isfinite(post_var), post_var, 0.0))
            pvar = np.where(np.isfinite(pvar), pvar, 0.0)

            w = float(posterior_weights[j, p])
            res_post_mean[j] += w * muA
            res_post_mean2[j] += w * (muA * muA + post_var)

            null_cond = post_var == 0.0
            if np.any(null_cond):
                res_post_zero[j, null_cond] += w
            nonnull = ~null_cond
            if np.any(nonnull):
                with np.errstate(all="ignore"):
                    neg_prob = pnorm(0.0, mean=muA[nonnull], sd=np.sqrt(np.maximum(post_var[nonnull], tiny)))
                neg_prob = np.where(np.isfinite(neg_prob), neg_prob, 0.0)
                res_post_neg[j, nonnull] += w * neg_prob

            if output_posterior_cov:
                assert post_sec_w_sum is not None
                post_sec_w_sum[j] += w * (pvar + np.outer(muA, muA))

            mean_list.append(muA)
            cov_list.append(pvar)

        counts = rng.multinomial(posterior_samples, posterior_weights[j])
        draws: list[np.ndarray] = []
        for p in range(P):
            c = int(counts[p])
            if c == 0:
                continue
            sample = rng.multivariate_normal(
                mean_list[p],
                cov_list[p],
                size=c,
                check_valid="ignore",
                method="eigh",
            )
            draws.append(np.atleast_2d(sample))
        merged = np.vstack(draws)
        perm = rng.permutation(posterior_samples)
        res_post_samples[j] = merged[perm].T

    res_post_var = np.maximum(0.0, res_post_mean2 - res_post_mean * res_post_mean)
    res_post_sd = np.sqrt(res_post_var)
    res_lfsr = compute_lfsr(res_post_neg, res_post_zero)

    res_post_cov = None
    if output_posterior_cov:
        assert post_sec_w_sum is not None
        res_post_cov = np.transpose(post_sec_w_sum - np.einsum("jq,jk->jqk", res_post_mean, res_post_mean), (1, 2, 0))

    return PosteriorMatrices(
        posterior_mean=res_post_mean,
        posterior_sd=res_post_sd,
        negative_prob=res_post_neg,
        zero_prob=res_post_zero,
        lfsr=res_lfsr,
        lfdr=res_post_zero,
        posterior_cov=res_post_cov,
        posterior_samples=res_post_samples,
    )


def compute_posterior_general_from_loglik(
    data: MashData,
    A: np.ndarray,
    Ulist: list[np.ndarray],
    component_loglik: np.ndarray,
    component_pi: np.ndarray,
    output_posterior_cov: bool,
    cov_stack: np.ndarray | None = None,
    u_factors: np.ndarray | None = None,
    u_ranks: np.ndarray | None = None,
) -> PosteriorMatrices:
    if not _use_cpp_general_posterior_from_loglik():
        raise RuntimeError(
            "C++ posterior-from-loglik backend is required for effect-specific covariance mode. "
            "Reinstall pymash with extension build support."
        )

    cov_stack = _resolve_general_cov_stack(data, cov_stack)

    u_stack = np.asarray(Ulist, dtype=float)
    if u_factors is None or u_ranks is None:
        u_factors, u_ranks = _prepare_u_factors(Ulist)
    else:
        u_factors = np.asarray(u_factors, dtype=float)
        u_ranks = np.asarray(u_ranks, dtype=np.int32)

    component_loglik = np.asarray(component_loglik, dtype=float)
    component_pi = np.asarray(component_pi, dtype=float)
    expected_llik = (data.n_effects, len(Ulist))
    if component_loglik.shape != expected_llik:
        raise ValueError(f"component_loglik must have shape {expected_llik}")
    if component_pi.shape != (len(Ulist),):
        raise ValueError(f"component_pi must have shape ({len(Ulist)},)")

    stats = _edcpp_backend.compute_posterior_general_stats_from_loglik(
        data.Bhat,
        data.Shat_alpha,
        cov_stack,
        u_stack,
        u_factors,
        u_ranks,
        component_loglik,
        component_pi,
        A,
        output_posterior_cov,
    )
    return _posterior_matrices_from_backend_stats(stats, output_posterior_cov)


def compute_posterior_matrices(
    data: MashData,
    Ulist: list[np.ndarray],
    posterior_weights: np.ndarray,
    A: np.ndarray | None = None,
    output_posterior_cov: bool = False,
    posterior_samples: int = 0,
    seed: int = 123,
    cov_stack: np.ndarray | None = None,
    u_factors: np.ndarray | None = None,
    u_ranks: np.ndarray | None = None,
) -> PosteriorMatrices:
    R = data.n_conditions

    if A is None:
        A = np.eye(R, dtype=float)
    else:
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.shape[1] != R:
            raise ValueError("A has invalid dimensions")

    common_cov_shat = data.is_common_cov_shat()
    common_cov_shat_alpha = True if data.alpha == 0 else data.is_common_cov_shat_alpha()
    is_common_cov = data.common_V and common_cov_shat and common_cov_shat_alpha

    if is_common_cov:
        return _compute_posterior_common_cov(
            data,
            A,
            Ulist,
            posterior_weights,
            output_posterior_cov,
            posterior_samples,
            seed,
        )

    if posterior_samples > 0:
        return _compute_posterior_general_with_sampling(
            data,
            A,
            Ulist,
            posterior_weights,
            output_posterior_cov,
            posterior_samples,
            seed,
            cov_stack=cov_stack,
        )

    return _compute_posterior_general(
        data,
        A,
        Ulist,
        posterior_weights,
        output_posterior_cov,
        cov_stack=cov_stack,
        u_factors=u_factors,
        u_ranks=u_ranks,
    )


__all__ = [
    "PosteriorMatrices",
    "posterior_cov",
    "posterior_mean",
    "posterior_mean_matrix",
    "compute_posterior_weights",
    "compute_posterior_from_component_terms",
    "compute_posterior_general_from_loglik",
    "compute_posterior_matrices",
]
