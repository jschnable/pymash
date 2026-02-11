from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import numpy as np

from .mash import MashResult
from .posterior import PosteriorMatrices

ResultLike = Union[MashResult, PosteriorMatrices]


@dataclass
class _ChunkedResultView:
    posterior_mean: np.ndarray | None
    posterior_sd: np.ndarray | None
    lfsr: np.ndarray | None
    lfdr: np.ndarray | None
    negative_prob: np.ndarray | None
    posterior_samples: np.ndarray | None
    null_loglik: np.ndarray | None
    alt_loglik: np.ndarray | None
    fitted_g: Any


def _normalize_result_container(m: Any) -> Any:
    # TrainApplyResult: expose the full-data result by default.
    if hasattr(m, "apply_result") and hasattr(m, "train_result"):
        return m.apply_result

    # ChunkedApplyResult: adapt array dict to a result-like view.
    if hasattr(m, "arrays") and hasattr(m, "fitted_g"):
        arrays = getattr(m, "arrays")
        return _ChunkedResultView(
            posterior_mean=arrays.get("posterior_mean"),
            posterior_sd=arrays.get("posterior_sd"),
            lfsr=arrays.get("lfsr"),
            lfdr=arrays.get("lfdr"),
            negative_prob=arrays.get("negative_prob"),
            posterior_samples=arrays.get("posterior_samples"),
            null_loglik=arrays.get("null_loglik"),
            alt_loglik=arrays.get("alt_loglik"),
            fitted_g=getattr(m, "fitted_g"),
        )
    return m


def get_log10bf(m: Any) -> np.ndarray | None:
    mm = _normalize_result_container(m)
    if mm.null_loglik is None or mm.alt_loglik is None:
        return None
    return (mm.alt_loglik - mm.null_loglik) / np.log(10.0)


def _require_matrix(x: np.ndarray | None, name: str) -> np.ndarray:
    if x is None:
        raise ValueError(f"{name} is not available in this result")
    return x


def get_pm(m: ResultLike) -> np.ndarray:
    """Extract posterior mean matrix from a mash result.

    Parameters
    ----------
    m : MashResult or PosteriorMatrices
        Fitted mash result or posterior summary object.

    Returns
    -------
    np.ndarray
        Posterior means, shape ``(J, R)``.
    """
    mm = _normalize_result_container(m)
    return _require_matrix(mm.posterior_mean, "posterior_mean")


def get_psd(m: ResultLike) -> np.ndarray:
    """Extract posterior standard deviation matrix from a mash result.

    Parameters
    ----------
    m : MashResult or PosteriorMatrices
        Fitted mash result or posterior summary object.

    Returns
    -------
    np.ndarray
        Posterior standard deviations, shape ``(J, R)``.
    """
    mm = _normalize_result_container(m)
    return _require_matrix(mm.posterior_sd, "posterior_sd")


def get_lfsr(m: ResultLike) -> np.ndarray:
    """Extract local false sign rate (lfsr) matrix from a mash result.

    The lfsr is the posterior probability that the sign of the effect
    is incorrect, providing a Bayesian measure of significance.

    Parameters
    ----------
    m : MashResult or PosteriorMatrices
        Fitted mash result or posterior summary object.

    Returns
    -------
    np.ndarray
        Local false sign rates, shape ``(J, R)``.
    """
    mm = _normalize_result_container(m)
    return _require_matrix(mm.lfsr, "lfsr")


def get_lfdr(m: ResultLike) -> np.ndarray:
    """Extract local false discovery rate (lfdr) matrix from a mash result.

    The lfdr is the posterior probability that the true effect is exactly
    zero. This differs from :func:`get_lfsr` (local false *sign* rate),
    which is the probability of getting the sign wrong. In practice,
    lfsr is more commonly used because it is more robust when the
    point-mass at zero is not a good model for small effects.

    Only available when ``mash()`` was called with ``output_lfdr=True``.

    Parameters
    ----------
    m : MashResult or PosteriorMatrices
        Fitted mash result or posterior summary object.

    Returns
    -------
    np.ndarray
        Local false discovery rates, shape ``(J, R)``.
    """
    mm = _normalize_result_container(m)
    return _require_matrix(mm.lfdr, "lfdr")


def get_ncond(m: ResultLike) -> int:
    return get_pm(m).shape[1]


def get_significant_results(
    m: ResultLike,
    thresh: float = 0.05,
    conditions: np.ndarray | list[int] | None = None,
    sig_fn=get_lfsr,
) -> np.ndarray:
    """Get indices of effects that are significant in at least one condition.

    Returns effect indices where the significance measure (default: lfsr)
    is below ``thresh`` in at least one of the specified conditions,
    sorted from most to least significant.

    Parameters
    ----------
    m : MashResult
        Fitted mash result.
    thresh : float
        Significance threshold (default 0.05).
    conditions : array-like of int, optional
        Column indices to consider. Defaults to all conditions.
    sig_fn : callable
        Function to extract the significance matrix (default :func:`get_lfsr`).

    Returns
    -------
    np.ndarray
        Sorted array of row indices of significant effects.

    Examples
    --------
    >>> sig = get_significant_results(result, thresh=0.05)
    >>> len(sig)
    150
    """
    sig = sig_fn(m)
    if conditions is None:
        conditions_arr = np.arange(sig.shape[1])
    else:
        conditions_arr = np.asarray(conditions, dtype=int)
    top = np.min(sig[:, conditions_arr], axis=1)
    idx = np.where(top < thresh)[0]
    ord_idx = np.argsort(top[idx])
    return idx[ord_idx]


def get_n_significant_conditions(
    m: ResultLike,
    thresh: float = 0.05,
    conditions: np.ndarray | list[int] | None = None,
    sig_fn=get_lfsr,
) -> np.ndarray:
    """Count the number of significant conditions for each effect.

    For each of J effects, returns how many conditions have a
    significance measure (default: lfsr) below ``thresh``.

    Parameters
    ----------
    m : MashResult or PosteriorMatrices
        Fitted mash result.
    thresh : float
        Significance threshold (default 0.05).
    conditions : array-like of int, optional
        Column indices to consider. Defaults to all conditions.
    sig_fn : callable
        Function to extract the significance matrix (default :func:`get_lfsr`).

    Returns
    -------
    np.ndarray
        Integer array of length J with per-effect counts.

    Examples
    --------
    >>> n_sig = get_n_significant_conditions(result)
    >>> pleiotropic = np.sum(n_sig >= 2)
    """
    sig = sig_fn(m)
    if conditions is None:
        conditions_arr = np.arange(sig.shape[1])
    else:
        conditions_arr = np.asarray(conditions, dtype=int)
    return np.sum(sig[:, conditions_arr] < thresh, axis=1)


def get_estimated_pi(m: MashResult, dimension: str = "cov") -> np.ndarray:
    """Extract estimated mixture proportions from a mash result.

    Parameters
    ----------
    m : MashResult
        Fitted mash result.
    dimension : str
        How to collapse the mixture proportions:

        - ``"cov"``: sum over grid values for each covariance matrix.
        - ``"grid"``: sum over covariance matrices for each grid value.
        - ``"all"``: return the full (uncollapsed) pi vector.

    Returns
    -------
    np.ndarray
        Estimated mixture proportions.

    Examples
    --------
    >>> pi_cov = get_estimated_pi(result, dimension="cov")
    """
    dimension = dimension.lower()
    if dimension not in {"cov", "grid", "all"}:
        raise ValueError("dimension must be one of 'cov', 'grid', 'all'")

    mm = _normalize_result_container(m)
    if not hasattr(mm, "fitted_g"):
        raise ValueError("fitted_g is not available in this result")
    g = mm.fitted_g
    pihat = np.asarray(g.pi, dtype=float)

    if dimension == "all":
        return pihat

    pi_null = None
    if g.usepointmass:
        pi_null = pihat[0]
        pihat = pihat[1:]

    K = len(g.Ulist)
    if pihat.size % K != 0:
        raise ValueError("pi shape does not match Ulist x grid dimensions")

    pimat = pihat.reshape((-1, K)).T
    if dimension == "cov":
        collapsed = np.sum(pimat, axis=1)
    else:
        collapsed = np.sum(pimat, axis=0)

    if pi_null is not None:
        collapsed = np.concatenate(([pi_null], collapsed))
    return collapsed


def get_pairwise_sharing(
    m: ResultLike,
    factor: float = 0.5,
    lfsr_thresh: float = 0.05,
    FUN=lambda x: x,
) -> np.ndarray:
    """Compute pairwise sharing of significant effects between conditions.

    For each pair of conditions, identifies effects that are significant
    in at least one of the two conditions, then computes the fraction
    whose posterior means are the same sign and within a factor of each
    other.

    Parameters
    ----------
    m : MashResult
        Fitted mash result.
    factor : float
        Sharing factor. Effects are "shared" if their ratio is between
        ``factor`` and ``1/factor``. Use 0 to assess only sign sharing.
    lfsr_thresh : float
        Threshold for significance.
    FUN : callable
        Transformation applied to posterior means before comparison
        (e.g., ``np.abs``).

    Returns
    -------
    np.ndarray
        Symmetric sharing matrix of shape ``(R, R)``.

    Examples
    --------
    >>> sharing = get_pairwise_sharing(result, factor=0.5)
    """
    R = get_ncond(m)
    lfsr = get_lfsr(m)
    pm = get_pm(m)
    S = np.full((R, R), np.nan, dtype=float)

    for i in range(R):
        for j in range(i, R):
            sig_i = get_significant_results(m, thresh=lfsr_thresh, conditions=[i])
            sig_j = get_significant_results(m, thresh=lfsr_thresh, conditions=[j])
            a = np.union1d(sig_i, sig_j)
            if a.size == 0:
                S[i, j] = np.nan
                continue
            denom = FUN(pm[a, j])
            numer = FUN(pm[a, i])
            ratio = np.divide(
                numer,
                denom,
                out=np.full_like(numer, np.nan, dtype=float),
                where=denom != 0,
            )
            if factor == 0:
                S[i, j] = np.mean(ratio > 0)
            else:
                S[i, j] = np.mean((ratio > factor) & (ratio < (1.0 / factor)))

    lower = np.tril_indices(R, k=-1)
    S[lower] = S.T[lower]
    return S


def get_pairwise_sharing_from_samples(
    m: ResultLike,
    factor: float = 0.5,
    lfsr_thresh: float = 0.05,
    FUN=lambda x: x,
) -> np.ndarray:
    mm = _normalize_result_container(m)
    samples = mm.posterior_samples
    if samples is None:
        raise ValueError("No posterior samples available")

    J, R, M = samples.shape
    _ = J
    S = np.full((R, R), np.nan, dtype=float)

    for i in range(R):
        for j in range(i, R):
            sig_i = get_significant_results(m, thresh=lfsr_thresh, conditions=[i])
            sig_j = get_significant_results(m, thresh=lfsr_thresh, conditions=[j])
            a = np.union1d(sig_i, sig_j)
            if a.size == 0:
                S[i, j] = np.nan
                continue
            numer = FUN(samples[a, i, :])
            denom = FUN(samples[a, j, :])
            ratio = np.divide(
                numer,
                denom,
                out=np.full_like(numer, np.nan, dtype=float),
                where=denom != 0,
            )
            if factor == 0:
                S[i, j] = np.mean(np.mean(ratio > 0, axis=1))
            else:
                S[i, j] = np.mean(np.mean((ratio > factor) & (ratio < (1.0 / factor)), axis=1))

    lower = np.tril_indices(R, k=-1)
    S[lower] = S.T[lower]
    return S


__all__ = [
    "get_log10bf",
    "get_significant_results",
    "get_n_significant_conditions",
    "get_estimated_pi",
    "get_pairwise_sharing",
    "get_pairwise_sharing_from_samples",
    "get_pm",
    "get_psd",
    "get_lfsr",
    "get_lfdr",
    "get_ncond",
]
