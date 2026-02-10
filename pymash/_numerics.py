from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import ndtr


def mvn_logpdf_batch(X: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Evaluate multivariate normal log-pdf for all rows of X."""
    X = np.asarray(X, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if mean.ndim != 1:
        raise ValueError("mean must be 1D")
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("sigma must be square")
    if X.shape[1] != mean.size or sigma.shape[0] != mean.size:
        raise ValueError("shape mismatch between X, mean, and sigma")

    J, R = X.shape
    try:
        L = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        # Mirror mashr C++ edge-case behavior for singular/non-PD covariance:
        # exact (or near-exact) match to the mean has infinite density.
        diff_l1 = np.sum(np.abs(X - mean[None, :]), axis=1)
        out = np.full(J, -np.inf)
        out[diff_l1 < 1e-6] = np.inf
        return out

    centered = (X - mean).T
    z = solve_triangular(L, centered, lower=True, check_finite=False)
    quad = np.sum(z * z, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    const = R * np.log(2.0 * np.pi)
    return -0.5 * (const + log_det + quad)


def pnorm(x: np.ndarray, mean: np.ndarray | float = 0.0, sd: np.ndarray | float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mean_arr = np.asarray(mean, dtype=float)
    sd_arr = np.asarray(sd, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x - mean_arr) / sd_arr
    return ndtr(z)


def compute_lfsr(neg_prob: np.ndarray, zero_prob: np.ndarray) -> np.ndarray:
    threshold = 0.5 * (1.0 - zero_prob)
    lfsr = np.where(neg_prob > threshold, 1.0 - neg_prob, neg_prob + zero_prob)
    return np.maximum(lfsr, 0.0)


__all__ = ["mvn_logpdf_batch", "pnorm", "compute_lfsr"]
