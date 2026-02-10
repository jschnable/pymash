from __future__ import annotations

import numpy as np

from .data import MashData
from .data import mash_update_data
from .mash import MashResult, mash


def estimate_null_correlation_simple(data: MashData, z_thresh: float = 2.0, est_cor: bool = True) -> np.ndarray:
    """Estimate the null correlation structure among conditions.

    Identifies putatively null effects (those with |z| < ``z_thresh`` in
    all conditions) and estimates their correlation (or covariance) matrix.
    This captures residual correlations among conditions that are not due
    to true effects.

    Parameters
    ----------
    data : MashData
        Data object created by :func:`~pymash.data.mash_set_data`.
    z_thresh : float
        Z-score threshold for selecting null-ish effects.
    est_cor : bool
        If True, return a correlation matrix; if False, return covariance.

    Returns
    -------
    np.ndarray
        Estimated null correlation (or covariance) matrix, shape ``(R, R)``.

    Examples
    --------
    >>> Vhat = estimate_null_correlation_simple(data, z_thresh=2.0)
    >>> data_v = mash_update_data(data, V=Vhat)
    """
    z = data.Bhat / data.Shat
    nullish = np.max(np.abs(z), axis=1) < z_thresh
    if np.sum(nullish) < data.n_conditions:
        raise ValueError("Not enough null effects to estimate correlation")

    z_null = z[nullish]
    if est_cor:
        return np.corrcoef(z_null, rowvar=False)
    return np.cov(z_null, rowvar=False)


def _cov2cor(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    d = np.sqrt(np.maximum(np.diag(V), 0.0))
    denom = np.outer(d, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = V / denom
    C[~np.isfinite(C)] = 0.0
    np.fill_diagonal(C, 1.0)
    return C


def _fit_mash_V(
    data: MashData,
    Ulist: dict[str, np.ndarray] | list[np.ndarray],
    V: np.ndarray,
    prior: str | np.ndarray = "nullbiased",
    **kwargs,
) -> MashResult:
    data_V = mash_update_data(data, V=V)
    # Need posterior covariance for E-step update.
    return mash(data_V, Ulist=Ulist, prior=prior, outputlevel=3, **kwargs)


def E_V(data: MashData, m_model: MashResult) -> np.ndarray:
    if m_model.posterior_mean is None or m_model.posterior_cov is None:
        raise ValueError("mash model must include posterior mean and posterior covariance (outputlevel >= 3)")

    Z = data.Bhat / data.Shat
    Shat = data.Shat * data.Shat_alpha
    post_m_shat = m_model.posterior_mean / Shat

    J, R = post_m_shat.shape
    temp3 = np.zeros((R, R), dtype=float)
    for i in range(J):
        s_outer = np.outer(Shat[i], Shat[i])
        post_cov_scaled = np.divide(
            m_model.posterior_cov[:, :, i],
            s_outer,
            out=np.zeros_like(m_model.posterior_cov[:, :, i]),
            where=s_outer != 0,
        )
        temp3 += post_cov_scaled + np.outer(post_m_shat[i], post_m_shat[i])

    temp1 = Z.T @ Z
    temp2 = post_m_shat.T @ Z + Z.T @ post_m_shat
    V = temp1 - temp2 + temp3
    return 0.5 * (V + V.T)


def mash_estimate_corr_em(
    data: MashData,
    Ulist: dict[str, np.ndarray] | list[np.ndarray],
    init: np.ndarray | None = None,
    max_iter: int = 30,
    tol: float = 1.0,
    est_cor: bool = True,
    track_fit: bool = False,
    prior: str | np.ndarray = "nullbiased",
    details: bool = True,
    **kwargs,
) -> dict | np.ndarray:
    """Estimate null correlation via EM algorithm.

    Iteratively estimates the residual correlation structure using an
    EM algorithm that alternates between fitting the mash model and
    updating the correlation matrix.

    Parameters
    ----------
    data : MashData
        Data object (must not have contrast transformation).
    Ulist : dict or list of np.ndarray
        Covariance matrices for the mash model.
    init : np.ndarray, optional
        Initial correlation matrix. Defaults to
        :func:`estimate_null_correlation_simple` or identity.
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance on log-likelihood improvement.
    est_cor : bool
        If True, estimate correlation; if False, estimate covariance.
    track_fit : bool
        If True, store intermediate results in ``"trace"``.
    prior : str or np.ndarray
        Prior for the mash model at each iteration.
    details : bool
        If True, return a dict with ``"V"``, ``"mash_model"``, etc.
        If False, return only the estimated matrix.
    **kwargs
        Additional arguments passed to :func:`~pymash.mash.mash`.

    Returns
    -------
    dict or np.ndarray
        If ``details=True``: dict with keys ``"V"``, ``"mash_model"``,
        ``"loglik"``, ``"niter"``, and optionally ``"trace"``.
        If ``details=False``: the estimated correlation/covariance matrix.
    """
    if data.L is not None:
        raise ValueError("Cannot estimate null correlation for contrast-transformed mash data")

    if init is None:
        try:
            init = estimate_null_correlation_simple(data, est_cor=est_cor)
        except Exception:
            init = np.eye(data.n_conditions, dtype=float)
    init = np.asarray(init, dtype=float)

    J = data.n_effects
    m_model = _fit_mash_V(data, Ulist, V=init, prior=prior, **kwargs)

    log_liks = [float(m_model.loglik)]
    V_cur = init
    result: dict = {"V": V_cur, "mash_model": m_model}
    tracking: list[dict] = []

    niter = 0
    while niter < max_iter:
        niter += 1
        if track_fit:
            tracking.append(result.copy())

        V_new = E_V(data, m_model) / float(J)
        if est_cor:
            V_new = _cov2cor(V_new)

        m_new = _fit_mash_V(data, Ulist, V=V_new, prior=prior, **kwargs)
        delta_ll = float(m_new.loglik) - log_liks[-1]
        if delta_ll < 0:
            break

        log_liks.append(float(m_new.loglik))
        result = {"V": V_new, "mash_model": m_new}
        V_cur = V_new
        m_model = m_new
        if delta_ll <= tol:
            break

    result["V"] = V_cur
    result["mash_model"] = m_model
    result["loglik"] = np.asarray(log_liks, dtype=float)
    result["niter"] = int(niter)
    if track_fit:
        result["trace"] = tracking

    if details:
        return result
    return result["V"]


__all__ = ["estimate_null_correlation_simple", "mash_estimate_corr_em", "E_V"]
