from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from ._numerics import compute_lfsr, pnorm
from .covariances import expand_cov, normalize_Ulist
from .data import MashData, build_cov_stack
from .likelihoods import (
    RelativeLikelihoodResult,
    _prepare_u_factors,
    calc_relative_lik_matrix,
    require_cpp_backend,
)
from .optimize import optimize_pi
from .posterior import (
    PosteriorMatrices,
    compute_posterior_general_from_loglik,
    compute_posterior_matrices,
    compute_posterior_weights,
)


@dataclass
class FittedG:
    """Fitted prior distribution (learned mixture weights and covariances).

    This is the portable, reusable part of a mash fit. Pass it to
    :func:`mash` via ``g=fitted_g, fixg=True`` to apply a previously
    learned model to new data without re-estimating mixture weights.
    You can also pass a full :class:`MashResult` to
    :func:`mash_compute_posterior_matrices` and it will extract
    ``fitted_g`` automatically.

    Attributes
    ----------
    pi : np.ndarray
        Mixture proportions over (point-mass +) grid-scaled covariances.
    Ulist : list of np.ndarray
        Base covariance matrices (before grid scaling).
    grid : np.ndarray
        Grid of scaling factors applied to each covariance matrix.
    usepointmass : bool
        Whether a point-mass (null) component is included.
    """

    pi: np.ndarray
    Ulist: list[np.ndarray]
    grid: np.ndarray
    usepointmass: bool


@dataclass
class MashResult:
    """Full result object from fitting a mash model.

    The main fields for downstream analysis are ``posterior_mean``,
    ``posterior_sd``, ``lfsr``, and ``loglik``. Extract them with the
    helper functions :func:`get_pm`, :func:`get_psd`, :func:`get_lfsr`.

    To reuse the learned model on new data, pass either this object or
    its ``fitted_g`` attribute to :func:`mash` (with ``fixg=True``) or
    to :func:`mash_compute_posterior_matrices`.

    Attributes
    ----------
    posterior_mean : np.ndarray or None
        Posterior mean effects, shape ``(J, R)``.
    posterior_sd : np.ndarray or None
        Posterior standard deviations, shape ``(J, R)``.
    lfsr : np.ndarray or None
        Local false sign rates, shape ``(J, R)``.
    lfdr : np.ndarray or None
        Local false discovery rates (only if ``output_lfdr=True``).
    negative_prob : np.ndarray or None
        Posterior probability that the effect is negative.
    loglik : float
        Total log-likelihood of the fitted model.
    vloglik : np.ndarray
        Per-effect log-likelihoods, shape ``(J,)``.
    null_loglik : np.ndarray or None
        Per-effect null log-likelihoods (if ``usepointmass=True``).
    alt_loglik : np.ndarray or None
        Per-effect alternative log-likelihoods (if ``usepointmass=True``).
    fitted_g : FittedG
        The fitted prior distribution.
    posterior_weights : np.ndarray
        Posterior mixture weights, shape ``(J, K_active)``.
    alpha : float
        The alpha parameter used for effect/LFSR scaling.
    posterior_cov : np.ndarray or None
        Posterior covariance matrices (if ``outputlevel >= 3``).
    posterior_samples : np.ndarray or None
        Posterior samples, shape ``(J, R, M)`` (if ``posterior_samples > 0``).
    lik_matrix : np.ndarray or None
        Log-likelihood matrix (if ``outputlevel == 4``).
    """

    posterior_mean: np.ndarray | None
    posterior_sd: np.ndarray | None
    lfsr: np.ndarray | None
    lfdr: np.ndarray | None
    negative_prob: np.ndarray | None
    loglik: float
    vloglik: np.ndarray
    null_loglik: np.ndarray | None
    alt_loglik: np.ndarray | None
    fitted_g: FittedG
    posterior_weights: np.ndarray
    alpha: float
    posterior_cov: np.ndarray | None = None
    posterior_samples: np.ndarray | None = None
    lik_matrix: np.ndarray | None = None


def grid_min(Bhat: np.ndarray, Shat: np.ndarray) -> float:
    return float(np.min(Shat) / 10.0)


def grid_max(Bhat: np.ndarray, Shat: np.ndarray) -> float:
    if np.all(Bhat * Bhat <= Shat * Shat):
        return 8.0 * grid_min(Bhat, Shat)
    return float(2.0 * np.sqrt(np.max(Bhat * Bhat - Shat * Shat)))


def autoselect_grid(data: MashData, mult: float) -> np.ndarray:
    include = ~(np.isclose(data.Shat, 0.0) | ~np.isfinite(data.Shat) | np.isnan(data.Bhat))
    gmax = grid_max(data.Bhat[include], data.Shat[include])
    gmin = grid_min(data.Bhat[include], data.Shat[include])

    if mult == 0.0:
        return np.array([0.0, gmax / 2.0], dtype=float)

    npoint = int(np.ceil(np.log2(gmax / gmin) / np.log2(mult)))
    powers = np.arange(-npoint, 1)
    return (mult ** powers) * gmax


def set_prior(K: int, prior: str | np.ndarray, nullweight: float = 10.0) -> np.ndarray:
    if isinstance(prior, str):
        normalized = prior.lower()
        if normalized == "uniform":
            return np.ones(K, dtype=float)
        if normalized == "nullbiased":
            out = np.ones(K, dtype=float)
            out[0] = float(nullweight)
            return out
        raise ValueError("prior must be 'uniform', 'nullbiased', or numeric vector")

    arr = np.asarray(prior, dtype=float)
    if arr.shape != (K,):
        raise ValueError("prior has wrong length")
    return arr


def compute_vloglik_from_matrix_and_pi(pi_s: np.ndarray, lm: RelativeLikelihoodResult, Shat_alpha: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        log_pi = np.log(np.asarray(pi_s, dtype=float))
    return logsumexp(lm.loglik_matrix + log_pi[None, :], axis=1) + lm.lfactors - np.sum(np.log(Shat_alpha), axis=1)


def compute_null_loglik_from_matrix(lm: RelativeLikelihoodResult, Shat_alpha: np.ndarray) -> np.ndarray:
    return lm.loglik_matrix[:, 0] + lm.lfactors - np.sum(np.log(Shat_alpha), axis=1)


def compute_alt_loglik_from_matrix_and_pi(
    pi_s: np.ndarray,
    lm: RelativeLikelihoodResult,
    Shat_alpha: np.ndarray,
) -> np.ndarray:
    if pi_s[0] == 1.0:
        tmp = np.full(len(pi_s) - 1, 1.0 / (len(pi_s) - 1), dtype=float)
        mix = tmp
    else:
        mix = pi_s[1:] / (1.0 - pi_s[0])
    with np.errstate(divide="ignore"):
        log_mix = np.log(mix)
    return logsumexp(lm.loglik_matrix[:, 1:] + log_mix[None, :], axis=1) + lm.lfactors - np.sum(np.log(Shat_alpha), axis=1)


def _validate_ulist(Ulist: list[np.ndarray], R: int) -> None:
    for i, U in enumerate(Ulist):
        if U.shape != (R, R):
            raise ValueError(f"Ulist[{i}] must be of shape ({R}, {R})")
        if not np.allclose(U, U.T, atol=1e-10, rtol=0.0):
            raise ValueError(f"Ulist[{i}] must be symmetric")
        evals = np.linalg.eigvalsh(U)
        if np.min(evals) < -1e-8:
            raise ValueError(f"Ulist[{i}] must be positive semidefinite")


def _require_mash_cpp_backend() -> None:
    # Likelihood computation always goes through the C++ backend.
    require_cpp_backend(context="mash fit setup")


def _normalize_ash_mixsd(
    bhat: np.ndarray,
    shat: np.ndarray,
    mixsd: np.ndarray | None,
    gridmult: float,
    pointmass: bool,
) -> np.ndarray:
    if mixsd is None:
        b = np.asarray(bhat, dtype=float)
        s = np.asarray(shat, dtype=float)
        include = ~(np.isclose(s, 0.0) | ~np.isfinite(s) | np.isnan(b))
        if not np.any(include):
            raise ValueError("Cannot auto-select mixsd: no finite nonzero standard errors")
        gmax = grid_max(b[include], s[include])
        gmin = grid_min(b[include], s[include])
        if gridmult == 0.0:
            mixsd_base = np.array([0.0, gmax / 2.0], dtype=float)
        else:
            npoint = int(np.ceil(np.log2(gmax / gmin) / np.log2(gridmult)))
            mixsd_base = (gridmult ** np.arange(-npoint, 1)) * gmax
    else:
        mixsd_base = np.asarray(mixsd, dtype=float)
        if mixsd_base.ndim != 1 or mixsd_base.size == 0:
            raise ValueError("mixsd must be a non-empty 1D array")

    mixsd_base = mixsd_base[np.isfinite(mixsd_base)]
    if mixsd_base.size == 0:
        raise ValueError("mixsd has no finite values")
    if np.any(mixsd_base < 0.0):
        raise ValueError("mixsd must be non-negative")

    if pointmass:
        mixsd_full = np.concatenate([np.array([0.0], dtype=float), mixsd_base])
    else:
        mixsd_full = mixsd_base[mixsd_base > 0.0]
        if mixsd_full.size == 0:
            raise ValueError("mixsd must include at least one positive value when pointmass=False")

    return np.unique(mixsd_full.astype(float))


def _fit_ash_normal_1d(
    bhat: np.ndarray,
    shat: np.ndarray,
    mixsd: np.ndarray,
    prior: np.ndarray,
    optmethod: str,
    control: dict | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    b = np.asarray(bhat, dtype=float)
    s = np.asarray(shat, dtype=float)
    if b.ndim != 1 or s.ndim != 1 or b.shape != s.shape:
        raise ValueError("bhat and shat must be 1D arrays of equal length")
    if np.any(s <= 0.0) or np.any(~np.isfinite(s)):
        raise ValueError("shat must be finite and strictly positive")

    sd = np.asarray(mixsd, dtype=float)
    if sd.ndim != 1 or sd.size == 0:
        raise ValueError("mixsd must be a non-empty 1D array")
    if np.any(sd < 0.0) or np.any(~np.isfinite(sd)):
        raise ValueError("mixsd must contain finite non-negative values")

    s2 = s * s
    sd2 = sd * sd
    total_var = s2[:, None] + sd2[None, :]

    with np.errstate(all="ignore"):
        loglik_matrix = -0.5 * (np.log(2.0 * np.pi * total_var) + (b[:, None] * b[:, None]) / total_var)
    loglik_matrix = np.where(np.isfinite(loglik_matrix), loglik_matrix, -np.inf)

    lfactors = np.max(loglik_matrix, axis=1, keepdims=True)
    rel_lik = np.exp(loglik_matrix - lfactors)

    pi_s = optimize_pi(rel_lik, prior=prior, method=optmethod, control=control)
    post_w = compute_posterior_weights(pi_s, rel_lik)

    with np.errstate(all="ignore"):
        post_var_comp = (s2[:, None] * sd2[None, :]) / total_var
        post_mean_comp = b[:, None] * (sd2[None, :] / total_var)
    post_var_comp = np.where(np.isfinite(post_var_comp), post_var_comp, 0.0)
    post_mean_comp = np.where(np.isfinite(post_mean_comp), post_mean_comp, 0.0)

    post_mean = np.sum(post_w * post_mean_comp, axis=1)
    post_mean2 = np.sum(post_w * (post_mean_comp * post_mean_comp + post_var_comp), axis=1)
    post_sd = np.sqrt(np.maximum(0.0, post_mean2 - post_mean * post_mean))

    zero_mask = sd2 == 0.0
    if np.any(zero_mask):
        zero_prob = np.sum(post_w[:, zero_mask], axis=1)
    else:
        zero_prob = np.zeros_like(post_mean)

    neg_prob = np.zeros_like(post_mean)
    nonnull = ~zero_mask
    if np.any(nonnull):
        sd_nonnull = np.sqrt(np.maximum(post_var_comp[:, nonnull], np.finfo(float).tiny))
        neg_prob = np.sum(post_w[:, nonnull] * pnorm(0.0, mean=post_mean_comp[:, nonnull], sd=sd_nonnull), axis=1)
    neg_prob = np.where(np.isfinite(neg_prob), neg_prob, 0.0)

    lfsr = compute_lfsr(neg_prob, zero_prob)
    with np.errstate(divide="ignore"):
        log_pi = np.log(pi_s)
    vloglik = logsumexp(loglik_matrix + log_pi[None, :], axis=1)

    return post_mean, post_sd, neg_prob, zero_prob, lfsr, vloglik, pi_s


def mash(
    data: MashData,
    Ulist: dict[str, np.ndarray] | list[np.ndarray] | None = None,
    gridmult: float = np.sqrt(2.0),
    grid: np.ndarray | None = None,
    normalizeU: bool = True,
    usepointmass: bool = True,
    g: FittedG | None = None,
    fixg: bool = False,
    prior: str | np.ndarray = "nullbiased",
    nullweight: float = 10.0,
    optmethod: str = "slsqp",
    control: dict | None = None,
    pi_thresh: float = 1e-10,
    A: np.ndarray | None = None,
    posterior_samples: int = 0,
    seed: int = 123,
    outputlevel: int = 2,
    output_lfdr: bool = False,
) -> MashResult:
    """Fit a multivariate adaptive shrinkage (mash) model.

    This is the main entry point for fitting a mash model. It estimates
    mixture proportions over a grid of scaled covariance matrices and
    computes posterior summaries for each effect.

    Parameters
    ----------
    data : MashData
        Data object created by :func:`mash_set_data`.
    Ulist : dict or list of np.ndarray, optional
        Covariance matrices to use. Mutually exclusive with ``g``.
    gridmult : float
        Multiplicative factor for the automatically selected grid.
    grid : np.ndarray, optional
        Explicit grid of scaling factors. Overrides ``gridmult``.
    normalizeU : bool
        Whether to normalize covariance matrices before fitting.
    usepointmass : bool
        Whether to include a point-mass (null) component.
    g : FittedG, optional
        A previously fitted prior. Mutually exclusive with ``Ulist``.
    fixg : bool
        If True, use mixture proportions from ``g`` without re-estimating.
    prior : str or np.ndarray
        Prior on mixture proportions: ``"uniform"``, ``"nullbiased"``,
        or a numeric vector.
    nullweight : float
        Weight on the null component when ``prior="nullbiased"``.
    optmethod : str
        Optimization method for mixture proportions.
        ``"mixsqp"`` uses a projected Newton solver on the simplex
        (mixSQP-style). ``"squarem"`` uses accelerated EM.
        ``"slsqp"`` uses constrained SciPy SQP. ``"em"`` uses plain EM.
        ``"auto"`` picks automatically and falls back for robustness.
    control : dict, optional
        Additional options passed to the optimizer.
    pi_thresh : float
        Components with weight below this threshold are dropped.
    A : np.ndarray, optional
        Linear combination matrix for posterior computation, shape
        ``(Q, R)`` where Q is the number of output quantities. When
        provided, mash computes posteriors for ``A @ theta`` instead of
        ``theta`` itself. For example, to get posteriors on the
        difference between conditions 0 and 1 in a 3-condition model::

            A = np.array([[1, -1, 0]])  # shape (1, 3)

        This is a post-hoc transformation of the posteriors. For a full
        contrast analysis (which also transforms the *data* and
        covariances before fitting), use :func:`contrast_matrix` with
        :func:`mash_update_data` instead. Defaults to the identity
        matrix (posteriors on the original conditions).
    posterior_samples : int
        Number of posterior samples to draw (0 for none).
    seed : int
        Random seed for posterior sampling.
    outputlevel : int
        Controls which posterior summaries are computed:

        - **1**: No posteriors — only mixture weights and log-likelihood.
          Use this when fitting on a random subset to learn ``fitted_g``,
          then apply to other data with ``fixg=True``.
        - **2** (default): Posterior mean, SD, and lfsr for each effect.
          This is the standard output for final results.
        - **3**: Like 2 plus full posterior covariance matrices per effect.
          Needed for ``E_V`` in null-correlation EM.
        - **4**: Like 2 plus the raw log-likelihood matrix (J x P).

    Returns
    -------
    MashResult
        Fitted model with posterior summaries and diagnostics.

    Examples
    --------
    >>> data = mash_set_data(sim["Bhat"], sim["Shat"])
    >>> U_c = cov_canonical(data)
    >>> result = mash(data, Ulist=U_c)
    """
    _require_mash_cpp_backend()

    using_fitted_g = g is not None
    if using_fitted_g:
        if Ulist is not None:
            raise ValueError("cannot supply both g and Ulist")
        if grid is not None:
            raise ValueError("cannot supply both g and grid")
        Ulist = g.Ulist
        grid = g.grid
        usepointmass = g.usepointmass
    else:
        if Ulist is None:
            raise ValueError("must supply Ulist (or g from previous mash fit)")

    if isinstance(Ulist, dict):
        ulist_raw = [np.asarray(x, dtype=float) for x in Ulist.values()]
    else:
        ulist_raw = [np.asarray(x, dtype=float) for x in Ulist]

    if grid is None:
        grid = autoselect_grid(data, gridmult)
    else:
        grid = np.asarray(grid, dtype=float)

    if normalizeU and not using_fitted_g:
        ulist_raw = normalize_Ulist(ulist_raw)

    _validate_ulist(ulist_raw, data.n_conditions)

    xUlist = expand_cov(ulist_raw, grid, usepointmass=usepointmass)
    use_common_lik = data.common_V and data.is_common_cov_shat()
    common_cov_shat_alpha = True if data.alpha == 0 else data.is_common_cov_shat_alpha()
    is_common_posterior = data.common_V and data.is_common_cov_shat() and common_cov_shat_alpha
    cov_stack = None if use_common_lik else build_cov_stack(data)
    u_factors = u_ranks = None
    if cov_stack is not None:
        u_factors, u_ranks = _prepare_u_factors(xUlist)

    lm = calc_relative_lik_matrix(
        data,
        xUlist,
        cov_stack=cov_stack,
        u_factors=u_factors,
        u_ranks=u_ranks,
    )

    if not fixg:
        prior_vec = set_prior(lm.loglik_matrix.shape[1], prior, nullweight=nullweight)
        pi_s = optimize_pi(np.exp(lm.loglik_matrix), prior=prior_vec, method=optmethod, control=control)
    else:
        if g is None:
            raise ValueError("cannot fix g if g is not supplied")
        pi_s = np.asarray(g.pi, dtype=float)

    which_comp = pi_s > pi_thresh
    if not np.any(which_comp):
        which_comp[np.argmax(pi_s)] = True

    posterior_weights = compute_posterior_weights(pi_s[which_comp], np.exp(lm.loglik_matrix[:, which_comp]))

    posterior_matrices: PosteriorMatrices | None
    if outputlevel > 1:
        if (not is_common_posterior) and cov_stack is None:
            cov_stack = build_cov_stack(data)
            u_factors, u_ranks = _prepare_u_factors(xUlist)
        selected_idx = np.where(which_comp)[0]
        selected_u_factors = None if u_factors is None else u_factors[selected_idx]
        selected_u_ranks = None if u_ranks is None else u_ranks[selected_idx]
        selected_ulist = [xUlist[i] for i in selected_idx]
        if (not is_common_posterior) and posterior_samples == 0:
            A_use = np.eye(data.n_conditions, dtype=float) if A is None else np.asarray(A, dtype=float)
            posterior_matrices = compute_posterior_general_from_loglik(
                data,
                A_use,
                selected_ulist,
                component_loglik=lm.loglik_matrix[:, selected_idx],
                component_pi=pi_s[selected_idx],
                output_posterior_cov=(outputlevel > 2),
                cov_stack=cov_stack,
                u_factors=selected_u_factors,
                u_ranks=selected_u_ranks,
            )
        else:
            posterior_matrices = compute_posterior_matrices(
                data,
                selected_ulist,
                posterior_weights,
                A=A,
                output_posterior_cov=(outputlevel > 2),
                posterior_samples=posterior_samples,
                seed=seed,
                cov_stack=cov_stack,
                u_factors=selected_u_factors,
                u_ranks=selected_u_ranks,
            )
    else:
        posterior_matrices = None

    vloglik = compute_vloglik_from_matrix_and_pi(pi_s, lm, data.Shat_alpha)
    loglik = float(np.sum(vloglik))

    if usepointmass:
        null_loglik = compute_null_loglik_from_matrix(lm, data.Shat_alpha)
        alt_loglik = compute_alt_loglik_from_matrix_and_pi(pi_s, lm, data.Shat_alpha)
    else:
        null_loglik = None
        alt_loglik = None

    fitted_g = FittedG(pi=pi_s, Ulist=ulist_raw, grid=grid, usepointmass=usepointmass)

    if posterior_matrices is None:
        pm = psd = lfsr = lfdr = neg = pcov = psamples = None
    else:
        pm = posterior_matrices.posterior_mean
        psd = posterior_matrices.posterior_sd
        lfsr = posterior_matrices.lfsr
        lfdr = posterior_matrices.lfdr if output_lfdr else None
        neg = posterior_matrices.negative_prob
        pcov = posterior_matrices.posterior_cov
        psamples = posterior_matrices.posterior_samples

    return MashResult(
        posterior_mean=pm,
        posterior_sd=psd,
        lfsr=lfsr,
        lfdr=lfdr,
        negative_prob=neg,
        loglik=loglik,
        vloglik=vloglik,
        null_loglik=null_loglik,
        alt_loglik=alt_loglik,
        fitted_g=fitted_g,
        posterior_weights=posterior_weights,
        alpha=data.alpha,
        posterior_cov=pcov,
        posterior_samples=psamples,
        lik_matrix=lm.loglik_matrix if outputlevel == 4 else None,
    )


def mash_compute_posterior_matrices(
    g: MashResult | FittedG,
    data: MashData,
    pi_thresh: float = 1e-10,
    A: np.ndarray | None = None,
    output_posterior_cov: bool = False,
    posterior_samples: int = 0,
    seed: int = 123,
) -> PosteriorMatrices:
    """Compute posterior matrices for new data using a fitted model.

    This is useful for applying a previously fitted mash model (its learned
    mixture proportions) to a different dataset, or for requesting posterior
    samples after the initial fit.

    Parameters
    ----------
    g : MashResult or FittedG
        A fitted mash result or its ``fitted_g`` component.
    data : MashData
        Data object to compute posteriors for.
    pi_thresh : float
        Components with weight below this are excluded.
    A : np.ndarray, optional
        Linear combination matrix for posterior computation, shape
        ``(Q, R)``. Computes posteriors for ``A @ theta`` instead of
        ``theta``. Defaults to the identity matrix.
    output_posterior_cov : bool
        Whether to output posterior covariance matrices.
    posterior_samples : int
        Number of posterior samples to draw (0 for none).
    seed : int
        Random seed for posterior sampling.

    Returns
    -------
    PosteriorMatrices
        Posterior means, SDs, lfsr, and optionally covariances/samples.

    Examples
    --------
    >>> pm = mash_compute_posterior_matrices(result, new_data,
    ...                                      posterior_samples=200)
    """
    _require_mash_cpp_backend()

    if isinstance(g, MashResult):
        fitted = g.fitted_g
        if g.alpha != data.alpha:
            raise ValueError("The alpha in data does not match the one used to fit mash")
    else:
        fitted = g

    xUlist = expand_cov(fitted.Ulist, fitted.grid, fitted.usepointmass)
    use_common_lik = data.common_V and data.is_common_cov_shat()
    common_cov_shat_alpha = True if data.alpha == 0 else data.is_common_cov_shat_alpha()
    is_common_posterior = data.common_V and data.is_common_cov_shat() and common_cov_shat_alpha
    cov_stack = None if use_common_lik else build_cov_stack(data)
    u_factors = u_ranks = None
    if cov_stack is not None:
        u_factors, u_ranks = _prepare_u_factors(xUlist)

    lm = calc_relative_lik_matrix(
        data,
        xUlist,
        cov_stack=cov_stack,
        u_factors=u_factors,
        u_ranks=u_ranks,
    )

    which_comp = fitted.pi > pi_thresh
    if not np.any(which_comp):
        which_comp[np.argmax(fitted.pi)] = True

    posterior_weights = compute_posterior_weights(
        fitted.pi[which_comp],
        np.exp(lm.loglik_matrix[:, which_comp]),
    )

    if (not is_common_posterior) and cov_stack is None:
        cov_stack = build_cov_stack(data)
        u_factors, u_ranks = _prepare_u_factors(xUlist)

    selected_idx = np.where(which_comp)[0]
    selected_u_factors = None if u_factors is None else u_factors[selected_idx]
    selected_u_ranks = None if u_ranks is None else u_ranks[selected_idx]
    selected_ulist = [xUlist[i] for i in selected_idx]
    if (not is_common_posterior) and posterior_samples == 0:
        A_use = np.eye(data.n_conditions, dtype=float) if A is None else np.asarray(A, dtype=float)
        return compute_posterior_general_from_loglik(
            data,
            A_use,
            selected_ulist,
            component_loglik=lm.loglik_matrix[:, selected_idx],
            component_pi=fitted.pi[selected_idx],
            output_posterior_cov=output_posterior_cov,
            cov_stack=cov_stack,
            u_factors=selected_u_factors,
            u_ranks=selected_u_ranks,
        )

    return compute_posterior_matrices(
        data,
        selected_ulist,
        posterior_weights,
        A=A,
        output_posterior_cov=output_posterior_cov,
        posterior_samples=posterior_samples,
        seed=seed,
        cov_stack=cov_stack,
        u_factors=selected_u_factors,
        u_ranks=selected_u_ranks,
    )


def mash_1by1(
    data: MashData,
    alpha: float = 0.0,
    mixsd: np.ndarray | None = None,
    gridmult: float = np.sqrt(2.0),
    prior: str | np.ndarray = "nullbiased",
    nullweight: float = 10.0,
    optmethod: str = "slsqp",
    control: dict | None = None,
    pointmass: bool = True,
) -> MashResult:
    """Run a simple condition-by-condition (univariate) analysis.

    This provides a lightweight baseline analysis that treats each condition
    independently, without cross-condition shrinkage. It is primarily used
    to identify strong signals for data-driven covariance estimation.

    Parameters
    ----------
    data : MashData
        Data object created by :func:`mash_set_data`.
    alpha : float
        Alpha parameter used for data scaling. If this differs from
        ``data.alpha``, only the default ``alpha=0`` mismatch is
        auto-corrected by using ``data.alpha``.
    mixsd : np.ndarray, optional
        Standard-deviation grid for the univariate normal-mixture prior.
        If omitted, an automatic grid is selected.
    gridmult : float
        Multiplicative spacing for the automatic grid.
    prior : str or np.ndarray
        Prior over mixture weights (``"nullbiased"``, ``"uniform"``, or
        explicit vector).
    nullweight : float
        Null prior weight for ``prior="nullbiased"``.
    optmethod : str
        Mixture optimization method passed to :func:`optimize_pi`.
    control : dict, optional
        Optimization control dictionary passed to :func:`optimize_pi`.
    pointmass : bool
        Whether to include a point-mass-at-zero component.

    Returns
    -------
    MashResult
        Result with per-condition posterior summaries (no cross-condition
        shrinkage applied).

    Examples
    --------
    >>> m1 = mash_1by1(data)
    >>> strong = get_significant_results(m1, thresh=0.05)
    """

    alpha_use = float(alpha)
    if not np.isclose(alpha_use, data.alpha):
        if np.isclose(alpha_use, 0.0):
            pass
        else:
            raise ValueError("alpha must match data.alpha")

    J, R = data.Bhat.shape
    fixed_mixsd = None
    if mixsd is not None:
        fixed_mixsd = _normalize_ash_mixsd(
            data.Bhat[:, 0],
            data.Shat[:, 0],
            mixsd=mixsd,
            gridmult=gridmult,
            pointmass=pointmass,
        )

    post_mean_theta = np.zeros((J, R), dtype=float)
    post_sd_theta = np.zeros((J, R), dtype=float)
    neg = np.zeros((J, R), dtype=float)
    lfdr = np.zeros((J, R), dtype=float)
    lfsr = np.zeros((J, R), dtype=float)
    vloglik = np.zeros(J, dtype=float)

    for r in range(R):
        mixsd_use = fixed_mixsd
        if mixsd_use is None:
            mixsd_use = _normalize_ash_mixsd(
                data.Bhat[:, r],
                data.Shat[:, r],
                mixsd=None,
                gridmult=gridmult,
                pointmass=pointmass,
            )
        if (not pointmass) and isinstance(prior, str) and prior.lower() == "nullbiased":
            prior_use: str | np.ndarray = "uniform"
        else:
            prior_use = prior
        prior_vec = set_prior(mixsd_use.size, prior_use, nullweight=nullweight)

        pm_r, psd_r, neg_r, lfdr_r, lfsr_r, vloglik_r, _ = _fit_ash_normal_1d(
            data.Bhat[:, r],
            data.Shat[:, r],
            mixsd_use,
            prior_vec,
            optmethod,
            control,
        )
        post_mean_theta[:, r] = pm_r
        post_sd_theta[:, r] = psd_r
        neg[:, r] = neg_r
        lfdr[:, r] = lfdr_r
        lfsr[:, r] = lfsr_r
        vloglik += vloglik_r

    # Convert posterior summaries from theta-scale back to effect-size scale.
    post_mean = post_mean_theta * data.Shat_alpha
    post_sd = post_sd_theta * data.Shat_alpha

    return MashResult(
        posterior_mean=post_mean,
        posterior_sd=post_sd,
        lfsr=lfsr,
        lfdr=lfdr,
        negative_prob=neg,
        loglik=float(np.sum(vloglik)),
        vloglik=vloglik,
        null_loglik=None,
        alt_loglik=None,
        fitted_g=FittedG(pi=np.array([1.0]), Ulist=[np.eye(data.n_conditions)], grid=np.array([1.0]), usepointmass=True),
        posterior_weights=np.ones((data.n_effects, 1), dtype=float),
        alpha=data.alpha,
        posterior_cov=None,
        posterior_samples=None,
        lik_matrix=None,
    )


__all__ = [
    "FittedG",
    "MashResult",
    "grid_min",
    "grid_max",
    "autoselect_grid",
    "mash",
    "mash_compute_posterior_matrices",
    "mash_1by1",
]
