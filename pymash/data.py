from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.stats import norm, t


@dataclass
class MashData:
    """Container for mash input data, created by :func:`mash_set_data`.

    **User-facing properties** (use these for inspection):

    - :attr:`n_effects` — number of tests (J, rows of Bhat)
    - :attr:`n_conditions` — number of conditions (R, columns of Bhat)
    - :attr:`Bhat` — effect-size matrix ``(J, R)`` (possibly alpha-scaled)
    - :attr:`Shat` — standard-error matrix ``(J, R)`` (possibly alpha-scaled)

    The remaining fields (``Shat_alpha``, ``V``, ``common_V``, ``alpha``,
    ``L``, ``Shat_orig``, ``LSVSLt``) are internal bookkeeping used by
    :func:`~pymash.mash.mash` and should not normally be modified
    directly.

    pymash does not store condition labels. Track them yourself::

        trait_names = ["liver", "brain", "heart"]
        # then use trait_names when plotting or exporting results
    """

    Bhat: np.ndarray
    Shat: np.ndarray
    Shat_alpha: np.ndarray
    V: np.ndarray
    common_V: bool
    alpha: float
    L: np.ndarray | None = None
    Shat_orig: np.ndarray | None = None
    LSVSLt: np.ndarray | None = None

    @property
    def n_effects(self) -> int:
        return int(self.Bhat.shape[0])

    @property
    def n_conditions(self) -> int:
        return int(self.Bhat.shape[1])

    def get_cov(self, j: int) -> np.ndarray:
        """Return per-effect covariance matrix for effect j."""
        if j < 0 or j >= self.n_effects:
            raise IndexError("j out of bounds")

        if self.common_V:
            V = self.V
        else:
            V = self.V[j]

        if self.L is None:
            s = self.Shat[j]
            return np.outer(s, s) * V

        if self.Shat_orig is None:
            raise ValueError("Shat_orig is required when L is set")
        s = self.Shat_orig[j]
        sigma = np.outer(s, s) * V
        return self.L @ sigma @ self.L.T

    def is_common_cov_shat(self) -> bool:
        if self.L is None:
            S = self.Shat
        else:
            if self.Shat_orig is None:
                return False
            S = self.Shat_orig
        if S.shape[0] <= 1:
            return True
        return bool(np.all(np.isclose(S, S[0], equal_nan=True), axis=1).all())

    def is_common_cov_shat_alpha(self) -> bool:
        S = self.Shat_alpha
        if S.shape[0] <= 1:
            return True
        return bool(np.all(np.isclose(S, S[0], equal_nan=True), axis=1).all())


def _as_2d_float_array(x: np.ndarray | float, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        if shape is None:
            raise ValueError(f"{name} scalar requires target shape")
        arr = np.full(shape, float(arr), dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    return arr


def _check_positive_definite(x: np.ndarray, name: str) -> None:
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(x, x.T, atol=1e-10, rtol=0.0):
        raise ValueError(f"{name} must be symmetric")
    try:
        np.linalg.cholesky(x)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc


def regularize_cov(V: np.ndarray, ridge: float = 1e-6, to_correlation: bool = False) -> np.ndarray:
    """Add diagonal ridge regularization to covariance/correlation matrices."""
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    arr = np.asarray(V, dtype=float)
    if arr.ndim == 2:
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("V must be a square matrix")
        out = arr + ridge * np.eye(arr.shape[0], dtype=float)
        out = 0.5 * (out + out.T)
        if to_correlation:
            d = np.sqrt(np.maximum(np.diag(out), np.finfo(float).tiny))
            out = out / np.outer(d, d)
            np.fill_diagonal(out, 1.0)
        return out

    if arr.ndim == 3:
        if arr.shape[1] != arr.shape[2]:
            raise ValueError("V must have shape (J, R, R)")
        out = np.array(arr, copy=True)
        eye = np.eye(arr.shape[1], dtype=float)
        for j in range(arr.shape[0]):
            out[j] = arr[j] + ridge * eye
            out[j] = 0.5 * (out[j] + out[j].T)
            if to_correlation:
                d = np.sqrt(np.maximum(np.diag(out[j]), np.finfo(float).tiny))
                out[j] = out[j] / np.outer(d, d)
                np.fill_diagonal(out[j], 1.0)
        return out

    raise ValueError("V must be either 2D or 3D")


def check_mash_data(
    Bhat: np.ndarray,
    Shat: np.ndarray | float,
    *,
    near_zero_tol: float = 1e-6,
    se_cv_thresh: float = 1.0,
    corr_cond_thresh: float = 1e4,
    verbose: bool = True,
) -> dict[str, str]:
    """Diagnose common data-quality issues before fitting mash."""
    bhat = _as_2d_float_array(Bhat, "Bhat")
    shat = _as_2d_float_array(Shat, "Shat", shape=bhat.shape)
    if bhat.shape != shat.shape:
        raise ValueError("dimensions of Bhat and Shat must match")

    issues: dict[str, str] = {}

    finite_shat = shat[np.isfinite(shat)]
    if finite_shat.size == 0:
        issues["invalid_shat"] = "Shat has no finite entries."
    else:
        n_nonpos = int(np.sum(finite_shat <= 0.0))
        if n_nonpos > 0:
            issues["nonpositive_se"] = (
                f"Shat contains {n_nonpos} non-positive values. "
                "Standard errors must be strictly positive."
            )

        positive_shat = finite_shat[finite_shat > 0.0]
        if positive_shat.size > 1:
            mean_se = float(np.mean(positive_shat))
            cv_se = float(np.std(positive_shat) / max(mean_se, np.finfo(float).tiny))
            if cv_se > se_cv_thresh:
                issues["high_se_variability"] = (
                    f"SE coefficient of variation is {cv_se:.2f}. "
                    "Consider alpha=1 (z-score scale), especially for GWAS/eQTL."
                )

    near_zero = int(np.sum(np.isfinite(shat) & (shat <= near_zero_tol)))
    if near_zero > 0:
        issues["near_zero_se"] = (
            f"{near_zero} entries have Shat <= {near_zero_tol:g}. "
            "Set zero_Shat_reset (or zero_Bhat_Shat_reset) before fitting."
        )

    finite_mask = np.isfinite(bhat) & np.isfinite(shat) & (np.abs(shat) > near_zero_tol)
    if np.any(finite_mask) and bhat.shape[0] > 1 and bhat.shape[1] > 1:
        z = np.zeros_like(bhat, dtype=float)
        z[finite_mask] = bhat[finite_mask] / shat[finite_mask]
        col_sd = np.std(z, axis=0)
        valid_cols = col_sd > np.finfo(float).tiny
        if int(np.sum(valid_cols)) >= 2:
            corr = np.corrcoef(z[:, valid_cols], rowvar=False)
            if np.all(np.isfinite(corr)):
                cond = float(np.linalg.cond(corr))
                if cond > corr_cond_thresh:
                    issues["ill_conditioned_traits"] = (
                        f"Trait z-score correlation condition number is {cond:.1f}. "
                        "Traits are highly collinear; consider covariance regularization."
                    )

    if verbose:
        for msg in issues.values():
            print(f"[check_mash_data] {msg}")

    return issues


def p2z(pval: np.ndarray, bhat: np.ndarray) -> np.ndarray:
    z = np.abs(norm.ppf(pval / 2.0))
    return np.where(bhat < 0, -z, z)


def mash_set_data(
    Bhat: np.ndarray,
    Shat: np.ndarray | float | None = None,
    alpha: float = 0.0,
    df: np.ndarray | float = np.inf,
    pval: np.ndarray | None = None,
    V: np.ndarray | None = None,
    v_ridge: float = 0.0,
    zero_check_tol: float = np.finfo(float).eps,
    zero_Bhat_Shat_reset: float = 0.0,
    zero_Shat_reset: float = 0.0,
) -> MashData:
    """Create and validate a MashData object for use with mash.

    Parameters
    ----------
    Bhat : np.ndarray
        Matrix of observed effect sizes, shape ``(J, R)``.
    Shat : np.ndarray or float, optional
        Matrix of standard errors, shape ``(J, R)``, or a scalar.
        Defaults to 1 if neither ``Shat`` nor ``pval`` is given.
    alpha : float
        Controls how effects are standardized before modeling:

        - ``alpha=0`` (default): model the raw effect sizes. Appropriate
          when standard errors are roughly constant across markers (e.g.,
          balanced designed experiments, RNA-seq with equal sample sizes).
        - ``alpha=1``: model z-scores (Bhat/Shat). Appropriate when
          standard errors vary across markers, which is the common case
          for **GWAS** and **eQTL** studies (where SE depends on allele
          frequency and sample size).

        **Rule of thumb:** if your Shat values span more than a 2-fold
        range across markers, use ``alpha=1`` or equivalently convert to
        z-scores yourself (``Zhat = Bhat / Shat``, ``Shat = ones``).
    df : np.ndarray or float
        Degrees of freedom for t-distribution conversion. Use ``np.inf``
        (default) for normally distributed effects.
    pval : np.ndarray, optional
        P-value matrix. If provided, standard errors are derived from
        p-values and effect sizes. Mutually exclusive with ``Shat``.
    V : np.ndarray, optional
        Correlation matrix among conditions, shape ``(R, R)``, or
        per-effect correlation matrices, shape ``(J, R, R)``.
        Defaults to identity.
    v_ridge : float
        Optional diagonal ridge added to ``V`` before validation. Use
        this when ``V`` is nearly singular.
    zero_check_tol : float
        Tolerance for checking near-zero standard errors.
    zero_Bhat_Shat_reset : float
        If > 0, replace entries where both Bhat and Shat are near zero
        with this Shat value.
    zero_Shat_reset : float
        If > 0, replace near-zero Shat entries with this value.

    Returns
    -------
    MashData
        Validated data object ready for :func:`~pymash.mash.mash`.

    Examples
    --------
    >>> data = mash_set_data(Bhat=Bhat, Shat=Shat)
    >>> data.n_effects, data.n_conditions
    (2000, 5)
    """
    bhat = _as_2d_float_array(Bhat, "Bhat")
    J, R = bhat.shape

    if Shat is None and pval is None:
        Shat = 1.0
    if pval is not None and Shat is not None:
        raise ValueError("Either Shat or pval can be specified but not both")
    if pval is not None and not np.isinf(df).all():
        raise ValueError("Either df or pval can be specified but not both")

    if pval is not None:
        pval_arr = _as_2d_float_array(pval, "pval", shape=(J, R))
        if np.any(np.isfinite(pval_arr) & (pval_arr <= 0.0)):
            raise ValueError("p-values cannot contain zero or negative values")
        shat = bhat / p2z(pval_arr, bhat)
    else:
        shat = _as_2d_float_array(Shat, "Shat", shape=(J, R))

    if shat.shape != bhat.shape:
        raise ValueError("dimensions of Bhat and Shat must match")
    if np.any(np.isinf(bhat)):
        raise ValueError("Bhat cannot contain Inf values")
    if np.any(np.isinf(shat)):
        raise ValueError("Shat cannot contain Inf values")
    if np.any(shat < -zero_check_tol):
        raise ValueError(
            "Shat contains negative values. Standard errors must be positive; "
            "check that Bhat and Shat were not swapped."
        )

    near_zero = shat <= zero_check_tol
    if np.any(near_zero):
        both_zero = near_zero & (np.abs(bhat) <= zero_check_tol)
        if np.any(both_zero):
            if zero_Bhat_Shat_reset > 0:
                shat[both_zero] = zero_Bhat_Shat_reset
            else:
                raise ValueError(
                    "Both Bhat and Shat are zero (or near zero) for some entries; "
                    "set zero_Bhat_Shat_reset to replace them."
                )
        else:
            if zero_Shat_reset > 0:
                shat[near_zero] = zero_Shat_reset
            else:
                raise ValueError("Shat contains zero (or near zero) values; set zero_Shat_reset.")

    if V is None:
        vmat = np.eye(R, dtype=float)
    else:
        vmat = np.asarray(V, dtype=float)
        if v_ridge > 0:
            vmat = regularize_cov(vmat, ridge=v_ridge, to_correlation=False)

    if vmat.ndim == 2:
        if vmat.shape != (R, R):
            raise ValueError("dimension of correlation matrix does not match the number of conditions")
        _check_positive_definite(vmat, "V")
        common_V = True
        v_store = vmat
    elif vmat.ndim == 3:
        if vmat.shape == (R, R, J):
            vmat = np.transpose(vmat, (2, 0, 1))
        if vmat.shape != (J, R, R):
            raise ValueError("V must have shape (J, R, R) or (R, R, J) for effect-specific covariances")
        for j in range(J):
            _check_positive_definite(vmat[j], f"V[{j}]")
        common_V = False
        v_store = vmat
    else:
        raise ValueError("V must be either 2D or 3D")

    df_arr = np.asarray(df, dtype=float)
    if not np.all(np.isinf(df_arr)):
        if df_arr.ndim == 0:
            df_arr = np.full((J, R), float(df_arr), dtype=float)
        elif df_arr.shape != (J, R):
            raise ValueError("df must be scalar or match Bhat shape")
        pvals = 2.0 * t.cdf(-np.abs(bhat / shat), df=df_arr)
        shat = bhat / p2z(pvals, bhat)

    na_bhat = np.isnan(bhat)
    na_shat = np.isnan(shat)
    sbhat_not_null = not np.allclose(shat, 1.0, equal_nan=True)
    if sbhat_not_null and not np.array_equal(na_bhat, na_shat):
        raise ValueError("Missing data pattern is inconsistent between Bhat and Shat")

    if alpha != 0 and sbhat_not_null:
        shat_alpha = shat**alpha
        bhat = bhat / shat_alpha
        shat = shat ** (1.0 - alpha)
    else:
        shat_alpha = np.ones_like(shat)

    bhat[na_bhat] = 0.0
    shat[na_bhat] = 1e6
    shat_alpha[na_bhat] = 1.0

    positive_shat = shat[np.isfinite(shat) & (shat > 0.0)]
    if alpha == 0 and positive_shat.size > 1:
        mean_se = float(np.mean(positive_shat))
        cv_se = float(np.std(positive_shat) / max(mean_se, np.finfo(float).tiny))
        fold = float(np.max(positive_shat) / max(np.min(positive_shat), np.finfo(float).tiny))
        if cv_se > 1.0:
            warnings.warn(
                f"Standard errors vary widely (CV={cv_se:.2f}, max/min={fold:.2f}). "
                "Consider alpha=1 (z-score scale), especially for GWAS/eQTL.",
                RuntimeWarning,
                stacklevel=2,
            )

    return MashData(
        Bhat=bhat,
        Shat=shat,
        Shat_alpha=shat_alpha,
        V=v_store,
        common_V=common_V,
        alpha=float(alpha),
    )


def _cov2cor(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(d, d)
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, 1.0)
    return corr


def contrast_matrix(R: int, ref: int | str, names: list[str] | tuple[str, ...] | None = None) -> np.ndarray:
    """Create a contrast matrix for comparing conditions to a reference.

    Parameters
    ----------
    R : int
        Number of conditions.
    ref : int or str
        Reference condition: a 0-based integer index, a condition name,
        or ``"mean"`` for deviation-from-mean contrasts.
    names : list of str, optional
        Condition names (length ``R``). Defaults to ``["1", "2", ...]``.

    Returns
    -------
    np.ndarray
        Contrast matrix of shape ``(R-1, R)``.

    Examples
    --------
    >>> L = contrast_matrix(5, ref=0)
    >>> L.shape
    (4, 5)
    """
    if names is None:
        names = [str(i + 1) for i in range(R)]
    if len(names) != R:
        raise ValueError("names must have length R")

    if ref == "mean":
        L = np.full((R, R), -1.0 / R)
        np.fill_diagonal(L, (R - 1.0) / R)
        return L[:-1]

    if isinstance(ref, int):
        if ref < 0 or ref >= R:
            raise ValueError("ref must be between 0 and R-1")
        idx = ref
    else:
        if ref not in names:
            raise ValueError("The ref group is not in the given conditions")
        idx = names.index(ref)

    L = np.eye(R, dtype=float)
    L[:, idx] = -1.0
    return np.delete(L, idx, axis=0)


def mash_set_data_contrast(data: MashData, L: np.ndarray) -> MashData:
    """Internal helper to transform mash data into contrast space."""
    L = np.asarray(L, dtype=float)
    if L.ndim != 2 or L.shape[1] != data.n_conditions:
        raise ValueError("The contrast matrix has invalid dimensions")

    bhat = data.Bhat @ L.T

    if data.common_V and data.is_common_cov_shat():
        cov0 = data.get_cov(0)
        shat = np.tile(np.sqrt(np.maximum(np.diag(cov0), 0.0)), (data.n_effects, 1))
        lsvslt = _cov2cor(cov0)
    else:
        q = L.shape[0]
        shat = np.zeros((data.n_effects, q), dtype=float)
        lsvslt = np.zeros((data.n_effects, q, q), dtype=float)
        for j in range(data.n_effects):
            cov_j = data.get_cov(j)
            shat[j] = np.sqrt(np.maximum(np.diag(cov_j), 0.0))
            lsvslt[j] = _cov2cor(cov_j)

    return MashData(
        Bhat=bhat,
        Shat=shat,
        Shat_alpha=np.ones_like(shat),
        V=data.V,
        common_V=data.common_V,
        alpha=0.0,
        L=L,
        Shat_orig=data.Shat,
        LSVSLt=lsvslt,
    )


def build_cov_stack(data: MashData) -> np.ndarray:
    """Build per-effect covariance stack with shape (J, R, R)."""
    J = data.n_effects
    R = data.n_conditions
    cov_stack = np.empty((J, R, R), dtype=float)
    for j in range(J):
        cov_stack[j] = data.get_cov(j)
    return cov_stack


def mash_update_data(
    mashdata: MashData,
    ref: int | str | None = None,
    V: np.ndarray | None = None,
    v_ridge: float = 0.0,
) -> MashData:
    """Update an existing MashData object with a new correlation structure.

    Creates a copy of ``mashdata`` with an updated correlation matrix ``V``
    and/or a contrast transformation applied via ``ref``.

    Parameters
    ----------
    mashdata : MashData
        Original data object.
    ref : int or str, optional
        Reference condition for contrast analysis. An integer (0-based)
        or ``"mean"`` for deviation-from-mean contrasts.
    V : np.ndarray, optional
        New correlation matrix, shape ``(R, R)`` or ``(J, R, R)``.
    v_ridge : float
        Optional diagonal ridge added to ``V`` before validation.

    Returns
    -------
    MashData
        Updated data object (the original is not modified).

    Examples
    --------
    >>> Vhat = estimate_null_correlation_simple(data)
    >>> data_updated = mash_update_data(data, V=Vhat)
    """
    data = MashData(
        Bhat=np.array(mashdata.Bhat, copy=True),
        Shat=np.array(mashdata.Shat, copy=True),
        Shat_alpha=np.array(mashdata.Shat_alpha, copy=True),
        V=np.array(mashdata.V, copy=True),
        common_V=mashdata.common_V,
        alpha=mashdata.alpha,
        L=None if mashdata.L is None else np.array(mashdata.L, copy=True),
        Shat_orig=None if mashdata.Shat_orig is None else np.array(mashdata.Shat_orig, copy=True),
        LSVSLt=None if mashdata.LSVSLt is None else np.array(mashdata.LSVSLt, copy=True),
    )

    if V is not None:
        vmat = np.asarray(V, dtype=float)
        if v_ridge > 0:
            vmat = regularize_cov(vmat, ridge=v_ridge, to_correlation=False)
        R = data.n_conditions
        if vmat.ndim == 2:
            if vmat.shape != (R, R):
                raise ValueError("The dimension of correlation matrix does not match the data")
            _check_positive_definite(vmat, "V")
            data.V = vmat
            data.common_V = True
        elif vmat.ndim == 3:
            if vmat.shape == (R, R, data.n_effects):
                vmat = np.transpose(vmat, (2, 0, 1))
            if vmat.shape != (data.n_effects, R, R):
                raise ValueError("V must have shape (J, R, R)")
            for j in range(vmat.shape[0]):
                _check_positive_definite(vmat[j], f"V[{j}]")
            data.V = vmat
            data.common_V = False
        else:
            raise ValueError("V must be 2D or 3D")

    if data.L is not None:
        data = mash_set_data_contrast(data, data.L)

    if ref is not None:
        if data.L is not None:
            raise ValueError("The data is already configured for contrast analysis")
        L = contrast_matrix(data.n_conditions, ref)
        data = mash_set_data_contrast(data, L)

    return data


__all__ = [
    "MashData",
    "p2z",
    "check_mash_data",
    "regularize_cov",
    "mash_set_data",
    "mash_update_data",
    "contrast_matrix",
    "mash_set_data_contrast",
    "build_cov_stack",
]
