from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from ._numerics import mvn_logpdf_batch
from .data import MashData

try:
    from . import _edcpp
except Exception:  # pragma: no cover
    _edcpp = None


@dataclass
class TeemResult:
    w: np.ndarray
    U: list[np.ndarray]
    objective: np.ndarray
    maxd: np.ndarray


def _as_ulist(Ulist_init: dict[str, np.ndarray] | list[np.ndarray]) -> tuple[list[str], list[np.ndarray]]:
    if isinstance(Ulist_init, dict):
        names = list(Ulist_init.keys())
        mats = [np.asarray(x, dtype=float) for x in Ulist_init.values()]
    else:
        mats = [np.asarray(x, dtype=float) for x in Ulist_init]
        names = [str(i + 1) for i in range(len(mats))]
    if not mats:
        raise ValueError("Ulist_init cannot be empty")
    return names, mats


def _softmax_rows(logP: np.ndarray) -> np.ndarray:
    return np.exp(logP - logsumexp(logP, axis=1, keepdims=True))


def shrink_cov(V: np.ndarray, eps: float) -> np.ndarray:
    eigval, eigvec = np.linalg.eigh(V)
    eigval = np.where(eigval > 1.0, eigval, 1.0 + eps)
    return eigvec @ np.diag(eigval) @ eigvec.T


def _compute_loglik(X: np.ndarray, w: np.ndarray, T: list[np.ndarray]) -> float:
    comp = []
    for j, cov in enumerate(T):
        ll = mvn_logpdf_batch(X, np.zeros(X.shape[1], dtype=float), cov)
        comp.append(np.log(np.maximum(w[j], np.finfo(float).tiny)) + ll)
    stacked = np.column_stack(comp)
    return float(np.sum(logsumexp(stacked, axis=1)))


def _normalize_ycovar(ycovar: np.ndarray | list[np.ndarray], n: int, d: int) -> np.ndarray:
    if isinstance(ycovar, list):
        arr = np.stack([np.asarray(s, dtype=float) for s in ycovar], axis=0)
    else:
        arr = np.asarray(ycovar, dtype=float)

    if arr.ndim == 2:
        if arr.shape == (n, d):
            out = np.zeros((n, d, d), dtype=float)
            idx = np.arange(d)
            out[:, idx, idx] = arr
            return out
        if arr.shape == (d, d):
            return np.repeat(arr[None, :, :], repeats=n, axis=0)
        raise ValueError("2D ycovar must have shape (n, d) or (d, d)")

    if arr.ndim == 3:
        if arr.shape == (n, d, d):
            return arr
        if arr.shape == (d, d, n):
            return np.transpose(arr, (2, 0, 1))
        raise ValueError("3D ycovar must have shape (n, d, d) or (d, d, n)")

    raise ValueError("ycovar must be 2D, 3D, or list of matrices")


def extreme_deconvolution(
    ydata: np.ndarray,
    ycovar: np.ndarray | list[np.ndarray],
    xamp: np.ndarray,
    xmean: np.ndarray,
    xcovar: np.ndarray | list[np.ndarray],
    *,
    fixmean: bool | np.ndarray = True,
    tol: float = 1e-6,
    maxiter: int = 500,
    w: float = 0.0,
    verbose: bool = False,
    **_ignored,
) -> dict:
    """Internal ED solver specialized for mashr use cases.

    This implementation currently supports the common mashr setup of fixed-zero
    component means.
    """
    if _edcpp is None:
        raise ImportError(
            "pymash._edcpp is not available. Reinstall with build dependencies to enable bovy ED."
        )

    y = np.asarray(ydata, dtype=float)
    if y.ndim != 2:
        raise ValueError("ydata must be 2D")
    n, d = y.shape

    means = np.asarray(xmean, dtype=float)
    if means.shape != (len(np.asarray(xamp).ravel()), d):
        raise ValueError("xmean must have shape (K, d)")

    # Current C++ implementation fixes means at zero, matching mashr::bovy_wrapper.
    if isinstance(fixmean, (bool, np.bool_)):
        fixed = bool(fixmean)
    else:
        fixed = bool(np.all(np.asarray(fixmean).astype(bool)))
    if not fixed or not np.allclose(means, 0.0):
        raise NotImplementedError("Only fixed zero-mean components are currently supported")

    if isinstance(xcovar, list):
        cov_arr = np.stack([np.asarray(c, dtype=float) for c in xcovar], axis=0)
    else:
        cov_arr = np.asarray(xcovar, dtype=float)
    if cov_arr.ndim != 3:
        raise ValueError("xcovar must be 3D or list of matrices")

    amp = np.asarray(xamp, dtype=float).ravel()
    if cov_arr.shape[0] != amp.shape[0] or cov_arr.shape[1] != d or cov_arr.shape[2] != d:
        raise ValueError("xcovar must have shape (K, d, d) consistent with xamp and ydata")

    ycov = _normalize_ycovar(ycovar, n=n, d=d)

    out = _edcpp.fit_extreme_deconvolution(
        y,
        ycov,
        amp,
        cov_arr,
        maxiter=int(maxiter),
        tol=float(tol),
        w=float(w),
        verbose=bool(verbose),
    )

    xcov_out = np.asarray(out["xcovar"], dtype=float)
    return {
        "xamp": np.asarray(out["xamp"], dtype=float),
        "xmean": np.zeros_like(means),
        "xcovar": [xcov_out[k] for k in range(xcov_out.shape[0])],
        "avgloglikedata": float(out["avgloglikedata"]),
        "objective": np.asarray(out["objective"], dtype=float),
        "status": int(out["status"]),
    }


def bovy_wrapper(
    data: MashData,
    Ulist_init: dict[str, np.ndarray] | list[np.ndarray],
    subset: np.ndarray | list[int] | None = None,
    **kwargs,
) -> dict:
    names, U0 = _as_ulist(Ulist_init)

    if subset is None:
        subset_idx = np.arange(data.n_effects)
    else:
        subset_idx = np.asarray(subset, dtype=int)

    K = len(U0)
    R = data.n_conditions
    pi_init = np.full(K, 1.0 / K, dtype=float)

    ydata = data.Bhat[subset_idx]
    ycovar = np.stack([data.get_cov(int(i)) for i in subset_idx], axis=0)

    ed_res = extreme_deconvolution(
        ydata,
        ycovar,
        xamp=pi_init,
        xmean=np.zeros((K, R), dtype=float),
        xcovar=U0,
        fixmean=True,
        **kwargs,
    )

    epsilon = np.eye(R, dtype=float) / np.sqrt(float(len(subset_idx)))
    Ulist = [U + epsilon for U in ed_res["xcovar"]]
    return {
        "pi": np.asarray(ed_res["xamp"], dtype=float),
        "Ulist": Ulist,
        "av_loglik": float(ed_res["avgloglikedata"]),
        "objective": np.asarray(ed_res["objective"], dtype=float),
        "names": names,
    }


def teem_wrapper(
    data: MashData,
    Ulist_init: dict[str, np.ndarray] | list[np.ndarray],
    subset: np.ndarray | list[int] | None = None,
    w_init: np.ndarray | None = None,
    max_iter: int = 5000,
    converge_tol: float = 1e-7,
    eigen_tol: float = 1e-7,
    verbose: bool = False,
) -> TeemResult:
    """Fit TEEM (truncated-eigenvalue EM) on z-scores."""
    _, U_init = _as_ulist(Ulist_init)

    if subset is None:
        subset_idx = np.arange(data.n_effects)
    else:
        subset_idx = np.asarray(subset, dtype=int)

    X = data.Bhat[subset_idx] / data.Shat[subset_idx]
    n, R = X.shape
    k = len(U_init)

    for i, U in enumerate(U_init):
        if U.shape != (R, R):
            raise ValueError(f"Ulist_init[{i}] must have shape ({R}, {R})")

    if w_init is None:
        w = np.full(k, 1.0 / k, dtype=float)
    else:
        w = np.asarray(w_init, dtype=float)
        if w.shape != (k,):
            raise ValueError("w_init has wrong length")
        if np.any(w < 0):
            raise ValueError("w_init cannot contain negatives")
        if np.sum(w) == 0:
            raise ValueError("w_init must have positive sum")
        w = w / np.sum(w)

    T = [U + np.eye(R, dtype=float) for U in U_init]

    objectives: list[float] = []
    maxd_list: list[float] = []

    for it in range(max_iter):
        w0 = w.copy()

        logP = np.empty((n, k), dtype=float)
        for j in range(k):
            ll = mvn_logpdf_batch(X, np.zeros(R, dtype=float), T[j])
            logP[:, j] = np.log(np.maximum(w[j], np.finfo(float).tiny)) + ll

        P = _softmax_rows(logP)

        XT = X.T
        for j in range(k):
            pj = P[:, j]
            denom = float(np.sum(pj))
            if denom <= np.finfo(float).tiny:
                continue
            cov = XT @ (pj[:, None] * X) / denom
            T[j] = shrink_cov(cov, eigen_tol)

        w = np.mean(P, axis=0)

        f = _compute_loglik(X, w, T)
        d = float(np.max(np.abs(w - w0)))
        objectives.append(f)
        maxd_list.append(d)

        if verbose and (it % 50 == 0 or d < converge_tol):
            print(f"[TEEM] iter={it} objective={f:.6f} maxd={d:.3e}")

        if d < converge_tol:
            break

    U_out = [cov - np.eye(R, dtype=float) for cov in T]
    return TeemResult(
        w=w,
        U=U_out,
        objective=np.array(objectives, dtype=float),
        maxd=np.array(maxd_list, dtype=float),
    )


__all__ = [
    "TeemResult",
    "shrink_cov",
    "teem_wrapper",
    "bovy_wrapper",
    "extreme_deconvolution",
]
