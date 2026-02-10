from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._numerics import mvn_logpdf_batch
from .data import MashData, build_cov_stack

try:
    from . import _edcpp as _edcpp_backend
except Exception:  # pragma: no cover - import failure fallback
    _edcpp_backend = None


@dataclass
class RelativeLikelihoodResult:
    loglik_matrix: np.ndarray
    lfactors: np.ndarray


def _use_cpp_general_lik() -> bool:
    return _edcpp_backend is not None and hasattr(_edcpp_backend, "calc_lik_matrix_general")


def require_cpp_backend(context: str = "likelihood") -> None:
    if _use_cpp_general_lik():
        return
    raise RuntimeError(
        "pymash C++ backend is required for mash fitting but is not available. "
        f"Missing backend detected while preparing {context}. "
        "Reinstall with extension build support (for example: pip install -e .)."
    )


def calc_lik_vector(bhat: np.ndarray, V: np.ndarray, Ulist: list[np.ndarray], log: bool = False) -> np.ndarray:
    bhat = np.asarray(bhat, dtype=float)
    out = np.empty(len(Ulist), dtype=float)
    X = bhat[None, :]
    for i, U in enumerate(Ulist):
        out[i] = mvn_logpdf_batch(X, np.zeros_like(bhat), U + V)[0]
    if log:
        return out
    return np.exp(out)


def calc_lik_matrix_common_cov(data: MashData, Ulist: list[np.ndarray], log: bool = False) -> np.ndarray:
    # Keep API compatibility while delegating execution to the shared C++ path.
    return _calc_lik_matrix_general_cpp(data, Ulist, log=log, cov_stack=np.asarray(data.get_cov(0), dtype=float))


def _prepare_u_factors(
    Ulist: list[np.ndarray],
    rank_tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
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


def _calc_lik_matrix_general_cpp(
    data: MashData,
    Ulist: list[np.ndarray],
    log: bool = False,
    cov_stack: np.ndarray | None = None,
    u_factors: np.ndarray | None = None,
    u_ranks: np.ndarray | None = None,
) -> np.ndarray:
    require_cpp_backend(context="likelihood computation")
    if cov_stack is None:
        if data.common_V and data.is_common_cov_shat():
            cov_stack = np.asarray(data.get_cov(0), dtype=float)
        else:
            cov_stack = build_cov_stack(data)
    else:
        cov_stack = np.asarray(cov_stack, dtype=float)
        expected3 = (data.n_effects, data.n_conditions, data.n_conditions)
        expected2 = (data.n_conditions, data.n_conditions)
        if cov_stack.shape != expected3 and cov_stack.shape != expected2:
            raise ValueError(f"cov_stack must have shape {expected3} or {expected2}")
    u_stack = np.asarray(Ulist, dtype=float)
    if u_factors is None or u_ranks is None:
        factors, ranks = _prepare_u_factors(Ulist)
    else:
        factors = np.asarray(u_factors, dtype=float)
        ranks = np.asarray(u_ranks, dtype=np.int32)
    matrix_llik = _edcpp_backend.calc_lik_matrix_general(data.Bhat, cov_stack, u_stack, factors, ranks)
    if log:
        return matrix_llik
    return np.exp(matrix_llik)


def calc_lik_matrix(
    data: MashData,
    Ulist: list[np.ndarray],
    log: bool = False,
    cov_stack: np.ndarray | None = None,
    u_factors: np.ndarray | None = None,
    u_ranks: np.ndarray | None = None,
) -> np.ndarray:
    """Compute JxP matrix of component likelihoods p(bhat_j | U_p, V_j)."""
    res = _calc_lik_matrix_general_cpp(
        data,
        Ulist,
        log=log,
        cov_stack=cov_stack,
        u_factors=u_factors,
        u_ranks=u_ranks,
    )

    if np.any(~np.isfinite(res)):
        cols = np.where(np.any(~np.isfinite(res), axis=0))[0]
        if cols.size > 0:
            import warnings

            warnings.warn(
                "Some mixture components produced non-finite likelihoods; "
                f"columns: {', '.join(map(str, cols.tolist()))}",
                RuntimeWarning,
                stacklevel=2,
            )

    return res


def calc_relative_lik_matrix(
    data: MashData,
    Ulist: list[np.ndarray],
    cov_stack: np.ndarray | None = None,
    u_factors: np.ndarray | None = None,
    u_ranks: np.ndarray | None = None,
) -> RelativeLikelihoodResult:
    matrix_llik = calc_lik_matrix(
        data,
        Ulist,
        log=True,
        cov_stack=cov_stack,
        u_factors=u_factors,
        u_ranks=u_ranks,
    )
    lfactors = np.max(matrix_llik, axis=1)
    matrix_llik = matrix_llik - lfactors[:, None]
    return RelativeLikelihoodResult(loglik_matrix=matrix_llik, lfactors=lfactors)


__all__ = [
    "RelativeLikelihoodResult",
    "require_cpp_backend",
    "calc_lik_vector",
    "calc_lik_matrix_common_cov",
    "calc_lik_matrix",
    "calc_relative_lik_matrix",
]
