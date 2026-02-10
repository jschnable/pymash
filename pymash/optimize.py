from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def initialize_pi(K: int) -> np.ndarray:
    if K <= 0:
        raise ValueError("K must be positive")
    return np.full(K, 1.0 / K, dtype=float)


def _sanitize_pi(pi: np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(pi, dtype=float)
    x = np.where(np.isfinite(x), x, eps)
    x = np.maximum(x, eps)
    s = np.sum(x)
    if not np.isfinite(s) or s <= 0:
        return np.full_like(x, 1.0 / x.size)
    return x / s


def _objective_value(
    matrix_lik: np.ndarray,
    pi: np.ndarray,
    prior_alpha: np.ndarray,
    eps: float,
) -> float:
    p = _sanitize_pi(pi, eps)
    with np.errstate(all="ignore"):
        denom = matrix_lik @ p
    denom = np.where(np.isfinite(denom), denom, eps)
    denom = np.maximum(denom, eps)
    value = -np.sum(np.log(denom))
    if np.any(prior_alpha != 0.0):
        value -= np.dot(prior_alpha, np.log(np.maximum(p, eps)))
    return float(value)


def _objective_grad(
    matrix_lik: np.ndarray,
    pi: np.ndarray,
    prior_alpha: np.ndarray,
    eps: float,
) -> np.ndarray:
    p = _sanitize_pi(pi, eps)
    with np.errstate(all="ignore"):
        denom = matrix_lik @ p
    denom = np.where(np.isfinite(denom), denom, eps)
    denom = np.maximum(denom, eps)
    with np.errstate(all="ignore"):
        frac = matrix_lik / denom[:, None]
    frac = np.where(np.isfinite(frac), frac, 0.0)
    grad = -np.sum(frac, axis=0)
    if np.any(prior_alpha != 0.0):
        grad -= prior_alpha / np.maximum(p, eps)
    return grad


def _objective_hessian(
    matrix_lik: np.ndarray,
    pi: np.ndarray,
    prior_alpha: np.ndarray,
    eps: float,
) -> np.ndarray:
    p = _sanitize_pi(pi, eps)
    with np.errstate(all="ignore"):
        denom = matrix_lik @ p
    denom = np.where(np.isfinite(denom), denom, eps)
    denom = np.maximum(denom, eps)
    w = 1.0 / (denom * denom)
    H = matrix_lik.T @ (matrix_lik * w[:, None])
    if np.any(prior_alpha != 0.0):
        H = H + np.diag(prior_alpha / np.maximum(p, eps) ** 2)
    return H


def _prescreen_components(
    matrix_lik: np.ndarray,
    prior: np.ndarray,
    pi_init: np.ndarray,
    log_tol: float,
    sample_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Group near-duplicate likelihood columns and build reduced problem."""
    J, K = matrix_lik.shape
    if K <= 1:
        groups = [np.array([0], dtype=int)] if K == 1 else []
        return matrix_lik, prior, pi_init, groups, [np.array([1.0], dtype=float)] if K == 1 else []

    rows = min(J, max(32, int(sample_rows)))
    if rows == J:
        idx = np.arange(J, dtype=int)
    else:
        idx = np.linspace(0, J - 1, rows, dtype=int)

    tiny = np.finfo(float).tiny
    log_sub = np.log(np.maximum(matrix_lik[idx], tiny))

    assigned = np.zeros(K, dtype=bool)
    groups: list[np.ndarray] = []
    for i in range(K):
        if assigned[i]:
            continue
        members = [i]
        assigned[i] = True
        remaining = np.where(~assigned)[0]
        if remaining.size > 0:
            diffs = np.max(np.abs(log_sub[:, remaining] - log_sub[:, i][:, None]), axis=0)
            close = remaining[diffs <= log_tol]
            if close.size > 0:
                assigned[close] = True
                members.extend(close.tolist())
        groups.append(np.asarray(members, dtype=int))

    if len(groups) == K:
        group_props = [np.array([1.0], dtype=float) for _ in groups]
        return matrix_lik, prior, pi_init, groups, group_props

    reduced_cols: list[np.ndarray] = []
    reduced_prior = np.zeros(len(groups), dtype=float)
    reduced_pi_init = np.zeros(len(groups), dtype=float)
    group_props: list[np.ndarray] = []

    for gidx, g in enumerate(groups):
        pg = np.asarray(prior[g], dtype=float)
        wg = np.maximum(pg, 0.0)
        sw = float(np.sum(wg))
        if sw <= 0.0:
            props = np.full(g.size, 1.0 / g.size, dtype=float)
        else:
            props = wg / sw
        group_props.append(props)

        reduced_cols.append(matrix_lik[:, g] @ props)
        reduced_prior[gidx] = float(np.sum(prior[g]))
        reduced_pi_init[gidx] = float(np.sum(pi_init[g]))

    reduced_matrix = np.column_stack(reduced_cols)
    return reduced_matrix, reduced_prior, reduced_pi_init, groups, group_props


def _expand_grouped_pi(
    pi_reduced: np.ndarray,
    groups: list[np.ndarray],
    group_props: list[np.ndarray],
    K: int,
) -> np.ndarray:
    out = np.zeros(K, dtype=float)
    for grp_w, g, props in zip(pi_reduced, groups, group_props):
        out[g] = float(grp_w) * props
    return out


def _em_single_step(
    matrix_lik: np.ndarray,
    pi: np.ndarray,
    prior: np.ndarray,
    denom_const: float,
    eps: float,
) -> np.ndarray:
    weighted = matrix_lik * pi[None, :]
    norm = np.sum(weighted, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    w = weighted / norm
    new_pi = (np.sum(w, axis=0) + prior - 1.0) / denom_const
    new_pi = _sanitize_pi(new_pi, eps)
    return new_pi


def _em_optimize_pi(
    matrix_lik: np.ndarray,
    pi_init: np.ndarray,
    prior: np.ndarray,
    max_iter: int = 2000,
    tol: float = 1e-8,
) -> np.ndarray:
    J, K = matrix_lik.shape
    pi = np.asarray(pi_init, dtype=float)
    denom_const = J + np.sum(prior) - K
    if denom_const <= 0:
        raise ValueError("Invalid prior: denominator for M-step is non-positive")

    eps = np.finfo(float).tiny
    matrix_lik = np.maximum(matrix_lik, 0.0)

    for _ in range(max_iter):
        new_pi = _em_single_step(matrix_lik, pi, prior, denom_const, eps)

        if np.max(np.abs(new_pi - pi)) < tol:
            pi = new_pi
            break
        pi = new_pi

    return pi


def _squarem_optimize_pi(
    matrix_lik: np.ndarray,
    pi_init: np.ndarray,
    prior: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> np.ndarray:
    J, K = matrix_lik.shape
    denom_const = J + np.sum(prior) - K
    if denom_const <= 0:
        raise ValueError("Invalid prior: denominator for M-step is non-positive")

    eps = np.finfo(float).tiny
    matrix_lik = np.maximum(matrix_lik, 0.0)
    prior_alpha = prior - 1.0

    x = _sanitize_pi(pi_init, eps)
    f_prev = _objective_value(matrix_lik, x, prior_alpha, eps)

    for _ in range(max_iter):
        x1 = _em_single_step(matrix_lik, x, prior, denom_const, eps)
        r = x1 - x
        if np.max(np.abs(r)) < tol:
            x = x1
            break

        x2 = _em_single_step(matrix_lik, x1, prior, denom_const, eps)
        v = (x2 - x1) - r

        sv2 = float(np.dot(v, v))
        if sv2 <= eps:
            x_prop = x2
        else:
            sr2 = float(np.dot(r, r))
            step = -np.sqrt(sr2 / sv2)
            step = float(np.clip(step, -10.0, -1e-4))
            x_sq = x - 2.0 * step * r + (step * step) * v
            x_sq = _sanitize_pi(x_sq, eps)
            x_prop = _em_single_step(matrix_lik, x_sq, prior, denom_const, eps)

        f_prop = _objective_value(matrix_lik, x_prop, prior_alpha, eps)
        if not np.isfinite(f_prop) or f_prop > f_prev:
            # Monotone safeguard.
            x_prop = x2
            f_prop = _objective_value(matrix_lik, x_prop, prior_alpha, eps)

        if abs(f_prev - f_prop) <= tol * (1.0 + abs(f_prev)):
            x = x_prop
            break

        x = x_prop
        f_prev = f_prop

    return _sanitize_pi(x, eps)


def _mixsqp_newton_optimize_pi(
    matrix_lik: np.ndarray,
    pi_init: np.ndarray,
    prior: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> np.ndarray:
    """Projected Newton solver on the simplex (mixSQP-style)."""
    _, K = matrix_lik.shape
    eps = 1e-12
    prior_alpha = prior - 1.0
    pi = _sanitize_pi(pi_init, eps)

    for _ in range(max_iter):
        g = _objective_grad(matrix_lik, pi, prior_alpha, eps)
        g_tan = g - np.mean(g)
        kkt = np.max(np.abs(g_tan))
        if kkt < tol:
            break

        H = _objective_hessian(matrix_lik, pi, prior_alpha, eps)
        reg = max(1e-12, 1e-9 * float(np.trace(H) / max(K, 1)))
        H = H + np.eye(K) * reg

        ones = np.ones((K, 1), dtype=float)
        kkt_mat = np.block([[H, ones], [ones.T, np.zeros((1, 1), dtype=float)]])
        rhs = np.concatenate([-g, np.array([0.0], dtype=float)])
        try:
            sol = np.linalg.solve(kkt_mat, rhs)
            d = sol[:K]
        except np.linalg.LinAlgError:
            # Fallback: tangent-space gradient direction.
            d = -g_tan

        if np.all(np.abs(d) < tol):
            break

        neg = d < 0.0
        if np.any(neg):
            tmax = np.min((pi[neg] - eps) / (-d[neg]))
            tmax = float(max(0.0, tmax))
        else:
            tmax = 1.0
        step = min(1.0, 0.99 * tmax if tmax > 0.0 else 0.0)
        if step <= 0.0:
            break

        f0 = _objective_value(matrix_lik, pi, prior_alpha, eps)
        gd = float(np.dot(g, d))
        c1 = 1e-4
        accepted = False
        for _ in range(30):
            cand = _sanitize_pi(pi + step * d, eps)
            fc = _objective_value(matrix_lik, cand, prior_alpha, eps)
            if np.isfinite(fc) and fc <= f0 + c1 * step * gd:
                pi = cand
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break

    return _sanitize_pi(pi, eps)


def _mixsqp_optimize_pi(
    matrix_lik: np.ndarray,
    pi_init: np.ndarray,
    prior: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> np.ndarray:
    """mixSQP-style optimization: Newton step + active-set polishing."""
    eps = 1e-12
    K = matrix_lik.shape[1]

    try:
        pi = _mixsqp_newton_optimize_pi(matrix_lik, pi_init, prior, max_iter=max_iter, tol=tol)
    except Exception:
        pi = _sanitize_pi(pi_init, eps)

    # Polish with constrained SQP, then progressively refit active components.
    pi = _slsqp_optimize_pi(matrix_lik, pi, prior, max_iter=max_iter, tol=tol)

    active_thresh = max(1e-6, 10.0 * tol)
    for _ in range(6):
        active = pi > active_thresh
        if not np.any(active):
            active[np.argmax(pi)] = True
        if int(np.sum(active)) == K:
            break

        pi_active = _sanitize_pi(pi[active], eps)
        pi_active = _slsqp_optimize_pi(
            matrix_lik[:, active],
            pi_active,
            prior[active],
            max_iter=max_iter,
            tol=tol,
        )
        new_pi = np.zeros(K, dtype=float)
        new_pi[active] = pi_active
        new_pi = _sanitize_pi(new_pi, eps)

        if np.max(np.abs(new_pi - pi)) < max(tol, 1e-9):
            pi = new_pi
            break
        pi = new_pi

    return _sanitize_pi(pi, eps)


def _slsqp_optimize_pi(
    matrix_lik: np.ndarray,
    pi_init: np.ndarray,
    prior: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> np.ndarray:
    """Optimize mixture proportions using constrained SQP (SLSQP)."""

    _, K = matrix_lik.shape
    eps = 1e-12
    pi0 = _sanitize_pi(pi_init, eps)

    alpha = np.asarray(prior, dtype=float) - 1.0

    def objective(pi: np.ndarray) -> float:
        return _objective_value(matrix_lik, pi, alpha, eps)

    def gradient(pi: np.ndarray) -> np.ndarray:
        return _objective_grad(matrix_lik, pi, alpha, eps)

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    bounds = [(eps, 1.0) for _ in range(K)]

    result = minimize(
        objective,
        pi0,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": int(max_iter), "ftol": float(tol), "disp": False},
    )

    if not result.success:
        raise RuntimeError(f"SLSQP optimization failed: {result.message}")

    pi = np.maximum(np.asarray(result.x, dtype=float), eps)
    pi /= np.sum(pi)
    return pi


def optimize_pi(
    matrix_lik: np.ndarray,
    pi_init: np.ndarray | None = None,
    prior: np.ndarray | None = None,
    method: str = "slsqp",
    control: dict | None = None,
) -> np.ndarray:
    """Estimate mixture proportions from likelihood matrix.

    Supported methods:
    - ``mixsqp``: projected Newton on simplex (mixSQP-style)
    - ``squarem``: accelerated EM fixed-point iteration
    - ``slsqp``: constrained SciPy SLSQP
    - ``em``: plain EM
    - ``auto``: route by problem size and fall back for robustness
    """

    matrix_lik = np.asarray(matrix_lik, dtype=float)
    if matrix_lik.ndim != 2:
        raise ValueError("matrix_lik must be 2D")
    if np.any(matrix_lik < 0) or np.any(~np.isfinite(matrix_lik)):
        raise ValueError("matrix_lik must contain finite non-negative values")

    _, K = matrix_lik.shape
    if pi_init is None:
        pi_init = initialize_pi(K)
    else:
        pi_init = np.asarray(pi_init, dtype=float)

    if prior is None:
        prior = np.ones(K, dtype=float)
    else:
        prior = np.asarray(prior, dtype=float)

    if pi_init.shape != (K,):
        raise ValueError("pi_init has wrong length")
    if prior.shape != (K,):
        raise ValueError("prior has wrong length")

    control = control or {}
    max_iter = int(control.get("max_iter", 2000))
    tol = float(control.get("tol", 1e-8))

    normalized = method.lower()
    if normalized not in {"mixsqp", "squarem", "slsqp", "em", "auto"}:
        raise ValueError("method must be one of 'mixsqp', 'squarem', 'slsqp', 'em', or 'auto'")

    prescreen = bool(control.get("prescreen", True))
    prescreen_min_k = int(control.get("prescreen_min_k", 64))
    prescreen_log_tol = float(control.get("prescreen_log_tol", 1e-3))
    prescreen_rows = int(control.get("prescreen_rows", 512))
    prescreen_polish = bool(control.get("prescreen_polish", True))
    prescreen_polish_iter = int(control.get("prescreen_polish_iter", min(max_iter, 300)))
    prescreen_active_thresh = float(control.get("prescreen_active_thresh", max(1e-6, 10.0 * tol)))

    if prescreen and K >= prescreen_min_k:
        (
            reduced_matrix,
            reduced_prior,
            reduced_pi_init,
            groups,
            group_props,
        ) = _prescreen_components(
            matrix_lik=matrix_lik,
            prior=prior,
            pi_init=pi_init,
            log_tol=prescreen_log_tol,
            sample_rows=prescreen_rows,
        )
        K_reduced = reduced_matrix.shape[1]
        if K_reduced < K:
            child_control = dict(control)
            child_control["prescreen"] = False
            pi_reduced = optimize_pi(
                reduced_matrix,
                pi_init=reduced_pi_init,
                prior=reduced_prior,
                method=normalized,
                control=child_control,
            )
            pi_full = _expand_grouped_pi(pi_reduced, groups, group_props, K)
            pi_full = _sanitize_pi(pi_full, np.finfo(float).tiny)

            if not prescreen_polish:
                return pi_full

            active = pi_full > prescreen_active_thresh
            if not np.any(active):
                active[np.argmax(pi_full)] = True
            polish_control = dict(control)
            polish_control["prescreen"] = False
            polish_control["max_iter"] = prescreen_polish_iter
            pi_active = optimize_pi(
                matrix_lik[:, active],
                pi_init=pi_full[active],
                prior=prior[active],
                method=normalized,
                control=polish_control,
            )
            out = np.zeros(K, dtype=float)
            out[active] = pi_active
            return _sanitize_pi(out, np.finfo(float).tiny)

    if normalized == "mixsqp":
        try:
            return _mixsqp_optimize_pi(matrix_lik, pi_init, prior, max_iter=max_iter, tol=tol)
        except Exception:
            return _squarem_optimize_pi(matrix_lik, pi_init, prior, max_iter=max_iter, tol=tol)

    if normalized == "squarem":
        try:
            return _squarem_optimize_pi(matrix_lik, pi_init, prior, max_iter=max_iter, tol=tol)
        except Exception:
            return _em_optimize_pi(matrix_lik, pi_init, prior, max_iter=max_iter, tol=tol)

    if normalized == "slsqp":
        try:
            return _slsqp_optimize_pi(matrix_lik, pi_init, prior, max_iter=max_iter, tol=tol)
        except Exception:
            return _squarem_optimize_pi(matrix_lik, pi_init, prior, max_iter=max_iter, tol=tol)

    if normalized == "em":
        return _em_optimize_pi(matrix_lik, pi_init, prior, max_iter=max_iter, tol=tol)

    # auto
    if K <= 128:
        order = ("mixsqp", "squarem", "slsqp", "em")
    else:
        order = ("squarem", "mixsqp", "slsqp", "em")

    last_exc: Exception | None = None
    for candidate in order:
        try:
            return optimize_pi(
                matrix_lik,
                pi_init=pi_init,
                prior=prior,
                method=candidate,
                control=control,
            )
        except Exception as exc:  # pragma: no cover - fallback chain
            last_exc = exc

    raise RuntimeError(f"All optimization methods failed; last error: {last_exc}")


__all__ = ["initialize_pi", "optimize_pi"]
