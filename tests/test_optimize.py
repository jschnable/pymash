import numpy as np

from pymash.optimize import optimize_pi


def _loglik(matrix_lik: np.ndarray, pi: np.ndarray) -> float:
    with np.errstate(all="ignore"):
        mix = matrix_lik @ pi
    mix = np.where(np.isfinite(mix), mix, np.finfo(float).tiny)
    return float(np.sum(np.log(np.maximum(mix, np.finfo(float).tiny))))


def _assert_valid_pi(pi: np.ndarray) -> None:
    assert np.all(np.isfinite(pi))
    assert np.isclose(np.sum(pi), 1.0)
    assert np.all(pi >= 0.0)


def _active_count(pi: np.ndarray, thresh: float = 1e-4) -> int:
    return int(np.sum(pi > thresh))


def test_mixsqp_matches_em_objective_or_better():
    rng = np.random.default_rng(10)
    # Positive likelihood matrix.
    matrix_lik = np.exp(rng.normal(size=(300, 20)))

    pi_em = optimize_pi(matrix_lik, method="em", control={"max_iter": 800, "tol": 1e-9})
    pi_sqp = optimize_pi(matrix_lik, method="slsqp", control={"max_iter": 800, "tol": 1e-9})

    assert np.isclose(np.sum(pi_em), 1.0)
    assert np.isclose(np.sum(pi_sqp), 1.0)
    assert np.all(pi_em > 0)
    assert np.all(pi_sqp > 0)

    ll_em = _loglik(matrix_lik, pi_em)
    ll_sqp = _loglik(matrix_lik, pi_sqp)
    # SQP should be at least as good up to tiny numerical tolerance.
    assert ll_sqp >= ll_em - 1e-6


def test_mixsqp_respects_prior_bias():
    rng = np.random.default_rng(11)
    matrix_lik = np.exp(rng.normal(size=(200, 8)))

    pi_uniform = optimize_pi(matrix_lik, method="slsqp", prior=np.ones(8))
    prior = np.ones(8)
    prior[0] = 20.0
    pi_biased = optimize_pi(matrix_lik, method="slsqp", prior=prior)

    assert pi_biased[0] > pi_uniform[0]


def test_mixsqp_method_matches_slsqp_close():
    rng = np.random.default_rng(12)
    matrix_lik = np.exp(rng.normal(size=(100, 6)))

    pi_mixsqp = optimize_pi(matrix_lik, method="mixsqp")
    pi_slsqp = optimize_pi(matrix_lik, method="slsqp")

    ll_mix = _loglik(matrix_lik, pi_mixsqp)
    ll_slsqp = _loglik(matrix_lik, pi_slsqp)
    assert ll_mix >= ll_slsqp - 1e-5


def test_squarem_improves_or_matches_em_objective():
    rng = np.random.default_rng(13)
    matrix_lik = np.exp(rng.normal(size=(400, 30)))

    pi_em = optimize_pi(matrix_lik, method="em", control={"max_iter": 1200, "tol": 1e-9})
    pi_sq = optimize_pi(matrix_lik, method="squarem", control={"max_iter": 1200, "tol": 1e-9})

    ll_em = _loglik(matrix_lik, pi_em)
    ll_sq = _loglik(matrix_lik, pi_sq)
    assert ll_sq >= ll_em - 1e-6


def test_stress_ill_conditioned_likelihood_methods_stable():
    rng = np.random.default_rng(123)
    J, K = 400, 40
    u = rng.normal(size=J)
    v = rng.normal(size=K)
    log_lik = 7.0 * np.outer(u / np.std(u), v / np.std(v)) + 0.02 * rng.normal(size=(J, K))
    log_lik = np.clip(log_lik, -60.0, 40.0)
    matrix_lik = np.exp(log_lik)

    methods = ("em", "squarem", "slsqp", "mixsqp", "auto")
    out = {m: optimize_pi(matrix_lik, method=m, control={"max_iter": 2000, "tol": 1e-10}) for m in methods}
    ll = {m: _loglik(matrix_lik, p) for m, p in out.items()}

    for p in out.values():
        _assert_valid_pi(p)

    best_ll = max(ll.values())
    assert ll["auto"] >= best_ll - 1e-5
    assert ll["mixsqp"] >= ll["em"] - 1e-5


def test_stress_near_duplicate_components_prefers_sparse_mixsqp():
    rng = np.random.default_rng(456)
    J = 600
    base = np.exp(rng.normal(size=(J, 6)))
    cols = []
    for i in range(6):
        for _ in range(8):
            col = base[:, i] * (1.0 + 1e-4 * rng.normal(size=J))
            cols.append(np.maximum(col, 1e-30))
    matrix_lik = np.column_stack(cols)

    pi_em = optimize_pi(matrix_lik, method="em", control={"max_iter": 3000, "tol": 1e-10})
    pi_sq = optimize_pi(matrix_lik, method="squarem", control={"max_iter": 3000, "tol": 1e-10})
    pi_slsqp = optimize_pi(matrix_lik, method="slsqp", control={"max_iter": 3000, "tol": 1e-10})
    pi_mix = optimize_pi(matrix_lik, method="mixsqp", control={"max_iter": 3000, "tol": 1e-10})
    pi_auto = optimize_pi(matrix_lik, method="auto", control={"max_iter": 3000, "tol": 1e-10})

    for p in (pi_em, pi_sq, pi_slsqp, pi_mix, pi_auto):
        _assert_valid_pi(p)

    ll_em = _loglik(matrix_lik, pi_em)
    ll_sq = _loglik(matrix_lik, pi_sq)
    ll_slsqp = _loglik(matrix_lik, pi_slsqp)
    ll_mix = _loglik(matrix_lik, pi_mix)
    ll_auto = _loglik(matrix_lik, pi_auto)

    assert ll_mix >= ll_em + 1e-3
    assert ll_mix >= ll_sq + 1e-3
    assert ll_auto >= ll_mix - 1e-5
    assert _active_count(pi_mix) <= 12
    assert _active_count(pi_slsqp) <= 12
    assert _active_count(pi_em) >= 30


def test_prescreen_duplicate_components_keeps_objective_and_sparsity():
    rng = np.random.default_rng(789)
    J = 500
    base = np.exp(rng.normal(size=(J, 5)))
    cols = []
    for i in range(5):
        for _ in range(10):
            cols.append(np.maximum(base[:, i] * (1.0 + 5e-4 * rng.normal(size=J)), 1e-30))
    matrix_lik = np.column_stack(cols)

    ctrl_no = {"max_iter": 2000, "tol": 1e-10, "prescreen": False}
    ctrl_yes = {
        "max_iter": 2000,
        "tol": 1e-10,
        "prescreen": True,
        "prescreen_min_k": 16,
        "prescreen_log_tol": 5e-3,
        "prescreen_rows": 256,
        "prescreen_polish_iter": 200,
    }

    pi_no = optimize_pi(matrix_lik, method="mixsqp", control=ctrl_no)
    pi_yes = optimize_pi(matrix_lik, method="mixsqp", control=ctrl_yes)

    _assert_valid_pi(pi_no)
    _assert_valid_pi(pi_yes)

    ll_no = _loglik(matrix_lik, pi_no)
    ll_yes = _loglik(matrix_lik, pi_yes)

    assert ll_yes >= ll_no - 1e-4
    assert _active_count(pi_yes) <= _active_count(pi_no)
