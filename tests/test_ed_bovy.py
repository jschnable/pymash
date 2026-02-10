import numpy as np
import pytest

from pymash.covariances import cov_ed, cov_pca
from pymash.data import mash_set_data
from pymash.ed import _edcpp, bovy_wrapper, extreme_deconvolution
from pymash.simulations import simple_sims


pytestmark = pytest.mark.skipif(_edcpp is None, reason="C++ ED backend not built")


def test_extreme_deconvolution_cpp_basic_properties():
    rng = np.random.default_rng(7)
    n = 120
    d = 3
    k = 2

    amps_true = np.array([0.65, 0.35])
    cov_true = np.array(
        [
            [[0.8, 0.2, 0.0], [0.2, 0.5, 0.1], [0.0, 0.1, 0.3]],
            [[1.2, -0.1, 0.0], [-0.1, 0.7, 0.05], [0.0, 0.05, 0.4]],
        ],
        dtype=float,
    )

    z = rng.choice(k, size=n, p=amps_true)
    x = np.zeros((n, d), dtype=float)
    for i in range(n):
        x[i] = rng.multivariate_normal(np.zeros(d), cov_true[z[i]])

    noise_diag = np.exp(rng.normal(-1.2, 0.2, size=(n, d)))
    y = x + rng.normal(0.0, np.sqrt(noise_diag), size=(n, d))

    xamp0 = np.array([0.5, 0.5], dtype=float)
    xmean0 = np.zeros((k, d), dtype=float)
    xcov0 = np.array([np.eye(d), np.eye(d) * 0.5], dtype=float)

    out = extreme_deconvolution(
        y,
        noise_diag,
        xamp=xamp0,
        xmean=xmean0,
        xcovar=xcov0,
        maxiter=150,
        tol=1e-6,
    )

    assert out["status"] == 0
    assert out["xamp"].shape == (k,)
    assert np.isclose(np.sum(out["xamp"]), 1.0)
    assert np.all(out["xamp"] > 0)
    assert len(out["xcovar"]) == k
    assert np.all(np.isfinite(out["objective"]))
    assert out["objective"].shape[0] >= 2

    # Objective should be non-decreasing up to tiny numerical tolerance.
    diffs = np.diff(out["objective"])
    assert np.all(diffs >= -1e-7)

    for U in out["xcovar"]:
        assert U.shape == (d, d)
        assert np.allclose(U, U.T, atol=1e-8)
        evals = np.linalg.eigvalsh(U)
        assert np.min(evals) > -1e-8


def test_bovy_wrapper_and_cov_ed_bovy():
    sim = simple_sims(nsamp=20, ncond=4, err_sd=0.5, seed=21)
    data = mash_set_data(sim["Bhat"], sim["Shat"])

    U_init = cov_pca(data, npc=2)

    out = bovy_wrapper(data, U_init, maxiter=80, tol=1e-5)
    assert out["pi"].shape[0] == len(U_init)
    assert np.isclose(np.sum(out["pi"]), 1.0)
    assert len(out["Ulist"]) == len(U_init)
    assert np.isfinite(out["av_loglik"])

    U_ed = cov_ed(data, U_init, algorithm="bovy", maxiter=80, tol=1e-5)
    assert set(U_ed.keys()) == {"ED_PCA_1", "ED_PCA_2", "ED_tPCA"}
    for U in U_ed.values():
        assert U.shape == (4, 4)
        assert np.allclose(U, U.T, atol=1e-8)
