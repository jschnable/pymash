import numpy as np

from pymash.covariances import cov_ed, cov_pca
from pymash.data import mash_set_data
from pymash.ed import teem_wrapper
from pymash.simulations import simple_sims


def test_cov_pca_returns_expected_shapes():
    sim = simple_sims(nsamp=30, ncond=5, err_sd=0.5, seed=11)
    data = mash_set_data(sim["Bhat"], sim["Shat"])

    U = cov_pca(data, npc=3)

    assert set(U.keys()) == {"PCA_1", "PCA_2", "PCA_3", "tPCA"}
    for mat in U.values():
        assert mat.shape == (5, 5)
        assert np.allclose(mat, mat.T, atol=1e-10)


def test_teem_wrapper_returns_valid_mixture():
    sim = simple_sims(nsamp=20, ncond=4, err_sd=0.3, seed=12)
    data = mash_set_data(sim["Bhat"], sim["Shat"])

    U_init = list(cov_pca(data, npc=2).values())
    out = teem_wrapper(data, U_init, max_iter=100, converge_tol=1e-6)

    assert out.w.shape == (len(U_init),)
    assert np.isclose(np.sum(out.w), 1.0)
    assert len(out.U) == len(U_init)
    assert np.all(np.isfinite(out.objective))
    assert out.maxd[-1] <= out.maxd[0]
    for u in out.U:
        assert u.shape == (4, 4)
        assert np.allclose(u, u.T, atol=1e-10)


def test_cov_ed_teem_naming_and_shapes():
    sim = simple_sims(nsamp=25, ncond=4, err_sd=0.4, seed=13)
    data = mash_set_data(sim["Bhat"], sim["Shat"])

    U_init = cov_pca(data, npc=2)
    U_ed = cov_ed(data, U_init, algorithm="teem", max_iter=100, converge_tol=1e-6)

    assert set(U_ed.keys()) == {"ED_PCA_1", "ED_PCA_2", "ED_tPCA"}
    for u in U_ed.values():
        assert u.shape == (4, 4)
        assert np.allclose(u, u.T, atol=1e-10)
