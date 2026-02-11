import numpy as np
import pytest

from pymash.correlation import estimate_null_correlation_simple
from pymash.covariances import cov_canonical, cov_pca
from pymash.data import check_mash_data, mash_set_data
from pymash.mash import mash


def test_near_singular_trait_covariance_can_be_regularized():
    rng = np.random.default_rng(3001)
    bhat = rng.normal(size=(40, 6))
    shat = np.ones((40, 6), dtype=float)
    v_singular = np.ones((6, 6), dtype=float)

    with pytest.raises(ValueError, match="positive definite"):
        mash_set_data(Bhat=bhat, Shat=shat, V=v_singular)

    data = mash_set_data(Bhat=bhat, Shat=shat, V=v_singular, v_ridge=1e-6)
    assert data.common_V
    assert data.V.shape == (6, 6)
    assert np.all(np.isfinite(data.V))


def test_check_mash_data_flags_common_issues():
    bhat = np.array([[0.0, 0.1], [0.0, -0.2], [0.0, 0.3]], dtype=float)
    shat = np.array([[1e-8, 0.5], [1e-8, 2.0], [1e-8, 8.0]], dtype=float)

    issues = check_mash_data(
        Bhat=bhat,
        Shat=shat,
        near_zero_tol=1e-6,
        se_cv_thresh=0.5,
        verbose=False,
    )
    assert "near_zero_se" in issues
    assert "high_se_variability" in issues


def test_cov_pca_warns_when_too_few_effects():
    rng = np.random.default_rng(3002)
    bhat = rng.normal(size=(24, 4))
    shat = np.exp(rng.normal(loc=0.0, scale=0.2, size=(24, 4)))
    data = mash_set_data(Bhat=bhat, Shat=shat, alpha=1.0)

    with pytest.warns(RuntimeWarning, match="may be unstable"):
        U = cov_pca(data, npc=2, subset=np.arange(10))
    assert "PCA_1" in U


def test_null_correlation_can_fallback_to_identity():
    bhat = np.full((10, 4), 8.0, dtype=float)
    shat = np.ones((10, 4), dtype=float)
    data = mash_set_data(Bhat=bhat, Shat=shat)

    with pytest.warns(RuntimeWarning, match="Falling back to identity"):
        vhat = estimate_null_correlation_simple(
            data,
            z_thresh=0.1,
            est_cor=True,
            on_insufficient_null="identity",
        )
    assert np.allclose(vhat, np.eye(4), atol=1e-10)


def test_mixed_signal_and_noise_traits_fit_without_crashing():
    rng = np.random.default_rng(3003)
    J, R = 60, 5
    bhat = rng.normal(scale=0.2, size=(J, R))
    bhat[:10, 0] += 2.5
    bhat[:10, 1] += 2.0
    shat = np.exp(rng.normal(loc=-0.2, scale=0.4, size=(J, R)))

    data = mash_set_data(Bhat=bhat, Shat=shat, alpha=1.0)
    U = cov_canonical(data)
    out = mash(data, Ulist=U, grid=np.array([0.5, 1.0]), outputlevel=2)

    assert out.posterior_mean is not None
    assert out.lfsr is not None
    assert out.posterior_mean.shape == (J, R)
    assert np.all(np.isfinite(out.posterior_mean))
    assert np.all(np.isfinite(out.lfsr))
