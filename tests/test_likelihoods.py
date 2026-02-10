import numpy as np
from scipy.stats import multivariate_normal

from pymash._numerics import mvn_logpdf_batch
from pymash.data import mash_set_data
from pymash.likelihoods import calc_lik_matrix


def test_calc_lik_matrix_matches_scipy_common_cov():
    Bhat = np.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.2]])
    data = mash_set_data(Bhat, Shat=1.0)

    U = [np.eye(2), np.array([[1.0, 0.3], [0.3, 1.0]])]
    ll = calc_lik_matrix(data, U, log=True)

    expected = np.column_stack(
        [
            multivariate_normal(mean=np.zeros(2), cov=np.eye(2) + Uk).logpdf(Bhat)
            for Uk in U
        ]
    )
    assert np.allclose(ll, expected, atol=1e-10)


def test_mvn_logpdf_batch_singular_cov_matches_near_mean_edge_case():
    mean = np.array([0.2, -0.1], dtype=float)
    sigma = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=float)  # singular PSD
    X = np.array(
        [
            [0.2, -0.1],  # exact mean
            [0.2 + 4e-7, -0.1 - 3e-7],  # within 1e-6 L1 tolerance
            [0.2 + 2e-6, -0.1],  # outside tolerance
        ],
        dtype=float,
    )
    ll = mvn_logpdf_batch(X, mean, sigma)
    assert np.isposinf(ll[0])
    assert np.isposinf(ll[1])
    assert np.isneginf(ll[2])
