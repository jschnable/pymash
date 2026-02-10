"""Python implementation of multivariate adaptive shrinkage (mashr).

Data setup
----------
mash_set_data
    Create a MashData object from Bhat/Shat matrices.
mash_update_data
    Copy a MashData with a new correlation matrix V or contrast ref.
contrast_matrix
    Build a contrast matrix for comparing conditions to a reference.

Covariance matrices
-------------------
cov_canonical
    Canonical (hypothesis-based) covariance matrices.
cov_pca
    PCA-based data-driven covariance matrices.
cov_ed
    Refine covariance matrices with Extreme Deconvolution.
cov_udi
    Build UDI (unassociated/direct/indirect) covariance models.

Model fitting
-------------
mash
    Fit the mash model (main entry point).
mash_1by1
    Simple condition-by-condition baseline (no cross-condition shrinkage).
mash_compute_posterior_matrices
    Compute posteriors for new data using a previously fitted model.
fit_mash_prior, apply_mash_prior, mash_train_apply
    Two-stage train/apply helpers for large-scale workflows.

Result extraction
-----------------
get_pm, get_psd
    Posterior means / standard deviations.
get_lfsr, get_lfdr
    Local false sign rate / local false discovery rate.
get_significant_results
    Indices of effects significant in at least one condition.
get_n_significant_conditions
    Count of significant conditions per effect.
get_estimated_pi
    Estimated mixture proportions.
get_pairwise_sharing
    Pairwise sharing matrix between conditions.
get_pairwise_sharing_from_samples
    Pairwise sharing matrix from posterior samples.
get_log10bf
    Log10 Bayes factors.

Correlation estimation
----------------------
estimate_null_correlation_simple
    Estimate null correlation from putatively null effects.
mash_estimate_corr_em
    Iterative EM estimation of null correlation.

Plotting
--------
mash_plot_meta
    Forest plot of posterior effects (requires ``pip install pymash[plot]``).

Simulation
----------
simple_sims, simple_sims2
    Simulate test data with known effect structure.
"""

import warnings
from importlib.metadata import version as _get_version

try:
    __version__ = _get_version("pymash")
except Exception:
    __version__ = "0.0.0"

from .data import MashData, contrast_matrix, mash_set_data, mash_update_data
from .correlation import estimate_null_correlation_simple, mash_estimate_corr_em
from .covariances import cov_canonical, cov_ed, cov_pca, cov_udi
from .mash import FittedG, MashResult, mash, mash_compute_posterior_matrices, mash_1by1
from .workflow import TrainApplyResult, apply_mash_prior, fit_mash_prior, mash_train_apply, select_training_effects
from .results import (
    get_estimated_pi,
    get_lfdr,
    get_lfsr,
    get_log10bf,
    get_n_significant_conditions,
    get_pairwise_sharing,
    get_pairwise_sharing_from_samples,
    get_pm,
    get_psd,
    get_significant_results,
)
from .plots import mash_plot_meta
from .simulations import simple_sims, simple_sims2

try:
    from . import _edcpp as _edcpp  # noqa: F401
except Exception:
    warnings.warn(
        "pymash C++ extension not found. This is needed for mash() fitting "
        "but not for data loading or result inspection. If you installed from "
        "a pre-built wheel this should not happen — try: pip install --force-reinstall pymash. "
        "If building from source, ensure a C++ compiler is available (see README).",
        RuntimeWarning,
        stacklevel=2,
    )

__all__ = [
    "MashData",
    "FittedG",
    "MashResult",
    "TrainApplyResult",
    "mash_set_data",
    "mash_update_data",
    "contrast_matrix",
    "cov_canonical",
    "cov_pca",
    "cov_ed",
    "cov_udi",
    "estimate_null_correlation_simple",
    "mash_estimate_corr_em",
    "mash",
    "mash_compute_posterior_matrices",
    "mash_1by1",
    "select_training_effects",
    "fit_mash_prior",
    "apply_mash_prior",
    "mash_train_apply",
    "get_log10bf",
    "get_significant_results",
    "get_n_significant_conditions",
    "get_estimated_pi",
    "get_pairwise_sharing",
    "get_pairwise_sharing_from_samples",
    "get_pm",
    "get_psd",
    "get_lfsr",
    "get_lfdr",
    "mash_plot_meta",
    "simple_sims",
    "simple_sims2",
]
