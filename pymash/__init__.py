"""Python implementation of multivariate adaptive shrinkage (mashr).

Data setup
----------
mash_set_data
    Create a MashData object from Bhat/Shat matrices.
mash_update_data
    Copy a MashData with a new correlation matrix V or contrast ref.
check_mash_data
    Run pre-fit diagnostics for common data quality issues.
regularize_cov
    Add diagonal ridge to covariance/correlation matrices.
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
fit_mash_prior, apply_mash_prior, apply_mash_prior_chunked, mash_train_apply
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
renumber_credible_sets_by_logbf, renumber_credible_sets_from_result
    Rank and renumber credible sets by descending Log10 BF.
format_credible_set_report
    Text report for ranked credible sets.

Correlation estimation
----------------------
estimate_null_correlation_simple
    Estimate null correlation from putatively null effects.
mash_estimate_corr_em
    Iterative EM estimation of null correlation.

Plotting
--------
mash_plot_meta
    Forest plot of posterior effects (requires ``pip install pymashrink[plot]``).

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

from .data import (
    MashData,
    check_mash_data,
    contrast_matrix,
    mash_set_data,
    mash_update_data,
    regularize_cov,
)
from .correlation import estimate_null_correlation_simple, mash_estimate_corr_em
from .covariances import cov_canonical, cov_ed, cov_pca, cov_udi
from .mash import FittedG, MashResult, mash, mash_compute_posterior_matrices, mash_1by1
from .workflow import (
    ChunkedApplyResult,
    TrainApplyResult,
    apply_mash_prior,
    apply_mash_prior_chunked,
    fit_mash_prior,
    mash_train_apply,
    select_training_effects,
)
from .results import (
    CredibleSetSummary,
    format_credible_set_report,
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
    renumber_credible_sets_by_logbf,
    renumber_credible_sets_from_result,
)
from .plots import mash_plot_meta
from .simulations import simple_sims, simple_sims2

import sys as _sys

# C++ extension status - populated below
_cpp_backend = None
_cpp_openmp_enabled = False
_cpp_openmp_status_known = False

try:
    # Import the C++ extension module
    from . import _edcpp

    _cpp_backend = _edcpp
    try:
        _cpp_openmp_enabled = bool(_edcpp.openmp_enabled())
        _cpp_openmp_status_known = True
    except Exception:
        _cpp_openmp_enabled = False
        _cpp_openmp_status_known = False
        warnings.warn(
            "pymash C++ extension loaded, but OpenMP status could not be queried. "
            "Assuming single-threaded mode.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Warn macOS users if running single-threaded (OpenMP disabled in wheel)
    if _sys.platform == "darwin" and _cpp_openmp_status_known and not _cpp_openmp_enabled:
        warnings.warn(
            "pymash is running single-threaded on macOS (OpenMP not enabled). "
            "This is normal for pre-built wheels. For better performance on large datasets, "
            "rebuild with OpenMP: brew install libomp && pip install --force-reinstall "
            "--no-binary=pymashrink pymashrink. Run pymash.check_threading() for details.",
            RuntimeWarning,
            stacklevel=2,
        )
except ImportError:
    warnings.warn(
        "pymash C++ extension not found. This is needed for mash() fitting "
        "but not for data loading or result inspection. If you installed from "
        "a pre-built wheel this should not happen — try: pip install --force-reinstall pymashrink."
        "If building from source, ensure a C++ compiler is available (see README).",
        RuntimeWarning,
        stacklevel=2,
    )


def check_threading(verbose: bool = False) -> dict:
    """Check and report pymash threading status.

    Returns a dictionary with threading diagnostics and optionally prints a summary.
    Useful for verifying that OpenMP is enabled for multi-threaded performance.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``"cpp_backend_available"``: bool, whether C++ extension loaded
        - ``"openmp_enabled"``: bool, whether OpenMP is active
        - ``"openmp_status_known"``: bool, whether OpenMP status was queryable
        - ``"platform"``: str, operating system
        - ``"recommendation"``: str or None, suggested action if applicable

    Examples
    --------
    >>> import pymash
    >>> info = pymash.check_threading(verbose=True)
    pymash threading status:
      Platform: darwin
      C++ backend: available
      OpenMP: disabled (single-threaded)
      Recommendation: For multi-threaded performance, run:
        brew install libomp
        pip install --force-reinstall --no-binary=pymashrink pymashrink
    """
    info = {
        "cpp_backend_available": _cpp_backend is not None,
        "openmp_enabled": _cpp_openmp_enabled,
        "openmp_status_known": _cpp_openmp_status_known,
        "platform": _sys.platform,
        "recommendation": None,
    }

    if _cpp_backend is None:
        info["recommendation"] = (
            "C++ extension failed to load. Reinstall with: pip install --force-reinstall pymashrink"
        )
    else:
        if not _cpp_openmp_status_known:
            info["recommendation"] = (
                "OpenMP status is unknown. Ensure your installed pymash wheel/source build "
                "matches your current Python environment."
            )
        elif not _cpp_openmp_enabled and _sys.platform == "darwin":
            info["recommendation"] = (
                "For multi-threaded performance on macOS, run:\n"
                "    brew install libomp\n"
                "    pip install --force-reinstall --no-binary=pymashrink pymashrink"
            )
        elif not _cpp_openmp_enabled and _sys.platform == "linux":
            info["recommendation"] = (
                "OpenMP should be enabled on Linux. Try rebuilding:\n"
                "    pip install --force-reinstall --no-binary=pymashrink pymashrink"
            )

    if verbose:
        print("pymash threading status:")
        print(f"  Platform: {_sys.platform}")
        if _cpp_backend is None:
            print("  C++ backend: NOT AVAILABLE")
        else:
            print("  C++ backend: available")
            if not _cpp_openmp_status_known:
                print("  OpenMP: unknown (status query unavailable)")
            elif _cpp_openmp_enabled:
                print("  OpenMP: enabled (multi-threaded)")
            else:
                print("  OpenMP: disabled (single-threaded)")
        if info["recommendation"] is not None:
            print(f"  Recommendation: {info['recommendation']}")

    return info

__all__ = [
    "MashData",
    "FittedG",
    "MashResult",
    "TrainApplyResult",
    "ChunkedApplyResult",
    "check_mash_data",
    "check_threading",
    "regularize_cov",
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
    "apply_mash_prior_chunked",
    "mash_train_apply",
    "CredibleSetSummary",
    "get_log10bf",
    "get_significant_results",
    "get_n_significant_conditions",
    "get_estimated_pi",
    "get_pairwise_sharing",
    "get_pairwise_sharing_from_samples",
    "renumber_credible_sets_by_logbf",
    "renumber_credible_sets_from_result",
    "format_credible_set_report",
    "get_pm",
    "get_psd",
    "get_lfsr",
    "get_lfdr",
    "mash_plot_meta",
    "simple_sims",
    "simple_sims2",
]
