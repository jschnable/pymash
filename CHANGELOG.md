# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Pre-fit diagnostics helper `check_mash_data()` to flag SE heterogeneity,
  near-zero standard errors, and trait-correlation ill-conditioning.
- Covariance regularization utility `regularize_cov()` with optional
  `v_ridge` support in `mash_set_data()` and `mash_update_data()`.
- Plant-genetics scenario test coverage in `tests/test_plant_genetics_scenarios.py`.
- Result-helper compatibility for workflow result containers
  (`TrainApplyResult` and `ChunkedApplyResult`) via `get_*` extractors.

### Changed

- `estimate_null_correlation_simple()` now supports explicit fallback modes
  when too few null effects are available (`error`, `identity`, `all`) and
  optional ridge regularization of the returned matrix.
- `cov_pca()` and `cov_ed()` now warn when data-driven covariance learning is
  attempted with very small subsets (`n < 20 * R`).
- `mash_set_data()` now raises a clearer error when `Shat` contains negative
  values, including guidance about possible `Bhat`/`Shat` swaps.
- README examples now use keyword arguments for `mash_set_data(Bhat=..., Shat=...)`
  and recommend `get_*` extraction helpers as the primary result interface.

## [0.1.0] - 2026-02-10

### Added

- Initial public `pymash` package and CLI.
- Core mash fitting workflow (`mash`, `mash_1by1`, posterior extraction helpers).
- Canonical and data-driven covariance generation.
- C++ extension backend (`pymash._edcpp`) for performance-critical kernels.
- End-to-end tests, including R-reference comparison tests.

### Changed

- Project rebranded from `pymashr` to `pymash`.

[0.1.0]: https://github.com/jschnable/pymash/releases/tag/v0.1.0
