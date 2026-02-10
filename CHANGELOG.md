# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
