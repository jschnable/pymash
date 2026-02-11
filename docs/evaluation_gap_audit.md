# pymash Evaluation Gap Audit (2026-02-11)

This checklist maps concerns from the external expert review to current repository status and acceptance criteria.

## Robustness

| Concern | Status | Evidence | Acceptance Criteria |
|---|---|---|---|
| Warn when data-driven covariance learning is underpowered (`n < 20 * R`) | Closed | `pymash/covariances.py` (`cov_pca`, `cov_ed`) | Runtime warning emitted for small subsets; tests assert warning. |
| Auto-guidance for heterogeneous SE with `alpha=0` | Closed | `pymash/data.py` (`mash_set_data`) | Warning includes CV and fold-range guidance toward `alpha=1`. |
| Near-singular covariance support via ridge regularization | Closed | `pymash/data.py` (`regularize_cov`, `mash_set_data`, `mash_update_data`) | Optional `v_ridge` stabilizes V before PD checks. |
| Better behavior when null-correlation has too few null effects | Closed | `pymash/correlation.py` (`estimate_null_correlation_simple`) | Supports `on_insufficient_null={"error","identity","all"}` with explicit warning/fallback. |
| Pre-fit dataset diagnostics helper | Closed | `pymash/data.py` (`check_mash_data`) | Returns issue dictionary for near-zero SE, SE variability, and ill-conditioning. |

## API Usability

| Concern | Status | Evidence | Acceptance Criteria |
|---|---|---|---|
| Inconsistent result access across `MashResult` / `TrainApplyResult` / `ChunkedApplyResult` | Closed | `pymash/results.py` (`_normalize_result_container`, `get_*`) | `get_pm/get_psd/get_lfsr/get_lfdr/get_estimated_pi/get_log10bf` accept all three result containers. |
| Bhat/Shat swap risk with positional arguments | Closed | `pymash/data.py` (`mash_set_data`) | Negative `Shat` fails fast with explicit swap hint in error message. |
| Recommended extraction style unclear | Closed | `README.md` (Result extraction section) | README explicitly recommends `get_*` helpers as canonical interface. |

## Documentation (Out of Current Scope)

| Concern | Status | Evidence | Acceptance Criteria |
|---|---|---|---|
| Full PLINK/GCTA conversion cookbook | Open | Not yet added | Add end-to-end conversion examples with schema assumptions and pitfalls. |
| LD clumping and multi-trait FDR workflow | Open | Not yet added | Add documented workflow guidance and references in README/docs. |
| Pleiotropic hotspot playbook | Open | Not yet added | Add reproducible pipeline using `get_n_significant_conditions` and region aggregation. |

## Testing

| Concern | Status | Evidence | Acceptance Criteria |
|---|---|---|---|
| Plant-genetics edge case tests | Closed | `tests/test_plant_genetics_scenarios.py` | Tests cover highly correlated traits, small strong-effect sets, null-correlation fallbacks, and mixed signal/noise stability. |
| Regression safety for new warnings/fallbacks | Closed | `tests/test_plant_genetics_scenarios.py` | Warnings/fallback paths are assertion-tested with `pytest.warns`. |
