# pymash

## What is mash?

When you run GWAS or eQTL analysis across multiple traits or tissues, you get
a table of effect sizes and standard errors for every marker in every
condition. Each condition is analyzed independently, so the estimates are
noisy and you lose the information that many effects are *shared* across
conditions.

**Multivariate Adaptive Shrinkage (mash)** fixes this. It learns which
patterns of sharing actually exist in your data — e.g., "this group of
markers affects all tissues equally" or "these markers are liver-specific" —
and then uses those patterns to improve every estimate. The result:

- **Better effect-size estimates** — borrowing strength across conditions
  reduces noise, especially for small effects
- **Calibrated significance** — the local false sign rate (lfsr) tells you
  the probability that you have the *sign* of the effect wrong, accounting
  for all conditions jointly
- **Interpretable sharing patterns** — the fitted mixture proportions
  directly tell you what fraction of effects are shared vs. condition-specific

The inputs are two matrices: **Bhat** (effect sizes, J markers x R conditions)
and **Shat** (standard errors, same shape). The outputs are shrunken posterior
means, posterior SDs, and lfsr for every marker in every condition.

See: Urbut, S. M., Wang, G., Carbonetto, P., & Stephens, M. (2019). Flexible statistical methods for estimating and testing effects in genomic studies with multiple conditions. Nature genetics. doi: https://doi.org/10.1038/s41588-018-0268-8

Note: this is an independent reimplementation. Any errors are mine and not those of the authors of Urbut et al.

## Quick Start

```python
import pymash as mash

# 1. Simulate example data (2000 effects x 5 conditions: 4 effect types x 500 each)
sim = mash.simple_sims(nsamp=500, ncond=5, err_sd=1.0, seed=1)

# 2. Set up the data
data = mash.mash_set_data(sim["Bhat"], sim["Shat"])

# 3. Build covariance matrices and fit the model
U_c = mash.cov_canonical(data)
result = mash.mash(data, Ulist=U_c)

# 4. Extract results
pm = mash.get_pm(result)       # Posterior means (J x R)
lfsr = mash.get_lfsr(result)   # Local false sign rates (J x R)
sig = mash.get_significant_results(result, thresh=0.05)
```

> **The Crucial Rule:** Keep covariance learning and prior-weight learning
> separate. Use strong signals to learn data-driven covariance matrices
> (`cov_pca`, `cov_ed`), but fit mixture weights (`pi`) on all tests or a
> large random subset. You can also use `select_method="topz_random"` to blend
> top-|z| effects with random background when `n_train` is small.

### Two-Stage Train/Apply Workflow (Recommended for very large GWAS/eQTL)

For large studies, a robust two-stage pattern is:

1. Learn data-driven covariance matrices on strong signals.
2. Fit `g` (mixture weights) on a random or mixed (`topz_random`) subset.
3. Apply the fixed `g` genome-wide.

```python
import numpy as np
import pymash as mash

data = mash.mash_set_data(Bhat, Shat, alpha=1)

# Stage 0: learn covariance patterns on strong effects
m1 = mash.mash_1by1(data)
strong_idx = mash.get_significant_results(m1, thresh=0.05)
data_strong = mash.mash_set_data(Bhat[strong_idx], Shat[strong_idx], alpha=1)

U_c = mash.cov_canonical(data)
U_pca = mash.cov_pca(data_strong, npc=5)
U_ed = mash.cov_ed(data_strong, U_pca)
U_all = {**U_c, **U_ed}

# Stage 1: train g on a manageable subset
workflow = mash.mash_train_apply(
    data,
    U_all,
    n_train=50000,                         # choose based on compute budget
    select_method="topz_random",           # strong + random background
    background_fraction=0.2,               # keep random background for calibration
    select_seed=1,
    train_mash_kwargs={"grid": np.array([0.5, 1.0]), "outputlevel": 1},
    apply_mash_kwargs={"outputlevel": 2, "output_lfdr": True},
)

result = workflow.apply_result            # full-data posterior summaries
train_idx = workflow.train_indices        # indices used for fitting g
fitted_g = workflow.train_result.fitted_g # reusable prior
```

This gives the same statistical model (`mash(..., g=fitted_g, fixg=True)`)
while making large analyses easier to scale and reproduce.

By default, `mash()` uses `chunk_size=250000`. For very large `J`, this
automatically switches to a two-stage train/apply execution to avoid OOM
while preserving the same workflow pattern.

For very large `J`, you can apply the fitted prior in chunks to reduce
peak memory:

```python
chunked = mash.apply_mash_prior_chunked(
    data,
    fitted_g,
    chunk_size=250_000,
    mash_kwargs={"outputlevel": 2, "output_lfdr": False},
    out_prefix="results/gwas40",  # writes results/gwas40.<name>.npy
)

pm = chunked.arrays["posterior_mean"]  # memmap if out_prefix is set
lfsr = chunked.arrays["lfsr"]
```

## Interpreting Results

- **`get_pm(result)`** — Posterior mean effect sizes (shrunken). These are
  your improved estimates. Null effects are shrunk toward zero; real effects
  are shrunk toward the most likely sharing pattern.

- **`get_lfsr(result)`** — Local false sign rate. The probability that you
  have the *sign* of the effect wrong. `lfsr < 0.05` means "95% confident
  this effect is in the stated direction." This is the recommended
  significance measure — it is more conservative and interpretable than
  p-values because it accounts for effect size, not just whether the effect
  is nonzero.

- **`get_lfdr(result)`** — Local false discovery rate. The probability the
  true effect is exactly zero. Less commonly used than lfsr because the
  point-mass-at-zero assumption is often unrealistic.

- **`get_psd(result)`** — Posterior standard deviations. Use for confidence
  intervals: `pm ± 1.96 * psd`.

- **`get_pairwise_sharing(result)`** — For each pair of conditions, the
  fraction of significant effects that are the same sign and similar
  magnitude. Values near 1.0 mean the two conditions behave similarly.

## Choosing Covariance Matrices

mash models effects as drawn from a mixture of multivariate normals. The
covariance matrices in that mixture represent hypotheses about how effects
are shared across conditions. You need to tell mash which patterns to
consider.

| Function | What it provides | When to use |
|----------|-----------------|-------------|
| `cov_canonical(data)` | Pre-defined patterns: null, identity, equal-effects, and condition-specific. | Always include these as a baseline. Sufficient for a first pass. |
| `cov_pca(data_strong, npc=5)` | Data-driven patterns from PCA of the strong effects. | When you have enough strong signals (~100+) to estimate real sharing patterns. |
| `cov_ed(data_strong, U_pca)` | Refines PCA matrices using Extreme Deconvolution (EM on the mixture). | When you want the best data-driven covariances. Use after `cov_pca`. |

**Typical recipe:** Use canonical alone for quick exploration. For a real
analysis, combine canonical + ED-refined PCA. First, identify strong signals
using `mash_1by1` and build a separate `MashData` for them:

```python
# Screen for strong signals (no cross-condition shrinkage)
m1 = mash.mash_1by1(data)
strong_idx = mash.get_significant_results(m1, thresh=0.05)
data_strong = mash.mash_set_data(Bhat[strong_idx], Shat[strong_idx])

# Build covariances from the strong signals
U_c = mash.cov_canonical(data)
U_pca = mash.cov_pca(data_strong, npc=5)
U_ed = mash.cov_ed(data_strong, U_pca)

# Fit on *all* data (not just strong), combining canonical + data-driven
U_all = {**U_c, **U_ed}
result = mash.mash(data, Ulist=U_all)
```

## Choosing alpha (z-scores vs. raw effects)

The `alpha` parameter in `mash_set_data()` controls how effects are
standardized before modeling. In practice there are two choices:

| | `alpha=0` (raw effects) | `alpha=1` (z-scores) |
|---|---|---|
| **Assumption** | True effect sizes are comparable across markers regardless of SE | Standardized effects (z-scores) are comparable |
| **Best for** | Designed experiments with equal sample sizes per condition (e.g., RNA-seq with balanced design) | GWAS/eQTL where SEs vary widely across markers due to allele frequency or sample size differences |
| **In practice** | Use when Shat is constant or nearly constant across markers | Use when Shat varies a lot (the common case for GWAS) |

For most GWAS analyses, convert to z-scores before calling mash:

```python
Zhat = Bhat / Shat
data = mash.mash_set_data(Zhat, np.ones_like(Zhat))
# Posterior means are in z-score space; convert back: pm_effect = pm_z * Shat
```

This is equivalent to `mash_set_data(Bhat, Shat, alpha=1)`.

## mash_1by1 vs. mash

- **`mash_1by1(data)`** — Fast, no cross-condition shrinkage. Treats each
  condition independently (like running R conditions of univariate tests).
  Use this only as a screening step to identify strong signals for
  covariance learning. Its results should not be used as final estimates.

- **`mash(data, Ulist=...)`** — The real model. Learns sharing patterns
  and produces shrunken multi-condition posteriors. This is what you
  report in your paper.

## Example Notebooks

See the [`examples/`](examples/) directory for Jupyter notebooks covering:

1. **[Introduction](examples/01_introduction.ipynb)** — Full 4-step workflow with canonical covariances
2. **[Data-Driven Covariances](examples/02_data_driven_covariances.ipynb)** — PCA and Extreme Deconvolution
3. **[eQTL Workflow](examples/03_eqtl_workflow.ipynb)** — Scaling to real data with separate strong/random subsets
4. **[Posterior Sampling](examples/04_posterior_sampling.ipynb)** — Drawing posterior samples and computing sharing
5. **[GWAS to Pleiotropic Markers](examples/05_gwas_to_mash.ipynb)** — Complete pipeline from VCF + phenotypes through per-trait GWAS (PANICLE) to multi-trait mash analysis

## Loading Your Own Data

pymash takes two numpy arrays: **Bhat** (effect sizes) and **Shat**
(standard errors), both shaped `(J, R)` where J is the number of
tests (markers, genes, ...) and R is the number of conditions (tissues,
traits, ...). If your GWAS or eQTL tool writes summary statistics to a
CSV or TSV, loading them looks like this:

```python
import numpy as np
import pandas as pd
import pymash as mash

# Suppose you ran GWAS for 5 traits and saved results per trait
trait_files = ["trait1.csv", "trait2.csv", "trait3.csv", "trait4.csv", "trait5.csv"]

frames = []
for i, f in enumerate(trait_files):
    df = pd.read_csv(f)           # columns: SNP, BETA, SE, ...
    frame = df.set_index("SNP")[["BETA", "SE"]].rename(
        columns={"BETA": f"BETA_{i}", "SE": f"SE_{i}"}
    )
    frames.append(frame)

# Align on SNP IDs and stack into J x R matrices
merged = frames[0]
for frame in frames[1:]:
    merged = merged.join(frame, how="inner")

beta_cols = [f"BETA_{i}" for i in range(len(trait_files))]
se_cols = [f"SE_{i}" for i in range(len(trait_files))]
Bhat = merged[beta_cols].to_numpy()
Shat = merged[se_cols].to_numpy()

# For GWAS data with varying SEs, convert to z-scores (standard practice)
Zhat = Bhat / Shat
data = mash.mash_set_data(Zhat, np.ones_like(Zhat))
```

See [notebook 03](examples/03_eqtl_workflow.ipynb) for the full eQTL
workflow and [notebook 05](examples/05_gwas_to_mash.ipynb) for a
complete GWAS pipeline.

## Installation

```bash
pip install pymash
```

pymash includes a compiled C++ extension that is built automatically from
source during installation. This requires:

- A **C++ compiler** with C++17 support (GCC >= 7, Clang >= 5, or MSVC 2017+)
- **pybind11** (installed automatically as a build dependency)
- On **macOS**, OpenMP support is provided via `libomp` from Homebrew
  (`brew install libomp`). This is optional — pymash will build without it
  but will not use multi-threading.

Pre-built wheels (which need no compiler) are available for common
platforms via `pip install pymash` once published to PyPI.

For development:

```bash
# Install editable with test dependencies
python -m pip install -e '.[test]'

# Run tests
python -m pytest

# Build wheel and source distribution
python -m build --sdist --wheel .
```

## Command-Line Interface

After installation, the `pymash` CLI is available. Run `pymash --help` for
the full list of commands and options.

### Minimal CLI tutorial

Suppose you have per-trait GWAS summary statistics already merged into two
matrices — `bhat.csv` (effect sizes, J rows x R columns) and `shat.csv`
(standard errors, same shape). Input formats: `.npy`, `.npz` (single-array),
`.csv`, `.tsv`.

```bash
# Step 1: Fit the mash model (canonical covariances, default settings)
pymash fit \
  --bhat bhat.csv \
  --shat shat.csv \
  --out my_result

# This writes my_result.npz (arrays) and my_result.json (summary metadata).
```

`pymash fit` now defaults to chunked large-scale execution with
`--chunk-size 250000`. For very large `J`, it automatically runs a two-stage
train/apply workflow to avoid out-of-memory failures, then applies the fixed
prior in chunks. Override with `--chunk-size`, and set `--chunk-size 0` to
disable chunking.

Load and interpret the outputs in Python:

```python
import numpy as np, json

res = np.load("my_result.npz")
pm   = res["posterior_mean"]   # Shrunken effect sizes (J x R)
lfsr = res["lfsr"]             # Local false sign rate (J x R)

# Which markers are significant in at least one condition?
sig = np.any(lfsr < 0.05, axis=1)
print(f"{sig.sum()} significant markers out of {pm.shape[0]}")

with open("my_result.json") as f:
    meta = json.load(f)
print(f"Log-likelihood: {meta['loglik']:.1f}")
```

### Other commands

```bash
# Run mash_1by1 baseline (screening step for covariance learning)
pymash onebyone --bhat bhat.csv --shat shat.csv --out onebyone_out

# Estimate null correlation matrix
pymash estimate-null-corr-simple --bhat bhat.csv --shat shat.csv --out vhat.npy
```

### `--cov-methods`

The `fit` command accepts `--cov-methods` (comma-separated) to choose which
canonical covariance patterns to include. Accepted values:

| Value | Meaning |
|-------|---------|
| `identity` | Independent effects (diagonal covariance) |
| `singletons` | One condition-specific matrix per condition |
| `equal_effects` | All conditions share the same effect |
| `simple_het` | Heterogeneous sharing at correlations 0.25, 0.5, 0.75 |

Default: `--cov-methods identity,singletons,equal_effects,simple_het`
(all four, matching `cov_canonical(data)` in the Python API).

### Output file schema

**`<out>.npz`** — NumPy compressed archive. Load with `np.load("result.npz")`.

| Key | Shape | Always present | Description |
|-----|-------|----------------|-------------|
| `posterior_mean` | `(J, R)` | outputlevel >= 2 | Shrunken effect-size estimates |
| `posterior_sd` | `(J, R)` | outputlevel >= 2 | Posterior standard deviations |
| `lfsr` | `(J, R)` | outputlevel >= 2 | Local false sign rate |
| `lfdr` | `(J, R)` | if `--output-lfdr` | Local false discovery rate |
| `negative_prob` | `(J, R)` | outputlevel >= 2 | P(true effect < 0) |
| `posterior_cov` | `(J, R, R)` | outputlevel >= 3 | Full posterior covariance per effect |
| `posterior_samples` | `(J, R, M)` | if `--posterior-samples M` | Posterior draws |
| `pi` | `(K,)` | always | Fitted mixture weights |
| `grid` | `(G,)` | always | Grid scaling factors |
| `fitted_u_stack` | `(P, R, R)` | always | Base covariance matrices (before grid) |
| `posterior_weights` | `(J, K)` | if available | Per-effect component responsibilities (omitted in chunked apply mode) |
| `vloglik` | `(J,)` | always | Per-effect log-likelihoods |
| `lik_matrix` | `(J, K)` | outputlevel >= 4 | Full likelihood matrix |
| `null_loglik` | scalar | always | Log-likelihood under null model |
| `alt_loglik` | scalar | always | Log-likelihood under fitted model |

**`<out>.json`** — Human-readable summary metadata.

| Key | Type | Description |
|-----|------|-------------|
| `command` | string | Command that produced this file (`"fit"` or `"onebyone"`) |
| `loglik` | float | Total log-likelihood of the fitted model |
| `n_effects` | int | Number of effects (J) |
| `n_components_active` | int | Number of mixture components after thresholding |
| `alpha` | float | Alpha scaling used (0 = raw effects, 1 = z-scores) |
| `usepointmass` | bool | Whether a null point-mass component was included |

## Troubleshooting and Common Pitfalls

### Build / install issues

**"pymash._edcpp backend is not available" warning at import:**
This means the C++ extension did not compile during installation. The
core fitting function (`mash()`) requires this extension. To fix:

1. Ensure you have a C++ compiler installed (`gcc --version` or
   `clang --version`).
2. On macOS, install Xcode command line tools: `xcode-select --install`.
3. Reinstall: `pip install --force-reinstall pymash` (or `pip install -e .`
   for development).

**"matplotlib is required for plotting":** Install the plotting extra:
`pip install pymash[plot]`.

### Statistical / methodological pitfalls

**Fitting on significant markers only:**
The most common mistake. If you select markers by significance and then
fit mash on that subset, the mixture weights will be wrong and
posteriors will be miscalibrated. Fit on all markers, a random subset,
or a mixed topz+random subset (`select_method="topz_random"`). Do not
fit on only significance-selected markers. Use strong signals for
covariance learning (`cov_pca`, `cov_ed`).

**Too few conditions (R < 3):**
mash is designed for multivariate problems. With only 2 conditions,
the covariance structure is a single correlation coefficient and there
is little to learn. Consider whether a simpler method (e.g., `ashr`)
would suffice.

**Wrong alpha for your data type:**
If your standard errors vary widely across markers (common in GWAS) and
you use `alpha=0`, the model may not shrink appropriately because it
treats all markers as having comparable effect-size priors. Use
`alpha=1` (z-scores) for GWAS. See "Choosing alpha" above.

**Grid too coarse:**
The default `gridmult=sqrt(2)` works well in most cases. If you see
poor fit (low log-likelihood) or unexpected results, try a finer grid
with `gridmult=1.25`.

## API Overview

**Data setup:**
`mash_set_data`, `mash_update_data`, `contrast_matrix`

**Covariance matrices:**
`cov_canonical`, `cov_pca`, `cov_ed`

**Model fitting:**
`mash` (main entry point), `mash_1by1` (screening step),
`mash_compute_posterior_matrices` (apply fitted model to new data)

**Large-scale workflow:**
`fit_mash_prior`, `apply_mash_prior`, `apply_mash_prior_chunked`, `mash_train_apply`

**Result extraction:**
`get_pm`, `get_psd`, `get_lfsr`, `get_lfdr`,
`get_significant_results`, `get_n_significant_conditions`,
`get_estimated_pi`, `get_pairwise_sharing`, `get_log10bf`

**Correlation estimation:**
`estimate_null_correlation_simple`, `mash_estimate_corr_em`

**Plotting:** `mash_plot_meta` (requires `pip install pymash[plot]`)

**Simulation:** `simple_sims`, `simple_sims2`

**Advanced** (not exported at top level — import from submodules if needed):
`pymash.covariances.expand_cov`, `scale_cov`, `normalize_Ulist`;
`pymash.likelihoods.calc_lik_matrix`, `calc_relative_lik_matrix`

## Test CI

Push/PR test runs are configured in:

- `.github/workflows/ci.yml`

## Wheel CI

Cross-platform wheel builds are configured via `cibuildwheel` and GitHub Actions in:

- `.github/workflows/wheels.yml`

That workflow also performs a smoke test by installing the just-built wheel
and running a minimal `mash()` fit on each platform.

## PyPI Release

Tag-driven release publishing (wheels + sdist) is configured in:

- `.github/workflows/release.yml`

To publish:

1. Configure PyPI Trusted Publishing for this repository.
2. Create and push a tag like `v0.1.0`.

## Citation

If you use pymash in your research, please cite:

> Urbut, S.M., Wang, G., Carbonetto, P. & Stephens, M. (2019).
> Flexible statistical methods for estimating and testing effects in genomic
> studies with multiple conditions. *Nature Genetics*, 51, 187–195.
> https://doi.org/10.1038/s41588-018-0268-8
