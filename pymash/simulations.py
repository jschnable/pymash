from __future__ import annotations

import numpy as np


def simple_sims(nsamp: int = 100, ncond: int = 5, err_sd: float = 0.01, seed: int | None = None) -> dict[str, np.ndarray]:
    """Simulate data with four effect types for testing mash.

    Generates ``4 * nsamp`` effects across ``ncond`` conditions with four
    types: null, independent, condition-1-specific, and equal (shared).

    Parameters
    ----------
    nsamp : int
        Number of effects per type (total effects = 4 * nsamp).
    ncond : int
        Number of conditions.
    err_sd : float
        Standard error applied to all observations.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``{"B": true_effects, "Bhat": observed, "Shat": standard_errors}``,
        each an ``(4*nsamp, ncond)`` array.

    Examples
    --------
    >>> sim = simple_sims(500, ncond=5, err_sd=1.0, seed=42)
    >>> sim["Bhat"].shape
    (2000, 5)
    """
    rng = np.random.default_rng(seed)

    B_id = rng.normal(size=(nsamp, ncond))
    b = rng.normal(size=nsamp)
    B_all = np.tile(b[:, None], (1, ncond))
    B_zero = np.zeros((nsamp, ncond), dtype=float)
    B_one = np.zeros((nsamp, ncond), dtype=float)
    B_one[:, 0] = rng.normal(size=nsamp)

    B = np.vstack([B_zero, B_id, B_one, B_all])
    Shat = np.full_like(B, float(err_sd))
    E = rng.normal(loc=0.0, scale=Shat)
    Bhat = B + E

    return {"B": B, "Bhat": Bhat, "Shat": Shat}


def simple_sims2(nsamp: int = 100, err_sd: float = 0.01, seed: int | None = None) -> dict[str, np.ndarray]:
    """Simulate data with two block-structured effect types.

    Generates ``2 * nsamp`` effects across 5 conditions. Block 1 has shared
    effects in conditions 1-2; block 2 has shared effects in conditions 3-5.

    Parameters
    ----------
    nsamp : int
        Number of effects per block (total effects = 2 * nsamp).
    err_sd : float
        Standard error applied to all observations.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``{"B": true_effects, "Bhat": observed, "Shat": standard_errors}``,
        each a ``(2*nsamp, 5)`` array.

    Examples
    --------
    >>> sim = simple_sims2(200, err_sd=1.0, seed=42)
    >>> sim["Bhat"].shape
    (400, 5)
    """
    rng = np.random.default_rng(seed)

    ncond = 5
    b1 = rng.normal(size=nsamp)
    B1 = np.column_stack([b1, b1, np.zeros(nsamp), np.zeros(nsamp), np.zeros(nsamp)])

    b2 = rng.normal(size=nsamp)
    B2 = np.column_stack([np.zeros(nsamp), np.zeros(nsamp), b2, b2, b2])

    B = np.vstack([B1, B2])
    Shat = np.full_like(B, float(err_sd))
    E = rng.normal(loc=0.0, scale=Shat)
    Bhat = B + E

    return {"B": B, "Bhat": Bhat, "Shat": Shat}


__all__ = ["simple_sims", "simple_sims2"]
