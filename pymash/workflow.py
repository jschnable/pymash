from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import MashData
from .mash import FittedG, MashResult, mash


@dataclass
class TrainApplyResult:
    train_indices: np.ndarray
    train_result: MashResult
    apply_result: MashResult


def _subset_mash_data(data: MashData, indices: np.ndarray) -> MashData:
    idx = np.asarray(indices, dtype=int)
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError("indices must be a non-empty 1D integer array")
    if np.any(idx < 0) or np.any(idx >= data.n_effects):
        raise ValueError("indices out of bounds")

    def _subset_optional_rows(arr: np.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        out = np.asarray(arr)
        if out.ndim >= 1 and out.shape[0] == data.n_effects:
            return np.array(out[idx], copy=True)
        return np.array(out, copy=True)

    if data.common_V:
        v_sub = np.array(data.V, copy=True)
    else:
        v_sub = np.array(data.V[idx], copy=True)

    return MashData(
        Bhat=np.array(data.Bhat[idx], copy=True),
        Shat=np.array(data.Shat[idx], copy=True),
        Shat_alpha=np.array(data.Shat_alpha[idx], copy=True),
        V=v_sub,
        common_V=data.common_V,
        alpha=data.alpha,
        L=_subset_optional_rows(data.L),
        Shat_orig=_subset_optional_rows(data.Shat_orig),
        LSVSLt=_subset_optional_rows(data.LSVSLt),
    )


def select_training_effects(
    data: MashData,
    n_train: int,
    method: str = "random",
    seed: int = 123,
    background_fraction: float = 0.2,
) -> np.ndarray:
    """Select effect indices for prior training.

    Parameters
    ----------
    data
        Full mash dataset.
    n_train
        Number of effects to use for training.
    method
        ``"random"`` for unbiased sampling, or ``"topz_random"``
        to include top-|z| effects plus random background.
    seed
        RNG seed for selection.
    background_fraction
        Fraction of random background effects when
        ``method="topz_random"``.
    """
    J = data.n_effects
    if n_train <= 0:
        raise ValueError("n_train must be positive")
    if n_train > J:
        raise ValueError("n_train cannot exceed the number of effects")

    rng = np.random.default_rng(seed)
    normalized = method.lower()

    if normalized == "random":
        idx = rng.choice(J, size=n_train, replace=False)
        return np.sort(idx.astype(int))

    if normalized == "topz_random":
        if background_fraction < 0.0 or background_fraction > 1.0:
            raise ValueError("background_fraction must be in [0, 1]")
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.abs(data.Bhat) / np.maximum(data.Shat, np.finfo(float).tiny)
        z = np.where(np.isfinite(z), z, 0.0)
        score = np.max(z, axis=1)

        n_bg = int(round(background_fraction * n_train))
        n_top = max(0, n_train - n_bg)

        order = np.argsort(-score)
        top_idx = order[:n_top]
        used = np.zeros(J, dtype=bool)
        used[top_idx] = True

        if n_bg > 0:
            pool = np.where(~used)[0]
            bg_idx = rng.choice(pool, size=n_bg, replace=False)
            idx = np.concatenate([top_idx, bg_idx])
        else:
            idx = top_idx
        return np.sort(idx.astype(int))

    raise ValueError("method must be one of {'random', 'topz_random'}")


def fit_mash_prior(
    data: MashData,
    Ulist: dict[str, np.ndarray] | list[np.ndarray],
    *,
    train_indices: np.ndarray | None = None,
    n_train: int | None = None,
    select_method: str = "random",
    select_seed: int = 123,
    background_fraction: float = 0.2,
    mash_kwargs: dict | None = None,
) -> tuple[FittedG, np.ndarray, MashResult]:
    """Fit mash prior on a subset and return fitted g."""
    if train_indices is None:
        if n_train is None:
            raise ValueError("Provide train_indices or n_train")
        train_indices = select_training_effects(
            data,
            n_train=n_train,
            method=select_method,
            seed=select_seed,
            background_fraction=background_fraction,
        )
    else:
        train_indices = np.asarray(train_indices, dtype=int)

    data_train = _subset_mash_data(data, train_indices)
    kwargs = dict(mash_kwargs or {})
    kwargs["outputlevel"] = 1
    train_result = mash(data_train, Ulist=Ulist, **kwargs)
    return train_result.fitted_g, np.sort(train_indices), train_result


def apply_mash_prior(
    data: MashData,
    fitted_g: FittedG,
    *,
    mash_kwargs: dict | None = None,
) -> MashResult:
    """Apply a pre-fitted mash prior to full data."""
    kwargs = dict(mash_kwargs or {})
    return mash(data, g=fitted_g, fixg=True, **kwargs)


def mash_train_apply(
    data: MashData,
    Ulist: dict[str, np.ndarray] | list[np.ndarray],
    *,
    train_indices: np.ndarray | None = None,
    n_train: int | None = None,
    select_method: str = "random",
    select_seed: int = 123,
    background_fraction: float = 0.2,
    train_mash_kwargs: dict | None = None,
    apply_mash_kwargs: dict | None = None,
) -> TrainApplyResult:
    """Two-stage workflow: fit prior on subset, apply to full data."""
    fitted_g, idx, train_result = fit_mash_prior(
        data,
        Ulist,
        train_indices=train_indices,
        n_train=n_train,
        select_method=select_method,
        select_seed=select_seed,
        background_fraction=background_fraction,
        mash_kwargs=train_mash_kwargs,
    )
    apply_result = apply_mash_prior(
        data,
        fitted_g,
        mash_kwargs=apply_mash_kwargs,
    )
    return TrainApplyResult(
        train_indices=idx,
        train_result=train_result,
        apply_result=apply_result,
    )


__all__ = [
    "TrainApplyResult",
    "select_training_effects",
    "fit_mash_prior",
    "apply_mash_prior",
    "mash_train_apply",
]
