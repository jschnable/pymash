from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import MashData
from .mash import FittedG, MashResult, mash


@dataclass
class TrainApplyResult:
    train_indices: np.ndarray
    train_result: MashResult
    apply_result: MashResult


@dataclass
class ChunkedApplyResult:
    arrays: dict[str, np.ndarray]
    output_paths: dict[str, str]
    loglik: float
    n_effects: int
    n_conditions: int
    chunk_size: int
    n_chunks: int
    alpha: float
    fitted_g: FittedG


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


def _subset_mash_data_slice(data: MashData, start: int, stop: int) -> MashData:
    if start < 0 or stop > data.n_effects or start >= stop:
        raise ValueError("invalid slice bounds")

    def _slice_optional_rows(arr: np.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        out = np.asarray(arr)
        if out.ndim >= 1 and out.shape[0] == data.n_effects:
            return np.array(out[start:stop], copy=True)
        return np.array(out, copy=True)

    if data.common_V:
        v_sub = np.array(data.V, copy=True)
    else:
        v_sub = np.array(data.V[start:stop], copy=True)

    return MashData(
        Bhat=np.array(data.Bhat[start:stop], copy=True),
        Shat=np.array(data.Shat[start:stop], copy=True),
        Shat_alpha=np.array(data.Shat_alpha[start:stop], copy=True),
        V=v_sub,
        common_V=data.common_V,
        alpha=data.alpha,
        L=_slice_optional_rows(data.L),
        Shat_orig=_slice_optional_rows(data.Shat_orig),
        LSVSLt=_slice_optional_rows(data.LSVSLt),
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
    chunk_size: int | None = 250_000,
    mash_kwargs: dict | None = None,
) -> MashResult:
    """Apply a pre-fitted mash prior to full data."""
    kwargs = dict(mash_kwargs or {})
    kwargs.setdefault("chunk_size", chunk_size)
    return mash(data, g=fitted_g, fixg=True, **kwargs)


def apply_mash_prior_chunked(
    data: MashData,
    fitted_g: FittedG,
    *,
    chunk_size: int = 250_000,
    mash_kwargs: dict | None = None,
    out_prefix: str | Path | None = None,
) -> ChunkedApplyResult:
    """Apply a pre-fitted prior in chunks to reduce peak memory usage.

    This function is designed for large ``J`` where a full in-memory
    application would materialize very large ``J x P`` intermediates.

    Parameters
    ----------
    data
        Full mash data to evaluate.
    fitted_g
        Fitted prior from a previous mash run.
    chunk_size
        Number of effects to process per chunk.
    mash_kwargs
        Additional kwargs forwarded to :func:`~pymash.mash.mash`.
        ``g`` and ``fixg`` are controlled internally and cannot be set.
    out_prefix
        If provided, outputs are written as ``<out_prefix>.<name>.npy``
        via ``numpy`` memmap and returned as mapped arrays.

    Returns
    -------
    ChunkedApplyResult
        Chunk-wise output arrays and summary metadata.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    kwargs = dict(mash_kwargs or {})
    kwargs.setdefault("chunk_size", None)
    if "g" in kwargs or "fixg" in kwargs:
        raise ValueError("mash_kwargs must not include g or fixg")

    outputlevel = int(kwargs.get("outputlevel", 2))
    if outputlevel < 1 or outputlevel > 2:
        raise ValueError("apply_mash_prior_chunked currently supports outputlevel 1 or 2")
    if int(kwargs.get("posterior_samples", 0)) > 0:
        raise ValueError("apply_mash_prior_chunked does not support posterior_samples > 0")

    J = data.n_effects
    R = data.n_conditions
    n_chunks = int(np.ceil(J / chunk_size))
    prefix = Path(out_prefix) if out_prefix is not None else None
    if prefix is not None:
        prefix.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    output_paths: dict[str, str] = {}

    def _alloc(name: str, shape: tuple[int, ...]) -> np.ndarray:
        if prefix is None:
            return np.empty(shape, dtype=float)
        path = Path(f"{str(prefix)}.{name}.npy")
        output_paths[name] = str(path)
        return np.lib.format.open_memmap(path, mode="w+", dtype=float, shape=shape)

    total_loglik = 0.0
    for start in range(0, J, chunk_size):
        stop = min(start + chunk_size, J)
        chunk_data = _subset_mash_data_slice(data, start, stop)
        chunk = mash(chunk_data, g=fitted_g, fixg=True, **kwargs)
        total_loglik += float(chunk.loglik)

        chunk_arrays: dict[str, np.ndarray | None] = {
            "vloglik": chunk.vloglik,
            "posterior_mean": chunk.posterior_mean,
            "posterior_sd": chunk.posterior_sd,
            "lfsr": chunk.lfsr,
            "lfdr": chunk.lfdr,
            "negative_prob": chunk.negative_prob,
            "null_loglik": chunk.null_loglik,
            "alt_loglik": chunk.alt_loglik,
        }

        if not arrays:
            for name, arr in chunk_arrays.items():
                if arr is None:
                    continue
                arr_np = np.asarray(arr, dtype=float)
                if arr_np.ndim == 1:
                    shape = (J,)
                elif arr_np.ndim == 2:
                    shape = (J, arr_np.shape[1])
                else:
                    raise ValueError(f"Unsupported output shape for {name}: {arr_np.shape}")
                arrays[name] = _alloc(name, shape)

        for name, buf in arrays.items():
            val = chunk_arrays[name]
            if val is None:
                continue
            val_np = np.asarray(val, dtype=float)
            if val_np.ndim == 1:
                buf[start:stop] = val_np
            else:
                buf[start:stop, :] = val_np

    for arr in arrays.values():
        if isinstance(arr, np.memmap):
            arr.flush()

    return ChunkedApplyResult(
        arrays=arrays,
        output_paths=output_paths,
        loglik=total_loglik,
        n_effects=J,
        n_conditions=R,
        chunk_size=chunk_size,
        n_chunks=n_chunks,
        alpha=data.alpha,
        fitted_g=fitted_g,
    )


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
    apply_chunk_size: int | None = 250_000,
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
        chunk_size=apply_chunk_size,
        mash_kwargs=apply_mash_kwargs,
    )
    return TrainApplyResult(
        train_indices=idx,
        train_result=train_result,
        apply_result=apply_result,
    )


__all__ = [
    "TrainApplyResult",
    "ChunkedApplyResult",
    "select_training_effects",
    "fit_mash_prior",
    "apply_mash_prior",
    "apply_mash_prior_chunked",
    "mash_train_apply",
]
