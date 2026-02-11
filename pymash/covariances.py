from __future__ import annotations

from collections.abc import Iterable
from itertools import product
import warnings

import numpy as np

from .data import MashData
from .ed import bovy_wrapper, teem_wrapper


def _to_ulist(Ulist: dict[str, np.ndarray] | Iterable[np.ndarray]) -> list[np.ndarray]:
    if isinstance(Ulist, dict):
        mats = list(Ulist.values())
    else:
        mats = list(Ulist)
    if not mats:
        raise ValueError("Ulist cannot be empty")
    return [np.asarray(u, dtype=float) for u in mats]


def cov_identity(data: MashData) -> np.ndarray:
    return np.eye(data.n_conditions, dtype=float)


def cov_all_zeros(data: MashData) -> np.ndarray:
    return np.zeros((data.n_conditions, data.n_conditions), dtype=float)


def cov_singletons(data: MashData) -> list[np.ndarray]:
    R = data.n_conditions
    out: list[np.ndarray] = []
    for r in range(R):
        mat = np.zeros((R, R), dtype=float)
        mat[r, r] = 1.0
        out.append(mat)
    return out


def cov_equal_effects(data: MashData) -> np.ndarray:
    R = data.n_conditions
    return np.ones((R, R), dtype=float)


def cov_simple_het(data: MashData, corr: Iterable[float] = (0.25, 0.5, 0.75)) -> list[np.ndarray]:
    R = data.n_conditions
    mats: list[np.ndarray] = []
    for c in corr:
        if c < -1.0 or c > 1.0:
            raise ValueError("corr must be between -1 and 1")
        mat = np.full((R, R), float(c), dtype=float)
        np.fill_diagonal(mat, 1.0)
        mats.append(mat)
    return mats


def udi_model_matrix(R: int) -> np.ndarray:
    if R <= 0:
        raise ValueError("R must be positive")
    rows = [tuple(reversed(row)) for row in product(("U", "D", "I"), repeat=R)]
    rows = [row for row in rows if "D" in row]
    return np.asarray(rows, dtype=object)


def _normalize_udi_model(model: np.ndarray | list[str] | tuple[str, ...], R: int) -> np.ndarray:
    arr = np.asarray(model, dtype=object)
    if arr.ndim == 1:
        if arr.shape[0] != R:
            raise ValueError("model must be vector of length R with elements U, D, I")
        arr = arr.reshape(1, R)
    if arr.ndim != 2 or arr.shape[1] != R:
        raise ValueError("model must be vector of length R with elements U, D, I")
    return arr


def _cov_udi_single(data: MashData, model_row: np.ndarray) -> np.ndarray:
    R = data.n_conditions
    if data.V.ndim != 2:
        raise ValueError("cov_udi currently requires a common 2D correlation matrix V")
    V = np.asarray(data.V, dtype=float)
    if V.shape != (R, R):
        raise ValueError("data V has invalid dimensions")

    model_vec = np.asarray(model_row, dtype=object)
    if model_vec.shape != (R,):
        raise ValueError("model must be vector of length R with elements U, D, I")

    D = np.where(model_vec == "D")[0]
    U = np.where(model_vec == "U")[0]
    I = np.where(model_vec == "I")[0]
    if (len(D) + len(U) + len(I)) != R:
        raise ValueError("model must be vector of length R with elements U, D, I")
    if len(D) == 0:
        raise ValueError("model must have at least one direct association")

    res = np.zeros((R, R), dtype=float)

    VDD = V[np.ix_(D, D)]
    if len(U) > 0:
        VDU = V[np.ix_(D, U)]
        VUU = V[np.ix_(U, U)]
        U0 = VDD - VDU @ np.linalg.solve(VUU, VDU.T)
    else:
        U0 = VDD
    res[np.ix_(D, D)] = U0

    Ic = np.concatenate([U, D])
    if len(I) > 0:
        VIIc = V[np.ix_(I, Ic)]
        VIcIc = V[np.ix_(Ic, Ic)]
        inv_vicic = np.linalg.solve(VIcIc, np.eye(len(Ic), dtype=float))
        BD = inv_vicic[:, len(U) : len(U) + len(D)]
        id_block = VIIc @ BD @ U0
        res[np.ix_(I, D)] = id_block
        res[np.ix_(D, I)] = id_block.T
        res[np.ix_(I, I)] = VIIc @ BD @ U0 @ BD.T @ VIIc.T

    return res


def cov_udi(
    data: MashData,
    model: np.ndarray | list[str] | tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    R = data.n_conditions
    model_matrix = udi_model_matrix(R) if model is None else _normalize_udi_model(model, R)
    out: dict[str, np.ndarray] = {}
    for row in model_matrix:
        key = "cov_udi_" + "".join(str(x) for x in row)
        out[key] = _cov_udi_single(data, row)
    return out


def cov_canonical(
    data: MashData,
    methods: Iterable[str] = ("identity", "singletons", "equal_effects", "simple_het"),
) -> dict[str, np.ndarray]:
    """Build a set of canonical covariance matrices for mash.

    Generates a named dictionary of covariance matrices representing
    common effect-sharing patterns: identity (independent effects),
    singletons (condition-specific), equal effects (fully shared),
    and simple heterogeneous (shared with varying correlation).

    Parameters
    ----------
    data : MashData
        Data object (used to determine the number of conditions).
    methods : iterable of str
        Which canonical types to include. Options: ``"identity"``,
        ``"singletons"``, ``"equal_effects"``, ``"simple_het"``,
        ``"null"``.

    Returns
    -------
    dict of str to np.ndarray
        Named covariance matrices, each of shape ``(R, R)``.

    Examples
    --------
    >>> U_c = cov_canonical(data)
    >>> list(U_c.keys())[:3]
    ['identity', 'singleton_1', 'singleton_2']
    """
    out: dict[str, np.ndarray] = {}
    for method in methods:
        key = method.lower()
        if key == "identity":
            out["identity"] = cov_identity(data)
        elif key == "singletons":
            for i, mat in enumerate(cov_singletons(data), start=1):
                out[f"singleton_{i}"] = mat
        elif key == "equal_effects":
            out["equal_effects"] = cov_equal_effects(data)
        elif key == "simple_het":
            for i, mat in enumerate(cov_simple_het(data), start=1):
                out[f"simple_het_{i}"] = mat
        elif key == "null":
            out["null"] = cov_all_zeros(data)
        else:
            raise ValueError(f"Unknown covariance method: {method}")
    return out


def r1cov(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    return np.outer(x, x)


def cov_from_factors(f: np.ndarray, name: str) -> dict[str, np.ndarray]:
    f = np.asarray(f, dtype=float)
    if f.ndim != 2:
        raise ValueError("f must be 2D")
    out: dict[str, np.ndarray] = {}
    for i in range(f.shape[0]):
        out[f"{name}_{i + 1}"] = r1cov(f[i])
    return out


def normalize_cov(U: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    max_diag = float(np.max(np.diag(U)))
    if max_diag == 0.0:
        return U.copy()
    return U / max_diag


def normalize_Ulist(Ulist: dict[str, np.ndarray] | Iterable[np.ndarray]) -> list[np.ndarray]:
    mats = _to_ulist(Ulist)
    return [normalize_cov(U) for U in mats]


def scale_cov(Ulist: dict[str, np.ndarray] | Iterable[np.ndarray], grid: Iterable[float]) -> list[np.ndarray]:
    mats = _to_ulist(Ulist)
    grid_arr = np.asarray(list(grid), dtype=float)
    if grid_arr.ndim != 1 or grid_arr.size == 0:
        raise ValueError("grid must be a non-empty 1D array")
    out: list[np.ndarray] = []
    for g in grid_arr:
        out.extend([(g * g) * U for U in mats])
    return out


def expand_cov(
    Ulist: dict[str, np.ndarray] | Iterable[np.ndarray],
    grid: Iterable[float],
    usepointmass: bool = True,
) -> list[np.ndarray]:
    """Scale covariance matrices by grid and optionally prepend a null.

    Produces the full list of covariance matrices used internally by
    :func:`~pymash.mash.mash`: each base matrix scaled by each grid
    value squared, with an optional zero matrix (point mass) prepended.

    Parameters
    ----------
    Ulist : dict or iterable of np.ndarray
        Base covariance matrices.
    grid : iterable of float
        Scaling factors.
    usepointmass : bool
        Whether to prepend a zero (null) covariance matrix.

    Returns
    -------
    list of np.ndarray
        Expanded list of covariance matrices.
    """
    mats = scale_cov(Ulist, grid)
    if usepointmass:
        R = mats[0].shape[0]
        mats = [np.zeros((R, R), dtype=float)] + mats
    return mats


def cov_pca(data: MashData, npc: int, subset: np.ndarray | list[int] | None = None) -> dict[str, np.ndarray]:
    """Compute data-driven covariance matrices using PCA.

    Performs SVD on (a subset of) the effect-size matrix to produce
    rank-1 covariance matrices from each of the top principal components,
    plus a combined "tPCA" matrix.

    Parameters
    ----------
    data : MashData
        Data object.
    npc : int
        Number of principal components to use (must be > 1 and <= R).
    subset : array-like of int, optional
        Row indices to use for PCA (e.g., indices of strong signals).
        Defaults to all effects.

    Returns
    -------
    dict of str to np.ndarray
        Named covariance matrices: ``"PCA_1"`` through ``"PCA_{npc}"``
        plus ``"tPCA"``.

    Examples
    --------
    >>> strong = get_significant_results(m1, thresh=0.05)
    >>> U_pca = cov_pca(data, npc=5, subset=strong)
    """
    if npc <= 1:
        raise ValueError("npc must be > 1")
    if npc > data.n_conditions:
        raise ValueError("npc cannot exceed the number of conditions")

    if subset is None:
        subset_idx = np.arange(data.n_effects)
    else:
        subset_idx = np.asarray(subset, dtype=int)
    if subset_idx.size == 0:
        raise ValueError("subset cannot be empty")
    min_recommended = int(data.n_conditions * 20)
    if subset_idx.size < min_recommended:
        warnings.warn(
            f"Only {subset_idx.size} effects for {data.n_conditions} conditions. "
            "Data-driven covariances from PCA may be unstable; consider canonical covariances.",
            RuntimeWarning,
            stacklevel=2,
        )

    X = data.Bhat[subset_idx]
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    f = vt[:npc].T

    Ulist = cov_from_factors(f.T, "PCA")
    d2 = np.diag(s[:npc] ** 2)
    Ulist["tPCA"] = f @ d2 @ f.T / float(subset_idx.size)
    return Ulist


def cov_ed(
    data: MashData,
    Ulist_init: dict[str, np.ndarray] | Iterable[np.ndarray],
    subset: np.ndarray | list[int] | None = None,
    algorithm: str = "teem",
    **kwargs,
) -> dict[str, np.ndarray]:
    """Refine covariance matrices using Extreme Deconvolution (ED).

    Takes initial covariance estimates (e.g., from :func:`cov_pca`) and
    refines them by fitting a mixture of multivariate normals to the
    observed data using the ED algorithm, which accounts for measurement
    error.

    Parameters
    ----------
    data : MashData
        Data object.
    Ulist_init : dict or iterable of np.ndarray
        Initial covariance matrices to refine.
    subset : array-like of int, optional
        Row indices to use for ED fitting (e.g., strong signals).
    algorithm : str
        ED algorithm: ``"teem"`` (default) or ``"bovy"``.
    **kwargs
        Additional arguments passed to the ED algorithm.

    Returns
    -------
    dict of str to np.ndarray
        Refined covariance matrices, named ``"ED_{original_name}"``.

    Examples
    --------
    >>> U_pca = cov_pca(data, npc=5, subset=strong)
    >>> U_ed = cov_ed(data, U_pca, subset=strong)
    """
    names: list[str]
    if isinstance(Ulist_init, dict):
        names = list(Ulist_init.keys())
        mats = list(Ulist_init.values())
    else:
        mats = list(Ulist_init)
        names = [str(i + 1) for i in range(len(mats))]
    if not mats:
        raise ValueError("Ulist_init cannot be empty")
    subset_size = data.n_effects if subset is None else int(np.asarray(subset).size)
    min_recommended = int(data.n_conditions * 20)
    if subset_size < min_recommended:
        warnings.warn(
            f"Only {subset_size} effects for {data.n_conditions} conditions. "
            "Extreme Deconvolution may be unstable with so few effects.",
            RuntimeWarning,
            stacklevel=2,
        )

    algo = algorithm.lower()
    if algo == "teem":
        result = teem_wrapper(data, mats, subset=subset, **kwargs)
        U_ed = result.U
    elif algo == "bovy":
        out = bovy_wrapper(data, mats, subset=subset, **kwargs)
        U_ed = out["Ulist"]
    else:
        raise ValueError("algorithm must be one of {'teem', 'bovy'}")

    return {f"ED_{name}": U for name, U in zip(names, U_ed)}


def cov_flash(*args, **kwargs):
    raise NotImplementedError("cov_flash is optional and not yet implemented")


__all__ = [
    "cov_canonical",
    "cov_identity",
    "cov_singletons",
    "cov_equal_effects",
    "cov_simple_het",
    "r1cov",
    "cov_from_factors",
    "cov_all_zeros",
    "normalize_cov",
    "normalize_Ulist",
    "scale_cov",
    "expand_cov",
    "cov_pca",
    "cov_ed",
    "cov_udi",
    "cov_flash",
]
