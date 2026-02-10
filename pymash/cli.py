from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .correlation import estimate_null_correlation_simple
from .covariances import cov_canonical
from .data import mash_set_data
from .mash import mash, mash_1by1


def _guess_delimiter(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return ","
    if suffix in {".tsv", ".tab"}:
        return "\t"
    return None


def _load_array(path_str: str) -> np.ndarray:
    path = Path(path_str)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as zf:
            if len(zf.files) != 1:
                raise ValueError(f"{path} contains multiple arrays; expected exactly one")
            arr = zf[zf.files[0]]
    else:
        arr = np.loadtxt(path, delimiter=_guess_delimiter(path))
    return np.asarray(arr, dtype=float)


def _load_2d_matrix(path_str: str, name: str) -> np.ndarray:
    arr = _load_array(path_str)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D; got shape {arr.shape}")
    return np.asarray(arr, dtype=float)


def _load_v(path_str: str) -> np.ndarray:
    arr = _load_array(path_str)
    if arr.ndim not in {2, 3}:
        raise ValueError(f"V must be 2D or 3D; got shape {arr.shape}")
    return np.asarray(arr, dtype=float)


def _load_ulist(path_str: str, expected_r: int) -> list[np.ndarray]:
    path = Path(path_str)
    suffix = path.suffix.lower()
    ulist: list[np.ndarray] = []

    if suffix == ".npy":
        arr = np.load(path)
        if arr.ndim != 3:
            raise ValueError(f".npy Ulist must be 3D (K, R, R); got shape {arr.shape}")
        ulist = [np.asarray(arr[k], dtype=float) for k in range(arr.shape[0])]
    elif suffix == ".npz":
        with np.load(path) as zf:
            if "u_stack" in zf.files:
                arr = np.asarray(zf["u_stack"], dtype=float)
                if arr.ndim != 3:
                    raise ValueError(f"u_stack must be 3D (K, R, R); got shape {arr.shape}")
                ulist = [arr[k] for k in range(arr.shape[0])]
            else:
                for key in zf.files:
                    mat = np.asarray(zf[key], dtype=float)
                    if mat.ndim != 2:
                        raise ValueError(f"Ulist[{key}] must be 2D; got shape {mat.shape}")
                    ulist.append(mat)
    else:
        raise ValueError("Ulist file must be .npy or .npz")

    if not ulist:
        raise ValueError("Ulist cannot be empty")

    for i, u in enumerate(ulist):
        if u.shape != (expected_r, expected_r):
            raise ValueError(
                f"Ulist[{i}] has shape {u.shape}; expected ({expected_r}, {expected_r})"
            )
    return ulist


def _save_result(out_prefix: str, result, command: str) -> None:
    out = Path(out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "pi": np.asarray(result.fitted_g.pi, dtype=float),
        "grid": np.asarray(result.fitted_g.grid, dtype=float),
        "posterior_weights": np.asarray(result.posterior_weights, dtype=float),
        "vloglik": np.asarray(result.vloglik, dtype=float),
        "fitted_u_stack": np.stack(
            [np.asarray(u, dtype=float) for u in result.fitted_g.Ulist],
            axis=0,
        ),
    }

    optional = {
        "posterior_mean": result.posterior_mean,
        "posterior_sd": result.posterior_sd,
        "lfsr": result.lfsr,
        "lfdr": result.lfdr,
        "negative_prob": result.negative_prob,
        "posterior_cov": result.posterior_cov,
        "posterior_samples": result.posterior_samples,
        "lik_matrix": result.lik_matrix,
        "null_loglik": result.null_loglik,
        "alt_loglik": result.alt_loglik,
    }
    for key, val in optional.items():
        if val is not None:
            arrays[key] = np.asarray(val, dtype=float)

    np.savez_compressed(str(out) + ".npz", **arrays)

    meta = {
        "command": command,
        "loglik": float(result.loglik),
        "n_effects": int(result.posterior_weights.shape[0]),
        "n_components_active": int(result.posterior_weights.shape[1]),
        "alpha": float(result.alpha),
        "usepointmass": bool(result.fitted_g.usepointmass),
    }
    with (Path(str(out) + ".json")).open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def _cmd_fit(args: argparse.Namespace) -> int:
    bhat = _load_2d_matrix(args.bhat, "Bhat")
    shat = _load_2d_matrix(args.shat, "Shat")
    v = _load_v(args.v) if args.v is not None else None

    data = mash_set_data(bhat, shat, alpha=args.alpha, V=v)
    if args.ulist is not None:
        ulist = _load_ulist(args.ulist, expected_r=data.n_conditions)
    else:
        methods = [m.strip() for m in args.cov_methods.split(",") if m.strip()]
        ulist = cov_canonical(data, methods=methods)

    grid = np.asarray(args.grid, dtype=float) if args.grid is not None else None
    result = mash(
        data,
        Ulist=ulist,
        grid=grid,
        gridmult=args.gridmult,
        normalizeU=not args.no_normalize_u,
        usepointmass=not args.no_pointmass,
        prior=args.prior,
        nullweight=args.nullweight,
        optmethod=args.optmethod,
        pi_thresh=args.pi_thresh,
        posterior_samples=args.posterior_samples,
        seed=args.seed,
        outputlevel=args.outputlevel,
        output_lfdr=args.output_lfdr,
    )
    _save_result(args.out, result, command="fit")
    return 0


def _cmd_onebyone(args: argparse.Namespace) -> int:
    bhat = _load_2d_matrix(args.bhat, "Bhat")
    shat = _load_2d_matrix(args.shat, "Shat")
    v = _load_v(args.v) if args.v is not None else None
    data = mash_set_data(bhat, shat, alpha=args.alpha, V=v)
    result = mash_1by1(data, alpha=args.alpha)
    _save_result(args.out, result, command="onebyone")
    return 0


def _cmd_estimate_null_corr_simple(args: argparse.Namespace) -> int:
    bhat = _load_2d_matrix(args.bhat, "Bhat")
    shat = _load_2d_matrix(args.shat, "Shat")
    data = mash_set_data(bhat, shat, alpha=args.alpha)
    vhat = estimate_null_correlation_simple(
        data,
        z_thresh=args.z_thresh,
        est_cor=not args.est_cov,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.asarray(vhat, dtype=float))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pymash",
        description="Command-line interface for fitting and using pymash models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="Fit a mash model and save result arrays.")
    fit.add_argument("--bhat", required=True, help="Path to Bhat matrix (.npy/.npz/.csv/.tsv).")
    fit.add_argument("--shat", required=True, help="Path to Shat matrix (.npy/.npz/.csv/.tsv).")
    fit.add_argument("--out", required=True, help="Output prefix (writes <out>.npz and <out>.json).")
    fit.add_argument("--alpha", type=float, default=0.0, help="Alpha scaling parameter (default: 0.0).")
    fit.add_argument("--v", help="Optional covariance/correlation matrix V (.npy/.npz/.csv/.tsv).")
    fit.add_argument("--ulist", help="Optional Ulist file (.npy KxRxR or .npz).")
    fit.add_argument(
        "--cov-methods",
        default="identity,singletons,equal_effects,simple_het",
        help=(
            "Canonical covariance methods when --ulist is not provided "
            "(comma-separated)."
        ),
    )
    fit.add_argument("--grid", nargs="+", type=float, help="Explicit grid values, e.g. --grid 0.5 1.0")
    fit.add_argument("--gridmult", type=float, default=np.sqrt(2.0), help="Auto-grid multiplier (default: sqrt(2)).")
    fit.add_argument("--no-normalize-u", action="store_true", help="Disable covariance normalization.")
    fit.add_argument("--no-pointmass", action="store_true", help="Disable null point-mass component.")
    fit.add_argument("--prior", default="nullbiased", help="Prior on mixture weights (uniform or nullbiased).")
    fit.add_argument("--nullweight", type=float, default=10.0, help="Null prior weight for nullbiased prior.")
    fit.add_argument(
        "--optmethod",
        default="slsqp",
        choices=["mixsqp", "squarem", "slsqp", "em", "auto"],
        help="Optimizer for mixture weights (default: slsqp).",
    )
    fit.add_argument("--pi-thresh", type=float, default=1e-10, help="Drop components with pi <= this threshold.")
    fit.add_argument("--posterior-samples", type=int, default=0, help="Number of posterior samples.")
    fit.add_argument("--seed", type=int, default=123, help="Random seed.")
    fit.add_argument("--outputlevel", type=int, default=2, choices=[1, 2, 3, 4], help="Output detail level.")
    fit.add_argument("--output-lfdr", action="store_true", help="Compute lfdr output.")
    fit.set_defaults(func=_cmd_fit)

    ob = sub.add_parser("onebyone", help="Run mash_1by1 baseline and save result arrays.")
    ob.add_argument("--bhat", required=True, help="Path to Bhat matrix (.npy/.npz/.csv/.tsv).")
    ob.add_argument("--shat", required=True, help="Path to Shat matrix (.npy/.npz/.csv/.tsv).")
    ob.add_argument("--out", required=True, help="Output prefix (writes <out>.npz and <out>.json).")
    ob.add_argument("--alpha", type=float, default=0.0, help="Alpha scaling parameter (default: 0.0).")
    ob.add_argument("--v", help="Optional covariance/correlation matrix V (.npy/.npz/.csv/.tsv).")
    ob.set_defaults(func=_cmd_onebyone)

    ncs = sub.add_parser(
        "estimate-null-corr-simple",
        help="Estimate null correlation/covariance matrix with simple z-threshold rule.",
    )
    ncs.add_argument("--bhat", required=True, help="Path to Bhat matrix (.npy/.npz/.csv/.tsv).")
    ncs.add_argument("--shat", required=True, help="Path to Shat matrix (.npy/.npz/.csv/.tsv).")
    ncs.add_argument("--out", required=True, help="Output .npy path for estimated matrix.")
    ncs.add_argument("--alpha", type=float, default=0.0, help="Alpha scaling parameter (default: 0.0).")
    ncs.add_argument("--z-thresh", type=float, default=2.0, help="Null z-score threshold.")
    ncs.add_argument("--est-cov", action="store_true", help="Estimate covariance (default is correlation).")
    ncs.set_defaults(func=_cmd_estimate_null_corr_simple)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # pragma: no cover - exercised via subprocess in tests
        print(f"pymash: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
