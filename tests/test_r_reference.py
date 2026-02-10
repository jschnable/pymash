from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from pymash.covariances import cov_canonical, expand_cov
from pymash.data import mash_set_data
from pymash.mash import FittedG, mash


HAS_RSCRIPT = shutil.which("Rscript") is not None


def _run_r_reference(
    repo_root: Path,
    Bhat: np.ndarray,
    Shat: np.ndarray,
    pi: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    r_script = repo_root / "pymash" / "tests" / "r" / "run_mashr_reference.R"

    with TemporaryDirectory() as td:
        tmp = Path(td)
        bhat_csv = tmp / "bhat.csv"
        shat_csv = tmp / "shat.csv"
        out_dir = tmp / "r_out"

        np.savetxt(bhat_csv, Bhat, delimiter=",")
        np.savetxt(shat_csv, Shat, delimiter=",")

        cmd = [
            "Rscript",
            str(r_script),
            str(repo_root),
            str(bhat_csv),
            str(shat_csv),
            str(out_dir),
        ]
        if pi is not None:
            pi_csv = tmp / "pi.csv"
            np.savetxt(pi_csv, pi.reshape(-1, 1), delimiter=",")
            cmd.append(str(pi_csv))

        subprocess.run(cmd, check=True, capture_output=True, text=True)

        return {
            "pi": np.loadtxt(out_dir / "pi.csv", delimiter=","),
            "posterior_mean": np.loadtxt(out_dir / "posterior_mean.csv", delimiter=","),
            "posterior_sd": np.loadtxt(out_dir / "posterior_sd.csv", delimiter=","),
            "lfsr": np.loadtxt(out_dir / "lfsr.csv", delimiter=","),
            "loglik": float(np.loadtxt(out_dir / "loglik.csv", delimiter=",")),
        }


@pytest.mark.skipif(not HAS_RSCRIPT, reason="Rscript is required for R reference comparison")
def test_outputs_match_r_reference_with_fixed_g():
    repo_root = Path(__file__).resolve().parents[2]

    rng = np.random.default_rng(42)
    Bhat = rng.normal(size=(40, 5))
    Shat = np.exp(rng.normal(loc=-0.2, scale=0.15, size=(40, 5)))

    data = mash_set_data(Bhat, Shat)
    U = cov_canonical(data)
    xU = expand_cov(U, grid=np.array([0.5, 1.0]), usepointmass=True)

    pi = np.linspace(1.0, float(len(xU)), len(xU), dtype=float)
    pi /= np.sum(pi)

    r_out = _run_r_reference(repo_root, Bhat, Shat, pi=pi)

    g = FittedG(pi=pi, Ulist=list(U.values()), grid=np.array([0.5, 1.0]), usepointmass=True)
    m = mash(data, g=g, fixg=True, outputlevel=2, output_lfdr=True)

    assert np.allclose(m.fitted_g.pi, r_out["pi"], atol=1e-12, rtol=1e-12)
    assert np.allclose(m.posterior_mean, r_out["posterior_mean"], atol=1e-8, rtol=1e-8)
    assert np.allclose(m.posterior_sd, r_out["posterior_sd"], atol=1e-8, rtol=1e-8)
    assert np.allclose(m.lfsr, r_out["lfsr"], atol=1e-8, rtol=1e-8)
    assert np.isclose(m.loglik, r_out["loglik"], atol=1e-8, rtol=1e-8)


@pytest.mark.skipif(not HAS_RSCRIPT, reason="Rscript is required for R reference comparison")
def test_end_to_end_matches_r_reference_approximately():
    repo_root = Path(__file__).resolve().parents[2]

    rng = np.random.default_rng(123)
    Bhat = rng.normal(size=(50, 5))
    Shat = np.exp(rng.normal(loc=-0.1, scale=0.2, size=(50, 5)))

    r_out = _run_r_reference(repo_root, Bhat, Shat, pi=None)

    data = mash_set_data(Bhat, Shat)
    U = cov_canonical(data)
    m = mash(
        data,
        Ulist=U,
        grid=np.array([0.5, 1.0]),
        prior="uniform",
        optmethod="em",
        outputlevel=2,
        output_lfdr=True,
    )

    assert np.allclose(m.fitted_g.pi, r_out["pi"], atol=3e-2, rtol=3e-2)
    assert np.allclose(m.posterior_mean, r_out["posterior_mean"], atol=2e-2, rtol=2e-2)
    assert np.allclose(m.posterior_sd, r_out["posterior_sd"], atol=4e-2, rtol=4e-2)
    assert np.allclose(m.lfsr, r_out["lfsr"], atol=4e-2, rtol=4e-2)
    assert np.isclose(m.loglik, r_out["loglik"], atol=2e-2, rtol=2e-3)
