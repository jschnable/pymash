from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from pymash.covariances import cov_canonical
from pymash.data import mash_set_data
from pymash.mash import FittedG, mash


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is required for R reference comparison")
def test_general_posterior_path_matches_r_with_fixed_g():
    repo_root = Path(__file__).resolve().parents[2]
    r_script = repo_root / "pymash" / "tests" / "r" / "run_mashr_reference.R"

    rng = np.random.default_rng(77)
    Bhat = rng.normal(size=(35, 5))
    # Varying Shat rows trigger the non-common-cov posterior path.
    Shat = np.exp(rng.normal(loc=-0.15, scale=0.25, size=(35, 5)))

    data = mash_set_data(Bhat, Shat)
    U = cov_canonical(data)
    P = 1 + len(U) * 2
    pi = np.linspace(1.0, float(P), P, dtype=float)
    pi /= np.sum(pi)

    with TemporaryDirectory() as td:
        tmp = Path(td)
        bhat_csv = tmp / "bhat.csv"
        shat_csv = tmp / "shat.csv"
        out_dir = tmp / "r_out"
        pi_csv = tmp / "pi.csv"

        np.savetxt(bhat_csv, Bhat, delimiter=",")
        np.savetxt(shat_csv, Shat, delimiter=",")
        np.savetxt(pi_csv, pi.reshape(-1, 1), delimiter=",")

        subprocess.run(
            [
                "Rscript",
                str(r_script),
                str(repo_root),
                str(bhat_csv),
                str(shat_csv),
                str(out_dir),
                str(pi_csv),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        pm_r = np.loadtxt(out_dir / "posterior_mean.csv", delimiter=",")
        psd_r = np.loadtxt(out_dir / "posterior_sd.csv", delimiter=",")
        lfsr_r = np.loadtxt(out_dir / "lfsr.csv", delimiter=",")

    g = FittedG(pi=pi, Ulist=list(U.values()), grid=np.array([0.5, 1.0]), usepointmass=True)
    m = mash(data, g=g, fixg=True, outputlevel=2, output_lfdr=True)

    assert np.allclose(m.posterior_mean, pm_r, atol=1e-6, rtol=1e-6)
    assert np.allclose(m.posterior_sd, psd_r, atol=1e-6, rtol=1e-6)
    assert np.allclose(m.lfsr, lfsr_r, atol=1e-6, rtol=1e-6)
