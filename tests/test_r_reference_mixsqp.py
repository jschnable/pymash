from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from pymash.covariances import cov_canonical
from pymash.data import mash_set_data
from pymash.mash import mash


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
R_SCRIPT = PROJECT_ROOT / "tests" / "r" / "run_mashr_reference.R"


def _has_r_mashr() -> bool:
    cmd = ["Rscript", "-e", "cat(as.integer(requireNamespace('mashr', quietly=TRUE)))"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0 and proc.stdout.strip() == "1"


HAS_MASHR_SOURCE = (REPO_ROOT / "mashr" / "R").is_dir()
HAS_R_REFERENCE_BACKEND = shutil.which("Rscript") is not None and (HAS_MASHR_SOURCE or _has_r_mashr())


@pytest.mark.skipif(
    not HAS_R_REFERENCE_BACKEND,
    reason="R reference backend requires either ../mashr source or installed R package 'mashr'",
)
def test_end_to_end_mixsqp_matches_r_mixsqp_approximately():
    rng = np.random.default_rng(1234)
    Bhat = rng.normal(size=(60, 5))
    Shat = np.exp(rng.normal(loc=-0.1, scale=0.2, size=(60, 5)))

    with TemporaryDirectory() as td:
        tmp = Path(td)
        bhat_csv = tmp / "bhat.csv"
        shat_csv = tmp / "shat.csv"
        out_dir = tmp / "r_out"

        np.savetxt(bhat_csv, Bhat, delimiter=",")
        np.savetxt(shat_csv, Shat, delimiter=",")

        subprocess.run(
            [
                "Rscript",
                str(R_SCRIPT),
                str(REPO_ROOT),
                str(bhat_csv),
                str(shat_csv),
                str(out_dir),
                "NA",
                "mixSQP",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        pi_r = np.loadtxt(out_dir / "pi.csv", delimiter=",")
        pm_r = np.loadtxt(out_dir / "posterior_mean.csv", delimiter=",")
        psd_r = np.loadtxt(out_dir / "posterior_sd.csv", delimiter=",")
        lfsr_r = np.loadtxt(out_dir / "lfsr.csv", delimiter=",")

    data = mash_set_data(Bhat, Shat)
    U = cov_canonical(data)
    m = mash(
        data,
        Ulist=U,
        grid=np.array([0.5, 1.0]),
        prior="uniform",
        optmethod="mixsqp",
        outputlevel=2,
        output_lfdr=True,
    )

    assert np.allclose(m.fitted_g.pi, pi_r, atol=2e-2, rtol=2e-2)
    assert np.allclose(m.posterior_mean, pm_r, atol=2e-2, rtol=2e-2)
    assert np.allclose(m.posterior_sd, psd_r, atol=3e-2, rtol=3e-2)
    assert np.allclose(m.lfsr, lfsr_r, atol=4e-2, rtol=4e-2)
