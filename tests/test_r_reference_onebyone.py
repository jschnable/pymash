from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from pymash.data import mash_set_data
from pymash.mash import mash_1by1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
R_SCRIPT = PROJECT_ROOT / "tests" / "r" / "run_mash1by1_reference.R"


def _has_r_mashr() -> bool:
    if shutil.which("Rscript") is None:
        return False
    cmd = ["Rscript", "-e", "cat(as.integer(requireNamespace('mashr', quietly=TRUE)))"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0 and proc.stdout.strip() == "1"


HAS_R_MASHR = _has_r_mashr()


@pytest.mark.skipif(not HAS_R_MASHR, reason="R package 'mashr' is required for mash_1by1 reference test")
def test_mash_1by1_approximately_matches_r() -> None:
    alpha = 0.0

    rng = np.random.default_rng(2027)
    J, R = 220, 4
    bhat = rng.normal(size=(J, R))
    shat = np.exp(rng.normal(loc=-0.1, scale=0.25, size=(J, R)))

    with TemporaryDirectory() as td:
        tmp = Path(td)
        bhat_csv = tmp / "bhat.csv"
        shat_csv = tmp / "shat.csv"
        out_dir = tmp / "r_out"

        np.savetxt(bhat_csv, bhat, delimiter=",")
        np.savetxt(shat_csv, shat, delimiter=",")

        subprocess.run(
            ["Rscript", str(R_SCRIPT), str(REPO_ROOT), str(bhat_csv), str(shat_csv), str(alpha), str(out_dir)],
            check=True,
            capture_output=True,
            text=True,
        )

        pm_r = np.loadtxt(out_dir / "posterior_mean.csv", delimiter=",")
        psd_r = np.loadtxt(out_dir / "posterior_sd.csv", delimiter=",")
        lfsr_r = np.loadtxt(out_dir / "lfsr.csv", delimiter=",")
        loglik_r = float(np.loadtxt(out_dir / "loglik.csv", delimiter=","))

    data = mash_set_data(bhat, shat, alpha=alpha)
    m = mash_1by1(data, alpha=alpha)
    assert m.posterior_mean is not None
    assert m.posterior_sd is not None
    assert m.lfsr is not None

    pm_p = m.posterior_mean
    psd_p = m.posterior_sd
    lfsr_p = m.lfsr

    assert np.corrcoef(pm_p.ravel(), pm_r.ravel())[0, 1] > 0.995
    assert np.corrcoef(psd_p.ravel(), psd_r.ravel())[0, 1] > 0.995
    assert np.corrcoef(lfsr_p.ravel(), lfsr_r.ravel())[0, 1] > 0.99

    assert np.mean(np.abs(pm_p - pm_r)) < 0.06
    assert np.mean(np.abs(psd_p - psd_r)) < 0.06
    assert np.mean(np.abs(lfsr_p - lfsr_r)) < 0.06
    assert np.isclose(float(m.loglik), loglik_r, atol=60.0, rtol=0.05)
