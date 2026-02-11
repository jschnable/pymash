from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from pymash.covariances import cov_pca
from pymash.data import mash_set_data
from pymash.ed import _edcpp, bovy_wrapper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
R_SCRIPT = PROJECT_ROOT / "tests" / "r" / "run_cov_ed_reference.R"


def _has_r_mashr() -> bool:
    if shutil.which("Rscript") is None:
        return False
    cmd = ["Rscript", "-e", "cat(as.integer(requireNamespace('mashr', quietly=TRUE)))"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0 and proc.stdout.strip() == "1"


HAS_R_MASHR = _has_r_mashr()


@pytest.mark.skipif(_edcpp is None, reason="C++ ED backend not built")
@pytest.mark.skipif(not HAS_R_MASHR, reason="R package 'mashr' is required for bovy regression test")
def test_bovy_wrapper_regression_against_r_mashr():
    rng = np.random.default_rng(202)
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
            ["Rscript", str(R_SCRIPT), str(bhat_csv), str(shat_csv), str(out_dir)],
            check=True,
            capture_output=True,
            text=True,
        )

        pi_r = np.loadtxt(out_dir / "pi.csv", delimiter=",")
        U_r = [np.loadtxt(out_dir / f"U_{i}.csv", delimiter=",") for i in range(1, len(pi_r) + 1)]
        av_loglik_r = float(np.loadtxt(out_dir / "av_loglik.csv", delimiter=","))

    data = mash_set_data(Bhat, Shat)
    U_init = cov_pca(data, npc=2)
    out = bovy_wrapper(data, U_init, maxiter=80, tol=1e-5)

    pi_p = out["pi"]
    U_p = out["Ulist"]
    av_loglik_p = float(out["av_loglik"])

    # Tolerances reflect independent implementations (R mashr C++ vs pymash C++).
    assert np.allclose(pi_p, pi_r, atol=2e-2, rtol=2e-2)
    for Up, Ur in zip(U_p, U_r):
        assert np.allclose(Up, Ur, atol=6e-2, rtol=6e-2)

    assert np.isclose(av_loglik_p, av_loglik_r, atol=3e-2, rtol=3e-2)
