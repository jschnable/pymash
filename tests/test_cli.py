from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import pymash.likelihoods as lk
from pymash.simulations import simple_sims


def _run_cli(args: list[str], repo_root: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    return subprocess.run(
        [sys.executable, "-m", "pymash", *args],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_help_runs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    res = _run_cli(["--help"], repo_root=repo_root)
    assert res.returncode == 0, res.stderr
    assert "fit" in res.stdout
    assert "onebyone" in res.stdout


@pytest.mark.skipif(not lk._use_cpp_general_lik(), reason="C++ likelihood backend unavailable")
def test_cli_fit_writes_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sim = simple_sims(nsamp=4, ncond=3, err_sd=0.5, seed=10)
    bhat_path = tmp_path / "bhat.npy"
    shat_path = tmp_path / "shat.npy"
    np.save(bhat_path, sim["Bhat"])
    np.save(shat_path, sim["Shat"])

    out_prefix = tmp_path / "fit_out"
    res = _run_cli(
        [
            "fit",
            "--bhat",
            str(bhat_path),
            "--shat",
            str(shat_path),
            "--out",
            str(out_prefix),
            "--cov-methods",
            "identity",
            "--grid",
            "1.0",
            "--outputlevel",
            "1",
        ],
        repo_root=repo_root,
    )
    assert res.returncode == 0, res.stderr

    npz_path = Path(str(out_prefix) + ".npz")
    json_path = Path(str(out_prefix) + ".json")
    assert npz_path.exists()
    assert json_path.exists()

    with np.load(npz_path) as zf:
        assert "pi" in zf.files
        assert "grid" in zf.files
        assert "posterior_weights" in zf.files
        assert "vloglik" in zf.files
        assert "fitted_u_stack" in zf.files
        assert zf["posterior_weights"].shape[0] == sim["Bhat"].shape[0]

    with json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["command"] == "fit"
    assert np.isfinite(meta["loglik"])


@pytest.mark.skipif(not lk._use_cpp_general_lik(), reason="C++ likelihood backend unavailable")
def test_cli_fit_chunked_large_mode_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sim = simple_sims(nsamp=6, ncond=3, err_sd=0.6, seed=13)
    bhat_path = tmp_path / "bhat.npy"
    shat_path = tmp_path / "shat.npy"
    np.save(bhat_path, sim["Bhat"])
    np.save(shat_path, sim["Shat"])

    out_prefix = tmp_path / "fit_chunked_out"
    res = _run_cli(
        [
            "fit",
            "--bhat",
            str(bhat_path),
            "--shat",
            str(shat_path),
            "--out",
            str(out_prefix),
            "--cov-methods",
            "identity",
            "--grid",
            "1.0",
            "--chunk-size",
            "5",
            "--outputlevel",
            "2",
        ],
        repo_root=repo_root,
    )
    assert res.returncode == 0, res.stderr

    npz_path = Path(str(out_prefix) + ".npz")
    json_path = Path(str(out_prefix) + ".json")
    assert npz_path.exists()
    assert json_path.exists()

    with np.load(npz_path) as zf:
        assert "posterior_mean" in zf.files
        assert "posterior_sd" in zf.files
        assert "lfsr" in zf.files
        assert zf["posterior_mean"].shape == sim["Bhat"].shape
        assert "posterior_weights" not in zf.files

    with json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["n_effects"] == sim["Bhat"].shape[0]
    assert np.isfinite(meta["loglik"])


def test_cli_onebyone_writes_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sim = simple_sims(nsamp=4, ncond=3, err_sd=0.5, seed=11)
    bhat_path = tmp_path / "bhat.npy"
    shat_path = tmp_path / "shat.npy"
    np.save(bhat_path, sim["Bhat"])
    np.save(shat_path, sim["Shat"])

    out_prefix = tmp_path / "onebyone_out"
    res = _run_cli(
        [
            "onebyone",
            "--bhat",
            str(bhat_path),
            "--shat",
            str(shat_path),
            "--out",
            str(out_prefix),
        ],
        repo_root=repo_root,
    )
    assert res.returncode == 0, res.stderr

    npz_path = Path(str(out_prefix) + ".npz")
    assert npz_path.exists()
    with np.load(npz_path) as zf:
        assert "posterior_mean" in zf.files
        assert "posterior_sd" in zf.files
        assert "lfsr" in zf.files
        assert zf["posterior_mean"].shape == sim["Bhat"].shape


def test_cli_estimate_null_corr_simple_writes_matrix(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sim = simple_sims(nsamp=15, ncond=4, err_sd=1.0, seed=12)
    bhat_path = tmp_path / "bhat.npy"
    shat_path = tmp_path / "shat.npy"
    np.save(bhat_path, sim["Bhat"])
    np.save(shat_path, sim["Shat"])

    out_path = tmp_path / "vhat.npy"
    res = _run_cli(
        [
            "estimate-null-corr-simple",
            "--bhat",
            str(bhat_path),
            "--shat",
            str(shat_path),
            "--out",
            str(out_path),
            "--z-thresh",
            "3.0",
        ],
        repo_root=repo_root,
    )
    assert res.returncode == 0, res.stderr
    assert out_path.exists()
    vhat = np.load(out_path)
    assert vhat.shape == (4, 4)
    assert np.allclose(vhat, vhat.T, atol=1e-10)
