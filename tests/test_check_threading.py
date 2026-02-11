"""Tests for check_threading() and CLI check command."""

import subprocess
import sys

import pytest

import pymash

HAS_CPP_BACKEND = getattr(pymash, "_cpp_backend", None) is not None


def test_check_threading_returns_dict():
    """check_threading() should return a dict with expected keys."""
    info = pymash.check_threading()
    assert isinstance(info, dict)
    assert "cpp_backend_available" in info
    assert "openmp_enabled" in info
    assert "openmp_status_known" in info
    assert "platform" in info
    assert "recommendation" in info


def test_check_threading_reports_cpp_backend_as_bool():
    """check_threading() should report backend availability as a boolean."""
    info = pymash.check_threading()
    assert isinstance(info["cpp_backend_available"], bool)


def test_check_threading_platform_matches_sys():
    """check_threading() platform should match sys.platform."""
    info = pymash.check_threading()
    assert info["platform"] == sys.platform


def test_check_threading_openmp_is_bool():
    """openmp_enabled should be a boolean."""
    info = pymash.check_threading()
    assert isinstance(info["openmp_enabled"], bool)


def test_check_threading_quiet_by_default(capsys):
    """check_threading() should not print by default."""
    _ = pymash.check_threading()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_check_threading_verbose_prints(capsys):
    """check_threading(verbose=True) should print a summary."""
    _ = pymash.check_threading(verbose=True)
    captured = capsys.readouterr()
    assert "threading status" in captured.out.lower()


def test_cli_check_runs_successfully():
    """pymash check should return success iff C++ backend is available."""
    result = subprocess.run(
        [sys.executable, "-m", "pymash.cli", "check"],
        capture_output=True,
        text=True,
    )
    expected = 0 if HAS_CPP_BACKEND else 1
    assert result.returncode == expected, f"stderr: {result.stderr}"
    assert "pymash" in result.stdout
    assert "C++ backend" in result.stdout
    if HAS_CPP_BACKEND:
        assert "Installation verified" in result.stdout


def test_cli_check_reports_openmp_status():
    """pymash check should report OpenMP status."""
    result = subprocess.run(
        [sys.executable, "-m", "pymash.cli", "check"],
        capture_output=True,
        text=True,
    )
    if HAS_CPP_BACKEND:
        assert result.returncode == 0
        assert "OpenMP" in result.stdout
    else:
        assert result.returncode == 1


def test_cli_check_runs_smoke_test():
    """pymash check should run and pass a smoke test."""
    result = subprocess.run(
        [sys.executable, "-m", "pymash.cli", "check"],
        capture_output=True,
        text=True,
    )
    if HAS_CPP_BACKEND:
        assert result.returncode == 0
        assert "smoke test" in result.stdout.lower()
        assert "done" in result.stdout.lower()
    else:
        assert result.returncode == 1
