from __future__ import annotations

import os
import platform
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

compile_args = ["-O3"]
link_args: list[str] = []
include_dirs: list[str] = []
library_dirs: list[str] = []


def _first_existing_dir(candidates: list[str]) -> str | None:
    for c in candidates:
        if c and Path(c).is_dir():
            return c
    return None

if os.environ.get("PYMASH_OPENMP", "1") != "0":
    system = platform.system().lower()
    if system == "linux":
        compile_args.append("-fopenmp")
        link_args.append("-fopenmp")
    elif system == "darwin":
        # Darwin/Apple clang needs libomp. Try to auto-detect common install paths.
        # Set PYMASH_OPENMP_DARWIN=0 to force-disable if needed.
        darwin_enabled = os.environ.get("PYMASH_OPENMP_DARWIN", "1") != "0"
        if darwin_enabled:
            omp_prefix = os.environ.get("PYMASH_OMP_PREFIX") or os.environ.get("OMP_PREFIX")
            if not omp_prefix:
                omp_prefix = _first_existing_dir(
                    [
                        "/opt/homebrew/opt/libomp",
                        "/usr/local/opt/libomp",
                    ]
                )
            omp_include = None
            omp_lib = None
            if omp_prefix:
                omp_include = str(Path(omp_prefix) / "include")
                omp_lib = str(Path(omp_prefix) / "lib")
                if not Path(omp_include).is_dir():
                    omp_include = None
                if not Path(omp_lib).is_dir():
                    omp_lib = None

            if omp_include and omp_lib:
                compile_args.extend(["-Xpreprocessor", "-fopenmp"])
                link_args.append("-lomp")
                include_dirs.append(omp_include)
                library_dirs.append(omp_lib)
            else:
                print("pymash: libomp not found on macOS; building without OpenMP.")
    elif system == "windows":
        compile_args.append("/openmp")

ext_modules = [
    Pybind11Extension(
        "pymash._edcpp",
        ["pymash/_edcpp.cpp"],
        cxx_std=17,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
