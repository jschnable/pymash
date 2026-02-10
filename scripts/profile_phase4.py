from __future__ import annotations

import cProfile
import pstats
import time

import numpy as np

from pymash.covariances import cov_canonical
from pymash.data import mash_set_data
from pymash.mash import mash
from pymash.simulations import simple_sims


def run_profile(nsamp: int = 5000, ncond: int = 10, err_sd: float = 1.0) -> None:
    sim = simple_sims(nsamp=nsamp, ncond=ncond, err_sd=err_sd, seed=2026)
    data = mash_set_data(sim["Bhat"], sim["Shat"])
    U = cov_canonical(data)

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    _ = mash(
        data,
        Ulist=U,
        grid=np.array([0.5, 1.0]),
        prior="uniform",
        optmethod="mixsqp",
        outputlevel=2,
    )
    profiler.disable()
    t1 = time.perf_counter()

    print(f"Elapsed seconds: {t1 - t0:.3f}")
    print(f"Problem size: J={data.n_effects}, R={data.n_conditions}, K={len(U)}, P={1 + len(U) * 2}")
    print("Top 20 cumulative-time functions:")
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(20)


if __name__ == "__main__":
    run_profile()
