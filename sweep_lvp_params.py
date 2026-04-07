#!/usr/bin/env python3

import os
import sys
import multiprocessing as mp
from run import gem5Run

PROJ      = os.path.expanduser("~/Desktop/project markov")
GEM5_BIN  = os.environ.get(
    "GEM5_BIN",
    os.path.join(PROJ, "gem5-fresh/build/X86/gem5.opt")
)
RUN_SCRIPT   = os.path.join(PROJ, "gem5-config/run_micro.py")
MICRO_DIR    = os.path.join(PROJ, "microbenchmark")

# Sweep 1 — vary depth, fixed conf_threshold=1
DEPTH_RESULTS  = os.path.join(PROJ, "results/X86/lvp_depth_sweep")
DEPTHS         = [1, 2, 3, 4, 5, 6]
FIXED_CONF     = 1

# Sweep 2 — vary conf_threshold, fixed depth=2
CONF_RESULTS   = os.path.join(PROJ, "results/X86/lvp_conf_sweep")
CONF_THRESHOLDS = [1, 2, 3, 4]
FIXED_DEPTH    = 2

BP_TYPE   = "local"
MAX_INSTS = 10_000_000


def worker(run):
    run.run()
    print(run.dumpsJson())


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    benches = sorted(
        f for f in os.listdir(MICRO_DIR)
        if os.path.isdir(os.path.join(MICRO_DIR, f))
    )

    print(f"Benchmarks  : {benches}")
    print(f"Depth sweep : depths={DEPTHS}, conf_threshold={FIXED_CONF}")
    print(f"Conf sweep  : conf_thresholds={CONF_THRESHOLDS}, depth={FIXED_DEPTH}")
    print(f"Workers     : {n_workers}")
    print()

    jobs = []

    # Sweep 1 — depth
    for bench in benches:
        bench_bin = os.path.join(MICRO_DIR, bench, "bench.X86")
        for depth in DEPTHS:
            outdir = os.path.join(DEPTH_RESULTS, bench, f"depth{depth}")
            jobs.append(gem5Run.createSERun(
                f"{bench}_depth{depth}",
                GEM5_BIN, RUN_SCRIPT, outdir,
                bench_bin,
                f"--bp_type={BP_TYPE}",
                f"--max_insts={MAX_INSTS}",
                "--value-pred",
                f"--context_depth={depth}",
                f"--conf_threshold={FIXED_CONF}",
                timeout=60 * 30,
            ))

    # Sweep 2 — conf_threshold
    for bench in benches:
        bench_bin = os.path.join(MICRO_DIR, bench, "bench.X86")
        for conf in CONF_THRESHOLDS:
            outdir = os.path.join(CONF_RESULTS, bench, f"conf{conf}")
            jobs.append(gem5Run.createSERun(
                f"{bench}_conf{conf}",
                GEM5_BIN, RUN_SCRIPT, outdir,
                bench_bin,
                f"--bp_type={BP_TYPE}",
                f"--max_insts={MAX_INSTS}",
                "--value-pred",
                f"--context_depth={FIXED_DEPTH}",
                f"--conf_threshold={conf}",
                timeout=60 * 30,
            ))

    print(f"Total jobs: {len(jobs)}")
    with mp.Pool(n_workers) as pool:
        pool.map(worker, jobs)

    print(f"\nDone.")
    print(f"  Depth results : {DEPTH_RESULTS}")
    print(f"  Conf results  : {CONF_RESULTS}")


if __name__ == "__main__":
    main()
