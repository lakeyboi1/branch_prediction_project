from run import gem5Run
import os
import multiprocessing as mp
import argparse

# ---------------------------------------------------------------------------
# Branch predictors to sweep across — matches system.py choices
# ---------------------------------------------------------------------------
ALL_BP_TYPES = ["none", "local", "bimode", "tage_base", "multi_branch"]

def worker(run):
    run.run()
    print(run.dumpsJson())

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Sweep branch predictors across microbenchmarks"
    )

    parser.add_argument(
        "N", type=int, default=1, nargs="?",
        help="Number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--bp-types", nargs="+", default=ALL_BP_TYPES,
        choices=ALL_BP_TYPES,
        help="Branch predictor(s) to sweep (default: all)"
    )
    parser.add_argument(
        "--benches", nargs="*", default=None,
        help="Specific benchmarks to run (default: all found in --micro-dir)"
    )
    parser.add_argument(
        "--micro-dir", default="microbenchmark",
        help="Directory containing microbenchmark subdirectories"
    )
    parser.add_argument(
        "--bench-bin-name", default="bench.X86",
        help="Name of the compiled benchmark binary inside each subdir"
    )
    parser.add_argument(
        "--run-script", default="gem5-config/run_micro.py",
        help="gem5 config script to use"
    )
    parser.add_argument(
        "--out-root", default="results/X86/bp_sweep",
        help="Root directory for results output"
    )
    parser.add_argument(
        "--gem5-bin", default=None,
        help="Path to gem5 binary. Defaults to $GEM5_BIN, then $M5_PATH/build/X86/gem5.opt"
    )
    parser.add_argument(
        "--timeout", type=int, default=60 * 15,
        help="Per-run timeout in seconds (default: 900)"
    )

    args = parser.parse_args()

    # Resolve gem5 binary — prefer GEM5_BIN (local build dir), fall back to M5_PATH
    if args.gem5_bin is None:
        gem5_bin = os.getenv("GEM5_BIN")
        if gem5_bin:
            args.gem5_bin = gem5_bin
        else:
            m5 = os.getenv("M5_PATH")
            if m5 is None:
                raise SystemExit(
                    "Neither GEM5_BIN nor M5_PATH is set, and --gem5-bin was not provided"
                )
            args.gem5_bin = os.path.join(m5, "build/X86/gem5.opt")

    # Resolve benchmark list
    if args.benches:
        bm_list = sorted(args.benches)
    else:
        bm_list = sorted(
            f for f in os.listdir(args.micro_dir)
            if os.path.isdir(os.path.join(args.micro_dir, f))
            and not f.startswith(".")
        )

    if not bm_list:
        raise SystemExit(f"No benchmarks found in '{args.micro_dir}'")

    print(f"Benchmarks : {bm_list}")
    print(f"Predictors : {args.bp_types}")
    print(f"Workers    : {args.N}")
    print(f"gem5 binary: {args.gem5_bin}")
    print()

    # Build job list — one job per (benchmark x bp_type) combination
    jobs = []
    for bm in bm_list:
        bench_path = os.path.abspath(os.path.join(args.micro_dir, bm, args.bench_bin_name))
        for bp in args.bp_types:
            outdir = os.path.join(args.out_root, bm, bp)
            params = (
                bench_path,
                f"--bp_type={bp}",
            )
            run = gem5Run.createSERun(
                f"{bm}_{bp}",
                args.gem5_bin,
                args.run_script,
                outdir,
                *params,
                timeout=args.timeout,
            )
            jobs.append(run)

    print(f"Total jobs: {len(jobs)}")

    with mp.Pool(args.N) as pool:
        pool.map(worker, jobs)