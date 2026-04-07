from run import gem5Run
import os
import multiprocessing as mp
import argparse

ALL_BP_TYPES = ["none", "local", "tage_base"]

def worker(run):
    run.run()
    print(run.dumpsJson())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep branch predictors across microbenchmarks"
    )
    parser.add_argument("N", type=int, default=1, nargs="?",
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--bp-types", nargs="+", default=ALL_BP_TYPES,
                        choices=ALL_BP_TYPES,
                        help="Branch predictor(s) to sweep (default: all)")
    parser.add_argument("--benches", nargs="*", default=None)
    parser.add_argument("--micro-dir", default="microbenchmark")
    parser.add_argument("--bench-bin-name", default="bench.X86")
    parser.add_argument("--run-script", default="gem5-config/run_micro.py")
    parser.add_argument("--out-root", default="results/X86/bp_sweep")
    parser.add_argument("--gem5-bin", default=None)
    parser.add_argument("--timeout", type=int, default=60 * 15)
    parser.add_argument("--value-pred", action="store_true", default=False,
                        help="Enable Markov load value predictor")
    args = parser.parse_args()

    # Resolve gem5 binary
    if args.gem5_bin is None:
        gem5_bin = os.getenv("GEM5_BIN")
        if gem5_bin:
            args.gem5_bin = gem5_bin
        else:
            m5 = os.getenv("M5_PATH")
            if m5 is None:
                raise SystemExit(
                    "Neither GEM5_BIN nor M5_PATH is set"
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
    print(f"Value pred : {args.value_pred}")
    print(f"Workers    : {args.N}")
    print(f"gem5 binary: {args.gem5_bin}")
    print()

    jobs = []
    for bm in bm_list:
        bench_path = os.path.abspath(
            os.path.join(args.micro_dir, bm, args.bench_bin_name)
        )
        for bp in args.bp_types:
            outdir = os.path.join(args.out_root, bm, bp)
            vp_flag = ("--value-pred",) if args.value_pred else ()
            params = (bench_path, f"--bp_type={bp}") + vp_flag
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
