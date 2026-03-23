#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Record:
    sweep: str
    bench: str
    cpu: str
    mem: str
    freq: str
    l1: str
    l2: str
    outdir: Path
    ips: float


# Folder names we recognize in your tree
KNOWN_SWEEPS = {"matrix_fixedcache", "cache_sweep", "freq_sweep", "memtype_sweep", "run_micro", "run_mem"}
KNOWN_CPUS = {"Simple", "Minor4", "DefaultO3", "O3_W256", "O3_W2K"}
KNOWN_MEM_MODELS = {"Inf", "SingleCycle", "Slow"}
KNOWN_DRAMS = {"DDR3_1600_8x8", "DDR3_2133_8x8", "LPDDR2_S4_1066_1x32", "HBM_1000_4H_1x64"}


def parse_size_bytes(s: str) -> int:
    u = s.strip().lower()
    if u.endswith("kb"):
        return int(float(u[:-2]) * 1024)
    if u.endswith("mb"):
        return int(float(u[:-2]) * 1024 * 1024)
    if u.endswith("b"):
        return int(float(u[:-1]))
    return int(float(u))


def size_sort_key(s: str) -> Tuple[int, str]:
    try:
        return (parse_size_bytes(s), s)
    except Exception:
        return (10**18, s)


def freq_sort_key(s: str) -> Tuple[float, str]:
    u = s.strip().lower()
    try:
        if u.endswith("ghz"):
            return (float(u[:-3]), s)
        if u.endswith("mhz"):
            return (float(u[:-3]) / 1000.0, s)
        return (float(u), s)
    except Exception:
        return (10**18, s)


def is_freq_token(tok: str) -> bool:
    t = tok.strip().lower()
    return t.endswith("ghz") or t.endswith("mhz")


def read_stats(stats_path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not stats_path.exists():
        return out
    with stats_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            k = parts[0]
            v = parts[1]
            try:
                out[k] = float(v)
            except Exception:
                continue
    return out


def get_first(stats: Dict[str, float], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in stats:
            return stats[k]
    return None


def extract_ips(stats: Dict[str, float]) -> float:
    # Support both underscore and camelCase variants (gem5 varies by version/config)
    sim_seconds = get_first(stats, ["sim_seconds", "simSeconds"])
    inst = get_first(
        stats,
        [
            "sim_insts",
            "simInsts",
            "system.cpu.committedInsts",
            "system.cpu.numInsts",
            "system.cpu0.numInsts",
            "system.cpu.thread_0.numInsts",
        ],
    )

    if sim_seconds is None or inst is None:
        return float("nan")
    if not (sim_seconds == sim_seconds and sim_seconds > 0):
        return float("nan")
    return float(inst) / float(sim_seconds)


def parse_from_path(results_root: Path, outdir: Path) -> Tuple[str, str, str, str, str, str, str]:
    """
    Infers: sweep, bench, cpu, mem, freq, l1, l2 from directory layout.

    Works for your shown layouts, e.g.:
      - .../matrix_fixedcache/<bench>/<cpu>/<memModel>/stats.txt
      - .../cache_sweep/<bench>/<cpu>/l1_*/l2_*/<memModel>/stats.txt
      - .../freq_sweep/<bench>/<cpu>/<dram>/<freq>/stats.txt (or similar)
      - .../memtype_sweep/<bench>/<cpu>/<freq>/<dram>/stats.txt (or similar)
      - .../run_mem/<bench>/<cpu>/<dram>/<freq>/stats.txt
      - .../run_micro/<bench>/.../stats.txt (best-effort)
    """
    rel = outdir.relative_to(results_root)
    parts = list(rel.parts)

    sweep = ""
    bench = ""
    cpu = ""
    mem = ""
    freq = ""
    l1 = ""
    l2 = ""

    tail: List[str] = []

    # 1) Prefer known sweep tokens anywhere in path
    for i, p in enumerate(parts):
        if p in {"matrix_fixedcache", "cache_sweep", "freq_sweep", "memtype_sweep"}:
            sweep = p
            if i + 1 < len(parts):
                bench = parts[i + 1]
            tail = parts[i + 2 :]
            break

    # 2) Fallback: treat run_micro/run_mem as sweep-like
    if not sweep:
        if len(parts) >= 2 and parts[0] in {"run_micro", "run_mem"}:
            sweep = parts[0]
            bench = parts[1]
            tail = parts[2:]
        else:
            return "", "", "", "", "", "", ""

    # Pull tokens out of tail
    for p in tail:
        if p in KNOWN_CPUS and not cpu:
            cpu = p
        if p in KNOWN_MEM_MODELS and not mem:
            mem = p
        if p in KNOWN_DRAMS and not mem:
            mem = p
        if is_freq_token(p) and not freq:
            freq = p
        if p.startswith("l1_") and not l1:
            l1 = p.replace("l1_", "")
        if p.startswith("l2_") and not l2:
            l2 = p.replace("l2_", "")

    # cache_sweep: mem model is often last folder (Slow/Inf/etc)
    if sweep == "cache_sweep" and not mem:
        for p in reversed(tail):
            if p in KNOWN_MEM_MODELS:
                mem = p
                break

    # run_mem: run_mem/<bench>/<cpu>/<dram>/<freq>
    if sweep == "run_mem":
        if not cpu and len(tail) >= 1:
            cpu = tail[0]
        if not mem:
            for p in tail:
                if p in KNOWN_DRAMS:
                    mem = p
                    break
        if not freq:
            for p in tail:
                if is_freq_token(p):
                    freq = p
                    break

    return sweep, bench, cpu, mem, freq, l1, l2


def discover_records(results_root: Path) -> List[Record]:
    recs: List[Record] = []
    for stats_path in results_root.rglob("stats.txt"):
        outdir = stats_path.parent
        stats = read_stats(stats_path)
        ips = extract_ips(stats)

        sweep, bench, cpu, mem, freq, l1, l2 = parse_from_path(results_root, outdir)
        if not sweep or not bench:
            continue

        recs.append(Record(sweep, bench, cpu, mem, freq, l1, l2, outdir, ips))
    return recs


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def avg(vals: List[float]) -> float:
    v = [x for x in vals if x == x]
    if not v:
        return float("nan")
    return sum(v) / len(v)


def save_line_multi(
    path: Path,
    xlabels: List[str],
    series: List[Tuple[str, List[float]]],
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    x = list(range(len(xlabels)))
    plt.figure()
    for name, ys in series:
        plt.plot(x, ys, marker="o", label=name)
    plt.xticks(x, xlabels, rotation=45, ha="right")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if len(series) > 1:
        plt.legend()
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_grouped_bars(
    path: Path,
    xlabels: List[str],
    series: List[Tuple[str, List[float]]],
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    n = len(xlabels)
    m = max(1, len(series))
    width = 0.8 / m
    x = list(range(n))
    plt.figure()
    for i, (name, ys) in enumerate(series):
        xs = [xi - 0.4 + width / 2 + i * width for xi in x]
        plt.bar(xs, ys, width=width, label=name)
    plt.xticks(x, xlabels, rotation=45, ha="right")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if len(series) > 1:
        plt.legend()
    plt.grid(True, axis="y", linewidth=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ------------------- PLOTS -------------------


def plot_matrix_fixedcache(recs: List[Record], out_root: Path) -> None:
    """
    Produces the two BAR charts you showed:
      - CPU Comparison using Slow Memory
      - SingleCycle vs Slow Memory Model
    """
    r = [x for x in recs if x.sweep == "matrix_fixedcache" and x.ips == x.ips]
    if not r:
        return

    outdir = out_root / "matrix_fixedcache"
    ensure_dir(outdir)

    benches = sorted({x.bench for x in r})

    # A) CPU Comparison using Slow Memory
    slow = [x for x in r if x.mem == "Slow" and x.cpu]
    if slow:
        cpus = sorted({x.cpu for x in slow})
        series = []
        for cpu in cpus:
            ys = []
            for b in benches:
                ys.append(avg([x.ips for x in slow if x.bench == b and x.cpu == cpu]))
            series.append((cpu, ys))

        save_grouped_bars(
            outdir / "cpu_comparison_slow.png",
            benches,
            series,
            "Benchmark",
            "IPS",
            "CPU Comparison using Slow Memory",
        )

    # B) SingleCycle vs Slow Memory Model (use Simple if present, else first CPU)
    cpus_all = sorted({x.cpu for x in r if x.cpu})
    fixed_cpu = "Simple" if "Simple" in cpus_all else (cpus_all[0] if cpus_all else "")
    if fixed_cpu:
        rs = [x for x in r if x.cpu == fixed_cpu and x.mem in {"SingleCycle", "Slow"}]
        if rs:
            mems = ["SingleCycle", "Slow"]
            series = []
            for mem in mems:
                ys = []
                for b in benches:
                    ys.append(avg([x.ips for x in rs if x.bench == b and x.mem == mem]))
                series.append((mem, ys))

            save_grouped_bars(
                outdir / "singlecycle_vs_slow.png",
                benches,
                series,
                "Benchmark",
                "IPS",
                "SingleCycle vs Slow Memory Model",
            )


def plot_freq_sweep(recs: List[Record], out_root: Path) -> None:
    """
    LINE plot: IPS vs Frequency, one line per CPU (per benchmark).
    """
    r = [x for x in recs if x.sweep == "freq_sweep" and x.ips == x.ips and x.bench and x.cpu and x.freq]
    if not r:
        return

    outdir = out_root / "freq_sweep"
    ensure_dir(outdir)

    benches = sorted({x.bench for x in r})
    for b in benches:
        rb = [x for x in r if x.bench == b]
        freqs = sorted({x.freq for x in rb if x.freq}, key=freq_sort_key)
        cpus = sorted({x.cpu for x in rb if x.cpu})
        if not freqs or not cpus:
            continue

        series = []
        for cpu in cpus:
            ys = []
            for f in freqs:
                ys.append(avg([x.ips for x in rb if x.cpu == cpu and x.freq == f]))
            series.append((cpu, ys))

        save_line_multi(
            outdir / f"ips_vs_freq_{b}.png",
            freqs,
            series,
            "Frequency",
            "IPS",
            f"IPS vs Frequency: {b}",
        )


def plot_memtype_sweep(recs: List[Record], out_root: Path) -> None:
    """
    LINE plot: IPS vs Memory Type, one line per CPU (per benchmark).
    """
    r = [x for x in recs if x.sweep == "memtype_sweep" and x.ips == x.ips and x.bench and x.cpu and x.mem]
    if not r:
        return

    outdir = out_root / "memtype_sweep"
    ensure_dir(outdir)

    benches = sorted({x.bench for x in r})
    for b in benches:
        rb = [x for x in r if x.bench == b]
        mems = sorted({x.mem for x in rb if x.mem})
        cpus = sorted({x.cpu for x in rb if x.cpu})
        if not mems or not cpus:
            continue

        series = []
        for cpu in cpus:
            ys = []
            for m in mems:
                ys.append(avg([x.ips for x in rb if x.cpu == cpu and x.mem == m]))
            series.append((cpu, ys))

        save_line_multi(
            outdir / f"ips_vs_memtype_{b}.png",
            mems,
            series,
            "Memory Type",
            "IPS",
            f"IPS vs Memory Type: {b}",
        )


def plot_cache_sweep(recs: List[Record], out_root: Path) -> None:
    """
    LINE plot: IPS vs L2 size
      - one plot per (bench, cpu, mem)
      - x-axis: L2 sizes
      - one line per L1 size
    """
    r = [x for x in recs if x.sweep == "cache_sweep" and x.ips == x.ips and x.bench and x.cpu and x.l1 and x.l2]
    if not r:
        return

    outdir = out_root / "cache_sweep"
    ensure_dir(outdir)

    benches = sorted({x.bench for x in r})
    for b in benches:
        rb = [x for x in r if x.bench == b]
        cpus = sorted({x.cpu for x in rb})
        mems = sorted({x.mem for x in rb if x.mem}) or ["allmem"]

        for cpu in cpus:
            for mem in mems:
                if mem == "allmem":
                    rcm = [x for x in rb if x.cpu == cpu]
                else:
                    rcm = [x for x in rb if x.cpu == cpu and x.mem == mem]

                if not rcm:
                    continue

                l1s = sorted({x.l1 for x in rcm}, key=size_sort_key)
                l2s = sorted({x.l2 for x in rcm}, key=size_sort_key)
                if not l1s or not l2s:
                    continue

                series = []
                for l1 in l1s:
                    ys = []
                    for l2 in l2s:
                        ys.append(avg([x.ips for x in rcm if x.l1 == l1 and x.l2 == l2]))
                    series.append((f"L1 {l1}", ys))

                bench_cpu_dir = outdir / b / cpu / mem
                ensure_dir(bench_cpu_dir)

                save_line_multi(
                    bench_cpu_dir / f"ips_vs_l2_{b}_{cpu}_{mem}.png",
                    l2s,
                    series,
                    "L2 size",
                    "IPS",
                    f"Cache Sweep (IPS vs L2): {b} / {cpu} / {mem}",
                )

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/X86")
    ap.add_argument("--extraplots", default="extraplots")
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        choices=["matrix_fixedcache", "cache_sweep", "freq_sweep", "memtype_sweep", "run_micro", "run_mem"],
    )
    args = ap.parse_args()

    results_root = Path(args.results_root)
    extraplots = Path(args.extraplots)
    ensure_dir(extraplots)

    if not results_root.exists():
        raise SystemExit(f"results_root does not exist: {results_root.resolve()}")

    recs = discover_records(results_root)
    if args.only:
        allowed = set(args.only)
        recs = [r for r in recs if r.sweep in allowed]

    total = len(recs)
    valid = sum(1 for r in recs if r.ips == r.ips)
    print(f"Discovered {total} runs, {valid} with valid IPS. Saving plots to {extraplots.resolve()}")

    plot_matrix_fixedcache(recs, extraplots)
    plot_cache_sweep(recs, extraplots)
    plot_freq_sweep(recs, extraplots)
    plot_memtype_sweep(recs, extraplots)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
