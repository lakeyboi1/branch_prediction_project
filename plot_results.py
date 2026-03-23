#!/usr/bin/env python3
"""
plot_results.py — Branch predictor experiment plots
Reads gem5 stats.txt files from results/X86/bp_sweep/<bench>/<bp_type>/
and produces:
  1. IPC comparison grouped by benchmark
  2. Branch misprediction rate grouped by benchmark
  3. Simulated execution time grouped by benchmark
  4. Per-predictor summary (geomean across all benchmarks)
"""

import os
import math
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BP_ORDER = ["none", "local", "bimode", "tage_base", "multi_branch"]
BP_LABELS = {
    "none":         "No Pred",
    "local":        "Local 2-bit",
    "bimode":       "BiMode",
    "tage_base":    "TAGE",
    "multi_branch": "MultiBranch\n(ours)",
}
COLORS = {bp: c for bp, c in zip(BP_ORDER, cm.tab10.colors)}

# ---------------------------------------------------------------------------
# Stats parsing
# ---------------------------------------------------------------------------
def parse_stats(stats_path: Path) -> dict:
    """Parse a gem5 stats.txt into a flat key→float dict."""
    stats = {}
    with open(stats_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    stats[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return stats


def extract_metrics(stats: dict) -> dict:
    """Pull the metrics we care about from a parsed stats dict."""
    ipc         = stats.get("system.cpu.ipc",    float("nan"))
    sim_seconds = stats.get("simSeconds",        float("nan"))

    # Branch prediction accuracy from committed branch stats
    committed   = stats.get("system.cpu.branchPred.committed_0::total",    0)
    mispredicted = stats.get("system.cpu.branchPred.mispredicted_0::total", 0)

    # Fallback to condPredicted / condIncorrect if the above are zero
    if committed == 0:
        committed    = stats.get("system.cpu.branchPred.condPredicted",  0)
        mispredicted = stats.get("system.cpu.branchPred.condIncorrect",  0)

    if committed > 0:
        mispredict_rate = mispredicted / committed * 100.0
    else:
        mispredict_rate = float("nan")

    return {
        "ipc":             ipc,
        "sim_seconds":     sim_seconds,
        "mispredict_rate": mispredict_rate,
    }


def discover_results(results_root: Path) -> dict:
    """
    Walk results_root/<bench>/<bp_type>/stats.txt and return:
    { bench: { bp_type: { metric: value } } }
    """
    data = defaultdict(dict)
    for bench_dir in sorted(results_root.iterdir()):
        if not bench_dir.is_dir():
            continue
        bench = bench_dir.name
        for bp_dir in sorted(bench_dir.iterdir()):
            if not bp_dir.is_dir():
                continue
            bp = bp_dir.name
            stats_file = bp_dir / "stats.txt"
            if not stats_file.exists():
                print(f"  warn: missing {stats_file}")
                continue
            stats  = parse_stats(stats_file)
            data[bench][bp] = extract_metrics(stats)
    return data


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def grouped_bar(ax, benches, bp_types, values, ylabel, title, baseline=None):
    """
    Draw a grouped bar chart.
    values[bench][bp] = float
    If baseline bp is given, normalise all values to that predictor = 1.0
    """
    n_groups = len(benches)
    n_bars   = len(bp_types)
    width    = 0.8 / n_bars
    x        = np.arange(n_groups)

    for i, bp in enumerate(bp_types):
        ys = []
        for bench in benches:
            v = values.get(bench, {}).get(bp, float("nan"))
            if baseline is not None:
                base = values.get(bench, {}).get(baseline, float("nan"))
                v = v / base if (base and not math.isnan(base)) else float("nan")
            ys.append(v)

        xpos = x - 0.4 + width / 2 + i * width
        bars = ax.bar(xpos, ys, width=width * 0.9,
                      label=BP_LABELS[bp], color=COLORS[bp], zorder=2)

        # Label NaN bars
        for xi, yi in zip(xpos, ys):
            if math.isnan(yi):
                ax.text(xi, 0.02, "N/A", ha="center", va="bottom",
                        fontsize=6, color="grey", rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(benches, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, ncol=len(bp_types))
    ax.grid(axis="y", linewidth=0.4, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    if baseline is not None:
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)


def geomean(vals):
    vals = [v for v in vals if v and not math.isnan(v) and v > 0]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/X86/bp_sweep",
                        help="Root of results directory")
    parser.add_argument("--out-dir", default="plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_root.exists():
        raise SystemExit(f"Results directory not found: {results_root.resolve()}\n"
                         "Run the sweep first.")

    print(f"Reading results from {results_root.resolve()}")
    data = discover_results(results_root)

    if not data:
        raise SystemExit("No results found. Check that stats.txt files exist.")

    benches  = sorted(data.keys())
    bp_types = [bp for bp in BP_ORDER if any(bp in data[b] for b in benches)]

    print(f"Benchmarks : {benches}")
    print(f"Predictors : {bp_types}")

    # ------------------------------------------------------------------
    # Build metric tables
    # ------------------------------------------------------------------
    ipc_vals  = {b: {bp: data[b].get(bp, {}).get("ipc",             float("nan")) for bp in bp_types} for b in benches}
    time_vals = {b: {bp: data[b].get(bp, {}).get("sim_seconds",     float("nan")) for bp in bp_types} for b in benches}
    mpr_vals  = {b: {bp: data[b].get(bp, {}).get("mispredict_rate", float("nan")) for bp in bp_types} for b in benches}

    # ------------------------------------------------------------------
    # Plot 1 — IPC by benchmark
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(8, len(benches) * 1.6), 5))
    grouped_bar(ax, benches, bp_types, ipc_vals,
                "IPC (instructions per cycle)",
                "IPC by Benchmark and Branch Predictor")
    plt.tight_layout()
    p = out_dir / "ipc_by_benchmark.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")

    # ------------------------------------------------------------------
    # Plot 2 — IPC normalised to "none" baseline
    # ------------------------------------------------------------------
    if "none" in bp_types:
        fig, ax = plt.subplots(figsize=(max(8, len(benches) * 1.6), 5))
        grouped_bar(ax, benches, bp_types, ipc_vals,
                    "IPC normalised to No-Prediction baseline",
                    "Normalised IPC (higher = better)",
                    baseline="none")
        plt.tight_layout()
        p = out_dir / "ipc_normalised.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"Saved {p}")

    # ------------------------------------------------------------------
    # Plot 3 — Branch misprediction rate
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(8, len(benches) * 1.6), 5))
    grouped_bar(ax, benches, bp_types, mpr_vals,
                "Misprediction rate (%)",
                "Branch Misprediction Rate by Benchmark and Predictor")
    plt.tight_layout()
    p = out_dir / "mispredict_rate.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")

    # ------------------------------------------------------------------
    # Plot 4 — Simulated execution time
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(8, len(benches) * 1.6), 5))
    grouped_bar(ax, benches, bp_types, time_vals,
                "Simulated execution time (s)",
                "Simulated Execution Time by Benchmark and Predictor")
    plt.tight_layout()
    p = out_dir / "exec_time.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")

    # ------------------------------------------------------------------
    # Plot 5 — Geomean summary across all benchmarks
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = [
        (ipc_vals,  "IPC",                    "Geomean IPC"),
        (mpr_vals,  "Misprediction rate (%)", "Geomean Misprediction Rate (%)"),
        (time_vals, "Simulated time (s)",     "Geomean Simulated Time (s)"),
    ]
    for ax, (vals, ylabel, title) in zip(axes, metrics):
        gms = []
        labels = []
        cols = []
        for bp in bp_types:
            vs = [vals[b].get(bp, float("nan")) for b in benches]
            gms.append(geomean(vs))
            labels.append(BP_LABELS[bp])
            cols.append(COLORS[bp])
        bars = ax.bar(labels, gms, color=cols, zorder=2)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", linewidth=0.4, alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        for bar, gm in zip(bars, gms):
            if not math.isnan(gm):
                ax.text(bar.get_x() + bar.get_width() / 2, gm * 1.01,
                        f"{gm:.3f}", ha="center", va="bottom", fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)

    fig.suptitle("Geomean Summary Across All Benchmarks", fontsize=12)
    plt.tight_layout()
    p = out_dir / "geomean_summary.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")

if __name__ == "__main__":
    main()