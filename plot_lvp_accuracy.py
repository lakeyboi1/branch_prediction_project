#!/usr/bin/env python3

import math
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEPTHS  = [1, 2, 3, 4, 5, 6]
CONFS   = [1, 2, 3, 4]
COLORS  = list(plt.cm.tab10.colors)

def parse_stats(path):
    stats = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
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


def extract_metrics(stats):
    total     = stats.get("system.cpu.value_pred.totalLoads",     0)
    predicted = stats.get("system.cpu.value_pred.predicted",      0)
    correct   = stats.get("system.cpu.value_pred.correct",        0)
    checked   = stats.get("system.cpu.value_pred.pendingChecked", 0)
    incorrect = stats.get("system.cpu.value_pred.incorrect",      0)
    return {
        "accuracy":  correct / checked * 100
                     if checked             > 0 else float("nan"),
        "coverage":  predicted / total * 100
                     if total              > 0 else float("nan"),
        "precision": correct / (correct + incorrect) * 100
                     if (correct+incorrect) > 0 else float("nan"),
    }


def discover(root, prefix):
    """Walk root/<bench>/<prefix><N>/stats.txt → {bench: {label: metrics}}"""
    root = Path(root)
    data = {}
    if not root.exists():
        print(f"  warn: {root} not found — skipping")
        return data
    for bench_dir in sorted(root.iterdir()):
        if not bench_dir.is_dir():
            continue
        bench = bench_dir.name
        for cfg_dir in sorted(bench_dir.iterdir()):
            if not cfg_dir.is_dir():
                continue
            label      = cfg_dir.name
            stats_file = cfg_dir / "stats.txt"
            if not stats_file.exists():
                print(f"  warn: missing {stats_file}")
                continue
            data.setdefault(bench, {})[label] = extract_metrics(
                parse_stats(stats_file))
    return data

def geomean(vals):
    vals = [v for v in vals if v and not math.isnan(v) and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) \
           if vals else float("nan")


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

def bar_chart(data, benches, configs, metric, ylabel, title,
              out_path, color_offset=0, ylim=(0, 105),
              legend_prefix=""):
    fig, ax = plt.subplots(figsize=(max(10, len(benches)*1.8), 5))
    x = np.arange(len(benches))
    n = len(configs)
    w = 0.8 / n
    for i, cfg in enumerate(configs):
        vals = [data.get(b, {}).get(cfg, {}).get(metric, float("nan"))
                for b in benches]
        num  = int(''.join(filter(str.isdigit, cfg)))
        ax.bar(x - 0.4 + w/2 + i*w, vals, w*0.9,
               label=f"{legend_prefix}{num}",
               color=COLORS[(i + color_offset) % 10], zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(benches, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, ncol=min(6, n))
    ax.grid(axis="y", linewidth=0.4, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(*ylim)
    plt.tight_layout()
    save(fig, out_path)


def trend_line(data, benches, param_values, prefix, xlabel, title,
               out_path, color_offset=0):
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric, label, cidx in [
        ("accuracy",  "Accuracy",  0 + color_offset),
        ("coverage",  "Coverage",  1 + color_offset),
        ("precision", "Precision", 2 + color_offset),
    ]:
        gms = [geomean([data.get(b, {}).get(f"{prefix}{v}", {}).get(
                        metric, float("nan")) for b in benches])
               for v in param_values]
        ax.plot(param_values, gms, marker="o", label=label,
                color=COLORS[cidx % 10], linewidth=2, markersize=7)
        for v, gm in zip(param_values, gms):
            if not math.isnan(gm):
                ax.annotate(f"{gm:.1f}%", (v, gm),
                            textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Geomean % across all benchmarks", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(param_values)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(linewidth=0.4, alpha=0.6)
    plt.tight_layout()
    save(fig, out_path)


def geomean_bars(data, benches, configs, out_path, legend_prefix,
                 color_offset=0, title_suffix=""):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(f"Markov LVP Geomean Summary{title_suffix}", fontsize=12)
    for ax, (metric, ylabel, title) in zip(axes, [
        ("accuracy",  "Accuracy (%)",  "Geomean Accuracy"),
        ("coverage",  "Coverage (%)",  "Geomean Coverage"),
        ("precision", "Precision (%)", "Geomean Precision"),
    ]):
        gms  = [geomean([data.get(b, {}).get(c, {}).get(metric, float("nan"))
                         for b in benches]) for c in configs]
        num  = [int(''.join(filter(str.isdigit, c))) for c in configs]
        lbls = [f"{legend_prefix}{n}" for n in num]
        cols = [COLORS[(i + color_offset) % 10] for i in range(len(configs))]
        bars = ax.bar(lbls, gms, color=cols, zorder=2)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linewidth=0.4, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
        for bar, gm in zip(bars, gms):
            if not math.isnan(gm):
                ax.text(bar.get_x() + bar.get_width()/2, gm + 1,
                        f"{gm:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    save(fig, out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-results",
                        default="results/X86/lvp_depth_sweep")
    parser.add_argument("--conf-results",
                        default="results/X86/lvp_conf_sweep")
    parser.add_argument("--out-dir", default="plots/lvp")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    depth_data = discover(args.depth_results, "depth")
    conf_data  = discover(args.conf_results,  "conf")

    depth_benches = sorted(depth_data.keys())
    conf_benches  = sorted(conf_data.keys())

    depth_configs = [f"depth{d}" for d in DEPTHS
                     if any(f"depth{d}" in depth_data.get(b, {})
                            for b in depth_benches)]
    conf_configs  = [f"conf{c}" for c in CONFS
                     if any(f"conf{c}" in conf_data.get(b, {})
                            for b in conf_benches)]

    # ── Depth sweep charts ───────────────────────────────────────────────
    if depth_data:
        print("\n── Depth sweep plots ──")
        for metric, ylabel, fname in [
            ("accuracy",  "Accuracy (correct/evaluated) %",  "depth_accuracy.png"),
            ("coverage",  "Coverage (predicted/total loads) %", "depth_coverage.png"),
            ("precision", "Precision (correct/predicted) %",  "depth_precision.png"),
        ]:
            bar_chart(depth_data, depth_benches, depth_configs,
                      metric, ylabel,
                      f"Markov LVP {metric.capitalize()} vs Context Depth",
                      out_dir / fname,
                      legend_prefix="Depth ")

        trend_line(depth_data, depth_benches, DEPTHS, "depth",
                   "Context Depth",
                   "Markov LVP: Accuracy, Coverage & Precision vs Context Depth",
                   out_dir / "depth_trend.png")

        geomean_bars(depth_data, depth_benches, depth_configs,
                     out_dir / "depth_geomean.png",
                     legend_prefix="Depth ",
                     title_suffix=" — Context Depth Sweep")

    # ── Conf sweep charts ────────────────────────────────────────────────
    if conf_data:
        print("\n── Confidence threshold sweep plots ──")
        for metric, ylabel, fname in [
            ("accuracy",  "Accuracy (correct/evaluated) %",  "conf_accuracy.png"),
            ("coverage",  "Coverage (predicted/total loads) %", "conf_coverage.png"),
            ("precision", "Precision (correct/predicted) %",  "conf_precision.png"),
        ]:
            bar_chart(conf_data, conf_benches, conf_configs,
                      metric, ylabel,
                      f"Markov LVP {metric.capitalize()} vs Confidence Threshold",
                      out_dir / fname,
                      color_offset=4,
                      legend_prefix="Conf≥")

        trend_line(conf_data, conf_benches, CONFS, "conf",
                   "Confidence Threshold",
                   "Markov LVP: Accuracy, Coverage & Precision vs Confidence Threshold",
                   out_dir / "conf_trend.png",
                   color_offset=4)

        geomean_bars(conf_data, conf_benches, conf_configs,
                     out_dir / "conf_geomean.png",
                     legend_prefix="Conf≥",
                     color_offset=4,
                     title_suffix=" — Confidence Threshold Sweep")

    print(f"\nAll plots saved to {out_dir.resolve()}/")


if __name__ == "__main__":
    main()