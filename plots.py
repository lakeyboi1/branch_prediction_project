"""
Usage:
    python3 plot_lvp_accuracy.py \
        --depth-results results/X86/lvp_depth_sweep \
        --conf-results  results/X86/lvp_conf_sweep \
        --out-dir plots/lvp
"""

import math
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEPTHS = [1, 2, 3, 4, 5, 10, 64, 128, 256, 512, 1024, 2048, 4096]
CONFS  = [1, 2, 3, 5, 7, 9, 11, 13, 15]


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


def extract_metrics(stats, path_hint=""):
    total     = stats.get("system.cpu.value_pred.totalLoads",     0)
    predicted = stats.get("system.cpu.value_pred.predicted",      0)
    correct   = stats.get("system.cpu.value_pred.correct",        0)
    incorrect = stats.get("system.cpu.value_pred.incorrect",      0)
    checked   = stats.get("system.cpu.value_pred.pendingChecked", 0)

    true_accuracy = correct / predicted * 100  if predicted > 0 else float("nan")
    coverage      = predicted / total * 100    if total     > 0 else float("nan")
    precision     = correct / (correct + incorrect) * 100 \
                                               if (correct + incorrect) > 0 else float("nan")
    verified      = checked / predicted * 100  if predicted > 0 else float("nan")

    return {"accuracy": true_accuracy, "coverage": coverage,
            "precision": precision, "verified": verified}


def discover(root, prefix, param_values):
    root = Path(root)
    data = {v: {} for v in param_values}
    if not root.exists():
        print(f"  warn: {root} not found")
        return data
    for bench_dir in sorted(root.iterdir()):
        if not bench_dir.is_dir():
            continue
        bench = bench_dir.name
        for v in param_values:
            stats_file = bench_dir / f"{prefix}{v}" / "stats.txt"
            if not stats_file.exists():
                print(f"  warn: missing {stats_file}")
                continue
            data[v][bench] = extract_metrics(parse_stats(stats_file),
                                             path_hint=str(stats_file))
    return data


def geomean(vals):
    vals = [v for v in vals if v and not math.isnan(v) and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) \
           if vals else float("nan")


def make_trend_plot(data, param_values, xlabel, title, out_path):
    import numpy as np
    benches = sorted({b for v in data.values() for b in v})

    metrics = [
        ("accuracy",  "Accuracy",  "#1f77b4", "o", "-",  4),
        ("coverage",  "Coverage",  "#ff7f0e", "s", "--", 2),
        ("precision", "Precision", "#2ca02c", "^", "-",  3),
    ]

    positions = list(range(len(param_values)))

    fig, ax = plt.subplots(figsize=(max(9, len(param_values) * 0.8), 6))

    for metric, label, color, marker, ls, zo in metrics:
        gms = []
        for v in param_values:
            bench_vals = [data[v].get(b, {}).get(metric, float("nan"))
                          for b in benches]
            gms.append(geomean(bench_vals))

        if all(math.isnan(g) for g in gms):
            print(f"  warn: all NaN for {metric} — skipping line")
            continue

        ax.plot(positions, gms,
                marker=marker, label=label, color=color,
                linewidth=2.5, markersize=9,
                linestyle=ls, zorder=zo)

        offsets = {"accuracy": -18, "coverage": -32, "precision": 10}
        xytext = (0, offsets.get(metric, 10))
        for pos, gm in zip(positions, gms):
            if not math.isnan(gm):
                ax.annotate(f"{gm:.1f}%", (pos, gm),
                            textcoords="offset points",
                            xytext=xytext, ha="center",
                            fontsize=9, color=color, fontweight="bold")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Geomean % across all benchmarks", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(v) for v in param_values], rotation=45, ha="right")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


LOW_VERIFIED_BENCHES = ["CCa", "CCl", "ED1", "EI"]


def plot_combined_verified_geomean(depth_data, conf_data,
                                   depth_values, conf_values,
                                   out_path):
    """
    Single figure with two subplots showing geomean verification rate:
      Left:  vs context depth
      Right: vs confidence threshold
    Y-axis zoomed to 1% above max and 1% below min across both subplots.
    """
    benches_d = sorted({b for v in depth_data.values() for b in v})
    benches_c = sorted({b for v in conf_data.values()  for b in v})

    def gm_verified(data, param_values, benches):
        return [geomean([data[v].get(b, {}).get("verified", float("nan"))
                         for b in benches])
                for v in param_values]

    gms_depth = gm_verified(depth_data, depth_values, benches_d)
    gms_conf  = gm_verified(conf_data,  conf_values,  benches_c)

    all_vals = [v for v in gms_depth + gms_conf if not math.isnan(v)]
    if not all_vals:
        print("  warn: no verified data for combined plot — skipping")
        return
    ymin = min(all_vals) - 1
    ymax = max(all_vals) + 1

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Markov LVP: Geomean Verification Rate — Depth Sweep vs Conf Sweep",
        fontsize=13
    )

    for ax, param_values, gms, xlabel, prefix in [
        (ax_left,  depth_values, gms_depth, "Context Depth",        "Depth "),
        (ax_right, conf_values,  gms_conf,  "Confidence Threshold", "Conf≥"),
    ]:
        positions = list(range(len(param_values)))
        ax.plot(positions, gms,
                marker="o", color="#1f77b4",
                linewidth=2.5, markersize=9, zorder=3)
        for i, (pos, gm) in enumerate(zip(positions, gms)):
            if not math.isnan(gm):
                offset = 12 if i % 2 == 0 else -18
                ax.annotate(f"{gm:.2f}%", (pos, gm),
                            textcoords="offset points",
                            xytext=(0, offset), ha="center",
                            fontsize=9, color="#1f77b4", fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Geomean verification rate (%)", fontsize=11)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(v) for v in param_values],
                           rotation=45, ha="right")
        ax.set_ylim(ymin, ymax)
        ax.grid(linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_accuracy_vs_precision_by_bench(depth_data, conf_data,
                                         depth_values, conf_values,
                                         out_path):
    import numpy as np

    all_configs = (
        [(depth_data, v) for v in depth_values] +
        [(conf_data,  v) for v in conf_values]
    )

    benches = sorted({
        b
        for data, v in all_configs
        for b in data.get(v, {})
    })

    def avg_metric(bench, metric):
        vals = []
        for data, v in all_configs:
            m = data.get(v, {}).get(bench, {}).get(metric, float("nan"))
            if not math.isnan(m):
                vals.append(m)
        return sum(vals) / len(vals) if vals else float("nan")

    accuracy_vals  = [avg_metric(b, "accuracy")  for b in benches]
    precision_vals = [avg_metric(b, "precision") for b in benches]

    x     = np.arange(len(benches))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(9, len(benches) * 1.6), 6))

    bars_acc  = ax.bar(x - width/2, accuracy_vals,  width,
                       label="Accuracy",  color="#1f77b4", zorder=2)
    bars_prec = ax.bar(x + width/2, precision_vals, width,
                       label="Precision", color="#2ca02c", zorder=2)

    for bar, val in list(zip(bars_acc, accuracy_vals)) + list(zip(bars_prec, precision_vals)):
        if not math.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(benches, fontsize=11)
    ax.set_ylabel("Average % across all sweep configurations", fontsize=11)
    ax.set_title("Accuracy and Precision Comparison", fontsize=12)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linewidth=0.4, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


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

    depth_data = discover(args.depth_results, "depth", DEPTHS)
    if any(depth_data[v] for v in DEPTHS):
        make_trend_plot(
            depth_data, DEPTHS,
            xlabel="Context Depth",
            title="Markov LVP: Coverage & Precision vs Context Depth",
            out_path=out_dir / "depth_trend.png"
        )

    conf_data = discover(args.conf_results, "conf", CONFS)
    if any(conf_data[v] for v in CONFS):
        make_trend_plot(
            conf_data, CONFS,
            xlabel="Confidence Threshold",
            title="Markov LVP: Coverage & Precision vs Confidence Threshold",
            out_path=out_dir / "conf_trend.png"
        )

    if any(depth_data[v] for v in DEPTHS) and any(conf_data[v] for v in CONFS):
        plot_combined_verified_geomean(
            depth_data, conf_data,
            DEPTHS, CONFS,
            out_dir / "combined_verified_geomean.png"
        )

    if any(depth_data[v] for v in DEPTHS) and any(conf_data[v] for v in CONFS):
        plot_accuracy_vs_precision_by_bench(
            depth_data, conf_data,
            DEPTHS, CONFS,
            out_dir / "accuracy_vs_precision_by_bench.png"
        )

    print(f"\nDone. Plots in {out_dir.resolve()}/")


if __name__ == "__main__":
    main()