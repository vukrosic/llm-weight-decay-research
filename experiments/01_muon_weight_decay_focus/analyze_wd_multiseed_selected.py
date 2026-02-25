import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np


RUNS_DIR = Path("experiments/01_muon_weight_decay_focus/runs/wd_multiseed_selected")
OUT_DIR = RUNS_DIR


def parse_run(name: str):
    wd_token = name.split("_")[0].replace("wd", "")
    wd = float(wd_token.replace("m", "-").replace("p", "."))
    seed = int(name.split("_")[1].replace("seed", ""))
    return wd, seed


def cohen_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = mean(a), mean(b)
    sa, sb = stdev(a), stdev(b)
    n1, n2 = len(a), len(b)
    sp2 = ((n1 - 1) * sa * sa + (n2 - 1) * sb * sb) / (n1 + n2 - 2)
    if sp2 <= 0:
        return float("nan")
    return (mb - ma) / math.sqrt(sp2)


def main():
    rows = []
    for d in sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()]):
        m = d / "metrics.json"
        if not m.exists():
            continue
        wd, seed = parse_run(d.name)
        data = json.loads(m.read_text())
        fm = data.get("final_metrics", {})
        rows.append(
            {
                "run": d.name,
                "wd": wd,
                "seed": seed,
                "val_loss": float(fm["val_loss"]),
                "train_loss": float(fm["train_loss"]),
                "val_acc": float(fm["val_accuracy"]),
            }
        )
    if not rows:
        raise SystemExit("No runs found.")

    by_wd = {}
    for r in rows:
        by_wd.setdefault(r["wd"], []).append(r)

    summary = []
    for wd, rs in sorted(by_wd.items(), key=lambda x: x[0]):
        vals = [r["val_loss"] for r in rs]
        summary.append(
            {
                "wd": wd,
                "n": len(vals),
                "mean_val_loss": mean(vals),
                "std_val_loss": stdev(vals) if len(vals) > 1 else 0.0,
                "mean_val_acc": mean([r["val_acc"] for r in rs]),
            }
        )
    summary.sort(key=lambda x: x["mean_val_loss"])

    baseline = next((r for r in summary if abs(r["wd"]) < 1e-12), None)
    baseline_vals = [r["val_loss"] for r in by_wd.get(0.0, [])]
    for r in summary:
        if baseline:
            r["delta_vs_baseline"] = r["mean_val_loss"] - baseline["mean_val_loss"]
            r["pct_vs_baseline"] = 100 * r["delta_vs_baseline"] / baseline["mean_val_loss"]
            r["cohen_d_vs_baseline"] = cohen_d(baseline_vals, [x["val_loss"] for x in by_wd[r["wd"]]])
        else:
            r["delta_vs_baseline"] = float("nan")
            r["pct_vs_baseline"] = float("nan")
            r["cohen_d_vs_baseline"] = float("nan")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    labels = [f"wd={r['wd']}" for r in summary]
    means = [r["mean_val_loss"] for r in summary]
    stds = [r["std_val_loss"] for r in summary]
    x = np.arange(len(labels))
    plt.figure(figsize=(8, 5))
    bars = plt.bar(x, means, yerr=stds, capsize=6, color="#0ea5e9")
    plt.xticks(x, labels)
    plt.ylabel("Mean Val Loss (n=3 seeds)")
    plt.title("WD Multi-Seed Comparison (mean ± std)")
    for i, b in enumerate(bars):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001, f"{means[i]:.4f}", ha="center", fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    p1 = OUT_DIR / "multiseed_mean_std.png"
    plt.savefig(p1, dpi=200)
    plt.close()

    deltas = [r["delta_vs_baseline"] for r in summary]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(x, deltas, color="#f97316")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(x, labels)
    plt.ylabel("Delta Val Loss vs wd=0.0")
    plt.title("Real Impact vs Baseline (lower is better)")
    for i, b in enumerate(bars):
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + (0.0002 if b.get_height() >= 0 else -0.0006),
            f"{deltas[i]:+.4f}",
            ha="center",
            fontsize=8,
        )
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    p2 = OUT_DIR / "delta_vs_baseline.png"
    plt.savefig(p2, dpi=200)
    plt.close()

    md = OUT_DIR / "REPORT.md"
    with open(md, "w") as f:
        f.write("# WD Multi-Seed Impact Report\n\n")
        f.write("wd | n | mean_val_loss | std | delta_vs_baseline | pct_vs_baseline | cohen_d_vs_baseline\n")
        f.write("---|---:|---:|---:|---:|---:|---:\n")
        for r in summary:
            f.write(
                f"{r['wd']} | {r['n']} | {r['mean_val_loss']:.4f} | {r['std_val_loss']:.4f} | "
                f"{r['delta_vs_baseline']:+.4f} | {r['pct_vs_baseline']:+.2f}% | {r['cohen_d_vs_baseline']:+.3f}\n"
            )
        f.write("\n![Mean Std](./multiseed_mean_std.png)\n")
        f.write("\n![Delta](./delta_vs_baseline.png)\n")

    print(f"Wrote: {md}")
    print(f"Wrote: {p1}")
    print(f"Wrote: {p2}")


if __name__ == "__main__":
    main()
