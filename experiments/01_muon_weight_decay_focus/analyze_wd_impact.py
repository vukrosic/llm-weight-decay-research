import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np


RUNS_DIR = Path("experiments/01_muon_weight_decay_focus/runs/wd_impact")
OUT_DIR = RUNS_DIR


def parse_run(run_name: str):
    # wdm0p01_seed42, wd0p005_seed137, wd0p0_seed42
    wd_part, seed_part = run_name.split("_")
    wd_token = wd_part.replace("wd", "")
    wd_token = wd_token.replace("m", "-").replace("p", ".")
    wd = float(wd_token)
    seed = int(seed_part.replace("seed", ""))
    return wd, seed


def load_rows():
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
                "val_loss": float(fm.get("val_loss")),
                "train_loss": float(fm.get("train_loss")),
                "val_acc": float(fm.get("val_accuracy")),
                "tokens_seen": int(data.get("tokens_seen")),
                "steps": int(data.get("actual_steps")),
            }
        )
    return rows


def cohen_d(a, b):
    # effect of b relative to a
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    if not rows:
        raise SystemExit("No run metrics found.")

    by_wd = {}
    for r in rows:
        by_wd.setdefault(r["wd"], []).append(r)

    # Phase 1 table (seed 42 only)
    phase1 = sorted([r for r in rows if r["seed"] == 42], key=lambda x: x["val_loss"])

    # Phase 2 summary (only WDs that have >1 seed)
    phase2 = []
    for wd, rs in sorted(by_wd.items(), key=lambda x: x[0]):
        seeds = sorted({r["seed"] for r in rs})
        vals = [r["val_loss"] for r in rs]
        phase2.append(
            {
                "wd": wd,
                "n": len(vals),
                "seeds": seeds,
                "mean_val_loss": mean(vals),
                "std_val_loss": stdev(vals) if len(vals) > 1 else 0.0,
                "mean_val_acc": mean([r["val_acc"] for r in rs]),
            }
        )

    phase2_multi = [r for r in phase2 if r["n"] > 1]
    phase2_multi.sort(key=lambda x: x["mean_val_loss"])

    baseline = next((r for r in phase2_multi if abs(r["wd"]) < 1e-12), None)
    baseline_vals = by_wd.get(0.0, [])
    baseline_losses = [r["val_loss"] for r in baseline_vals]

    for r in phase2_multi:
        r["delta_vs_baseline"] = r["mean_val_loss"] - baseline["mean_val_loss"] if baseline else float("nan")
        r["pct_vs_baseline"] = (
            100.0 * (r["mean_val_loss"] - baseline["mean_val_loss"]) / baseline["mean_val_loss"]
            if baseline
            else float("nan")
        )
        r["cohen_d_vs_baseline"] = cohen_d(baseline_losses, [x["val_loss"] for x in by_wd[r["wd"]]]) if baseline else float("nan")

    # Plot 1: phase1 single-seed sorted
    plt.figure(figsize=(12, 6))
    labels = [f"wd={r['wd']}" for r in phase1]
    vals = [r["val_loss"] for r in phase1]
    y = np.arange(len(vals))
    bars = plt.barh(y, vals, color="#2563eb")
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Val Loss (seed=42)")
    plt.title("Phase 1: Single-Seed WD Screen")
    for i, b in enumerate(bars):
        plt.text(b.get_width() + 0.0002, b.get_y() + b.get_height() / 2, f"{vals[i]:.4f}", va="center", fontsize=8)
    plt.xlim(min(vals) - 0.002, max(vals) + 0.004)
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    p1_plot = OUT_DIR / "phase1_seed42_val_loss.png"
    plt.savefig(p1_plot, dpi=200)
    plt.close()

    # Plot 2: phase2 mean ± std for selected WDs
    if phase2_multi:
        plt.figure(figsize=(10, 5))
        labels2 = [f"wd={r['wd']}" for r in phase2_multi]
        means = [r["mean_val_loss"] for r in phase2_multi]
        stds = [r["std_val_loss"] for r in phase2_multi]
        x = np.arange(len(labels2))
        bars = plt.bar(x, means, yerr=stds, capsize=6, color="#059669")
        plt.xticks(x, labels2)
        plt.ylabel("Mean Val Loss (multi-seed)")
        plt.title("Phase 2: Multi-Seed Confirmation (mean ± std)")
        for i, b in enumerate(bars):
            plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001, f"{means[i]:.4f}", ha="center", fontsize=8)
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        p2_plot = OUT_DIR / "phase2_multiseed_mean_std.png"
        plt.savefig(p2_plot, dpi=200)
        plt.close()
    else:
        p2_plot = None

    report = OUT_DIR / "WD_IMPACT_REPORT.md"
    with open(report, "w") as f:
        f.write("# WD Impact Report (WD-only, fixed LR in phase 1)\n\n")
        f.write("## Phase 1 (seed=42) ranking\n\n")
        for i, r in enumerate(phase1, 1):
            f.write(
                f"{i}. `wd={r['wd']}` | val_loss={r['val_loss']:.4f} | "
                f"train_loss={r['train_loss']:.4f} | val_acc={r['val_acc']:.4f}\n"
            )
        f.write("\n")
        f.write("![Phase 1](./phase1_seed42_val_loss.png)\n\n")

        if phase2_multi:
            f.write("## Phase 2 (multi-seed) summary\n\n")
            f.write("wd | n | mean_val_loss | std | delta_vs_baseline | pct_vs_baseline | cohen_d_vs_baseline\n")
            f.write("---|---:|---:|---:|---:|---:|---:\n")
            for r in phase2_multi:
                f.write(
                    f"{r['wd']} | {r['n']} | {r['mean_val_loss']:.4f} | {r['std_val_loss']:.4f} | "
                    f"{r['delta_vs_baseline']:+.4f} | {r['pct_vs_baseline']:+.2f}% | {r['cohen_d_vs_baseline']:+.3f}\n"
                )
            f.write("\n")
            f.write("![Phase 2](./phase2_multiseed_mean_std.png)\n")

    print(f"Wrote: {report}")
    print(f"Wrote: {p1_plot}")
    if p2_plot:
        print(f"Wrote: {p2_plot}")


if __name__ == "__main__":
    main()
