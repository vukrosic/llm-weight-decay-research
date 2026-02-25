import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(base: Path):
    rows = []
    for run_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, "r") as f:
            data = json.load(f)
        cfg = data.get("experiment_config", {})
        fm = data.get("final_metrics", {})
        rows.append(
            {
                "run": run_dir.name,
                "muon_wd": cfg.get("muon_weight_decay"),
                "muon_lr": cfg.get("muon_lr"),
                "adamw_wd": cfg.get("adamw_weight_decay", cfg.get("weight_decay")),
                "val_loss": fm.get("val_loss"),
                "train_loss": fm.get("train_loss"),
                "val_acc": fm.get("val_accuracy"),
                "tokens": data.get("tokens_seen"),
            }
        )
    return rows


def main():
    base = Path("experiments/01_muon_weight_decay_focus/runs/short_ablation")
    out_dir = base
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(base)
    if not rows:
        raise SystemExit("No metrics found.")

    rows_sorted = sorted(rows, key=lambda x: x["val_loss"])

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(rows_sorted, f, indent=2)

    labels = [r["run"] for r in rows_sorted]
    vals = np.array([r["val_loss"] for r in rows_sorted], dtype=float)
    y = np.arange(len(vals))

    # Plot 1: absolute val loss, horizontal bars with per-bar values.
    plt.figure(figsize=(14, 7))
    bars = plt.barh(y, vals, color="#3b82f6")
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel("Final Validation Loss (lower is better)")
    plt.title("Short Regularization Ablation: Validation Loss Ranking")
    plt.gca().invert_yaxis()
    for i, b in enumerate(bars):
        plt.text(
            b.get_width() + 0.0002,
            b.get_y() + b.get_height() / 2,
            f"{vals[i]:.4f}",
            va="center",
            fontsize=8,
        )
    plt.xlim(vals.min() - 0.002, vals.max() + 0.004)
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plot_path = out_dir / "val_loss_ranking.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # Plot 2: zoomed difference to best for easier visual discrimination.
    deltas = vals - vals.min()
    plt.figure(figsize=(14, 7))
    bars = plt.barh(y, deltas, color="#ef4444")
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel("Delta vs Best Validation Loss")
    plt.title("Short Regularization Ablation: Gap to Best (zoomed)")
    plt.gca().invert_yaxis()
    for i, b in enumerate(bars):
        plt.text(
            b.get_width() + 0.00002,
            b.get_y() + b.get_height() / 2,
            f"+{deltas[i]:.4f}",
            va="center",
            fontsize=8,
        )
    plt.xlim(-0.0001, deltas.max() + 0.001)
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    delta_plot_path = out_dir / "val_loss_delta_to_best.png"
    plt.savefig(delta_plot_path, dpi=200)
    plt.close()

    md_path = out_dir / "REPORT.md"
    with open(md_path, "w") as f:
        f.write("# Short Regularization Ablation Report\n\n")
        f.write("## Ranking by Final Validation Loss\n\n")
        for i, r in enumerate(rows_sorted, 1):
            f.write(
                f"{i}. `{r['run']}` | val_loss={r['val_loss']:.4f} | "
                f"train_loss={r['train_loss']:.4f} | val_acc={r['val_acc']:.4f} | "
                f"muon_wd={r['muon_wd']} | muon_lr={r['muon_lr']} | adamw_wd={r['adamw_wd']}\n"
            )
        f.write("\n## Plot\n\n")
        f.write("![Val Loss Ranking](./val_loss_ranking.png)\n")
        f.write("\n![Val Loss Delta to Best](./val_loss_delta_to_best.png)\n")

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {plot_path}")
    print(f"Wrote: {delta_plot_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
