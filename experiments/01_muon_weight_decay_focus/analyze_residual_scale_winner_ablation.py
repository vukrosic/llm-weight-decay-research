import json
from pathlib import Path

import matplotlib.pyplot as plt


RUNS_DIR = Path("experiments/01_muon_weight_decay_focus/runs/residual_scale_winner_ablation")


def parse_scale(run_name: str) -> float:
    tok = run_name.split("_")[0].replace("res", "")
    return float(tok.replace("p", "."))


def parse_seed(run_name: str) -> int:
    return int(run_name.split("_")[1].replace("seed", ""))


def main():
    rows = []
    curve_rows = []
    for d in sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()]):
        m = d / "metrics.json"
        if not m.exists():
            continue
        data = json.loads(m.read_text())
        fm = data.get("final_metrics", {})
        h = data.get("history", {})
        rows.append(
            {
                "run": d.name,
                "residual_scale": parse_scale(d.name),
                "seed": parse_seed(d.name),
                "val_loss": float(fm.get("val_loss")),
                "train_loss": float(fm.get("train_loss")),
                "val_acc": float(fm.get("val_accuracy")),
                "steps": int(data.get("actual_steps")),
            }
        )
        curve_rows.append(
            {
                "run": d.name,
                "residual_scale": parse_scale(d.name),
                "train_steps": h.get("train_loss_steps", []),
                "train_losses": h.get("train_losses", []),
                "val_steps": h.get("steps", []),
                "val_losses": h.get("val_losses", []),
            }
        )
    if not rows:
        raise SystemExit("No completed runs found.")

    # Keep only seed-42 by default for this single-seed ablation report.
    seed42_rows = [r for r in rows if r["seed"] == 42]
    if seed42_rows:
        rows = seed42_rows
        curve_rows = [r for r in curve_rows if r["run"].endswith("_seed42")]

    rows_by_scale = sorted(rows, key=lambda x: x["residual_scale"])
    best = min(rows, key=lambda x: x["val_loss"])
    baseline = next((r for r in rows if abs(r["residual_scale"] - 1.0) < 1e-12), None)
    for r in rows_by_scale:
        r["delta_vs_scale1"] = r["val_loss"] - baseline["val_loss"] if baseline else float("nan")

    (RUNS_DIR / "summary.json").write_text(json.dumps(rows_by_scale, indent=2))

    xs = [r["residual_scale"] for r in rows_by_scale]
    ys = [r["val_loss"] for r in rows_by_scale]
    plt.figure(figsize=(10, 5.5))
    plt.plot(xs, ys, marker="o", linewidth=1.8)
    if baseline:
        plt.axhline(baseline["val_loss"], linestyle="--", linewidth=1, color="gray", label="scale=1 baseline")
    plt.scatter([best["residual_scale"]], [best["val_loss"]], color="red", zorder=3, label=f"best scale={best['residual_scale']}")
    plt.xlabel("Residual scale")
    plt.ylabel("Validation loss")
    plt.title("Residual-scale winner ablation (single seed)")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    p1 = RUNS_DIR / "val_loss_vs_scale.png"
    plt.savefig(p1, dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5.5))
    deltas = [r["delta_vs_scale1"] for r in rows_by_scale]
    colors = ["#16a34a" if d < 0 else "#dc2626" for d in deltas]
    plt.bar(xs, deltas, width=0.015, color=colors)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Residual scale")
    plt.ylabel("Val loss delta vs scale=1")
    plt.title("Impact vs baseline (negative is better)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    p2 = RUNS_DIR / "delta_vs_scale1.png"
    plt.savefig(p2, dpi=220)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    cmap = plt.get_cmap("viridis", len(curve_rows))
    curve_rows = sorted(curve_rows, key=lambda x: x["residual_scale"])
    for i, r in enumerate(curve_rows):
        color = cmap(i)
        if r["train_steps"] and r["train_losses"]:
            axes[0].plot(r["train_steps"], r["train_losses"], color=color, linewidth=1.2, alpha=0.9)
        if r["val_steps"] and r["val_losses"]:
            axes[1].plot(r["val_steps"], r["val_losses"], color=color, linewidth=1.5, alpha=0.9, label=f"s={r['residual_scale']}")
    axes[0].set_title("Train Loss Curves")
    axes[1].set_title("Validation Loss Curves")
    axes[0].set_xlabel("Step")
    axes[1].set_xlabel("Step")
    axes[0].set_ylabel("Train Loss")
    axes[1].set_ylabel("Validation Loss")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.suptitle("Residual-scale winner ablation: all curves")
    p3 = RUNS_DIR / "all_loss_curves.png"
    plt.savefig(p3, dpi=220, bbox_inches="tight")
    plt.close(fig)

    report = RUNS_DIR / "REPORT.md"
    with report.open("w") as f:
        f.write("# Residual-Scale Winner Ablation Report (Single Seed)\n\n")
        f.write("residual_scale | seed | val_loss | train_loss | val_acc | delta_vs_scale1\n")
        f.write("---|---:|---:|---:|---:|---:\n")
        for r in sorted(rows_by_scale, key=lambda x: x["val_loss"]):
            f.write(
                f"{r['residual_scale']} | {r['seed']} | {r['val_loss']:.4f} | "
                f"{r['train_loss']:.4f} | {r['val_acc']:.4f} | {r['delta_vs_scale1']:+.4f}\n"
            )
        f.write("\n")
        f.write(f"Best residual_scale: `{best['residual_scale']}` with val_loss `{best['val_loss']:.4f}`\n")
        if baseline:
            f.write(f"Baseline scale=1.0 val_loss: `{baseline['val_loss']:.4f}`\n")
            f.write(f"Best delta vs scale=1.0: `{best['val_loss'] - baseline['val_loss']:+.4f}`\n")
        f.write("\n![Val Loss vs Scale](./val_loss_vs_scale.png)\n")
        f.write("\n![Delta vs Scale1](./delta_vs_scale1.png)\n")
        f.write("\n![All Curves](./all_loss_curves.png)\n")

    print(f"Wrote: {report}")
    print(f"Wrote: {p1}")
    print(f"Wrote: {p2}")
    print(f"Wrote: {p3}")


if __name__ == "__main__":
    main()
