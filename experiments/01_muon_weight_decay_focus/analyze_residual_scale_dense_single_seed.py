import json
from pathlib import Path

import matplotlib.pyplot as plt


RUNS_DIR = Path("experiments/01_muon_weight_decay_focus/runs/residual_scale_dense_single_seed")


def parse_scale(name: str) -> float:
    # res1p10_seed42 -> 1.10
    tok = name.split("_")[0].replace("res", "")
    return float(tok.replace("p", "."))


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
                "train_loss": float(fm.get("train_loss")),
                "val_loss": float(fm.get("val_loss")),
                "val_acc": float(fm.get("val_accuracy")),
                "steps": int(data.get("actual_steps")),
                "tokens_seen": int(data.get("tokens_seen")),
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

    rows.sort(key=lambda x: x["residual_scale"])
    best = min(rows, key=lambda x: x["val_loss"])
    baseline = next((r for r in rows if abs(r["residual_scale"] - 1.0) < 1e-12), None)

    for r in rows:
        if baseline is not None:
            r["delta_vs_scale1"] = r["val_loss"] - baseline["val_loss"]
        else:
            r["delta_vs_scale1"] = float("nan")

    (RUNS_DIR / "summary.json").write_text(json.dumps(rows, indent=2))

    xs = [r["residual_scale"] for r in rows]
    ys = [r["val_loss"] for r in rows]
    plt.figure(figsize=(10, 5.5))
    plt.plot(xs, ys, marker="o", linewidth=1.8)
    if baseline is not None:
        plt.axhline(baseline["val_loss"], linestyle="--", linewidth=1, color="gray", label="scale=1 baseline")
    plt.scatter([best["residual_scale"]], [best["val_loss"]], color="red", zorder=3, label=f"best scale={best['residual_scale']}")
    plt.xlabel("Residual scale")
    plt.ylabel("Final validation loss")
    plt.title("Dense single-seed short-run residual-scale sweep")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    p1 = RUNS_DIR / "val_loss_vs_residual_scale.png"
    plt.savefig(p1, dpi=220)
    plt.close()

    ys2 = [r["delta_vs_scale1"] for r in rows]
    plt.figure(figsize=(10, 5.5))
    colors = ["#16a34a" if y < 0 else "#dc2626" for y in ys2]
    plt.bar(xs, ys2, width=0.02, color=colors)
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
    curve_rows_sorted = sorted(curve_rows, key=lambda r: r["residual_scale"])
    for i, r in enumerate(curve_rows_sorted):
        color = cmap(i)
        if r["train_steps"] and r["train_losses"]:
            axes[0].plot(r["train_steps"], r["train_losses"], color=color, linewidth=1.2, alpha=0.9)
            axes[0].scatter([r["train_steps"][-1]], [r["train_losses"][-1]], color=color, s=10)
        if r["val_steps"] and r["val_losses"]:
            axes[1].plot(r["val_steps"], r["val_losses"], color=color, linewidth=1.5, alpha=0.9, label=f"s={r['residual_scale']}")
            axes[1].scatter([r["val_steps"][-1]], [r["val_losses"][-1]], color=color, s=12)

    axes[0].set_title("Train Loss Curves (17 scale runs)")
    axes[1].set_title("Validation Loss Curves (17 scale runs)")
    axes[0].set_xlabel("Step")
    axes[1].set_xlabel("Step")
    axes[0].set_ylabel("Train Loss")
    axes[1].set_ylabel("Validation Loss")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.suptitle("Dense Single-Seed Residual-Scale Sweep: All Loss Curves")
    p3 = RUNS_DIR / "all_loss_curves_17scales.png"
    plt.savefig(p3, dpi=220, bbox_inches="tight")
    plt.close(fig)

    report = RUNS_DIR / "REPORT.md"
    with report.open("w") as f:
        f.write("# Dense Single-Seed Residual-Scale Sweep Report\n\n")
        f.write("residual_scale | val_loss | train_loss | val_acc | delta_vs_scale1\n")
        f.write("---|---:|---:|---:|---:\n")
        for r in sorted(rows, key=lambda x: x["val_loss"]):
            f.write(
                f"{r['residual_scale']} | {r['val_loss']:.4f} | {r['train_loss']:.4f} | "
                f"{r['val_acc']:.4f} | {r['delta_vs_scale1']:+.4f}\n"
            )
        f.write("\n")
        f.write(f"Best residual_scale: `{best['residual_scale']}` with val_loss `{best['val_loss']:.4f}`\n")
        if baseline is not None:
            f.write(f"Baseline scale=1.0 val_loss: `{baseline['val_loss']:.4f}`\n")
            f.write(f"Best delta vs scale=1.0: `{best['val_loss'] - baseline['val_loss']:+.4f}`\n")
        f.write("\n![Val Loss vs Scale](./val_loss_vs_residual_scale.png)\n")
        f.write("\n![Delta vs Scale1](./delta_vs_scale1.png)\n")
        f.write("\n![All Loss Curves](./all_loss_curves_17scales.png)\n")

    print(f"Wrote: {report}")
    print(f"Wrote: {p1}")
    print(f"Wrote: {p2}")
    print(f"Wrote: {p3}")


if __name__ == "__main__":
    main()
