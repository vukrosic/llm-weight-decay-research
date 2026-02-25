import json
from pathlib import Path

import matplotlib.pyplot as plt


RUNS_DIR = Path("experiments/01_muon_weight_decay_focus/runs/wd_dense_single_seed")


def parse_wd(name: str) -> float:
    wd_token = name.split("_")[0].replace("wd", "")
    return float(wd_token.replace("m", "-").replace("p", "."))


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
                "wd": parse_wd(d.name),
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
                "wd": parse_wd(d.name),
                "train_steps": h.get("train_loss_steps", []),
                "train_losses": h.get("train_losses", []),
                "val_steps": h.get("steps", []),
                "val_losses": h.get("val_losses", []),
            }
        )

    if not rows:
        raise SystemExit("No completed runs found.")

    rows.sort(key=lambda x: x["wd"])
    best = min(rows, key=lambda x: x["val_loss"])
    baseline = next((r for r in rows if abs(r["wd"]) < 1e-12), None)

    for r in rows:
        if baseline is not None:
            r["delta_vs_wd0"] = r["val_loss"] - baseline["val_loss"]
        else:
            r["delta_vs_wd0"] = float("nan")

    (RUNS_DIR / "summary.json").write_text(json.dumps(rows, indent=2))

    xs = [r["wd"] for r in rows]
    ys = [r["val_loss"] for r in rows]
    plt.figure(figsize=(10, 5.5))
    plt.plot(xs, ys, marker="o", linewidth=1.8)
    if baseline is not None:
        plt.axhline(baseline["val_loss"], linestyle="--", linewidth=1, color="gray", label="wd=0 baseline")
    plt.scatter([best["wd"]], [best["val_loss"]], color="red", zorder=3, label=f"best wd={best['wd']}")
    plt.xlabel("Muon weight decay")
    plt.ylabel("Final validation loss")
    plt.title("Dense single-seed short-run WD sweep")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    p1 = RUNS_DIR / "val_loss_vs_wd.png"
    plt.savefig(p1, dpi=220)
    plt.close()

    ys2 = [r["delta_vs_wd0"] for r in rows]
    plt.figure(figsize=(10, 5.5))
    colors = ["#16a34a" if y < 0 else "#dc2626" for y in ys2]
    plt.bar(xs, ys2, width=0.008, color=colors)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Muon weight decay")
    plt.ylabel("Val loss delta vs wd=0")
    plt.title("Impact vs baseline (negative is better)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    p2 = RUNS_DIR / "delta_vs_wd0.png"
    plt.savefig(p2, dpi=220)
    plt.close()

    # Single image with all loss curves for all WD points.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    cmap = plt.cm.get_cmap("coolwarm", len(curve_rows))
    curve_rows_sorted = sorted(curve_rows, key=lambda r: r["wd"])
    for i, r in enumerate(curve_rows_sorted):
        color = cmap(i)
        if r["train_steps"] and r["train_losses"]:
            axes[0].plot(r["train_steps"], r["train_losses"], color=color, linewidth=1.3, alpha=0.9)
            axes[0].scatter([r["train_steps"][-1]], [r["train_losses"][-1]], color=color, s=12)
        if r["val_steps"] and r["val_losses"]:
            axes[1].plot(r["val_steps"], r["val_losses"], color=color, linewidth=1.6, alpha=0.9, label=f"wd={r['wd']:g}")
            axes[1].scatter([r["val_steps"][-1]], [r["val_losses"][-1]], color=color, s=14)

    axes[0].set_title("Train Loss Curves (17 WD runs)")
    axes[1].set_title("Validation Loss Curves (17 WD runs)")
    axes[0].set_xlabel("Step")
    axes[1].set_xlabel("Step")
    axes[0].set_ylabel("Train Loss")
    axes[1].set_ylabel("Validation Loss")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.suptitle("Dense Single-Seed WD Sweep: All Loss Curves")
    p3 = RUNS_DIR / "all_loss_curves_17wd.png"
    plt.savefig(p3, dpi=220, bbox_inches="tight")
    plt.close(fig)

    report = RUNS_DIR / "REPORT.md"
    with report.open("w") as f:
        f.write("# Dense Single-Seed WD Sweep Report\n\n")
        f.write("wd | val_loss | train_loss | val_acc | delta_vs_wd0\n")
        f.write("---|---:|---:|---:|---:\n")
        for r in sorted(rows, key=lambda x: x["val_loss"]):
            f.write(
                f"{r['wd']} | {r['val_loss']:.4f} | {r['train_loss']:.4f} | "
                f"{r['val_acc']:.4f} | {r['delta_vs_wd0']:+.4f}\n"
            )
        f.write("\n")
        f.write(f"Best wd: `{best['wd']}` with val_loss `{best['val_loss']:.4f}`\n")
        if baseline is not None:
            f.write(f"Baseline wd=0 val_loss: `{baseline['val_loss']:.4f}`\n")
        f.write(f"Best delta vs wd=0: `{best['val_loss'] - baseline['val_loss']:+.4f}`\n")
        f.write("\n![Val Loss vs WD](./val_loss_vs_wd.png)\n")
        f.write("\n![Delta vs WD0](./delta_vs_wd0.png)\n")
        f.write("\n![All Loss Curves](./all_loss_curves_17wd.png)\n")

    print(f"Wrote: {report}")
    print(f"Wrote: {p1}")
    print(f"Wrote: {p2}")
    print(f"Wrote: {p3}")


if __name__ == "__main__":
    main()
