import json
from pathlib import Path

import matplotlib.pyplot as plt


RUNS_DIR = Path("experiments/01_muon_weight_decay_focus/runs/wd_dense_single_seed")


def parse_wd(name: str) -> float:
    wd_token = name.split("_")[0].replace("wd", "")
    return float(wd_token.replace("m", "-").replace("p", "."))


def main():
    rows = []
    for d in sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()]):
        m = d / "metrics.json"
        if not m.exists():
            continue
        data = json.loads(m.read_text())
        fm = data.get("final_metrics", {})
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

    print(f"Wrote: {report}")
    print(f"Wrote: {p1}")
    print(f"Wrote: {p2}")


if __name__ == "__main__":
    main()
