import json
from pathlib import Path

import matplotlib.pyplot as plt

RUNS_DIR = Path("experiments/learned_residual_scale_long/runs")


def main():
    rows = []
    curve_rows = []
    for d in sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()]):
        m = d / "metrics.json"
        if not m.exists():
            continue
        data = json.loads(m.read_text())
        fm = data.get("final_metrics", {})
        ec = data.get("experiment_config", {})
        ls = data.get("learned_scale_summary", {})
        mode = str(ec.get("residual_scale_mode", "fixed"))
        init = float(ec.get("residual_scale_init", 1.0))
        fixed_scale = float(ec.get("residual_scale", 1.0))

        if mode == "fixed":
            variant = f"fixed:{fixed_scale:g}"
        else:
            variant = f"{mode}:init{init:g}"

        rows.append(
            {
                "run": d.name,
                "variant": variant,
                "mode": mode,
                "init": init,
                "fixed_scale": fixed_scale,
                "val_loss": float(fm.get("val_loss")),
                "train_loss": float(fm.get("train_loss")),
                "val_acc": float(fm.get("val_accuracy")),
                "steps": int(data.get("actual_steps")),
                "tokens_seen": int(data.get("tokens_seen")),
                "attn_scale_mean": ls.get("attn_scale_mean"),
                "ff_scale_mean": ls.get("ff_scale_mean"),
                "layer_scale_mean": ls.get("layer_scale_mean"),
            }
        )
        h = data.get("history", {})
        curve_rows.append(
            {
                "variant": variant,
                "train_steps": h.get("train_loss_steps", []),
                "train_losses": h.get("train_losses", []),
                "val_steps": h.get("steps", []),
                "val_losses": h.get("val_losses", []),
            }
        )

    if not rows:
        raise SystemExit("No completed runs found.")

    baseline = next(
        (r for r in rows if r["mode"] == "fixed" and abs(r["fixed_scale"] - 1.0) < 1e-12),
        None,
    )
    for r in rows:
        r["delta_vs_baseline"] = r["val_loss"] - baseline["val_loss"] if baseline else float("nan")

    rows_sorted = sorted(rows, key=lambda x: x["val_loss"])
    out_dir = Path("experiments/learned_residual_scale_long")
    (out_dir / "summary.json").write_text(json.dumps(rows_sorted, indent=2))

    labels = [r["variant"] for r in rows_sorted]
    vals = [r["val_loss"] for r in rows_sorted]
    plt.figure(figsize=(10, 5.5))
    plt.bar(labels, vals, color="#2563eb")
    if baseline:
        plt.axhline(
            baseline["val_loss"], linestyle="--", linewidth=1.0, color="gray", label="fixed:1.0 baseline"
        )
        plt.legend(frameon=False)
    plt.ylabel("Final validation loss")
    plt.xticks(rotation=20, ha="right")
    plt.title("Learned residual scale at 35M tokens")
    plt.tight_layout()
    p1 = out_dir / "ranked_val_loss.png"
    plt.savefig(p1, dpi=220)
    plt.close()

    deltas = [r["delta_vs_baseline"] for r in rows_sorted]
    colors = ["#16a34a" if d < 0 else "#dc2626" for d in deltas]
    plt.figure(figsize=(10, 5.5))
    plt.bar(labels, deltas, color=colors)
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.ylabel("Delta vs fixed:1.0 baseline")
    plt.xticks(rotation=20, ha="right")
    plt.title("Ablation impact at 35M tokens")
    plt.tight_layout()
    p2 = out_dir / "delta_vs_baseline.png"
    plt.savefig(p2, dpi=220)
    plt.close()

    plt.figure(figsize=(11, 6.2))
    cmap = plt.get_cmap("tab10", len(rows_sorted))
    curves_by_variant = {c["variant"]: c for c in curve_rows}
    for i, r in enumerate(rows_sorted):
        color = cmap(i)
        c = curves_by_variant.get(r["variant"])
        if not c:
            continue
        if c["train_steps"] and c["train_losses"]:
            plt.plot(
                c["train_steps"],
                c["train_losses"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                color=color,
                label=f"{r['variant']} train",
            )
        if c["val_steps"] and c["val_losses"]:
            plt.plot(
                c["val_steps"],
                c["val_losses"],
                linestyle="-",
                linewidth=2.0,
                alpha=0.95,
                color=color,
                label=f"{r['variant']} val",
            )
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("All Loss Curves (solid=val, dashed=train)")
    plt.grid(alpha=0.25)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    plt.tight_layout()
    p3 = out_dir / "all_loss_lines.png"
    plt.savefig(p3, dpi=220, bbox_inches="tight")
    plt.close()

    report = out_dir / "REPORT.md"
    best = rows_sorted[0]
    with report.open("w") as f:
        f.write("# Learned Residual Scale Long-Run Report (Single Seed)\n\n")
        f.write(
            "variant | mode | init | val_loss | train_loss | val_acc | "
            "attn_scale_mean | ff_scale_mean | layer_scale_mean | delta_vs_baseline\n"
        )
        f.write("---|---|---:|---:|---:|---:|---:|---:|---:|---:\n")
        for r in rows_sorted:
            attn_mean = "" if r["attn_scale_mean"] is None else f"{r['attn_scale_mean']:.4f}"
            ff_mean = "" if r["ff_scale_mean"] is None else f"{r['ff_scale_mean']:.4f}"
            layer_mean = "" if r["layer_scale_mean"] is None else f"{r['layer_scale_mean']:.4f}"
            f.write(
                f"{r['variant']} | {r['mode']} | {r['init']} | {r['val_loss']:.4f} | {r['train_loss']:.4f} | "
                f"{r['val_acc']:.4f} | {attn_mean} | {ff_mean} | {layer_mean} | {r['delta_vs_baseline']:+.4f}\n"
            )

        f.write("\n")
        f.write(f"Best variant: `{best['variant']}` with val_loss=`{best['val_loss']:.4f}`\n")
        if baseline:
            f.write(f"Baseline (fixed:1.0) val_loss=`{baseline['val_loss']:.4f}`\n")
            f.write(f"Best delta vs baseline: `{best['delta_vs_baseline']:+.4f}`\n")
        f.write("\n![Ranked val loss](./ranked_val_loss.png)\n")
        f.write("\n![Delta vs baseline](./delta_vs_baseline.png)\n")
        f.write("\n![All loss lines](./all_loss_lines.png)\n")

    print(f"Wrote: {report}")
    print(f"Wrote: {p1}")
    print(f"Wrote: {p2}")
    print(f"Wrote: {p3}")


if __name__ == "__main__":
    main()
