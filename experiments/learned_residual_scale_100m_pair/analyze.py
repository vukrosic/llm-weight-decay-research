import json
from pathlib import Path

import matplotlib.pyplot as plt

RUNS_DIR = Path("experiments/learned_residual_scale_100m_pair/runs")
OUT_DIR = Path("experiments/learned_residual_scale_100m_pair")


def main():
    rows = []
    curves = []
    for d in sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()]):
        m = d / "metrics.json"
        if not m.exists():
            continue
        data = json.loads(m.read_text())
        fm = data.get("final_metrics", {})
        ec = data.get("experiment_config", {})
        ls = data.get("learned_scale_summary", {})
        h = data.get("history", {})

        mode = str(ec.get("residual_scale_mode", "fixed"))
        init = float(ec.get("residual_scale_init", 1.0))
        fixed_scale = float(ec.get("residual_scale", 1.0))
        variant = f"fixed:{fixed_scale:g}" if mode == "fixed" else f"{mode}:init{init:g}"

        rows.append(
            {
                "run": d.name,
                "variant": variant,
                "mode": mode,
                "init": init,
                "val_loss": float(fm.get("val_loss")),
                "train_loss": float(fm.get("train_loss")),
                "val_acc": float(fm.get("val_accuracy")),
                "steps": int(data.get("actual_steps")),
                "tokens_seen": int(data.get("tokens_seen")),
                "attn_scale_mean": ls.get("attn_scale_mean"),
                "ff_scale_mean": ls.get("ff_scale_mean"),
            }
        )
        curves.append(
            {
                "variant": variant,
                "train_steps": h.get("train_loss_steps", []),
                "train_losses": h.get("train_losses", []),
                "val_steps": h.get("steps", []),
                "val_losses": h.get("val_losses", []),
            }
        )

    if len(rows) < 2:
        raise SystemExit("Need at least two runs completed.")

    baseline = next((r for r in rows if r["mode"] == "fixed"), None)
    for r in rows:
        r["delta_vs_baseline"] = r["val_loss"] - baseline["val_loss"] if baseline else float("nan")

    rows_sorted = sorted(rows, key=lambda x: x["val_loss"])
    (OUT_DIR / "summary.json").write_text(json.dumps(rows_sorted, indent=2))

    labels = [r["variant"] for r in rows_sorted]
    vals = [r["val_loss"] for r in rows_sorted]
    bar_colors = ["#2563eb" if "fixed" in l else "#16a34a" for l in labels]
    plt.figure(figsize=(10.5, 5.3))
    plt.bar(labels, vals, color=bar_colors)
    if baseline:
        plt.axhline(baseline["val_loss"], linestyle="--", linewidth=1, color="gray")
    plt.ylabel("Final validation loss")
    plt.xticks(rotation=25, ha="right")
    plt.title("100M tokens: residual-scale ablations")
    plt.tight_layout()
    p1 = OUT_DIR / "final_val_loss_compare.png"
    plt.savefig(p1, dpi=220)
    plt.close()

    curve_map = {c["variant"]: c for c in curves}
    plt.figure(figsize=(10.5, 5.8))
    cmap = plt.get_cmap("tab10", len(rows_sorted))
    for i, r in enumerate(rows_sorted):
        c = curve_map[r["variant"]]
        color = cmap(i)
        if c["val_steps"] and c["val_losses"]:
            plt.plot(c["val_steps"], c["val_losses"], linestyle="-", linewidth=2.0, color=color, alpha=0.95, label=r["variant"])
    plt.xlabel("Step")
    plt.ylabel("Validation loss")
    plt.title("100M tokens validation-loss curves")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.tight_layout()
    p2 = OUT_DIR / "all_val_loss_lines.png"
    plt.savefig(p2, dpi=220, bbox_inches="tight")
    plt.close()

    report = OUT_DIR / "REPORT.md"
    with report.open("w") as f:
        f.write("# Learned Residual Scale 100M Ablation Report (Single Seed)\n\n")
        f.write("variant | val_loss | train_loss | val_acc | delta_vs_baseline | attn_scale_mean | ff_scale_mean\n")
        f.write("---|---:|---:|---:|---:|---:|---:\n")
        for r in rows_sorted:
            am = "" if r["attn_scale_mean"] is None else f"{r['attn_scale_mean']:.4f}"
            fm = "" if r["ff_scale_mean"] is None else f"{r['ff_scale_mean']:.4f}"
            f.write(f"{r['variant']} | {r['val_loss']:.4f} | {r['train_loss']:.4f} | {r['val_acc']:.4f} | {r['delta_vs_baseline']:+.4f} | {am} | {fm}\n")
        f.write("\n![Final val compare](./final_val_loss_compare.png)\n")
        f.write("\n![All val loss lines](./all_val_loss_lines.png)\n")

    print(f"Wrote: {report}")
    print(f"Wrote: {p1}")
    print(f"Wrote: {p2}")


if __name__ == "__main__":
    main()
