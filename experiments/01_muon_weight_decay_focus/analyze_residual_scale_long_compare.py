import json
from pathlib import Path

import matplotlib.pyplot as plt


RUNS_DIR = Path("experiments/01_muon_weight_decay_focus/runs/residual_scale_long_compare")


def parse_scale(name: str) -> float:
    tok = name.split("_")[0].replace("res", "")
    return float(tok.replace("p", "."))


def main():
    rows = []
    curves = []
    for d in sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()]):
        m = d / "metrics.json"
        if not m.exists():
            continue
        data = json.loads(m.read_text())
        h = data.get("history", {})
        fm = data.get("final_metrics", {})
        scale = parse_scale(d.name)
        rows.append(
            {
                "run": d.name,
                "residual_scale": scale,
                "train_loss": float(fm.get("train_loss")),
                "val_loss": float(fm.get("val_loss")),
                "val_acc": float(fm.get("val_accuracy")),
                "tokens_seen": int(data.get("tokens_seen")),
                "steps": int(data.get("actual_steps")),
            }
        )
        curves.append(
            {
                "run": d.name,
                "residual_scale": scale,
                "train_steps": h.get("train_loss_steps", []),
                "train_losses": h.get("train_losses", []),
                "val_steps": h.get("steps", []),
                "val_losses": h.get("val_losses", []),
            }
        )

    if len(rows) < 2:
        raise SystemExit("Need both runs completed.")

    rows = sorted(rows, key=lambda x: x["residual_scale"])
    base = next((r for r in rows if abs(r["residual_scale"] - 1.0) < 1e-12), None)
    best = next((r for r in rows if abs(r["residual_scale"] - 0.69) < 1e-12), None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    colors = {1.0: "#0f172a", 0.69: "#16a34a"}
    for c in curves:
        s = c["residual_scale"]
        label = f"scale={s:g}"
        axes[0].plot(c["train_steps"], c["train_losses"], linewidth=2.0, color=colors.get(s), label=label)
        axes[1].plot(c["val_steps"], c["val_losses"], linewidth=2.0, color=colors.get(s), label=label)

    axes[0].set_title("Train Loss (20M tokens)")
    axes[1].set_title("Validation Loss (20M tokens)")
    axes[0].set_xlabel("Step")
    axes[1].set_xlabel("Step")
    axes[0].set_ylabel("Train Loss")
    axes[1].set_ylabel("Validation Loss")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    out_plot = RUNS_DIR / "loss_compare_20m_tokens.png"
    plt.savefig(out_plot, dpi=220)
    plt.close(fig)

    out_json = RUNS_DIR / "summary.json"
    out_json.write_text(json.dumps(rows, indent=2))

    report = RUNS_DIR / "REPORT.md"
    with report.open("w") as f:
        f.write("# Residual-Scale Long Comparison (20M Tokens)\n\n")
        f.write("residual_scale | val_loss | train_loss | val_acc | tokens_seen | steps | delta_vs_scale1\n")
        f.write("---|---:|---:|---:|---:|---:|---:\n")
        for r in sorted(rows, key=lambda x: x["residual_scale"]):
            delta = r["val_loss"] - base["val_loss"] if base else float("nan")
            f.write(
                f"{r['residual_scale']} | {r['val_loss']:.4f} | {r['train_loss']:.4f} | {r['val_acc']:.4f} | "
                f"{r['tokens_seen']} | {r['steps']} | {delta:+.4f}\n"
            )
        if base and best:
            f.write("\n")
            f.write(f"Baseline (scale=1.0) val_loss: `{base['val_loss']:.4f}`\\\n\n")
            f.write(f"Best candidate (scale=0.69) val_loss: `{best['val_loss']:.4f}`\\\n\n")
            f.write(f"Delta (0.69 - 1.0): `{best['val_loss'] - base['val_loss']:+.4f}`\n")
        f.write("\n![Loss compare](./loss_compare_20m_tokens.png)\n")

    print(f"Wrote: {report}")
    print(f"Wrote: {out_plot}")
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
