import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_run_name(name: str):
    # expected: muon_wd0p05_seed42
    wd_part = name.split("_")[1]
    seed_part = name.split("_")[2]
    wd = float(wd_part.replace("wd", "").replace("p", "."))
    seed = int(seed_part.replace("seed", ""))
    return wd, seed


def load_metrics(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def color_for_wd(wd: float):
    palette = {
        0.0: "#d7301f",
        0.05: "#1f78b4",
        0.1: "#2ca25f",
    }
    return palette.get(wd, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir",
        default="experiments/01_muon_weight_decay_focus/runs/social",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/01_muon_weight_decay_focus/runs/social",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("muon_wd")])
    if not run_dirs:
        raise SystemExit(f"No runs found in {runs_dir}")

    run_data = []
    for run_dir in run_dirs:
        wd, seed = parse_run_name(run_dir.name)
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = load_metrics(metrics_path)
        hist = metrics.get("history", {})
        run_data.append(
            {
                "name": run_dir.name,
                "wd": wd,
                "seed": seed,
                "tokens": hist.get("train_loss_tokens", []),
                "train_losses": hist.get("train_losses", []),
                "val_steps": hist.get("steps", []),
                "val_losses": hist.get("val_losses", []),
            }
        )

    if not run_data:
        raise SystemExit("No completed runs with metrics.json found.")

    plt.figure(figsize=(9, 6))
    for r in sorted(run_data, key=lambda x: x["wd"]):
        xs = r["tokens"]
        ys = r["train_losses"]
        if not xs or not ys:
            continue
        label = f"wd={r['wd']:.2f}"
        plt.plot(xs, ys, marker="o", linewidth=2, label=label, color=color_for_wd(r["wd"]))
    plt.title("Train Loss vs Tokens")
    plt.xlabel("Tokens seen")
    plt.ylabel("Train loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    train_path = out_dir / "train_loss_comparison.png"
    plt.savefig(train_path, dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    for r in sorted(run_data, key=lambda x: x["wd"]):
        xs = r["val_steps"]
        ys = r["val_losses"]
        if not xs or not ys:
            continue
        label = f"wd={r['wd']:.2f}"
        plt.plot(xs, ys, marker="o", linewidth=2, label=label, color=color_for_wd(r["wd"]))
    plt.title("Validation Loss vs Training Step")
    plt.xlabel("Training step")
    plt.ylabel("Validation loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    val_path = out_dir / "val_loss_comparison.png"
    plt.savefig(val_path, dpi=180)
    plt.close()

    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")


if __name__ == "__main__":
    main()
