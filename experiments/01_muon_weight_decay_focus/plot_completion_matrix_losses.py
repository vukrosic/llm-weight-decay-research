import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE = Path("experiments/01_muon_weight_decay_focus/runs")

MATRICES = [
    ("wd_multiseed_selected", "WD Multi-Seed Selected (9/9)"),
    ("wd_extended_certainty", "WD Extended Certainty (15/15)"),
    ("wd_longer_certainty", "WD Longer Certainty (9/9)"),
]

COLORS = {
    -0.01: "#ef4444",
    -0.005: "#f97316",
    0.0: "#0f172a",
    0.01: "#22c55e",
}


def parse_wd(run_name: str) -> float:
    wd_token = run_name.split("_")[0].replace("wd", "")
    return float(wd_token.replace("m", "-").replace("p", "."))


def load_runs(matrix_dir: Path):
    rows = []
    for run_dir in sorted([p for p in matrix_dir.iterdir() if p.is_dir()]):
        mpath = run_dir / "metrics.json"
        if not mpath.exists():
            continue
        data = json.loads(mpath.read_text())
        h = data.get("history", {})
        rows.append(
            {
                "run": run_dir.name,
                "wd": parse_wd(run_dir.name),
                "train_steps": np.array(h.get("train_loss_steps", []), dtype=float),
                "train_losses": np.array(h.get("train_losses", []), dtype=float),
                "val_steps": np.array(h.get("steps", []), dtype=float),
                "val_losses": np.array(h.get("val_losses", []), dtype=float),
            }
        )
    return rows


def plot_metric(ax, runs, step_key, loss_key, ylabel):
    by_wd = {}
    for r in runs:
        if len(r[step_key]) == 0 or len(r[loss_key]) == 0:
            continue
        by_wd.setdefault(r["wd"], []).append(r)

    for wd in sorted(by_wd.keys()):
        items = by_wd[wd]
        min_len = min(len(x[loss_key]) for x in items)
        if min_len == 0:
            continue
        xs = items[0][step_key][:min_len]
        ys = np.stack([x[loss_key][:min_len] for x in items], axis=0)
        mean = ys.mean(axis=0)
        std = ys.std(axis=0)
        color = COLORS.get(wd, None)
        label = f"wd={wd:g} (n={len(items)})"
        ax.plot(xs, mean, linewidth=2.0, color=color, label=label)
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.18)

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)


def main():
    fig, axes = plt.subplots(3, 2, figsize=(15, 14), constrained_layout=True)

    for i, (matrix_name, title) in enumerate(MATRICES):
        matrix_dir = BASE / matrix_name
        runs = load_runs(matrix_dir)
        if not runs:
            continue
        plot_metric(axes[i, 0], runs, "train_steps", "train_losses", "Train Loss")
        plot_metric(axes[i, 1], runs, "val_steps", "val_losses", "Validation Loss")
        axes[i, 0].set_title(f"{title} - Train")
        axes[i, 1].set_title(f"{title} - Val")

    out = BASE / "completion_matrix_all_losses.png"
    fig.suptitle("Completion vs Planned Matrix: All Loss Curves", fontsize=16)
    plt.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
