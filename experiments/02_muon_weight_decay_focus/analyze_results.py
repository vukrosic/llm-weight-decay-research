import argparse
import json
import math
from pathlib import Path
from collections import defaultdict


def first_crossing(xs, ys, target):
    for i, y in enumerate(ys):
        if y <= target:
            return xs[i]
    return None


def auc_trapezoid(xs, ys):
    if len(xs) < 2:
        return float("nan")
    area = 0.0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        area += dx * (ys[i] + ys[i - 1]) * 0.5
    return area


def parse_run_name(name):
    # expected: muon_wd0p05_seed42
    wd_part = name.split("_")[1]  # wd0p05
    seed_part = name.split("_")[2]  # seed42
    wd = float(wd_part.replace("wd", "").replace("p", "."))
    seed = int(seed_part.replace("seed", ""))
    return wd, seed


def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)


def summarize_run(run_dir, target_loss):
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None

    data = load_metrics(metrics_path)
    hist = data.get("history", {})

    losses = hist.get("train_losses", [])
    tokens = hist.get("train_loss_tokens", [])
    minutes = hist.get("train_loss_elapsed_minutes", [])
    if not losses or not tokens or not minutes:
        return None

    token_hit = first_crossing(tokens, losses, target_loss)
    minute_hit = first_crossing(minutes, losses, target_loss)
    auc = auc_trapezoid(tokens, losses)
    final_val_loss = data.get("final_metrics", {}).get("val_loss", float("nan"))

    return {
        "tokens_to_target_loss": token_hit if token_hit is not None else float("inf"),
        "minutes_to_target_loss": minute_hit if minute_hit is not None else float("inf"),
        "auc_train_loss_early": auc,
        "final_val_loss": final_val_loss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["phase1", "phase2"], required=True)
    parser.add_argument("--base-dir", default="experiments/02_muon_weight_decay_focus/runs")
    parser.add_argument("--target-loss", type=float, required=True)
    args = parser.parse_args()

    phase_dir = Path(args.base_dir) / args.phase
    if not phase_dir.exists():
        raise SystemExit(f"Missing run directory: {phase_dir}")

    grouped = defaultdict(list)
    for run_dir in sorted([p for p in phase_dir.iterdir() if p.is_dir()]):
        if not run_dir.name.startswith("muon_wd"):
            continue
        parsed = parse_run_name(run_dir.name)
        wd, seed = parsed
        summary = summarize_run(run_dir, args.target_loss)
        if summary is None:
            continue
        grouped[wd].append((seed, summary))

    if not grouped:
        raise SystemExit("No valid runs found with train-loss history.")

    print(f"Phase: {args.phase}")
    print(f"Target loss: {args.target_loss}")
    print("")
    print("Per-decay summary")
    print("wd\tseeds\tmean_tokens_to_target\tmean_minutes_to_target\tmean_auc\tmean_final_val_loss")

    ranked = []
    for wd in sorted(grouped.keys()):
        rows = grouped[wd]
        n = len(rows)
        mean_tokens = sum(r["tokens_to_target_loss"] for _, r in rows) / n
        mean_minutes = sum(r["minutes_to_target_loss"] for _, r in rows) / n
        mean_auc = sum(r["auc_train_loss_early"] for _, r in rows) / n
        mean_vloss = sum(r["final_val_loss"] for _, r in rows) / n
        print(f"{wd:.4f}\t{n}\t{mean_tokens:.1f}\t{mean_minutes:.3f}\t{mean_auc:.3e}\t{mean_vloss:.4f}")
        ranked.append((mean_tokens, wd))

    ranked.sort()
    best_tokens, best_wd = ranked[0]
    print("")
    if math.isinf(best_tokens):
        print("No run reached target loss. Increase token budget or relax target.")
    else:
        print(f"Best by primary endpoint (tokens_to_target_loss): wd={best_wd:.4f}")


if __name__ == "__main__":
    main()
