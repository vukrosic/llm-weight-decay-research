import json
import matplotlib.pyplot as plt
import os
from pathlib import Path

def plot_single_experiment(metrics_path, output_dir):
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    
    history = data['history']
    steps = history['steps']
    losses = history['val_losses']
    accs = history['val_accuracies']
    
    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, marker='o', color='tab:blue', label='Val Loss')
    plt.title(f"Validation Loss - {metrics_path.parent.name}")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "loss_finished.png")
    plt.close()

    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(steps, accs, marker='o', color='tab:green', label='Val Accuracy')
    plt.title(f"Validation Accuracy - {metrics_path.parent.name}")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "accuracy_finished.png")
    plt.close()

if __name__ == "__main__":
    metrics_file = Path("experiments/01_muon_vs_adamw_baseline/muon_seed42/metrics.json")
    out_dir = Path("plots/finished_run")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_single_experiment(metrics_file, out_dir)
    print(f"Saved finished run plots to {out_dir}")
