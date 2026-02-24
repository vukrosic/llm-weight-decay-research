import os
import json
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def get_data_from_json(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data.get('history', {})
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def get_data_from_checkpoint(path):
    try:
        # Load on CPU, weights_only=False is needed for dicts/history
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        return ckpt.get('metrics_history', {})
    except Exception as e:
        print(f"Error loading checkpoint {path}: {e}")
        return None

def main():
    base_dir = Path("experiments/01_muon_vs_adamw_baseline")
    output_dir = Path("results/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # 1. Look for completed runs (metrics.json)
    for metrics_json in base_dir.glob("*/metrics.json"):
        exp_name = metrics_json.parent.name
        print(f"Adding finished run: {exp_name}")
        history = get_data_from_json(metrics_json)
        if history and 'steps' in history:
            plt.plot(history['steps'], history['val_losses'], label=f"{exp_name} (final)", marker='o', alpha=0.7)

    # 2. Look for active run in the global checkpoints dir
    # We saw checkpoints/latest_checkpoint.pt exists
    latest_ckpt = Path("checkpoints/latest_checkpoint.pt")
    if latest_ckpt.exists():
        print(f"Found latest checkpoint, checking for active run data...")
        history = get_data_from_checkpoint(latest_ckpt)
        if history and 'steps' in history:
            # We need to guess which experiment this is. 
            # In our case, muon_seed137 is the one currently running.
            # We can also check the config if it's in the checkpoint.
            plt.plot(history['steps'], history['val_losses'], label="Active Run (from checkpoint)", linestyle='--', marker='x', color='red')

    plt.title("Muon vs AdamW Baseline - Validation Loss", fontsize=16)
    plt.xlabel("Optimizer Steps", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log') # Loss often looks better on log scale if there's high initial loss
    
    out_path = output_dir / "suite_loss_comparison.png"
    plt.savefig(out_path)
    print(f"Saved suite comparison to {out_path}")

    # Plot Accuracy too
    plt.figure(figsize=(12, 8))
    for metrics_json in base_dir.glob("*/metrics.json"):
        exp_name = metrics_json.parent.name
        history = get_data_from_json(metrics_json)
        if history and 'steps' in history:
            plt.plot(history['steps'], history['val_accuracies'], label=f"{exp_name} (final)", marker='o', alpha=0.7)

    if latest_ckpt.exists():
        history = get_data_from_checkpoint(latest_ckpt)
        if history and 'steps' in history:
            plt.plot(history['steps'], history['val_accuracies'], label="Active Run (from checkpoint)", linestyle='--', marker='x', color='red')

    plt.title("Muon vs AdamW Baseline - Validation Accuracy", fontsize=16)
    plt.xlabel("Optimizer Steps", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path_acc = output_dir / "suite_accuracy_comparison.png"
    plt.savefig(out_path_acc)
    print(f"Saved suite accuracy comparison to {out_path_acc}")

if __name__ == "__main__":
    main()
