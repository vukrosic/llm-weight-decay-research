import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_jsonl(path):
    data = []
    if not os.path.exists(path): return data
    with open(path, 'r') as f:
        for line in f:
            try: data.append(json.loads(line))
            except: continue
    return data

def compute_entropy(singular_values):
    s = np.array(singular_values)
    if len(s) == 0: return 0
    s_norm = s / (np.sum(s) + 1e-12)
    return -np.sum(s_norm * np.log(s_norm + 1e-12))

def plot_heatmaps(records, exp_name, output_dir, metric='entropy'):
    projections = ['q', 'k', 'v', 'o', 'up', 'down']
    
    if metric == 'entropy':
        for r in records:
            if 'singular_values' in r:
                r['entropy'] = compute_entropy(r['singular_values'])

    data_dict = {(r['step'], r['layer'], r['proj']): r for r in records}
    steps = sorted(list(set([k[0] for k in data_dict.keys()])))
    layers = sorted(list(set([k[1] for k in data_dict.keys()])))

    fig, axes = plt.subplots(len(projections), 1, figsize=(10, 4 * len(projections)))
    
    for idx, proj in enumerate(projections):
        ax = axes[idx]
        heatmap = np.zeros((len(layers), len(steps)))
        for sid, s in enumerate(steps):
            for lid, l in enumerate(layers):
                key = (s, l, proj)
                if key in data_dict:
                    heatmap[lid, sid] = data_dict[key].get(metric, 0)
        
        im = ax.imshow(heatmap, aspect='auto', origin='lower', cmap='viridis',
                      extent=[steps[0], steps[-1], layers[0], layers[-1]])
        fig.colorbar(im, ax=ax)
        ax.set_title(f"{exp_name} - {proj.upper()} - {metric}")
        ax.set_ylabel("Layer Depth")
        if idx == len(projections) - 1: ax.set_xlabel("Step")

    plt.tight_layout()
    plt.savefig(output_dir / f"heatmap_{metric}_seed137.png")
    plt.close()

def plot_single_experiment(metrics_path, output_dir):
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    
    history = data['history']
    steps = history['steps']
    losses = history['val_losses']
    accs = history['val_accuracies']
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, marker='o', color='tab:blue', label='Val Loss')
    plt.title(f"Validation Loss - Muon Seed 137")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "loss_seed137.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(steps, accs, marker='o', color='tab:green', label='Val Accuracy')
    plt.title(f"Validation Accuracy - Muon Seed 137")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "accuracy_seed137.png")
    plt.close()

if __name__ == "__main__":
    exp_dir = Path("experiments/01_muon_vs_adamw_baseline/muon_seed137")
    out_dir = Path("experiments/01_muon_vs_adamw_baseline/plots_seed137")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Standard plots
    metrics_file = exp_dir / "metrics.json"
    if metrics_file.exists():
        plot_single_experiment(metrics_file, out_dir)
    
    # Heatmaps
    stats_file = exp_dir / "metrics" / "manifold_stats.jsonl"
    if stats_file.exists():
        records = load_jsonl(stats_file)
        for m in ['entropy', 'update_alignment', 'effective_rank']:
            print(f"Generating heatmap for {m}...")
            plot_heatmaps(records, "Muon Seed 137", out_dir, metric=m)
    
    print(f"Done. Plots saved to {out_dir}")
