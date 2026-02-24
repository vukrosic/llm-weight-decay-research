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

def compute_entropy_fixed(singular_values, top_k=50):
    # Always truncate to top_k to ensure consistency
    s = np.array(singular_values[:top_k])
    if len(s) <= 1: return 0
    s_norm = s / (np.sum(s) + 1e-12)
    entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
    return entropy

def plot_heatmaps_consistent(records, exp_name, output_dir, metric='entropy'):
    projections = ['q', 'k', 'v', 'o', 'up', 'down']
    
    if metric == 'entropy':
        for r in records:
            if 'singular_values' in r:
                # Use fixed top-50 entropy
                r['entropy_consistent'] = compute_entropy_fixed(r['singular_values'], top_k=50)
        work_metric = 'entropy_consistent'
        label_name = "Entropy (Top 50 SVs)"
    else:
        work_metric = metric
        label_name = metric.replace('_', ' ').title()

    all_steps = sorted(list(set([r['step'] for r in records])))
    all_layers = sorted(list(set([r['layer'] for r in records])))
    
    if not all_steps: return
    step_to_idx = {s: i for i, s in enumerate(all_steps)}
    layer_to_idx = {l: i for i, l in enumerate(all_layers)}

    fig, axes = plt.subplots(len(projections), 1, figsize=(12, 4 * len(projections)))
    
    for idx, proj in enumerate(projections):
        ax = axes[idx]
        heatmap = np.full((len(all_layers), len(all_steps)), np.nan)
        for r in records:
            if r['proj'] == proj:
                s_idx = step_to_idx[r['step']]
                l_idx = layer_to_idx[r['layer']]
                heatmap[l_idx, s_idx] = r.get(work_metric, 0)
        
        heatmap = np.ma.masked_invalid(heatmap)
        X, Y = np.meshgrid(all_steps, all_layers)
        im = ax.pcolormesh(X, Y, heatmap, shading='auto', cmap='magma' if 'entropy' in metric else 'viridis')
        fig.colorbar(im, ax=ax)
        
        ax.set_title(f"{exp_name} - {proj.upper()} - {label_name}")
        ax.set_ylabel("Layer Depth")
        if idx == len(projections) - 1: ax.set_xlabel("Optimizer Step")

    plt.tight_layout()
    plt.savefig(output_dir / f"heatmap_{metric}_consistent.png")
    plt.close()

if __name__ == "__main__":
    for seed in ["42", "137"]:
        exp_dir = Path(f"experiments/01_muon_vs_adamw_baseline/muon_seed{seed}")
        out_dir = Path(f"experiments/01_muon_vs_adamw_baseline/plots_seed{seed}_consistent")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        stats_file = exp_dir / "metrics" / "manifold_stats.jsonl"
        if stats_file.exists():
            records = load_jsonl(stats_file)
            for m in ['entropy', 'update_alignment', 'effective_rank']:
                print(f"Generating consistent {m} heatmap for Seed {seed}...")
                plot_heatmaps_consistent(records, f"Muon Seed {seed}", out_dir, metric=m)

    print("Done. Generated comparison-safe consistent plots (Top 50 SVs).")
