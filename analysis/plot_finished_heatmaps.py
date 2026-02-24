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
    
    # Process entropy
    if metric == 'entropy':
        for r in records:
            if 'singular_values' in r:
                r['entropy'] = compute_entropy(r['singular_values'])

    # Build dict
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
    plt.savefig(output_dir / f"heatmap_{metric}_finished.png")
    plt.close()

if __name__ == "__main__":
    exp_dir = Path("experiments/01_muon_vs_adamw_baseline/muon_seed42")
    stats_file = exp_dir / "metrics" / "manifold_stats.jsonl"
    records = load_jsonl(stats_file)
    
    out_dir = Path("plots/finished_run")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for m in ['entropy', 'update_alignment', 'effective_rank']:
        print(f"Generating heatmap for {m}...")
        plot_heatmaps(records, "Muon Seed 42", out_dir, metric=m)
