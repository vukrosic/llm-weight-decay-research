import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def load_jsonl(path):
    data = []
    if not os.path.exists(path):
        return data
    with open(path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def compute_entropy(singular_values):
    s = np.array(singular_values)
    if len(s) == 0: return 0
    s_norm = s / (np.sum(s) + 1e-12)
    entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
    return entropy

def get_experiment_data(exp_dir):
    stats_file = Path(exp_dir) / "metrics" / "manifold_stats.jsonl"
    if not stats_file.exists():
        # Fallback to any other possible location
        stats_file = Path(exp_dir) / "raw_metrics" / "manifold_stats.jsonl"
        if not stats_file.exists():
            stats_file = Path(exp_dir) / "manifold_stats.jsonl"
            
    records = load_jsonl(stats_file)
    if not records:
        print(f"Warning: No stats found for {exp_dir}")
        return None
    
    # Process entropy if requested metric is entropy
    for r in records:
        if 'singular_values' in r:
            r['entropy'] = compute_entropy(r['singular_values'])
            # Don't keep the full singular values in memory for all records if we don't need them
            # del r['singular_values'] 
            
    return records

def plot_grid(all_data, metric='entropy'):
    # all_data is a dict: { 'muon': [ [run1_records], [run2_records], ... ], 'adamw': [...] }
    
    layers_to_plot = [1, 3, 6, 9, 12]
    projections = ['q', 'k', 'v', 'o']
    
    fig, axes = plt.subplots(len(layers_to_plot), len(projections), figsize=(20, 15), sharex=True)
    
    colors = {'muon': 'blue', 'adamw': 'red'}
    
    for r_idx, layer in enumerate(layers_to_plot):
        for c_idx, proj in enumerate(projections):
            ax = axes[r_idx, c_idx]
            
            for opt, runs in all_data.items():
                run_values = [] # List of (steps, values) tuples
                
                for records in runs:
                    filtered = [r for r in records if r.get('layer') == layer and r.get('proj', '').lower() == proj]
                    if not filtered: continue
                    filtered.sort(key=lambda x: x['step'])
                    steps = [r['step'] for r in filtered]
                    values = [r.get(metric, 0) for r in filtered]
                    run_values.append((steps, values))
                
                if not run_values: continue
                
                for i, (steps, vals) in enumerate(run_values):
                    label = opt.upper() if i == 0 else None
                    ax.plot(steps, vals, color=colors[opt], alpha=0.6, linewidth=1.5, label=label)
            
            if r_idx == 0:
                ax.set_title(f"Projection: {proj.upper()}", fontsize=14)
            if c_idx == 0:
                ax.set_ylabel(f"Layer {layer}\n{metric.capitalize()}", fontsize=12)
            
            if r_idx == len(layers_to_plot) - 1:
                ax.set_xlabel("Step")
                
            if r_idx == 0 and c_idx == 0:
                ax.legend()
                
    plt.tight_layout()
    os.makedirs("results/analysis", exist_ok=True)
    plt.savefig(f"results/analysis/baseline_comparison_{metric}.png")
    print(f"Saved plot to results/analysis/baseline_comparison_{metric}.png")

def main():
    base_path = Path("experiments/01_muon_vs_adamw_baseline")
    opts = ['muon', 'adamw']
    seeds = [42, 137, 256]
    
    all_data = {'muon': [], 'adamw': []}
    
    for opt in opts:
        for seed in seeds:
            exp_dir = base_path / f"{opt}_seed{seed}"
            print(f"Loading {exp_dir}...")
            data = get_experiment_data(exp_dir)
            if data:
                all_data[opt].append(data)
                
    if not all_data['muon'] and not all_data['adamw']:
        print("No data found to plot. Ensure experiments completed.")
        return

    for metric in ['entropy', 'update_alignment', 'grad_norm', 'weight_norm', 'effective_rank']:
        print(f"Plotting grid for {metric}...")
        try:
            plot_grid(all_data, metric=metric)
        except Exception as e:
            print(f"Failed to plot {metric}: {e}")

if __name__ == "__main__":
    main()
