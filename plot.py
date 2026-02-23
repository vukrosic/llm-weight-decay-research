import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def compute_entropy(singular_values):
    s = np.array(singular_values)
    if len(s) == 0: return 0
    # Normalize to get a distribution
    s_norm = s / (np.sum(s) + 1e-12)
    # Shannon entropy
    entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
    return entropy

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True, help="Path to experiment directory")
    parser.add_argument("--metric", type=str, default="entropy", help="Metric to plot (entropy, rank, alignment, grad_norm, weight_norm, update_alignment)")
    parser.add_argument("--proj", type=str, default="q,v", help="Projections to plot (comma separated, e.g., q,v,k,o)")
    parser.add_argument("--layers", type=str, default="1,6,12", help="Layers to plot (comma separated, e.g., 1,6,12)")
    args = parser.parse_args()

    # Parse segments
    projs = [p.strip().lower() for p in args.proj.split(',')]
    try:
        layers = [int(l.strip()) for l in args.layers.split(',')]
    except ValueError:
        print(f"Warning: Could not parse layers '{args.layers}'. Ensure they are integers.")
        layers = []
        
    exp_path = Path(args.experiment)
    
    # Common locations for metrics
    possible_dirs = [
        exp_path / "metrics" / "raw",
        exp_path / "raw_metrics",
        exp_path
    ]
    
    raw_dir = None
    for d in possible_dirs:
        if d.exists() and (list(d.glob("*.jsonl")) or (d / "metrics.json").exists()):
            raw_dir = d
            break
            
    if not raw_dir:
        print(f"Error: Could not find metrics in {exp_path}")
        return

    metric = args.metric.lower()
    
    # Handle the unified format from Task 2 if it exists, or legacy split format
    unified_file = raw_dir / "manifold_stats.jsonl"
    all_records = []
    
    if unified_file.exists():
        print(f"Loading unified metrics from {unified_file}...")
        all_records = load_jsonl(unified_file)
    else:
        # Legacy/Current split format
        print("Using split metrics format...")
        if metric in ['entropy', 'singular_values']:
            all_records = load_jsonl(raw_dir / "singular_values.jsonl")
        elif metric in ['alignment', 'update_alignment']:
            # Try both possible names
            all_records = load_jsonl(raw_dir / "alignment.jsonl")
            if not all_records:
                all_records = load_jsonl(raw_dir / "manifold_stats.jsonl") 
        elif metric in ['rank', 'effective_rank', 'grad_norm', 'weight_norm', 'update_norm']:
            all_records = load_jsonl(raw_dir / "norms.jsonl")
            if not all_records:
                all_records = load_jsonl(raw_dir / "manifold_stats.jsonl")

    if not all_records:
        print(f"No records found for metric {metric}")
        return

    # Filter and process
    plt.figure(figsize=(12, 7))
    
    plot_count = 0
    for p in projs:
        for l in layers:
            filtered = [r for r in all_records if r.get('proj', '').lower() == p and r.get('layer') == l]
            if not filtered:
                continue
            
            filtered.sort(key=lambda x: x['step'])
            steps = [r['step'] for r in filtered]
            
            values = []
            for r in filtered:
                if metric == 'entropy':
                    if 'singular_values' in r:
                        values.append(compute_entropy(r['singular_values']))
                    else:
                        values.append(0)
                elif metric in ['rank', 'effective_rank']:
                    values.append(r.get('effective_rank', r.get('rank', 0)))
                elif metric in ['alignment', 'update_alignment']:
                    values.append(r.get('alignment', r.get('update_alignment', 0)))
                elif metric == 'grad_norm':
                    values.append(r.get('grad_norm', 0))
                elif metric == 'weight_norm':
                    values.append(r.get('weight_norm', 0))
                elif metric == 'update_norm':
                    values.append(r.get('update_norm', 0))
                else:
                    values.append(r.get(metric, 0))
            
            if values:
                plt.plot(steps, values, label=f"L{l}-{p.upper()}", marker='o', markersize=4, alpha=0.8)
                plot_count += 1

    if plot_count == 0:
        print(f"No data points to plot for projs={projs}, layers={layers}")
        return

    plt.title(f"{metric.upper()} vs Steps\nExperiment: {exp_path.name}", fontsize=14)
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel(metric.replace('_', ' ').capitalize(), fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    out_file = output_dir / f"plot_{exp_path.name}_{metric}.png"
    plt.savefig(out_file)
    print(f"Successfully saved plot to {out_file}")

if __name__ == "__main__":
    main()
