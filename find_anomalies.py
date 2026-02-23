import json
import numpy as np
from pathlib import Path

def load_jsonl(path):
    data = []
    if not path.exists(): return data
    with open(path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

opts = ['muon', 'adamw']
seeds = [42, 137, 256]
base_path = Path("experiments/01_muon_vs_adamw_baseline")

all_data = {opt: {} for opt in opts}
for opt in opts:
    for seed in seeds:
        stats_file = base_path / f"{opt}_seed{seed}" / "metrics" / "manifold_stats.jsonl"
        records = load_jsonl(stats_file)
        # map (step, layer, proj) -> record
        d = {}
        for r in records:
            d[(r['step'], r['layer'], r['proj'])] = r
        all_data[opt][seed] = d

metrics = ['grad_norm', 'weight_norm', 'update_alignment', 'effective_rank']
print("Finding anomalies (max differences between seeds at the same step/layer/proj)...")

for opt in opts:
    print(f"\n--- {opt.upper()} anomalies ---")
    for metric in metrics:
        max_diff = 0
        max_diff_info = None
        
        # We need common keys across all seeds
        ref_seed = seeds[0]
        if not all_data[opt][ref_seed]: continue
        keys = list(all_data[opt][ref_seed].keys())
        
        for k in keys:
            vals = []
            for s in seeds:
                if k in all_data[opt][s] and metric in all_data[opt][s][k]:
                    vals.append(all_data[opt][s][k][metric])
            
            if len(vals) == len(seeds):
                diff = max(vals) - min(vals)
                if diff > max_diff:
                    max_diff = diff
                    max_diff_info = (k, vals)
                    
        if max_diff_info:
            print(f"Max {metric} variance: {max_diff:.4f} at Step {max_diff_info[0][0]}, Layer {max_diff_info[0][1]}, Proj {max_diff_info[0][2]} | Values: {[round(v, 4) for v in max_diff_info[1]]}")
            
