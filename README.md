# Muon Training Dynamics: A Structural Investigation (88M)

This repository investigates how the **Muon optimizer** shapes transformer representations during training. We focus on the geometric evolution of weight manifolds, specifically looking for anomalies and learning signatures that differentiate manifold-aware optimization from standard AdamW.

## 🔬 Research Focus: Hierarchical Stretching

Our core investigation centers on "Hierarchical Stretching"—the observation that different layers in a transformer specialize spectrally over time. We track manifold spectral statistics (spectral norm, entropy, subspace alignment) to understand the "hidden geometry" of training.

## 🚀 Getting Started

### 1. Environment Setup
```bash
git clone https://github.com/vukrosic/muon-llm-research
cd muon-llm-research
pip install -r requirements.txt
```

### 2. Data Preparation
For the 2B research suite, you need the full token mix:
```bash
python3 -c "
from datasets import load_dataset
import os
print('Downloading 2B Pretraining Data...')
ds = load_dataset('vukrosic/blueberry-2B-pretrain')
os.makedirs('processed_data/pretrain_2B', exist_ok=True)
ds.save_to_disk('processed_data/pretrain_2B')
"
```

## 🔬 Running Research Experiments

For rigorous investigation, we use explicit YAML configs to ensure reproducibility. To launch the suite across multiple GPUs:

**GPU 0 (Muon Seed 42)**: 
```bash
CUDA_VISIBLE_DEVICES=0 python train_llm.py --config_yaml configs/experiments/muon_seed_42.yaml
```

**GPU 1 (AdamW Seed 42)**: 
```bash
CUDA_VISIBLE_DEVICES=1 python train_llm.py --config_yaml configs/experiments/adamw_seed_42.yaml
```

Each specific config in `configs/experiments/` defines its own `output_dir` and `seed` and inherits the base architecture from `base_2B.yaml`.

## 📊 Analysis & Visualization

After training, generate research visualizations from the logged manifold metrics:
```bash
python research_muon/track_manifold.py --plot_only
```
Plots will be saved to `results/research_plots/`.

## 🛠 Contribution Guidelines

We welcome contributions that improve our understanding of optimizer geometry or training efficiency.
- **Rigor over speed**: We value stable spectral signatures over minor FLOP reductions.
- **Reproducibility**: All experiments must be accompanied by a YAML config.
- **Single Variable**: Only change one hyperparameter or architectural feature at a time to isolate effects.

---
*Training Dynamics Investigation*
