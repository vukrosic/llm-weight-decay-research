# Muon Training Dynamics: A Structural Investigation (88M)

This repository investigates how the **Muon optimizer** shapes transformer representations during training. We focus on the geometric evolution of weight manifolds, specifically looking for anomalies and learning signatures that differentiate manifold-aware optimization from standard AdamW.

## 🔬 Research Focus: Hierarchical Stretching

Our primary research direction is **Hierarchical Stretching**—the observation that Muon-trained models dynamically modulate spectral signatures across layers. We provide tools to track these manifolds in real-time.

### Key Research Features:
- **Manifold Tracking**: Track spectral norms, spectral gaps, and singular value distributions per-layer.
- **Spectral Analysis Utilities**: High-performance SVD-based statistics calculation (CPU-offloaded).
- **Visualization Suite**: Generate hierarchical stretching reports and layer-wise heatmaps.

---

## 🚀 Getting Started

### 0. Setup

```bash
pip install -r requirements.txt
```

Download dataset:
```bash
python3 -c "
from datasets import load_dataset
import os
print('Downloading 2B Pretraining Data...')
ds = load_dataset('vukrosic/blueberry-2B-pretrain')
os.makedirs('processed_data/pretrain_2B', exist_ok=True)
ds.save_to_disk('processed_data/pretrain_2B')
print('✅ Full Data Ready!')
"
```

### 1. Training with Manifold Tracking
To train a model while tracking its manifold evolution, use the `--track_manifold true` flag:

```bash
python train_llm.py --train_tokens 8000000 --track_manifold true
```

### 2. Generating Research Plots
After training, you can generate the research visualizations using the tracking script:

```bash
python research_muon/track_manifold.py --plot_only
```

Plots will be saved to `results/research_plots/`.

---

## 📊 Experimental Setup
- **Model**: Blueberry (88M Params, 22 layers, 512 d_model)
- **Optimizer**: Muon (Orthogonalized 2D Updates)
- **Dataset**: Blueberry-2B (Full pre-training mix)

## 📈 Recent Findings
Check out our latest research reports in the `research_muon/` folder:
- [Hierarchical Stretching Report](research_muon/hierarchical_stretching.md)
- [Spectral Dynamics Update vs Weight](research_muon/spectral_dynamics_update_vs_weight.md)


