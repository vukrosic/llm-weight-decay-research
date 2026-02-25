# Muon Weight-Decay Research (88M)

This repository focuses on a single active study: reimplementing Muon decoupled weight decay from the Kimi paper and measuring early training-efficiency gains.

## Active experiment
- `experiments/01_muon_weight_decay_focus`

Read first:
- `experiments/01_muon_weight_decay_focus/PLAN.md`
- `experiments/01_muon_weight_decay_focus/README.md`

## Setup
```bash
pip install -r requirements.txt
```

## Run
Phase 1 screening:
```bash
python3 -c "
from datasets import load_dataset
import os
print('Downloading 1B Pretraining Data...')
ds = load_dataset('vukrosic/blueberry-1B-pretrain')
os.makedirs('processed_data/pretrain_1B', exist_ok=True)
ds.save_to_disk('processed_data/pretrain_1B')
"
```

## 🔬 Running Research Experiments

For rigorous investigation, we use explicit YAML configs to ensure reproducibility. To launch the suite across multiple GPUs:

**GPU 0 (Muon Seed 42)**: 
```bash
bash experiments/01_muon_weight_decay_focus/run_phase1_parallel_4gpu.sh
```

Phase 2 confirmation:
```bash
WD_A=0.05 WD_B=0.1 bash experiments/01_muon_weight_decay_focus/run_phase2.sh
```

## Analyze
```bash
python experiments/01_muon_weight_decay_focus/analyze_results.py --phase phase1 --target-loss 3.60
python experiments/01_muon_weight_decay_focus/analyze_results.py --phase phase2 --target-loss 3.60
```
