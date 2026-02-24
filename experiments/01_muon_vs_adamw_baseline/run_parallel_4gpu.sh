#!/bin/bash
# 4-GPU Parallel Launcher for remaining 4 experiments
# Each experiment is assigned to a specific GPU and logs to its own directory.

set -e

echo "Starting Parallel Execution on 4 GPUs..."

# GPU 0: Muon Seed 256
CUDA_VISIBLE_DEVICES=0 python train_llm.py --config_yaml experiments/01_muon_vs_adamw_baseline/configs/muon_seed256.yaml --track_manifold true > experiments/01_muon_vs_adamw_baseline/muon_seed256/training.log 2>&1 &
echo "Launched Muon Seed 256 on GPU 0"

# GPU 1: AdamW Seed 42
CUDA_VISIBLE_DEVICES=1 python train_llm.py --config_yaml experiments/01_muon_vs_adamw_baseline/configs/adamw_seed42.yaml --track_manifold true > experiments/01_muon_vs_adamw_baseline/adamw_seed42/training.log 2>&1 &
echo "Launched AdamW Seed 42 on GPU 1"

# GPU 2: AdamW Seed 137
CUDA_VISIBLE_DEVICES=2 python train_llm.py --config_yaml experiments/01_muon_vs_adamw_baseline/configs/adamw_seed137.yaml --track_manifold true > experiments/01_muon_vs_adamw_baseline/adamw_seed137/training.log 2>&1 &
echo "Launched AdamW Seed 137 on GPU 2"

# GPU 3: AdamW Seed 256
CUDA_VISIBLE_DEVICES=3 python train_llm.py --config_yaml experiments/01_muon_vs_adamw_baseline/configs/adamw_seed256.yaml --track_manifold true > experiments/01_muon_vs_adamw_baseline/adamw_seed256/training.log 2>&1 &
echo "Launched AdamW Seed 256 on GPU 3"

echo "All 4 background processes started. Use 'nvitop' or 'tail' on the logs to monitor."
echo "Waiting for all processes to complete..."
wait

echo "All 4 parallel runs have completed."
