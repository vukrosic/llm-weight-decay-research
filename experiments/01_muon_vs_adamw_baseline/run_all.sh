#!/bin/bash

# Experiment 01: Muon vs AdamW Baseline
# 6 runs total (2 optimizers x 3 seeds)

set -e

echo "Starting Experiment 01 Suite..."

CONFIGS=(
    "experiments/01_muon_vs_adamw_baseline/configs/muon_seed42.yaml"
    "experiments/01_muon_vs_adamw_baseline/configs/muon_seed137.yaml"
    "experiments/01_muon_vs_adamw_baseline/configs/muon_seed256.yaml"
    "experiments/01_muon_vs_adamw_baseline/configs/adamw_seed42.yaml"
    "experiments/01_muon_vs_adamw_baseline/configs/adamw_seed137.yaml"
    "experiments/01_muon_vs_adamw_baseline/configs/adamw_seed256.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do
    echo "----------------------------------------------------------------"
    echo "Running configuration: $CONFIG"
    echo "----------------------------------------------------------------"
    python train_llm.py --config_yaml "$CONFIG" --track_manifold true
done

echo "All 6 runs completed."
