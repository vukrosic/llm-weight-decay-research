#!/bin/bash
set -euo pipefail

BASE_DIR="experiments/01_muon_weight_decay_focus"
GEN_DIR="${BASE_DIR}/generated_configs/phase2"
RUNS_DIR="${BASE_DIR}/runs/phase2"

TRAIN_TOKENS="${TRAIN_TOKENS:-500000000}"
SEEDS=("42" "137" "256")

# Set from shell env, example:
# WD_A=0.05 WD_B=0.1 bash experiments/01_muon_weight_decay_focus/run_phase2.sh
WD_A="${WD_A:-0.05}"
WD_B="${WD_B:-0.1}"
WDS=("0.0" "${WD_A}" "${WD_B}")

mkdir -p "${GEN_DIR}" "${RUNS_DIR}"

echo "Phase 2: train_tokens=${TRAIN_TOKENS}, candidates=${WD_A},${WD_B}"
for WD in "${WDS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    WD_TAG="${WD/./p}"
    RUN_NAME="muon_wd${WD_TAG}_seed${SEED}"
    OUT_DIR="${RUNS_DIR}/${RUN_NAME}"
    CFG="${GEN_DIR}/${RUN_NAME}.yaml"

    mkdir -p "${OUT_DIR}"
    cat > "${CFG}" <<EOF
optimizer_type: "muon"
seed: ${SEED}
train_tokens: ${TRAIN_TOKENS}
muon_weight_decay: ${WD}
weight_decay: 0.2
log_every: 50
output_dir: "${OUT_DIR}"
EOF

    echo "------------------------------------------------------------"
    echo "Running ${RUN_NAME} (muon_weight_decay=${WD})"
    echo "------------------------------------------------------------"
    python train_llm.py --config_yaml "${CFG}"
  done
done

echo "Phase 2 complete."
