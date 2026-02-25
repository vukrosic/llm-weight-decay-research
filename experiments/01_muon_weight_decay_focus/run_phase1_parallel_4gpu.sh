#!/bin/bash
set -euo pipefail

BASE_DIR="experiments/01_muon_weight_decay_focus"
GEN_DIR="${BASE_DIR}/generated_configs/phase1"
RUNS_DIR="${BASE_DIR}/runs/phase1"

SEED=42
TRAIN_TOKENS="${TRAIN_TOKENS:-300000000}"
WDS=("0.0" "0.05" "0.1" "0.2")

mkdir -p "${GEN_DIR}" "${RUNS_DIR}"

echo "Phase 1 parallel: seed=${SEED}, train_tokens=${TRAIN_TOKENS}"
for i in "${!WDS[@]}"; do
  WD="${WDS[$i]}"
  WD_TAG="${WD/./p}"
  RUN_NAME="muon_wd${WD_TAG}_seed${SEED}"
  OUT_DIR="${RUNS_DIR}/${RUN_NAME}"
  RAW_DIR="${OUT_DIR}/metrics"
  CFG="${GEN_DIR}/${RUN_NAME}.yaml"
  LOG="${OUT_DIR}/training.log"

  mkdir -p "${OUT_DIR}" "${RAW_DIR}"
  cat > "${CFG}" <<EOF
optimizer_type: "muon"
seed: ${SEED}
train_tokens: ${TRAIN_TOKENS}
muon_weight_decay: ${WD}
weight_decay: 0.2
log_every: 50
detailed_log_every: 250
output_dir: "${OUT_DIR}"
raw_metrics_dir: "${RAW_DIR}"
EOF

  echo "Launching ${RUN_NAME} on GPU ${i}"
  CUDA_VISIBLE_DEVICES="${i}" \
    python train_llm.py --config_yaml "${CFG}" --track_manifold true > "${LOG}" 2>&1 &
done

echo "All runs launched. Waiting..."
wait
echo "Phase 1 parallel complete."
