#!/bin/bash
set -euo pipefail

BASE_DIR="experiments/01_muon_weight_decay_focus"
GEN_DIR="${BASE_DIR}/generated_configs/social"
RUNS_DIR="${BASE_DIR}/runs/social"

SEED="${SEED:-42}"
TRAIN_TOKENS="${TRAIN_TOKENS:-100000000}"
LOG_EVERY="${LOG_EVERY:-50}"
NUM_WORKERS="${NUM_WORKERS:-0}"
WDS=("0.0" "0.05" "0.1")

mkdir -p "${GEN_DIR}" "${RUNS_DIR}"

echo "Social matrix: seed=${SEED}, train_tokens=${TRAIN_TOKENS}"
for WD in "${WDS[@]}"; do
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
log_every: ${LOG_EVERY}
output_dir: "${OUT_DIR}"
EOF

  echo "------------------------------------------------------------"
  echo "Running ${RUN_NAME} (muon_weight_decay=${WD})"
  echo "------------------------------------------------------------"
  python train_llm.py \
    --config_yaml "${CFG}" \
    --log_every "${LOG_EVERY}" \
    --num_workers "${NUM_WORKERS}"
done

echo "All 3 runs complete."
