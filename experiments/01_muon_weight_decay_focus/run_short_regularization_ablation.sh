#!/bin/bash
set -euo pipefail

BASE_DIR="experiments/01_muon_weight_decay_focus"
GEN_DIR="${BASE_DIR}/generated_configs/short_ablation"
RUNS_DIR="${BASE_DIR}/runs/short_ablation"

SEED="${SEED:-42}"
TRAIN_TOKENS="${TRAIN_TOKENS:-300000}"
NUM_WORKERS="${NUM_WORKERS:-0}"

mkdir -p "${GEN_DIR}" "${RUNS_DIR}"

run_case() {
  local CASE="$1"
  local MUON_WD="$2"
  local MUON_LR="$3"
  local ADAMW_WD="$4"

  local RUN_NAME="${CASE}_seed${SEED}"
  local OUT_DIR="${RUNS_DIR}/${RUN_NAME}"
  local CFG="${GEN_DIR}/${RUN_NAME}.yaml"

  mkdir -p "${OUT_DIR}"
  cat > "${CFG}" <<EOF
optimizer_type: "muon"
seed: ${SEED}
train_tokens: ${TRAIN_TOKENS}
muon_weight_decay: ${MUON_WD}
muon_lr: ${MUON_LR}
weight_decay: ${ADAMW_WD}
log_every: 5
output_dir: "${OUT_DIR}"
EOF

  echo "============================================================"
  echo "Running ${RUN_NAME}"
  echo "  muon_weight_decay=${MUON_WD} muon_lr=${MUON_LR} adamw_weight_decay=${ADAMW_WD}"
  echo "============================================================"
  python train_llm.py --config_yaml "${CFG}" --num_workers "${NUM_WORKERS}" --compile false --log_every 5
}

# Baseline
run_case "baseline_wd0_lr024_adamw02" "0.0" "0.024" "0.2"

# Small positive decay
run_case "smallwd_0001_lr024_adamw02" "0.001" "0.024" "0.2"
run_case "smallwd_0005_lr024_adamw02" "0.005" "0.024" "0.2"
run_case "smallwd_001_lr024_adamw02" "0.01" "0.024" "0.2"

# Original larger decay references
run_case "refwd_005_lr024_adamw02" "0.05" "0.024" "0.2"
run_case "refwd_01_lr024_adamw02" "0.1" "0.024" "0.2"

# "Inverted decay" (negative decay / anti-decay)
run_case "negwd_0001_lr024_adamw02" "-0.001" "0.024" "0.2"
run_case "negwd_0005_lr024_adamw02" "-0.005" "0.024" "0.2"
run_case "negwd_001_lr024_adamw02" "-0.01" "0.024" "0.2"

# LR retunes with small decay
run_case "smallwd_001_lr016_adamw02" "0.01" "0.016" "0.2"
run_case "smallwd_001_lr02_adamw02" "0.01" "0.02" "0.2"
run_case "smallwd_001_lr028_adamw02" "0.01" "0.028" "0.2"

# Reduce "other regularization" (AdamW decay) while keeping Muon decay
run_case "smallwd_001_lr024_adamw00" "0.01" "0.024" "0.0"

echo "Short ablation complete."
