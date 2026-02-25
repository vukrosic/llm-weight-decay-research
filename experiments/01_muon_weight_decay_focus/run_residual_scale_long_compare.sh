#!/bin/bash
set -euo pipefail

BASE_DIR="experiments/01_muon_weight_decay_focus"
GEN_DIR="${BASE_DIR}/generated_configs/residual_scale_long_compare"
RUNS_DIR="${BASE_DIR}/runs/residual_scale_long_compare"

TRAIN_TOKENS="${TRAIN_TOKENS:-20000000}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MUON_LR="${MUON_LR:-0.024}"
MUON_WD="${MUON_WD:-0.0}"
SEED="${SEED:-42}"
SCALES=("1.00" "0.69")

mkdir -p "${GEN_DIR}" "${RUNS_DIR}"

scale_tag() {
  local s="$1"
  local tag="${s//./p}"
  echo "${tag}"
}

run_case() {
  local s="$1"
  local tag
  tag="$(scale_tag "${s}")"
  local run_name="res${tag}_seed${SEED}"
  local out_dir="${RUNS_DIR}/${run_name}"
  local cfg="${GEN_DIR}/${run_name}.yaml"

  if [[ -f "${out_dir}/metrics.json" ]]; then
    echo "Skipping existing ${run_name}"
    return
  fi

  mkdir -p "${out_dir}"
  cat > "${cfg}" <<CFG
optimizer_type: "muon"
seed: ${SEED}
train_tokens: ${TRAIN_TOKENS}
muon_lr: ${MUON_LR}
muon_weight_decay: ${MUON_WD}
residual_scale: ${s}
weight_decay: 0.2
log_every: 20
output_dir: "${out_dir}"
CFG

  echo "============================================================"
  echo "Running ${run_name} (residual_scale=${s}, seed=${SEED}, tokens=${TRAIN_TOKENS})"
  echo "============================================================"
  python train_llm.py --config_yaml "${cfg}" --num_workers "${NUM_WORKERS}" --compile false --log_every 20
}

echo "Residual-scale long comparison"
echo "  scales=${SCALES[*]}"
echo "  tokens=${TRAIN_TOKENS}, seed=${SEED}, muon_lr=${MUON_LR}, muon_wd=${MUON_WD}"

for s in "${SCALES[@]}"; do
  run_case "${s}"
done

python experiments/01_muon_weight_decay_focus/analyze_residual_scale_long_compare.py

echo "Residual-scale long comparison complete."
