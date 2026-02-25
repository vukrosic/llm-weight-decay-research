#!/bin/bash
set -euo pipefail

BASE_DIR="experiments/01_muon_weight_decay_focus"
GEN_DIR="${BASE_DIR}/generated_configs/residual_scale_winner_ablation"
RUNS_DIR="${BASE_DIR}/runs/residual_scale_winner_ablation"

TRAIN_TOKENS="${TRAIN_TOKENS:-300000}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MUON_LR="${MUON_LR:-0.024}"
MUON_WD="${MUON_WD:-0.0}"
SEED="${SEED:-42}"

# Focused sweep around previous winner (0.70), with baseline anchors.
SCALES=(
  "0.60" "0.64" "0.66" "0.68" "0.69" "0.70" "0.71" "0.72" "0.74" "0.76" "0.80" "0.90" "1.00"
)

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
log_every: 5
output_dir: "${out_dir}"
CFG

  echo "============================================================"
  echo "Running ${run_name} (residual_scale=${s}, seed=${SEED})"
  echo "============================================================"
  python train_llm.py --config_yaml "${cfg}" --num_workers "${NUM_WORKERS}" --compile false --log_every 5
}

echo "Single-seed local residual-scale ablation"
echo "  tokens=${TRAIN_TOKENS}, muon_lr=${MUON_LR}, muon_wd=${MUON_WD}, seed=${SEED}"
for s in "${SCALES[@]}"; do
  run_case "${s}"
done

python experiments/01_muon_weight_decay_focus/analyze_residual_scale_winner_ablation.py
echo "Residual-scale winner single-seed ablation complete."
