#!/bin/bash
set -euo pipefail

BASE_DIR="experiments/01_muon_weight_decay_focus"
GEN_DIR="${BASE_DIR}/generated_configs/muon_kimi_update_local_search"
RUNS_DIR="${BASE_DIR}/runs/muon_kimi_update_local_search"

TRAIN_TOKENS="${TRAIN_TOKENS:-300000}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MUON_LR="${MUON_LR:-0.024}"
SEED="${SEED:-42}"
RES_SCALE="${RES_SCALE:-1.0}"

WDS=("0.0" "-0.05" "-0.1" "-0.15" "-0.2" "-0.25" "-0.3" "-0.4")

mkdir -p "${GEN_DIR}" "${RUNS_DIR}"

run_case() {
  local wd="$1"
  local tag_wd="${wd//-/m}"
  tag_wd="${tag_wd//./p}"
  local run_name="update_wd${tag_wd}_seed${SEED}"
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
muon_decay_mode: "update"
muon_weight_decay: ${wd}
residual_scale: ${RES_SCALE}
weight_decay: 0.2
log_every: 5
output_dir: "${out_dir}"
CFG

  echo "============================================================"
  echo "Running ${run_name} (mode=update, wd=${wd})"
  echo "============================================================"
  python train_llm.py --config_yaml "${cfg}" --num_workers "${NUM_WORKERS}" --compile false --log_every 5
}

echo "Muon update-mode local search"
echo "  wds=${WDS[*]}"
for wd in "${WDS[@]}"; do
  run_case "$wd"
done

python experiments/01_muon_weight_decay_focus/analyze_muon_kimi_update_local_search.py

echo "Muon update-mode local search complete."
