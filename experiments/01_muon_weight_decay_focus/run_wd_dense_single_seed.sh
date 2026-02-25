#!/bin/bash
set -euo pipefail

BASE_DIR="experiments/01_muon_weight_decay_focus"
GEN_DIR="${BASE_DIR}/generated_configs/wd_dense_single_seed"
RUNS_DIR="${BASE_DIR}/runs/wd_dense_single_seed"

TRAIN_TOKENS="${TRAIN_TOKENS:-300000}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MUON_LR="${MUON_LR:-0.024}"
SEED="${SEED:-42}"

# Dense positive/negative sweep around zero plus outer values.
WDS=(
  "-0.2" "-0.1" "-0.05" "-0.02" "-0.01" "-0.005" "-0.002" "-0.001"
  "0.0"
  "0.001" "0.002" "0.005" "0.01" "0.02" "0.05" "0.1" "0.2"
)

mkdir -p "${GEN_DIR}" "${RUNS_DIR}"

wd_tag() {
  local wd="$1"
  local tag="${wd//-/m}"
  tag="${tag//./p}"
  echo "${tag}"
}

run_case() {
  local wd="$1"
  local tag
  tag="$(wd_tag "${wd}")"
  local run_name="wd${tag}_seed${SEED}"
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
muon_weight_decay: ${wd}
weight_decay: 0.2
log_every: 5
output_dir: "${out_dir}"
CFG

  echo "============================================================"
  echo "Running ${run_name} (muon_weight_decay=${wd}, seed=${SEED})"
  echo "============================================================"
  python train_llm.py --config_yaml "${cfg}" --num_workers "${NUM_WORKERS}" --compile false --log_every 5
}

echo "Dense single-seed WD sweep"
echo "  tokens=${TRAIN_TOKENS}, muon_lr=${MUON_LR}, seed=${SEED}"
echo "  num_wd_points=${#WDS[@]}"

for wd in "${WDS[@]}"; do
  run_case "${wd}"
done

python experiments/01_muon_weight_decay_focus/analyze_wd_dense_single_seed.py
echo "Dense single-seed WD sweep complete."
