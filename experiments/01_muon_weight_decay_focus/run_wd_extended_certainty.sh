#!/bin/bash
set -euo pipefail

BASE_DIR="experiments/01_muon_weight_decay_focus"
GEN_DIR="${BASE_DIR}/generated_configs/wd_extended_certainty"
RUNS_DIR="${BASE_DIR}/runs/wd_extended_certainty"

TRAIN_TOKENS="${TRAIN_TOKENS:-2400000}"
MUON_LR="${MUON_LR:-0.024}"
NUM_WORKERS="${NUM_WORKERS:-0}"

# Focused comparison: best inverse wd vs baseline vs symmetric positive wd.
WDS=("-0.01" "0.0" "0.01")
SEEDS_CSV="${SEEDS_CSV:-42,137,256,512,1024}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"

mkdir -p "${GEN_DIR}" "${RUNS_DIR}"

wd_tag() {
  local wd="$1"
  local tag="${wd//-/m}"
  tag="${tag//./p}"
  echo "${tag}"
}

run_case() {
  local wd="$1"
  local seed="$2"
  local tag
  tag="$(wd_tag "${wd}")"
  local run_name="wd${tag}_seed${seed}"
  local out_dir="${RUNS_DIR}/${run_name}"
  local cfg="${GEN_DIR}/${run_name}.yaml"

  if [[ -f "${out_dir}/metrics.json" ]]; then
    echo "Skipping existing ${run_name}"
    return
  fi

  mkdir -p "${out_dir}"
  cat > "${cfg}" <<EOF
optimizer_type: "muon"
seed: ${seed}
train_tokens: ${TRAIN_TOKENS}
muon_lr: ${MUON_LR}
muon_weight_decay: ${wd}
weight_decay: 0.2
log_every: 20
output_dir: "${out_dir}"
EOF

  echo "============================================================"
  echo "Running ${run_name} | tokens=${TRAIN_TOKENS}"
  echo "============================================================"
  python train_llm.py --config_yaml "${cfg}" --num_workers "${NUM_WORKERS}" --compile false --log_every 20
}

echo "Extended certainty run:"
echo "  train_tokens=${TRAIN_TOKENS}"
echo "  muon_lr=${MUON_LR}"
echo "  wds=${WDS[*]}"
echo "  seeds=${SEEDS[*]}"

for wd in "${WDS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_case "${wd}" "${seed}"
  done
done

python experiments/01_muon_weight_decay_focus/analyze_wd_extended_certainty.py
echo "Extended certainty experiment complete."
