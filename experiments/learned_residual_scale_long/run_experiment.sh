#!/bin/bash
set -euo pipefail

BASE_DIR="experiments/learned_residual_scale_long"
GEN_DIR="${BASE_DIR}/generated_configs"
RUNS_DIR="${BASE_DIR}/runs"

TRAIN_TOKENS="${TRAIN_TOKENS:-35000000}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"

mkdir -p "${GEN_DIR}" "${RUNS_DIR}"

make_tag() {
  local tok="$1"
  tok="${tok//-/m}"
  tok="${tok//./p}"
  echo "${tok}"
}

run_case() {
  local mode="$1"
  local init="$2"
  local fixed_scale="$3"
  local mode_tag init_tag fixed_tag
  mode_tag="$(make_tag "${mode}")"
  init_tag="$(make_tag "${init}")"
  fixed_tag="$(make_tag "${fixed_scale}")"

  local run_name="mode${mode_tag}_init${init_tag}_fixed${fixed_tag}_seed${SEED}"
  local out_dir="${RUNS_DIR}/${run_name}"
  local cfg="${GEN_DIR}/${run_name}.yaml"

  if [[ -f "${out_dir}/metrics.json" ]]; then
    echo "Skipping existing ${run_name}"
    return
  fi

  mkdir -p "${out_dir}"
  cat > "${cfg}" <<CFG
seed: ${SEED}
train_tokens: ${TRAIN_TOKENS}
residual_scale_mode: "${mode}"
residual_scale_init: ${init}
residual_scale: ${fixed_scale}
output_dir: "${out_dir}"
CFG

  echo "============================================================"
  echo "Running ${run_name}"
  echo "  train_tokens=${TRAIN_TOKENS}, residual_scale_mode=${mode}, residual_scale_init=${init}"
  echo "============================================================"
  python train_llm.py --config_yaml "${cfg}" --num_workers "${NUM_WORKERS}"
}

echo "Learned residual-scale long run"
echo "  seed=${SEED}, tokens=${TRAIN_TOKENS}"

CASES=(
  "fixed 1.0 1.0"
  "learned_layer 0.1 1.0"
  "learned_branch 0.1 1.0"
  "learned_branch 0.01 1.0"
)

for item in "${CASES[@]}"; do
  mode="$(echo "${item}" | awk '{print $1}')"
  init="$(echo "${item}" | awk '{print $2}')"
  fixed_scale="$(echo "${item}" | awk '{print $3}')"
  run_case "${mode}" "${init}" "${fixed_scale}"
done

python experiments/learned_residual_scale_long/analyze.py

echo "Long run complete."
