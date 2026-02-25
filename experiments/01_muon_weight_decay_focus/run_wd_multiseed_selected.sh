#!/bin/bash
set -euo pipefail

BASE_DIR="experiments/01_muon_weight_decay_focus"
SRC_SUMMARY="${BASE_DIR}/runs/short_ablation/summary.json"
GEN_DIR="${BASE_DIR}/generated_configs/wd_multiseed_selected"
RUNS_DIR="${BASE_DIR}/runs/wd_multiseed_selected"

TRAIN_TOKENS="${TRAIN_TOKENS:-300000}"
MUON_LR="${MUON_LR:-0.024}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEEDS=("42" "137" "256")

mkdir -p "${GEN_DIR}" "${RUNS_DIR}"

SELECTED_WDS="$(python - <<'PY'
import json
from pathlib import Path

p = Path("experiments/01_muon_weight_decay_focus/runs/short_ablation/summary.json")
rows = json.loads(p.read_text())

# Keep only WD-only, fixed-lr, fixed-adamw control runs from phase-1 seed42.
candidates = []
for r in rows:
    run = r["run"]
    if not run.endswith("_seed42"):
        continue
    if "_lr024_adamw02_" not in run:
        continue
    # Exclude lr sweep and adamw00 variants; keep baseline/small/ref/neg wd family.
    if not (run.startswith("baseline_") or run.startswith("smallwd_") or run.startswith("refwd_") or run.startswith("negwd_")):
        continue
    candidates.append((float(r["muon_wd"]), float(r["val_loss"]), run))

candidates.sort(key=lambda x: x[1])  # lower val loss is better

baseline = 0.0
top_non_baseline = []
for wd, _, _ in candidates:
    if abs(wd) < 1e-12:
        continue
    if wd not in top_non_baseline:
        top_non_baseline.append(wd)
    if len(top_non_baseline) == 2:
        break

selected = [baseline] + top_non_baseline
print(" ".join(str(x) for x in selected))
PY
)"

echo "Selected WDs for multi-seed confirmation: ${SELECTED_WDS}"

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
log_every: 5
output_dir: "${out_dir}"
EOF

  echo "Running ${run_name}"
  python train_llm.py --config_yaml "${cfg}" --num_workers "${NUM_WORKERS}" --compile false --log_every 5
}

for wd in ${SELECTED_WDS}; do
  for seed in "${SEEDS[@]}"; do
    run_case "${wd}" "${seed}"
  done
done

python experiments/01_muon_weight_decay_focus/analyze_wd_multiseed_selected.py
echo "Multi-seed WD confirmation complete."
