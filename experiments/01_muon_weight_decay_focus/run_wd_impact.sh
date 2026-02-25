#!/bin/bash
set -euo pipefail

BASE_DIR="experiments/01_muon_weight_decay_focus"
GEN_DIR="${BASE_DIR}/generated_configs/wd_impact"
RUNS_DIR="${BASE_DIR}/runs/wd_impact"

TRAIN_TOKENS="${TRAIN_TOKENS:-300000}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MUON_LR="${MUON_LR:-0.024}"

# WD-only sweep, includes inverse WD values.
WDS=("-0.01" "-0.005" "-0.001" "0.0" "0.001" "0.005" "0.01" "0.05" "0.1")
PHASE1_SEED=42
PHASE2_SEEDS=("42" "137" "256")

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
    echo "Skipping existing run ${run_name}"
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

  echo "============================================================"
  echo "Running ${run_name} (muon_weight_decay=${wd}, seed=${seed})"
  echo "============================================================"
  python train_llm.py --config_yaml "${cfg}" --num_workers "${NUM_WORKERS}" --compile false --log_every 5
}

echo "Phase 1: single-seed WD screen (seed=${PHASE1_SEED})"
for wd in "${WDS[@]}"; do
  run_case "${wd}" "${PHASE1_SEED}"
done

echo "Selecting top candidates from Phase 1..."
SELECTED_WDS="$(python - <<'PY'
import json
from pathlib import Path

base = Path("experiments/01_muon_weight_decay_focus/runs/wd_impact")
rows = []
for d in base.iterdir():
    if not d.is_dir() or not d.name.startswith("wd") or not d.name.endswith("_seed42"):
        continue
    m = d / "metrics.json"
    if not m.exists():
        continue
    data = json.loads(m.read_text())
    wd = data.get("experiment_config", {}).get("muon_weight_decay")
    vl = data.get("final_metrics", {}).get("val_loss")
    if wd is None or vl is None:
        continue
    rows.append((float(wd), float(vl)))

rows.sort(key=lambda x: x[1])
if not rows:
    raise SystemExit("No phase-1 rows found.")

# Keep baseline (0.0) + top-2 non-baseline.
baseline = [wd for wd, _ in rows if abs(wd) < 1e-12]
baseline_wd = baseline[0] if baseline else 0.0
top_non_baseline = []
for wd, _ in rows:
    if abs(wd) < 1e-12:
        continue
    top_non_baseline.append(wd)
    if len(top_non_baseline) == 2:
        break

selected = [baseline_wd] + top_non_baseline
print(" ".join(str(x) for x in selected))
PY
)"

echo "Phase 2 selected WDs: ${SELECTED_WDS}"
echo "Phase 2: multi-seed confirmation"
for wd in ${SELECTED_WDS}; do
  for seed in "${PHASE2_SEEDS[@]}"; do
    run_case "${wd}" "${seed}"
  done
done

python experiments/01_muon_weight_decay_focus/analyze_wd_impact.py
echo "WD impact experiment complete."
