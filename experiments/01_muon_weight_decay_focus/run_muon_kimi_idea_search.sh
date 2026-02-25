#!/bin/bash
set -euo pipefail

BASE_DIR="experiments/01_muon_weight_decay_focus"
GEN_DIR="${BASE_DIR}/generated_configs/muon_kimi_idea_search"
RUNS_DIR="${BASE_DIR}/runs/muon_kimi_idea_search"

TRAIN_TOKENS="${TRAIN_TOKENS:-300000}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MUON_LR="${MUON_LR:-0.024}"
SEED="${SEED:-42}"
RES_SCALE="${RES_SCALE:-1.0}"

mkdir -p "${GEN_DIR}" "${RUNS_DIR}"

run_case() {
  local mode="$1"
  local wd="$2"
  local tag_wd="${wd//-/m}"
  tag_wd="${tag_wd//./p}"
  local run_name="mode${mode}_wd${tag_wd}_seed${SEED}"
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
muon_decay_mode: "${mode}"
muon_weight_decay: ${wd}
residual_scale: ${RES_SCALE}
weight_decay: 0.2
log_every: 5
output_dir: "${out_dir}"
CFG

  echo "============================================================"
  echo "Running ${run_name} (mode=${mode}, wd=${wd})"
  echo "============================================================"
  python train_llm.py --config_yaml "${cfg}" --num_workers "${NUM_WORKERS}" --compile false --log_every 5
}

# Phase 1: broad idea sweep around Kimi-style decoupled decay placement.
PHASE1=(
  "param 0.0"
  "param 0.01"
  "param -0.01"
  "param 0.05"
  "param -0.05"
  "update 0.05"
  "update -0.05"
  "update 0.2"
  "update -0.2"
  "both 0.01"
  "both -0.01"
)

echo "Muon Kimi-style idea search - Phase 1"
echo "  tokens=${TRAIN_TOKENS}, seed=${SEED}, muon_lr=${MUON_LR}, residual_scale=${RES_SCALE}"
for item in "${PHASE1[@]}"; do
  mode=$(echo "$item" | awk '{print $1}')
  wd=$(echo "$item" | awk '{print $2}')
  run_case "$mode" "$wd"
done

# If phase1 best is still marginal, expand phase2 automatically.
BEST_DELTA=$(python - <<'PY'
import json
from pathlib import Path

base = Path('experiments/01_muon_weight_decay_focus/runs/muon_kimi_idea_search')
rows = []
for d in base.iterdir():
    if not d.is_dir() or not d.name.endswith('_seed42'):
        continue
    m = d / 'metrics.json'
    if not m.exists():
        continue
    data = json.loads(m.read_text())
    ec = data.get('experiment_config', {})
    mode = ec.get('muon_decay_mode', 'param')
    wd = float(ec.get('muon_weight_decay', 0.0))
    vl = float(data.get('final_metrics', {}).get('val_loss'))
    rows.append((mode, wd, vl))

base_row = next((r for r in rows if r[0] == 'param' and abs(r[1]) < 1e-12), None)
if base_row is None:
    print('nan')
else:
    base_v = base_row[2]
    best_v = min(v for _,_,v in rows)
    print(best_v - base_v)
PY
)

echo "Phase1 best delta vs baseline: ${BEST_DELTA}"

if python - <<PY
x = float('${BEST_DELTA}')
# Non-marginal threshold for these short runs.
raise SystemExit(0 if x <= -0.005 else 1)
PY
then
  echo "Found non-marginal gain in Phase 1; skipping Phase 2."
else
  echo "Phase 1 marginal; running Phase 2 expanded ideas."
  PHASE2=(
    "update 0.4"
    "update -0.4"
    "both 0.05"
    "both -0.05"
    "param 0.1"
    "param -0.1"
  )
  for item in "${PHASE2[@]}"; do
    mode=$(echo "$item" | awk '{print $1}')
    wd=$(echo "$item" | awk '{print $2}')
    run_case "$mode" "$wd"
  done
fi

python experiments/01_muon_weight_decay_focus/analyze_muon_kimi_idea_search.py

echo "Muon Kimi-style idea search complete."
