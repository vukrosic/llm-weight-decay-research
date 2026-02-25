# Experiment 02: Muon Weight Decay Focus

This experiment reimplements the Kimi paper Sec 2.2 Muon decay update:

`W <- W - lr * (O + lambda * W)`

and tests whether non-zero `muon_weight_decay` improves early efficiency on this 88M setup.

## Layout
- `PLAN.md`: research protocol and decision rules.
- `run_phase1.sh`: seed-42 screening sweep over decay values.
- `run_phase1_parallel_4gpu.sh`: same sweep, one run per GPU.
- `run_phase2.sh`: multi-seed confirmation for `wd=0` + two candidate non-zero decays.
- `analyze_results.py`: compute core endpoints from saved metrics.

## Phase 1 (screening)
Runs:
- `wd in {0.0, 0.05, 0.1, 0.2}`
- `seed=42`
- `train_tokens=300_000_000` (default)

Command:
```bash
bash experiments/02_muon_weight_decay_focus/run_phase1.sh
```

## Phase 2 (confirmation)
Runs:
- `wd=0.0` + two candidate non-zero decays from Phase 1
- `seeds={42,137,256}`
- `train_tokens=500_000_000` (default)

Command (example candidates):
```bash
WD_A=0.05 WD_B=0.1 bash experiments/02_muon_weight_decay_focus/run_phase2.sh
```

## Analysis
Use a fixed target loss threshold from your baseline policy.

Example:
```bash
python experiments/02_muon_weight_decay_focus/analyze_results.py \
  --phase phase1 \
  --target-loss 3.60
```

For phase2:
```bash
python experiments/02_muon_weight_decay_focus/analyze_results.py \
  --phase phase2 \
  --target-loss 3.60
```
