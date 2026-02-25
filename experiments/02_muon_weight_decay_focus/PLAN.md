# Experiment 02: Muon Weight Decay Reimplementation (Single-Question Plan)

## Single research question
At small scale (88M), does **decoupled weight decay inside Muon** improve **early training efficiency** (faster loss reduction per token) versus vanilla Muon?

## Pre-registered primary endpoint
- Primary endpoint: `tokens_to_target_loss` at a fixed target train loss.
- Target definition: choose one threshold from a short pilot (`wd=0.0`, seed 42), then freeze it for all runs.
- Interpretation: lower is better.

## What is learned from the Kimi paper
This plan is directly derived from the paper, with explicit mapping:

1. Core mechanism from paper Sec 2.2 (Eq. 3):
   - Paper change: `W <- W - lr * (O + lambda * W)`.
   - Our plan reimplements this exact Muon decay term and compares against `lambda=0`.

2. Paper motivation:
   - Paper reports that without decay, Muon weights/output RMS can grow and hurt performance; adding decay improves behavior.
   - Our measurements include weight-growth trends (`weight_norm` slopes/max) as the same failure-mode proxy.

3. Paper methodology style:
   - They isolate optimizer-rule changes while keeping architecture/training setup fixed.
   - Our plan also varies only one factor: `muon_weight_decay`.

4. Paper hyperparameter spirit:
   - Paper uses a single global decay value in training (e.g., fixed decay in stages).
   - Our dose-response (`0.0, 0.05, 0.1, 0.2`) is the minimal extension to find the useful range on our setup.

## Small-scale adaptation (explicitly not a paper claim)
- Paper’s strongest advantage appears in longer/over-train regimes at larger scale.
- On 88M, we first test an early proxy: loss reduction per token at fixed budget.
- If early signal is positive, we run a longer confirmation stage to check whether paper-like late behavior appears.

## Why this is the right focus
- Kimi reports weight decay as a key scaling fix for Muon.
- Current repo behavior is not a full reimplementation yet:
  - `weight_decay` is applied in AdamW groups.
  - Muon-optimized matrix params currently have no decay term in `optimizers/muon.py`.
- So the first meaningful experiment is to reimplement this exact mechanism, then measure impact.

## Scope (strictly one thing)
- Vary only Muon decay strength `muon_weight_decay` while keeping architecture/data/schedule fixed.
- Remove AdamW entirely from this experiment.

## Minimal implementation needed before runs
1. Add decoupled decay term to Muon step:
   - Update from `W <- W - lr * O` to `W <- W - lr * (O + lambda * W)`.
2. Add config knob `muon_weight_decay` (default `0.0` for backward compatibility).
3. Pass `muon_weight_decay` from trainer setup into Muon optimizer.
4. Log this field into run config/metrics metadata.

## Reimplementation experiment matrix (Muon only)
Phase 1 (cheap screening, early signal):
- Seed: 42 only.
- `muon_weight_decay in {0.0, 0.05, 0.1, 0.2}`.
- `train_tokens=300_000_000` (or 500M if affordable).
- Total: 4 runs.

Optional Phase 1b (only if needed):
- If both `0.05` and `0.1` outperform `0.0` in Phase 1, add one tie-break run at `wd=0.075` (seed 42).

Phase 2 (confirmation):
- Keep top 2 settings from Phase 1 + always include `wd=0.0`.
- Seeds: 42, 137, 256.
- `train_tokens=500_000_000`.
- Total: 9 runs max (often 6 if top2 includes `wd=0`).

## Small novel extension on top (still same question)
Dose-response curve of Muon decay at early stage:
- Quantify where decay starts helping/hurting efficiency (`wd=0.0/0.05/0.1/0.2`).

## Budget and schedule
Two-stage plan emphasizing early-emerging results:
- Stage A: very short Muon-only sweep for signal detection.
- Stage B: multi-seed confirmation at moderate token budget.

Optional Stage C (only if Phase 2 is positive):
- Long-run check (`1_700_000_000` tokens) for late-stage behavior.
- Trigger rule (pre-registered): run Stage C only if best non-zero `wd` improves mean Phase 2 `tokens_to_target_loss` by >=5% vs `wd=0.0` and has no instability flags.

## Measurements to collect
Use existing outputs:
- `metrics.json`: train/val curves and final metrics.
- `metrics/manifold_stats.jsonl`: `weight_norm`, `grad_norm`, `effective_rank`, `update_alignment`.

Primary efficiency metrics (early):
- `tokens_to_target_loss`: tokens needed to reach fixed loss threshold (pick one threshold from warmup run).
- `minutes_to_target_loss`: wall-clock minutes to same threshold (runtime fairness).
- `AUC_train_loss_early`: area under train-loss curve up to fixed token budget (secondary).
- `delta_val_loss_at_budget`: validation loss difference at same token budget (secondary).

Secondary stability metrics:
- late-interval slope of median layer `weight_norm`.
- max layer `weight_norm` near end of run.

## Decision criteria
Primary success criterion:
- Winner is the non-zero `wd` with lowest mean Phase 2 `tokens_to_target_loss` across seeds.
- Claim success only if improvement vs `wd=0.0` is >=5% and appears in at least 2/3 seeds.

Secondary criteria:
- `minutes_to_target_loss` does not regress by >5%.
- No instability signs (loss spikes/divergence/checkpoint NaNs).
- Same or lower weight-growth trend than `wd=0.0`.

## Run abort criteria (compute protection)
- Abort run if train loss is NaN/Inf at any step.
- Abort run if loss increases by >25% from its rolling 1k-step median for >=500 consecutive steps.
- Abort run if any logged weight norm exceeds 3x the median of first 10% of steps.
- Mark aborted runs as unstable and exclude from winner selection.

## Confound controls
- Keep non-Muon parameter-group behavior fixed across all arms (same AdamW settings for embeddings/norm/bias groups).
- Keep data order, tokenizer, batch size, LR schedule, and logging frequency identical across arms.

## Expected deliverables
1. `experiments/02_muon_weight_decay_focus/README.md` with setup and commands.
2. Config files for each arm/seed under `experiments/02_muon_weight_decay_focus/configs/`.
3. `run_all.sh` and optional `run_parallel_*.sh`.
4. `RESULTS.md` with tables + plots:
   - train/val loss vs tokens (early window highlighted)
   - tokens-to-threshold bar chart
   - minutes-to-threshold bar chart
   - dose-response chart for `muon_weight_decay`
   - weight-norm trend comparison (secondary).
   - reproducibility block: git commit hash, config checksum, seed table.

## Risks and controls
- Risk: early signal is noisy.
  - Control: fixed token-budget metrics + Phase 2 multi-seed confirmation.
- Risk: implementation bug in Muon decay path.
  - Control: unit sanity check: one-step update on toy matrix vs analytic decoupled formula.
- Risk: improvements only appear very late.
  - Control: run Optional Stage C only after early positive evidence.
