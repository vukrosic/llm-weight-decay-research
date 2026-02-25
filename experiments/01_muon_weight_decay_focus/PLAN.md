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

## Minimum Viable Experiment Matrix (for Social Media)
To achieve the clearest signal for a social media post with the lowest compute budget, we run a simple 3-way comparison over a short token horizon (e.g., 500M tokens):
- **Run 1**: `muon_weight_decay = 0.0` (Vanilla Muon Baseline)
- **Run 2**: `muon_weight_decay = 0.05` (Mild decay)
- **Run 3**: `muon_weight_decay = 0.1` (Kimi's recommended default)
- **Seed**: 42 for all runs.
- **Budget**: 100_000_000 tokens per run.
- **Total**: 3 independent runs.

## Tracking and Plotting (for Social Media)
To effectively demonstrate the value of Muon weight decay to the community, we will track and plot two key metrics:
1. **Validation Loss vs. Tokens/Steps**: A standard learning curve plot to show if adding weight decay maintains or accelerates loss reduction compared to the baseline. 
2. **Weight Norm Trajectory (Max/Median across layers)**: A plot showing `weight_norm` vs. `training_steps`. This is the visual "hook" — practically demonstrating the Kimi paper's claim that without decay (Run 1), weights grow unbounded, whereas with decay (Runs 2 & 3), weight norms plateau and stabilize appropriately.

## Measurements to collect
Use existing outputs:
- `metrics.json`: For `train_loss` and `val_loss`.
- `metrics/manifold_stats.jsonl`: For `weight_norm` logged per layer over time.

## Confound controls
- Keep non-Muon parameter-group behavior fixed across all arms.
- Keep data order, tokenizer, batch size, LR schedule, and logging frequency identical across arms.

## Expected deliverables
1. A reproducible shell script `run_experiments.sh` to execute the 3 runs.
2. A single Python plotting script `plot_results.py` that generates the two web-ready images: `val_loss_comparison.png` and `weight_norm_trajectory.png`.
3. A short `RESULTS.md` containing the social media draft text summarizing the findings (e.g., "Vanilla Muon's weights explode. Adding weight decay fixes it and improves loss...").
