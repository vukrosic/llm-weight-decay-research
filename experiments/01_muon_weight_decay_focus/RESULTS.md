# Muon Weight Decay Focus Results

Run date: 2026-02-25
Hardware: NVIDIA GeForce RTX 4090  
Command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
SEED=42 TRAIN_TOKENS=32768000 LOG_EVERY=50 NUM_WORKERS=0 \
bash experiments/01_muon_weight_decay_focus/run_experiments.sh  # 2000 steps/run
```

## Runs

- `muon_wd0p0_seed42` (`muon_weight_decay=0.0`)
- `muon_wd0p05_seed42` (`muon_weight_decay=0.05`)
- `muon_wd0p1_seed42` (`muon_weight_decay=0.1`)

All runs used `configs/llm_config.py` defaults for model/training shape and only changed experiment controls (weight decay, tokens, seed, logging/output).

## Key Metrics (from `metrics.json`)

- `wd=0.0`: final train loss `3.8066`, final val loss `4.2226`, final val acc `0.3058`, tokens seen `32,768,000`, steps `2000`
- `wd=0.05`: final train loss `3.9127`, final val loss `4.3159`, final val acc `0.2962`, tokens seen `32,768,000`, steps `2000`
- `wd=0.1`: final train loss `4.0185`, final val loss `4.4195`, final val acc `0.2856`, tokens seen `32,768,000`, steps `2000`

Primary-endpoint proxy (`tokens_to_target_loss` on train loss):

- Target `<=5.0`: `wd=0.0` reached at `9,846,784` tokens; `wd=0.05` and `wd=0.1` reached at `16,400,384` tokens.
- Target `<=4.8`: all reached at `16,400,384` tokens.

## Plots

- ![Train Loss Comparison](./runs/social/train_loss_comparison.png)
- ![Validation Loss Comparison](./runs/social/val_loss_comparison.png)

## Draft Social Summary

At 88M scale over 2,000 training steps (32.8M tokens) on RTX 4090, vanilla Muon (`wd=0.0`) outperformed non-zero Muon weight decay in this setup.  
`wd=0.0` gave the best final validation loss (`4.2226`) and reached the `train_loss <= 5.0` target earlier than `wd=0.05` and `wd=0.1`.  
In this experiment configuration, adding Muon decay degraded both final val loss and val accuracy monotonically as decay increased (`0.0 -> 0.05 -> 0.1`).

## Additional Certainty Sweeps (WD-only, fixed LR)

To reduce randomness sensitivity, we ran focused WD-only follow-ups with fixed `muon_lr=0.024` and compared `wd=-0.01`, `wd=0.0`, `wd=+0.01`.

### Extended certainty (2.4M tokens, 5 seeds)

Artifacts:
- `experiments/01_muon_weight_decay_focus/runs/wd_extended_certainty/REPORT.md`
- `experiments/01_muon_weight_decay_focus/runs/wd_extended_certainty/mean_std_val_loss.png`
- `experiments/01_muon_weight_decay_focus/runs/wd_extended_certainty/delta_vs_baseline.png`

Mean final val loss over seeds `{42,137,256,512,1024}`:
- `wd=+0.01`: `6.1501` (`delta=-0.0017` vs baseline, `-0.03%`)
- `wd=-0.01`: `6.1512` (`delta=-0.0006` vs baseline, `-0.01%`)
- `wd=0.0`: `6.1518` (baseline)

### Longer certainty (8.0M tokens, 3 seeds)

Artifacts:
- `experiments/01_muon_weight_decay_focus/runs/wd_longer_certainty/REPORT.md`
- `experiments/01_muon_weight_decay_focus/runs/wd_longer_certainty/mean_std_val_loss.png`
- `experiments/01_muon_weight_decay_focus/runs/wd_longer_certainty/delta_vs_baseline.png`

Mean final val loss over seeds `{42,137,256}`:
- `wd=+0.01`: `5.2327` (`delta=-0.0011` vs baseline, `-0.02%`)
- `wd=0.0`: `5.2338` (baseline)
- `wd=-0.01`: `5.2340` (`delta=+0.0002` vs baseline, `+0.00%`)

### Overall interpretation

Across multi-seed short and longer runs, WD effects are real but very small at this scale/configuration (order of `1e-3` val-loss).  
The strongest consistent signal in the new sweeps is a slight benefit for small positive decay (`wd=+0.01`) vs baseline.

## Recent Experiment Audit (Most Recent Runs)

Audit timestamp: 2026-02-25

### Completion check vs plan

- `wd_multiseed_selected`: expected `3 WD × 3 seeds = 9` runs, found `9/9` completed (`metrics.json` present).
- `wd_extended_certainty`: expected `3 WD × 5 seeds = 15` runs, found `15/15` completed.
- `wd_longer_certainty`: expected `3 WD × 3 seeds = 9` runs, found `9/9` completed.

All recent planned runs are finished.

### Key outcomes

- `wd_multiseed_selected` (short, 300k tokens): best mean val loss at `wd=-0.01` (`7.2888`) vs baseline `wd=0.0` (`7.2892`), delta `-0.0005`.
- `wd_extended_certainty` (2.4M tokens, 5 seeds): best mean val loss at `wd=+0.01` (`6.1501`) vs baseline (`6.1518`), delta `-0.0017`.
- `wd_longer_certainty` (8.0M tokens, 3 seeds): best mean val loss at `wd=+0.01` (`5.2327`) vs baseline (`5.2338`), delta `-0.0011`.

### Plots

Combined single image (all loss plots across completion matrices):
- ![completion_matrix_all_losses](./runs/completion_matrix_all_losses.png)

`wd_multiseed_selected`:
- ![wd_multiseed_selected_mean_std](./runs/wd_multiseed_selected/multiseed_mean_std.png)
- ![wd_multiseed_selected_delta](./runs/wd_multiseed_selected/delta_vs_baseline.png)

`wd_extended_certainty`:
- ![wd_extended_certainty_mean_std](./runs/wd_extended_certainty/mean_std_val_loss.png)
- ![wd_extended_certainty_delta](./runs/wd_extended_certainty/delta_vs_baseline.png)

`wd_longer_certainty`:
- ![wd_longer_certainty_mean_std](./runs/wd_longer_certainty/mean_std_val_loss.png)
- ![wd_longer_certainty_delta](./runs/wd_longer_certainty/delta_vs_baseline.png)

## Dense Short Single-Seed WD Search (Positive + Negative)

Goal: quickly test many WD values to see whether any non-marginal gain appears.

Setup:
- Seed: `42`
- Tokens/run: `300,000` (very short)
- `muon_lr=0.024`
- WD grid (17 points): `[-0.2, -0.1, -0.05, -0.02, -0.01, -0.005, -0.002, -0.001, 0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]`

Result summary:
- Best: `wd=-0.05` with `val_loss=7.2885`
- Baseline: `wd=0.0` with `val_loss=7.2899`
- Best improvement vs baseline: `-0.0015` (still small/marginal)
- Worst in sweep: `wd=0.2` with delta `+0.0059` vs baseline

Artifacts:
- `experiments/01_muon_weight_decay_focus/runs/wd_dense_single_seed/REPORT.md`
- ![wd_dense_single_seed_val_loss_vs_wd](./runs/wd_dense_single_seed/val_loss_vs_wd.png)
- ![wd_dense_single_seed_delta](./runs/wd_dense_single_seed/delta_vs_wd0.png)

## Dense Short Single-Seed Residual-Scale Search

Goal: sweep residual connection multiplier (`residual_scale`) to test if this knob gives larger-than-marginal effects.

Setup:
- Seed: `42`
- Tokens/run: `300,000`
- `muon_lr=0.024`, `muon_weight_decay=0.0`
- Scale grid (17 points): `[0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 1.00, 1.01, 1.02, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40]`

Result summary:
- Best: `residual_scale=0.70` with `val_loss=7.2817`
- Baseline: `residual_scale=1.00` with `val_loss=7.2914`
- Best improvement vs baseline: `-0.0097` (larger than the WD-only deltas)
- Worst in sweep: `residual_scale=1.40` with delta `+0.0133` vs baseline

Artifacts:
- `experiments/01_muon_weight_decay_focus/runs/residual_scale_dense_single_seed/REPORT.md`
- ![residual_scale_val_loss_vs_scale](./runs/residual_scale_dense_single_seed/val_loss_vs_residual_scale.png)
- ![residual_scale_delta](./runs/residual_scale_dense_single_seed/delta_vs_scale1.png)
- ![residual_scale_all_curves](./runs/residual_scale_dense_single_seed/all_loss_curves_17scales.png)

## Around-Winner Residual-Scale Ablation (Single Seed, No Multi-Seed)

Focused local sweep around the previous winner:
- Scale grid: `[0.60, 0.64, 0.66, 0.68, 0.69, 0.70, 0.71, 0.72, 0.74, 0.76, 0.80, 0.90, 1.00]`
- Seed: `42`
- Tokens/run: `300,000`

Result summary:
- Best local point: `residual_scale=0.69`, `val_loss=7.2777`
- Baseline `scale=1.00`: `val_loss=7.2913`
- Delta vs baseline: `-0.0136`

Artifacts:
- `experiments/01_muon_weight_decay_focus/runs/residual_scale_winner_ablation/REPORT.md`
- ![residual_scale_winner_val](./runs/residual_scale_winner_ablation/val_loss_vs_scale.png)
- ![residual_scale_winner_delta](./runs/residual_scale_winner_ablation/delta_vs_scale1.png)
- ![residual_scale_winner_all_curves](./runs/residual_scale_winner_ablation/all_loss_curves.png)

## Long Run Head-to-Head (20M Tokens, 2 Experiments)

As requested, trained only two runs for longer budget:
- Baseline: `residual_scale=1.0`
- Best candidate: `residual_scale=0.69`
- Seed: `42`
- Tokens each: `20,004,864` (target 20M)

Results:
- `scale=0.69`: `val_loss=4.5363`, `train_loss=4.4171`, `val_acc=0.2751`
- `scale=1.0`: `val_loss=4.5435`, `train_loss=4.4266`, `val_acc=0.2749`
- Delta `(0.69 - 1.0)` on val loss: `-0.0072`

Artifacts:
- `experiments/01_muon_weight_decay_focus/runs/residual_scale_long_compare/REPORT.md`
- ![residual_scale_20m_compare](./runs/residual_scale_long_compare/loss_compare_20m_tokens.png)

## Muon Kimi-Style Per-Update Idea Search (Single Seed)

Implemented Muon decay placement modes and searched quickly without seed changes:
- `muon_decay_mode` in `{param, update, both}`
- short runs (`300k` tokens), seed `42`, residual scale fixed at `1.0`

Phase-1 best (non-marginal):
- `mode=update`, `wd=-0.2`
- `val_loss=7.2825` vs baseline (`mode=param`, `wd=0`) `7.2895`
- Delta: `-0.0070`

Refined local search around update-mode winner:
- Grid: `wd ∈ {0.0, -0.05, -0.1, -0.15, -0.2, -0.25, -0.3, -0.4}`
- Best: `update wd=-0.4`, `val_loss=7.2774`
- Baseline (`update wd=0`): `7.2916`
- Delta: `-0.0142`

Artifacts:
- `experiments/01_muon_weight_decay_focus/runs/muon_kimi_idea_search/REPORT.md`
- ![muon_kimi_ranked](./runs/muon_kimi_idea_search/ranked_val_loss.png)
- ![muon_kimi_delta](./runs/muon_kimi_idea_search/delta_vs_baseline.png)
- `experiments/01_muon_weight_decay_focus/runs/muon_kimi_update_local_search/REPORT.md`
- ![muon_update_local_val](./runs/muon_kimi_update_local_search/val_loss_vs_wd.png)
- ![muon_update_local_delta](./runs/muon_kimi_update_local_search/delta_vs_wd0.png)
