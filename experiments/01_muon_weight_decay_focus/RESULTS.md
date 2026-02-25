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
