# Muon Weight Decay Focus Results

Run date: 2026-02-25  
Hardware: NVIDIA GeForce RTX 4090  
Command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
SEED=42 TRAIN_TOKENS=300000 LOG_EVERY=5 NUM_WORKERS=0 \
bash experiments/01_muon_weight_decay_focus/run_experiments.sh
```

## Runs

- `muon_wd0p0_seed42` (`muon_weight_decay=0.0`)
- `muon_wd0p05_seed42` (`muon_weight_decay=0.05`)
- `muon_wd0p1_seed42` (`muon_weight_decay=0.1`)

All runs used `configs/llm_config.py` defaults for model/training shape and only changed experiment controls (weight decay, tokens, seed, logging/output).

## Key Metrics (from `metrics.json`)

- `wd=0.0`: final train loss `7.4909`, final val loss `7.2910`, final val acc `0.1301`, tokens seen `311,296`
- `wd=0.05`: final train loss `7.4907`, final val loss `7.2911`, final val acc `0.1302`, tokens seen `311,296`
- `wd=0.1`: final train loss `7.4899`, final val loss `7.2901`, final val acc `0.1302`, tokens seen `311,296`

Primary-endpoint proxy at this short horizon:

- `tokens_to_train_loss<=8.0` was `262,144` for all three runs.

## Plots

- Validation-loss comparison: `experiments/01_muon_weight_decay_focus/runs/social/val_loss_comparison.png`

## Draft Social Summary

At 88M scale over ~0.3M tokens on RTX 4090, Muon with and without decoupled weight decay (`0.0`, `0.05`, `0.1`) shows very similar early loss behavior.  
`wd=0.1` is marginally best by final val loss in this short run, but the effect size is small; a longer token budget is needed to test the stronger late-training stabilization claim from Kimi.
