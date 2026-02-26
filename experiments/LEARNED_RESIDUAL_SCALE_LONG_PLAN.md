# Learned Residual Scale: Longer-Training Plan

## Goal
Validate whether the short-run win from learned residual scaling remains (or improves) at a meaningfully longer training horizon. [uncleal goal]

## What we already know (single seed, 2M tokens) [do 20m tokens for each ablation instead, just llm config, do not change batch size or anything else that you already have, just numer of tokens]
- Best: `learned_layer:init0.1` (`val_loss 6.0433`)
- Baseline: `fixed:1.0` (`val_loss 6.0872`)
- Delta: `-0.0439`

## Long-run experiment scope
- Keep seed fixed: `42` (no seed changes)
- Keep optimizer setup fixed to current winning Muon config:
  - `muon_decay_mode=update`
  - `muon_weight_decay=-0.2` [no delete all weight decay expeiments, purge experiments completely, i just want the base, i will delete it all, make new folder in experimetns for this filel and explerimt]
- Increase budget from `2M` to `20M` tokens per run

## Run matrix (focused)
1. `fixed:1.0` (baseline)
2. `learned_layer:init0.1` (current winner)
3. `learned_branch:init0.1` (close second)
4. `learned_branch:init0.01` (small-init stress test)
[what does this mean, where do you add learnable params,how many]