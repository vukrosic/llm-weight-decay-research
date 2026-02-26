# Learned Residual Scale: Long-Run Plan

## Goal
Test one thing only: at `35M` tokens, does learned residual scaling beat fixed residual scaling on final validation loss?

## Scope Rules
- Use base training setup from `configs/llm_config.py`.
- Change only:
  - `train_tokens` -> `35000000`
  - residual-scale mode/init fields per run
- Do not change batch size, LR, optimizer type, or other knobs.
- Do not run any weight-decay sweep in this plan.

## Where learned params are added
- Location: inside each `TransformerBlock`, on residual branches before skip summation:
  - `x = x + scale * attn_out`
  - `x = x + scale * ff_out`
- Note on `scale * x + ...`:
  - Good idea, but treated as a separate mechanism (skip-path scaling) and excluded from this phase to keep one-variable attribution.
  - If branch-only learned scaling wins at 35M, run a follow-up with skip-path scaling variants.
- Modes:
  1. `fixed`: no learnable scale params (uses constant `residual_scale`).
  2. `learned_layer`: one learnable scalar per layer, shared by both branches.
  3. `learned_branch`: two learnable scalars per layer (attention + FFN).
- Parameter count for current model (`n_layers=22`):
  - `learned_layer`: `22` extra params
  - `learned_branch`: `44` extra params

## Run Matrix (35M tokens each, seed 42)
1. `fixed:1.0` (baseline)
2. `learned_layer:init0.1`
3. `learned_branch:init0.1`
4. `learned_branch:init0.01`

## Success Criteria
1. Primary: at least one learned variant has lower final `val_loss` than `fixed:1.0`.
2. Secondary: learned scale values move from init without instability.
3. Secondary: no NaNs/divergence.

## Outputs
- New experiment folder: `experiments/learned_residual_scale_long/`
- Artifacts:
  - `REPORT.md` (ranked table + deltas vs baseline)
  - `summary.json`
  - `ranked_val_loss.png`
  - `delta_vs_baseline.png`
