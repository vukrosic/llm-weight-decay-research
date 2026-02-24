# Experiment 01: Muon vs AdamW Baseline

Investigation of spectral dynamics across layers and projection types comparing Muon and AdamW optimizers.

[**View Latest Results & Analysis**](RESULTS.md)


## Setup
- **Runs**: 6 runs total
- **Optimizers**: 2 (Muon, AdamW)
- **Seeds**: 3 (42, 137, 256)
- **Model**: 88M params (22 layers)
- **Data**: 2B tokens (Debug run: 2M tokens)

## Objective
To differentiate the manifold-aware optimization signatures of Muon from standard AdamW by tracking per-layer and per-projection spectral statistics.
