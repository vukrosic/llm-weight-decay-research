# Experiment 01: Baseline Anomaly Hunt Results (Debug 2M Tokens)

This document displays the comparative spectral dynamics between Muon (Blue) and AdamW (Red) across layers and projections.

## 1. Spectral Entropy
Rows: Layers (1, 3, 6, 9, 12) | Columns: Projections (Q, K, V, O)
![Spectral Entropy](baseline_comparison_entropy.png)

---

## 2. Update-Weight Alignment
Measures the geometric lock-in between the optimizer's updates and the weight manifold.
![Update Alignment](baseline_comparison_update_alignment.png)

---

## 3. Effective Rank
Tracking the "Needle vs Wave" hypothesis: how many dimensions the optimizer is utilizing.
![Effective Rank](baseline_comparison_effective_rank.png)

---

## 4. Gradient Norms
Spectral magnitude of the gradients.
![Grad Norm](baseline_comparison_grad_norm.png)

---

## 5. Weight Norms
Evolution of the parameter norms.
![Weight Norm](baseline_comparison_weight_norm.png)

---

# Experiment Critique for Anomaly Hunting

The primary goal of this experiment framework is to identify *anomalies*—sudden shifts, irregular alignments, and dimension collapse. The current setup, while robust for general metric tracking, has a few crucial blind spots for pure anomaly hunting.

### 1. The Averaging Trap (Resolved)
Initially, `analysis/compare_baseline.py` was interpolating and averaging metrics across all seeds, plotting the mean and standard deviation. 
**Critique:** Averaging destroys anomaly visibility. An optimizer instability (like a sudden spike in gradient norm) in `seed=137` will get completely diluted by the averages of `seed=42` and `seed=256`. 
**Resolution:** This has been resolved in `compare_baseline.py`. The runs are now plotted individually with slight transparency so exact paths and true chaotic divergences between identical seeds are preserved.

### 2. Missing Layers in Visualization
**Critique:** The plotting script currently only plots a subset of layers (`[1, 3, 6, 9, 12]`). However, empirical anomaly scans across the raw JSONL data point elsewhere entirely:
- **Max Muon `grad_norm` Variance:** Occurs at **Layer 0** (Proj V)
- **Max Muon `update_alignment` Variance:** Occurs at **Layer 20** (Proj O), varying wildly between runs [0.5239, 0.2631, 0.3789].
- **Max Muon `weight_norm` Variance:** Occurs at **Layer 10** (Proj UP).
By arbitrarily filtering to `1, 3, 6, 9, 12`, the most statistically volatile layers (the embeddings-proximate Layer 0 and exit-proximate Layer 20) are completely hidden from the plots. For anomaly hunting, we should be using Heatmaps of layer depth vs. step, rather than line plots of arbitrary layers.

### 3. Subspace Alignment Blind Spot
**Critique:** `compute_subspace_alignment(W, dW, k=5)` currently performs an SVD to fetch the *left* singular vectors (the `U` matrix). Because `W` represents the map from input to output features, `U` corresponds strictly to the **output subspace**.
An anomaly occurring in the **input subspace** (e.g., caused by an anomaly in a specific upstream routing or specific un-normalized token representations) will manifest in the *right* singular vectors (`V`). The alignment tracker is completely blind to right-side alignment collapses.
**Recommendation:** Track both `left_alignment` and `right_alignment` independently. 

### 4. Downsampling & Missing the Peak
**Critique:** Metrics are currently captured synchronously every N steps (`detailed_log_every`). An optimizer anomaly might manifest as a delta explosion that resolves itself within just 2 or 3 steps. If this spike happens between the intervals, the anomaly vanishes.
**Recommendation:** We should compute instantaneous metrics inside the inner step, track the `running_maximum`, and emit *that* every N steps, rather than taking a blind instantaneous sample every log interval.

### 5. Muon vs AdamW Volatility
Even on a brief 2M token run, exact point-to-point variance shows that Muon has roughly **10x higher variance** between seeds for `update_alignment` than AdamW (0.26 variance vs 0.03 variance). This is exactly what we are looking for: Muon is charting drastically different manifold trajectories based on seed initialization, whereas AdamW is highly deterministic. Future experiments should lengthen the sequence to see if Muon converging on divergent local minima actually yields different spectral structures (e.g., does high `update_alignment` variance predict early over-fitting?).
