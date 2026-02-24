# Experiment 01: Muon vs AdamW Baseline - Multi-Seed Comparison

This report compares the spectral dynamics and performance of the **Muon** optimizer across two different seeds (42 and 137) to evaluate stability and reproducibility.

---

## 📈 Validation Curves Comparison

| Metric | Muon Seed 42 (Seed 1) | Muon Seed 137 (Seed 2) |
| :--- | :---: | :---: |
| **Loss** | ![Loss S1](../../plots/finished_run/loss_finished.png) | ![Loss S2](./plots_seed137/loss_seed137.png) |
| **Accuracy** | ![Acc S1](../../plots/finished_run/accuracy_finished.png) | ![Acc S2](./plots_seed137/accuracy_seed137.png) |

**Final Performance Statistics:**
- **Seed 42**: Val Loss 3.3154, Acc 39.76%
- **Seed 137**: Val Loss 3.3098, Acc 39.83%

---

## 🗺️ Spectral Manifold Dynamics Comparison

### 1. Singular Value Entropy (Isotropy)
| Muon Seed 42 (Seed 1) | Muon Seed 137 (Seed 2) |
| :---: | :---: |
| ![Entropy S1](../../plots/finished_run/heatmap_entropy_finished.png) | ![Entropy S2](./plots_seed137/heatmap_entropy_seed137.png) |

### 2. Update Alignment
| Muon Seed 42 (Seed 1) | Muon Seed 137 (Seed 2) |
| :---: | :---: |
| ![Alignment S1](../../plots/finished_run/heatmap_update_alignment_finished.png) | ![Alignment S2](./plots_seed137/heatmap_update_alignment_seed137.png) |

### 3. Effective Rank
| Muon Seed 42 (Seed 1) | Muon Seed 137 (Seed 2) |
| :---: | :---: |
| ![Rank S1](../../plots/finished_run/heatmap_effective_rank_finished.png) | ![Rank S2](./plots_seed137/heatmap_effective_rank_seed137.png) |

---

## 📝 Multi-Seed Observations

**Reliability:**
- Across both seeds (42 and 137), final validation accuracy is extremely consistent (~39.8%).
- Validation loss shows minor stochastic variation (batching/init) but maintains high overall stability.

**Manifold Consistency:**
- Both runs exhibit identical spectral structural signatures, confirming that Muon's effects are deterministic with respect to architecture.
- The high-alignment band in layers 12-16 for Key (K) projections is a robust feature across seeds.
- The characteristic "rank tiers" (MLP > Q/O > K/V) are perfectly preserved.
- Entropy and Rank remain flat across time in both experiments, proving Muon's capability to maintain weight conditioning regardless of initialization.
