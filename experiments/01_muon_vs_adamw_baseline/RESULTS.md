# Finished Experiment Report: Muon Seed 42

This report focuses exclusively on the fully completed **Muon Seed 42** run from the Experiment 01 suite.

## 📈 Final Training Curves
Validation loss and accuracy across the full 1.7B tokens.

| Validation Loss | Validation Accuracy |
| :---: | :---: |
| ![Loss Finished](../../plots/finished_run/loss_finished.png) | ![Accuracy Finished](../../plots/finished_run/accuracy_finished.png) |

**Final Performance Metrics:**
- **Final Val Loss**: 3.3154
- **Final Val Accuracy**: 0.3976 (39.76%)
- **Total Steps**: 103,760
- **Total Tokens**: 1,700,003,840

---

## 🗺️ Spectral Manifold Dynamics
Exclusively for Seed 42, showing the clean evolution of depth-wise spectral statistics.

### 1. Singular Value Entropy (Isotropy)
Measures the health of the singular value distribution. High entropy throughout training suggests the weights remained well-conditioned.
![Entropy Heatmap](../../plots/finished_run/heatmap_entropy_finished.png)

**Image Description (For Screen Readers/Text-Only):**
The heatmap shows entropy values for layer weights (depths 0-21) over the full 100k training steps. The most notable feature is that entropy remains remarkably flat and stable horizontally across the entire duration with no vertical banding or sudden drop-offs. Up/Down (MLP) projections show the highest entropy (mostly green/yellow, values ~6.15-6.20). Q and O attention projections maintain strong mid-high entropy (values ~5.8-6.0). K and V projections operate with lower entropy (values ~5.2-5.45) but are equally stable. A localized, stable band of slightly lower entropy appears consistently around layer depths 16–18 for the Q and K projections.

### 2. Update Alignment
Shows how the Muon updates aligned with the principal subspaces of each layer's weight matrix.
![Alignment Heatmap](../../plots/finished_run/heatmap_update_alignment_finished.png)

**Image Description (For Screen Readers/Text-Only):**
The heatmap tracks alignment values (0.0 to 0.6) over the 100k steps for each projection. The strongest characteristic is a visually distinct, highly active bright yellow band in the Key (K) projections acting between layer depths 12 and 16, where update alignment consistently stays high (0.4 to 0.6). In contrast, the O, UP, and DOWN projections appear deeply purple/dark-blue, rarely exceeding 0.15, indicating low alignment and high dispersity across singular vectors. Nearly all alignment patterns lock into these distinct operational bands very early in training (first 2,000 steps) and hold perfectly steady until step 100,000 without shifting or oscillating.

### 3. Effective Rank
Tracks the effective dimensionality of the weights.
![Rank Heatmap](../../plots/finished_run/heatmap_effective_rank_finished.png)

**Image Description (For Screen Readers/Text-Only):**
The effective rank heatmap mirrors the structural stability seen in the entropy plot. There is no rank collapse visible at any level across the full 100k steps, maintaining practically perfect horizontal temporal consistency. The MLP projections (UP/DOWN) sustain the highest dimensionality, hovering steadily between ranks of 460–495. The Q and O attention layers sit in a rank band of 350-400. The K and V layers explicitly function at lower dimensionality, stabilized continuously between ranks 190 and 240. The network preserves its full conditioning span uninterrupted until the terminal step.

---

## 📝 Observations
- **Unwavering Stability**: The validation loss dropped precipitously from ~10.9 down to 3.6 within the first 10k steps and steadily asymptoted exactly to 3.3154, while accuracy climbed quickly to stabilize at ~39.7%. Most crucially, the training curves exhibit absolutely zero spikes, regressions, or anomalies. Seed 42 achieved an impeccably smooth convergence.
- **Protection Against Rank Collapse**: The spectral matrices confirmed what researchers hope for when using Muon: zero layer-wise rank degradation over 100k steps. The effective rank matrices reflect completely flat bands across time, proving the optimizer maintains full conditioning of the singular value distributions—especially in the late phases of training where AdamW often triggers representational collapse.
- **Projection Specialization**: The network displays highly specialized and dimensionally stable behavior uniquely tailored to matrix type. MLP (Up/Down) layers consistently consume the highest dimensionality (~460-495 rank), whereas Key (K) and Value (V) projections operate with much harder structural bottlenecks (190-240 rank).
- **Subspace Alignment**: Muon was observed heavily relying on the principal eigenspaces specific to making structured updates inside the Key (K) projections around mid-network depths (layers 12-16). In contrast, updates on Output (O) and MLP weights remained extremely dispersed.
- **Reference Baseline**: This run now serves as the "Golden Baseline" for Muon. Subsequent runs (Seeds 137 and 256) and optimizers (AdamW) will be rigidly evaluated against these clean layer-banding behaviors and temporal signatures.
