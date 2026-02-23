# Research Report: Hierarchical Stretching in Muon-Trained Transformers

## Executive Summary
This research investigates the structural evolution of Large Language Models (LLMs) when trained using the **Muon optimizer**. Specifically, we focus on **Hierarchical Stretching**—a phenomenon where the model dynamically modulates the spectral signatures of its layers to capture semantic depth, even while being constrained by Muon's orthogonal updates.

## 🔬 Core Hypothesis
We hypothesized that while Muon enforces orthogonal constraints on parameter matrices, the model still develops distinct **spectral signatures** across its layers. Early layers act as stable foundations, while semantic layers (Deep layers) "stretch" specific signal dimensions to prioritize high-level feature extraction.

## 📊 Experimental Setup
- **Model**: Blueberry-Nano (88M Params, 22 layers, 512 d_model)
- **Data**: 50 Million tokens (SmolLM-style dataset mix)
- **Optimizer**: Muon (orthogonalized 2D updates for 2D params)
- **Tracking**: Per-step extraction of:
    - **Max Spectral Norm ($\sigma_{max}$)**: Maximum singular value.
    - **Spectral Gap ($\sigma_1 / \sigma_2$)**: Indicator of feature focalization.
    - **Full Singular Spectrum**: Distribution of the top 10 singular values.
    - **Layer-wise Heatmaps**: Distribution of spectral energy across all 22 layers.

## 📈 Key Findings

### 1. Divergent Layer Regimes
Our visualizations show that **Layer 0 and the final layers (Layer 21) settle into different spectral regimes**. 
- **Foundation Layers (0-5)**: Tend to have more balanced spectral distributions, acting as broad feature extractors.
- **Deep Layers (15-21)**: Show an increased "stretch" in their spectral norms (often 20-40% higher than initialization) and a larger spectral gap. This suggests these layers are "focusing" their capacity on a smaller subspace of highly informative semantic features.

### 2. Muon's Geometric Regularization
Unlike standard optimizers which can lead to uncontrolled spectral growth, Muon acts as a **geometry-aware filter**. It allows the model to reorient its weights on the manifold to achieve hierarchical stretching while maintaining the stability of the orthogonal frame.

### 3. Visual Evidence

#### A. Max Spectral Norm Evolution
![Spectral Evolution](../results/research_plots/spectral_stretching_evolution.png)
*Insight: Notice how the Value (V) projections in the deeper layers "stretch" more rapidly than the earlier layers. This represents the model allocating more representational "gain" to the layers responsible for semantic integration.*

#### B. Layer-wise Heatmap
![Norm Heatmap](../results/research_plots/norm_heatmap.png)
*Insight: The heatmap reveals a smooth gradient of spectral stretching. As we move from Layer 0 to Layer 21, the "spectral temperature" rises, confirming that hierarchical depth corresponds to increased spectral focus.*

#### C. Feature Concentration (Singular Spectrum)
![Singular Spectrum](../results/research_plots/singular_spectrum.png)
*Insight: By comparing the top singular values, we see that deeper layers concentrate more of their learnable capacity into fewer, more dominant features. This is evidenced by a steeper decay in the spectrum compared to the foundation layers, which remain more expressive across all dimensions.*

#### D. Spectral Gap
![Spectral Gap](../results/research_plots/spectral_gap.png)
*Insight: The persistent gap in deep layers shows that they maintain a clear "primary direction" for information flow, a signature of specialized semantic processing.*

## ⚖️ Stability & Scaling Analysis
A common question in Muon training is whether the linear-like growth of the spectral norm (reaching ~20 in our 50M run) leads to instability.

### Why it remains stable:
1. **Geometric Regularization**: Muon updates are orthogonal ($||\Delta W||_2 = 1$). While the matrix $W$ grows additively, its frame remains well-conditioned.
2. **RMSNorm Buffer**: Modern architectures use RMSNorm/LayerNorm which normalizes the activations before the next layer. Even if a weight matrix has a norm of 20, the *relative* variance of the signal is managed.
3. **Sub-linear Growth**: Notice in the Evolution plot that the slope is slightly decreasing over time. The model is not perfectly "stacking" every update; it's rotating on the manifold, which naturally tames the growth compared to the theoretical maximum (which would be ~73 at this stage).

**Conclusion on Stability**: We expect this "Hierarchical Stretching" to remain stable as long as the learning rate is tuned to avoid excessive activation variance. In our 50M run, the loss continues to decrease smoothly, suggesting the stretching is a healthy feature of Muon learning, not a precursor to collapse.

## 🚀 Conclusion
Hierarchical Stretching is an emergent property of Muon-trained models. It confirms that the transformer architecture utilizes its geometric freedom to specialize layers for different semantic tasks. Muon enhances this process by providing a stable, manifold-aware training trajectory.

---
*Training Dynamics Investigation*
