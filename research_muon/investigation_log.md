# Investigation Log: The Geometry of Muon learning

## Current Observations (Preliminary 50M Run)

### 1. The "Scissors" Effect (Hierarchical Specialization)
We observe a clear divergence between early and late layers. 
- **Early Layers (0-5)**: Maintain high spectral entropy and low spectral norms. They act as "Generalists," spreading attention weights broadly across the context window.
- **Deep Layers (15-21)**: Undergo "Spectral Decay." The spectral norm ($\sigma_{max}$) grows aggressively (up to 50x), while the "tail" singular values collapse. They become "Extreme Specialists," focusing representational power into a tight singular subspace.

### 2. Q vs V Divergence
- **Query Matrices**: Show significantly larger spectral norms and faster rank decay than Value matrices.
- **Hypothesis**: Q matrices are responsible for the high-precision "routing" of information, requiring sharper specialization, while V matrices maintain the "content richness" and thus preserve more spectral entropy.

### 3. Training Phases
- **Discovery Phase (0-2k steps)**: High-rank updates across all layers. The optimizer is exploring the manifold.
- **Refinement Phase (2k+ steps)**: Deep layers begin to "focus," evidenced by a rising spectral gap and falling entropy.

## Research Questions for 2B Scale
1. Does the "Scissors Effect" scale linearly with depth, or is there a critical "semantic layer" where the shift occurs?
2. Can we "pre-sculpt" the model using **Stretched-Init** (polarizing deep layers to higher norms at Step 0) to accelerate specialization?
3. Does AdamW's rank collapse eventually lead to the same specialized geometry, or does Muon's orthogonality enable a *healthier* form of specialization?
