# Future Work: Modular Manifolds and Attention Geometry

Based on the theory of Modular Manifolds, there are two primary pathways for advancing our cross-layer head orthogonalization and Muon optimizer research:

### 1. Heterogeneous Modularity (Mix-and-Match Constraints)

In modern neural networks, different modules serve fundamentally different purposes and, consequently, may naturally belong on different mathematical manifolds. 

Currently, optimization techniques often apply a uniform constraint or preconditioning strategy across the entire network. However, a heterogeneous approach would involve heavily customizing the manifold constraints based on the module's exact role:

*   **Embeddings vs. Unembeddings:** An embedding layer maps discrete tokens into a continuous vector space, while an unembedding layer projects dense features back into a massive vocabulary distribution. Constraining them to the exact same Stiefel manifold or using the same orthogonalization rules ignores their asymmetric roles. Finding the "correct" manifold for each structural component is an open question.
*   **Layer-Depth Variations:** Do early layers need the same constraints as later layers? Early in the network, attention heads might need to collaborate to build basic, shared representations. Thus, applying a "soft" constraint (such as *Depth-Shared Feature Covariance*) makes sense. However, in the final layers, we might want heads to specialize in completely different, highly semantic concepts, which would necessitate a "hard" constraint (such as strict *Layer-Wise Head Repulsion*).
*   **Varying Constraints Within Attention:** Even inside a single attention block, the Query/Key matrices dictate routing (which often benefits from shared, low-rank subspaces) while the Value matrix dictates the payload (which benefits from high-rank diversity). Mix-and-matching different manifold constraints within the exact same layer is a strong candidate for future exploration.

### 2. Non-Riemannian Geometry for Attention

Most existing work in manifold optimization (including standard Stiefel manifold optimization) implicitly assumes a **Riemannian geometry**. In a Riemannian world, distances and lengths are induced by an inner product (like the standard Euclidean $L_2$ norm). If you visualize the "unit ball" of possible optimization steps in this space, it looks like a smooth, perfectly round ellipsoid.

However, parameter matrices in transformers (especially attention weights) are not simply vectors in space; they are **operators** that transform input vectors into output vectors. 
*   Because they are operators, their sizes and limits are best measured using operator norms (like the **spectral norm**, which measures the maximum stretching effect on any vector).
*   Crucially, operator norms *do not* emerge from inner products. This places them in the realm of **non-Riemannian geometry** (specifically, Finsler manifolds).

**Why does this matter for attention?** 
In non-Riemannian spaces, the "unit ball" of possible optimization steps doesn't have smooth, round edges—it has sharp corners. Because of these sharp corners, there is no single, unique gradient flow curve that the optimizer naturally follows. By forcing our head-orthogonalization algorithms into a classic Euclidean/Riemannian framework, we might be taking suboptimal gradient steps. Re-evaluating tensorized head orthogonalization through a natively non-Riemannian lens could yield new gradient flows that physically align much better with how attention matrices actually operate.
