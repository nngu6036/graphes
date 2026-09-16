# Heat-kernel spectral-space diffusion

> **Legacy representation.** Direct heat-kernel prediction is retained for reproducibility only.
> The maintained permutation-invariant eigenspace extension is documented in
> `EIGENSPACE_HISTOGRAM_SPECTRAL_DIFFUSION.md`.

## Motivation

The legacy GraphER spectral bridge represents each graph only by the
combinatorial-Laplacian eigenvalue vector.  Rewiring changes the eigenvectors as
well as the eigenvalues, so an eigenvalue-only target can underdetermine the
structural direction of the reverse process.

The heat-kernel representation

\[
H_\tau(G)=\exp(-\tau L(G)/s)
          =U\exp(-\tau\Lambda/s)U^\top
\]

contains both eigenvalue decay and eigenspace information.  It is invariant to
eigenvector sign changes and to rotations inside repeated eigenspaces.  GraphER
uses several diffusion times so both short- and long-range organization are
visible.

## Forward bridge

For a training graph `G*`, GraphER constructs an indexed degree-matched HH
source `G0`.  Heat-kernel mode computes

\[
S_0=(H_{\tau_1}(G_0),\ldots,H_{\tau_T}(G_0)),\qquad
S_1=(H_{\tau_1}(G^*),\ldots,H_{\tau_T}(G^*)).
\]

The endpoint-conditioned Brownian bridge is sampled directly in this matrix
space.  The intermediate continuous state is allowed to leave the exact
heat-kernel manifold; this is analogous to a continuous spectral bridge passing
through states that are not exactly realizable by a discrete graph.

The current default scales are:

```yaml
spectral_prediction:
  representation: heat_kernel
  heat_kernel_times: [0.25, 1.0, 4.0]
  heat_kernel_normalization: mean_degree
summary_diffusion:
  heat_kernel_sigma: 0.1
```

## Predictor

The current and source heat kernels are supplied as pair-level features to the
graph-context network.  A symmetric pair head predicts the clean multiscale
heat kernel.  A nonnegative symmetric Sinkhorn projection enforces approximate
unit row sums before the training loss is evaluated.

The projection intentionally does **not** impose PSD or exact heat-semigroup
consistency.  The final hard graph projection is performed by the constrained
rewiring process, so every realized candidate has a valid graph heat kernel.

## Reverse process and hard realization

Generation starts from the HH source and its exact heat kernel.  At each
continuous reverse step the model predicts the clean heat-kernel endpoint and
advances the matrix bridge.  The final heat-kernel state becomes the structural
target for rewiring.

For a candidate graph `G'`, the spectral component of the energy is

\[
D_H(G')=\operatorname{RMSE}(H(G'),\widehat H).
\]

Only valid degree-preserving swaps are considered, and connectivity checks are
unchanged.  Thus the diffusion state captures global spectral geometry while
the graph trajectory remains in the same degree fibre.

## Configurations

Full joint model:

```text
configs/experiments/grapher/community_small_joint_edge_heat_kernel_graphlets345_learned.yaml
```

True heat-kernel-only training ablation:

```text
configs/experiments/grapher/community_small_heat_kernel_only_learned.yaml
```

The second configuration disables edge diffusion, clustering, orbit and
induced-graphlet heads/losses and gives the refiner only the heat-kernel energy.
It should therefore be used when comparing a genuinely spectral-only GraphER
against the full global+local model.

## Alignment and limitations

A full heat-kernel matrix is node-indexed.  Training therefore uses GraphER's
existing degree-preserving source/target alignment before constructing the
matrix bridge.  Within equal-degree groups this correspondence is not unique;
the present implementation uses the existing deterministic/randomized indexed
alignment rather than solving a graph-matching problem.  This is the main
conceptual limitation of the first implementation.

The matrix representation is also more expensive than eigenvalues alone.
Candidate scoring requires an eigendecomposition and reconstruction of each
requested heat scale.  This is practical for Community-small but should be
profiled before applying the same exact candidate scoring to large graphs.
