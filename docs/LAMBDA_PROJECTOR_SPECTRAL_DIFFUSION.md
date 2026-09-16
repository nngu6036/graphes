# Lambda + projector spectral diffusion

GraphER's structured eigenspace representation is

\[
S(G) = (\Lambda(G), P_k(G)), \qquad P_k = U_k U_k^\top,
\]

where `U_k` contains the first `k` non-trivial eigenvectors of the
combinatorial Laplacian.  The constant zero mode is excluded.

## Why a projector

Raw eigenvectors are ambiguous under sign flips, and repeated eigenvalues allow
orthogonal rotations inside an eigenspace.  The projector is invariant to both:

\[
(-U_k)(-U_k)^\top = U_k U_k^\top,
\]

and

\[
(U_k R)(U_k R)^\top = U_k U_k^\top.
\]

The model therefore denoises eigenvalues and the selected eigenspace as two
coupled spectral variables instead of predicting several heat-kernel matrices
that may not correspond to one common Laplacian.

## Training bridge

For a target graph `G` and a connected HH realization `G_HH` with the same
degree multiset, GraphER forms

\[
(\Lambda_0, P_0) = S(G_{HH}), \qquad
(\Lambda_1, P_1) = S(G).
\]

The eigenvalue bridge is the maintained trace-preserving spectral Brownian
bridge.  The projector bridge uses

\[
P_t=(1-\alpha_t)P_0+\alpha_tP_1
+\sigma_P\sqrt{\alpha_t(1-\alpha_t)}\,E_t,
\]

where `E_t` is symmetrized and projected into the subspace orthogonal to the
constant vector.  Intermediate states need not be exact projectors.  The clean
prediction is projected back to the rank-k projector manifold.

Because `P_k` is node indexed, target nodes are aligned to source nodes within
equal-degree groups using a deterministic Hungarian assignment based on local
structural and projector-row signatures.  This replaces the random within-degree
alignment used by the legacy heat-kernel experiment.

## Projector head

The pair encoder receives current and source projector entries as two extra
pair channels.  A pair head emits a symmetric score matrix.  Projection uses a
Helmert basis of `1^perp`, performs an eigendecomposition there, takes the
largest `k` score modes, and reconstructs

\[
\widehat P_k=\widehat U_k\widehat U_k^\top.
\]

Thus the clean prediction satisfies, up to numerical precision,

\[
\widehat P_k^\top=\widehat P_k,\quad
\widehat P_k^2=\widehat P_k,\quad
\operatorname{tr}(\widehat P_k)=k,\quad
\widehat P_k\mathbf1=0.
\]

## Generation and rewiring

Generation initializes

\[
(\Lambda_t,P_t)=(\Lambda_{HH},P_{HH})
\]

and reverses both bridges.  The hard graph remains the HH/source graph until the
continuous endpoint is sampled.  Rewiring then scores a candidate graph with

\[
D_{spec}(G') =
\frac{w_\lambda D_\lambda(\Lambda(G'),\widehat\Lambda)
+w_P D_P(P_k(G'),\widehat P_k)}{w_\lambda+w_P},
\]

where

\[
D_P(P,Q)=\frac{\|P-Q\|_F}{\sqrt{2k}}
\]

is the chordal projector distance.  Every accepted swap still preserves the
sampled indexed degree sequence and connectivity.

## Community-small configs

Structured spectral-only:

```text
configs/experiments/grapher/community_small_lambda_projector_only_learned.yaml
```

Full edge + Lambda/P + structural summaries:

```text
configs/experiments/grapher/community_small_joint_edge_lambda_projector_graphlets345_learned.yaml
```

The legacy heat-kernel implementation remains available only for reproducing
existing checkpoints and ablations.
