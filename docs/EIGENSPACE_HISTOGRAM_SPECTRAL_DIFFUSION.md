# Permutation-invariant eigenspace-histogram spectral diffusion

## Representation

The maintained eigenspace extension avoids node-indexed heat kernels and
projectors.  For a connected graph with combinatorial Laplacian

\[
L = U\Lambda U^\top,
\]

let `U_k=[u_2,...,u_{k+1}]` contain the first `k` non-trivial Laplacian
eigenvectors.  The squared spectral-embedding distance for an unordered node
pair is

\[
d_{ij}^2 = \|U_k[i,:]-U_k[j,:]\|_2^2.
\]

This distance is invariant to eigenvector sign choices and to orthogonal
rotations inside the selected eigenspace.  Distances are normalized by

\[
s_{n,k}=\sqrt{\frac{2k}{n-1}},\qquad
\widetilde d_{ij}=d_{ij}/s_{n,k},
\]

where `k` is clipped to `min(k,n-1)`.  For an orthonormal non-trivial spectral
embedding the mean squared normalized pair distance is one.

To remove node-index dependence, pairs are grouped only by their unordered
endpoint degrees.  For every degree pair `(a,b)`, GraphER forms a probability
histogram

\[
h_{a,b}=\operatorname{Hist}\{\widetilde d_{ij}:d_i=a,d_j=b,i<j\}.
\]

The complete eigenspace summary is the ordered collection of all degree-pair
histograms.  It is permutation invariant, and its active blocks and block
weights are fixed by the degree sequence.  Therefore every degree-preserving
rewiring state has exactly the same histogram support.

The spectral state used by the new mode is

\[
S(G)=(\Lambda(G),h_{\rm eig}(G)).
\]

## Continuous bridge and prediction

For each target graph `G*`, GraphER constructs a degree-matched connected HH
source `G0`.  Training computes source and clean eigenvalues and eigenspace
histograms and samples endpoint-conditioned continuous bridge states for both.
The histogram bridge uses centered Gaussian noise independently inside every
active degree-pair block.  This keeps every active block on the affine
sum-one hyperplane; intermediate coordinates are allowed to leave the simplex.

The spectral transformer continues to predict the clean Laplacian spectrum.
The current/source histogram bridge states are encoded as graph-level spectral
conditioning, and a block-softmax head predicts a valid clean histogram.
The histogram loss is the degree-pair-weighted one-dimensional Wasserstein-1
distance between normalized-distance histograms.

No source/target node alignment is required by this representation.

## Generation and constrained realization

Generation samples a degree sequence, constructs its connected HH realization,
and initializes the reverse bridge from the exact HH spectrum and eigenspace
histogram.  The final predicted state contains a clean spectrum and clean
histogram target.

For every valid double-edge-swap candidate `G'`, GraphER recomputes

* the normalized Laplacian eigenvalue discrepancy; and
* the degree-pair-weighted Wasserstein-1 histogram discrepancy.

The two spectral subcomponents are normalized by their discrepancies at the HH
source before their configured weights are combined.  This prevents a larger
raw numerical scale from dominating merely because of units.  The constrained
rewiring operator still guarantees indexed-degree preservation, simplicity and
connectivity.

## Community-small configuration

True spectral-only experiment:

```text
configs/experiments/grapher/community_small_eigenspace_histogram_only_learned.yaml
```

Default summary settings:

```yaml
spectral_prediction:
  representation: lambda_eigenspace_histogram
  eigenspace_rank: 4
  eigenspace_histogram_bins: 16
  eigenspace_histogram_degree_max: 19
  eigenspace_histogram_max_distance: 3.0
  lambda_weight: 1.0
  eigenspace_histogram_weight: 1.0

summary_diffusion:
  spectral_sigma: 0.2
  eigenspace_histogram_sigma: 0.1

topology_predictor:
  loss_weights:
    spectrum: 1.0
    eigenspace_histogram: 1.0

topology_refiner:
  weights:
    edge: 0.0
    spectral: 1.0
    clustering: 0.0
    orbit: 0.0
    graphlet: 0.0
```

Full global+local experiment:

```text
configs/experiments/grapher/community_small_joint_edge_eigenspace_histogram_graphlets345_learned.yaml
```

## Commands

```bash
CFG=configs/experiments/grapher/community_small_eigenspace_histogram_only_learned.yaml
TRAIN=outputs/topology_grapher/community_small_eigenspace_histogram_only_learned/seed_42
GEN=outputs/topology_generation/community_small_eigenspace_histogram_only_learned/seed_42/learned

PYTHONPATH=src python scripts/train_topology_grapher.py \
  --config "$CFG" \
  --output-dir "$TRAIN" \
  --seed 42 \
  --device gpu

PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config "$CFG" \
  --checkpoint "$TRAIN/checkpoint.pt" \
  --output-dir "$GEN" \
  --num-generate 1024 \
  --seed 42 \
  --device gpu

PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config "$CFG" \
  --generated-dir "$GEN" \
  --reference-split test \
  --output-dir "$GEN/evaluation_test"
```

## Legacy modes

The previous node-indexed `heat_kernel` and `lambda_projector` representations
are retained only for reproducibility and direct ablation.  New eigenspace
experiments should use `lambda_eigenspace_histogram`.
