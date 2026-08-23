# Spectral + Graphlet-Logit Diffusion Guidance

This implementation extends the spectral GraphER topology generator with a
second denoising channel for local higher-order structure.

## State representation

The actual state is always a simple connected graph in the fixed indexed-degree
fibre. For every state `G_t` the joint predictor receives the graph, normalized
time, the current Laplacian eigenvalue sequence, and graphlet CLR/logit states.
It predicts the clean targets in one forward pass:

- `lambda_hat_0`: the full clean Laplacian spectrum, produced by the variable-
  length Spectral Transformer;
- `z_hat_{k,0}`: clean graphlet CLR/logits for every configured order `k`.

Eigenvectors are neither predicted nor fixed.

## Graphlet simplex and logits

For graphlet order `k`, each connected induced graphlet count is divided by
`C(n,k)`. A final `disconnected` coordinate contains the remaining probability
mass. Therefore each block is a proper probability simplex over all induced
`k`-node subsets. The block is transformed to centered log-ratio coordinates
using a small smoothing epsilon. A blockwise softmax maps predicted CLR/logits
back to probabilities.

This representation preserves both graphlet composition and connected-subgraph
mass, which the older connected-only normalized histogram discarded.

## Denoising and rewiring

The spectral and graphlet targets each use an x0-style deterministic bridge from
the current state toward the corresponding predicted clean state. Candidate
swaps are scored by

`w_spec(t) * D_spec + w_graphlet(t) * D_CLR`.

A configurable global-to-local schedule gives the spectrum the larger weight
early in generation and shifts emphasis to graphlet structure later. Only valid
degree-preserving double-edge swaps are considered, so hard invariants are never
relaxed by either learned target.

Candidate graphlet states use the existing exact local-delta update. The full
graphlet histogram is not recomputed for every candidate.

## Debugging

Set `topology_refiner.debug.enabled=true`. Lines prefixed with
`[GraphER/SpectralGraphlet]` report prediction refreshes, clean targets, current
and next targets, spectral/graphlet weights and bridge mixes, candidate counts,
rejection reasons, top candidate combined/spectral/graphlet gains, separate
projection residuals, and accepted states.

## Community-small commands

Train:

```bash
PYTHONPATH=src python scripts/train_topology_grapher.py \
  --config configs/experiments/grapher/community_small_topology_spectral_graphlet.yaml \
  --output-dir outputs/topology_grapher/community_small_spectral_graphlet/seed_42 \
  --seed 42 \
  --device gpu
```

Debug a small batch:

```bash
PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config configs/experiments/grapher/community_small_topology_spectral_graphlet.yaml \
  --output-dir outputs/topology_generation/community_small_spectral_graphlet/debug_seed_42 \
  --num-generate 5 \
  --seed 42 \
  --device gpu \
  --set topology_refiner.debug.enabled=true
```

Full generation:

```bash
PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config configs/experiments/grapher/community_small_topology_spectral_graphlet.yaml \
  --output-dir outputs/topology_generation/community_small_spectral_graphlet/seed_42 \
  --num-generate 1024 \
  --seed 42 \
  --device gpu
```

Evaluate:

```bash
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/grapher/community_small_topology_spectral_graphlet.yaml \
  --generated-dir outputs/topology_generation/community_small_spectral_graphlet/seed_42 \
  --output-dir outputs/topology_grapher/community_small_spectral_graphlet/seed_42/evaluation
```
