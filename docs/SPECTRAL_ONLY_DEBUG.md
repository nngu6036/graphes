# GraphER spectral-only debug path

This debug path removes graphlet prediction, source enrichment, and graph/GNN context from the denoiser. It is intended to isolate the minimum GraphER mechanism before reintroducing additional structure.

## Minimal model

For a clean training graph `G*`, construct a connected Havel-Hakimi source `G_HH` from the **same degree sequence**. Let `lambda_HH` and `lambda*` be the sorted combinatorial-Laplacian eigenvalues.

Training samples only a continuous spectral bridge state

```text
lambda_t = (1 - alpha_t) lambda_HH + alpha_t lambda* + sigma_t epsilon,
```

with `lambda_1 = 0` and spectral trace preserved. No intermediate graph is constructed for `lambda_t`. No node features, edge features, graphlets, clustering statistics, or eigenvectors are diffused.

The denoiser input is only:

```text
(noisy lambda_t, source lambda_HH, spectral rank, t, graph size, padding mask)
```

and it predicts the complete clean eigenvalue vector `lambda*` in one forward pass. The output parameterization enforces sorted nonnegative gaps, `lambda_1=0`, and `sum(lambda)=2m`.

At generation time:

```text
DH-VAE/train empirical degree sequence
        -> connected HH graph
        -> current Laplacian spectrum
        -> spectral-only denoiser predicts clean spectrum
        -> valid degree-preserving double-edge swaps
        -> choose the swap that reduces spectral RMSE
        -> repeat
```

The debug config uses `clean_mix=1`, refreshes the prediction after every accepted swap, and disables bridge-target expansion. Thus the projection stage is as direct as possible.

## Recommended debugging order

### 1. Test the rewiring projection with an oracle clean spectrum

This completely removes the neural network. If this fails to reduce spectral distance, debug candidate generation/search before training the denoiser.

```bash
PYTHONPATH=src python scripts/diagnose_spectral_oracle_projection.py \
  --config configs/experiments/grapher/community_small_topology_spectral_debug.yaml \
  --split val \
  --max-graphs 16 \
  --seed 42 \
  --device gpu \
  --output-dir outputs/debug/community_small_spectral_oracle
```

Expected: degree preservation = 1, connectedness = 1, and positive mean spectral gain.

### 2. Train only the spectral denoiser

```bash
PYTHONPATH=src python scripts/train_topology_grapher.py \
  --config configs/experiments/grapher/community_small_topology_spectral_debug.yaml \
  --output-dir outputs/topology_grapher/community_small_spectral_debug/seed_42 \
  --seed 42 \
  --device gpu
```

For a fast smoke test first, add `--epochs 10 --max-train-graphs 16 --max-val-graphs 8`.

### 3. Measure denoising without rewiring

```bash
PYTHONPATH=src python scripts/diagnose_spectral_denoiser.py \
  --config configs/experiments/grapher/community_small_topology_spectral_debug.yaml \
  --checkpoint outputs/topology_grapher/community_small_spectral_debug/seed_42/checkpoint.pt \
  --split val \
  --samples-per-graph 32 \
  --paths-per-graph 2 \
  --device gpu \
  --json-out outputs/topology_grapher/community_small_spectral_debug/seed_42/denoiser_val.json
```

The key quantity is `predicted_nrmse` versus `noisy_nrmse`. The predictor should reduce NRMSE across diffusion-time bins, not only near the clean endpoint.

### 4. Run end-to-end with empirical training degrees

The debug config defaults to `generation.degree_source=train_empirical`, which removes DH-VAE error while testing denoising + rewiring.

```bash
PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config configs/experiments/grapher/community_small_topology_spectral_debug.yaml \
  --checkpoint outputs/topology_grapher/community_small_spectral_debug/seed_42/checkpoint.pt \
  --output-dir outputs/topology_generation/community_small_spectral_debug_empirical/seed_42 \
  --num-generate 64 \
  --seed 42 \
  --device gpu
```

Then evaluate normally.

### 5. Only after the spectral path works, enable the learned DH-VAE

```bash
PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config configs/experiments/grapher/community_small_topology_spectral_debug.yaml \
  --checkpoint outputs/topology_grapher/community_small_spectral_debug/seed_42/checkpoint.pt \
  --output-dir outputs/topology_generation/community_small_spectral_debug_learned/seed_42 \
  --num-generate 1024 \
  --seed 42 \
  --device gpu \
  --set generation.degree_source=learned
```

The config points to the verified Community-small DH-VAE checkpoint `outputs/degree_generators/sbm/seed_42/checkpoint.pt` with edge-count and exact degree-sum conditioning.

## Failure localization

- Oracle projection poor -> rewiring/search cannot realize spectral targets efficiently.
- Oracle projection good, denoiser poor -> spectral diffusion/predictor/loss problem.
- Denoiser good, empirical-degree generation poor -> train/generation state mismatch or projection-policy problem.
- Empirical-degree generation good, learned-degree generation poor -> DH-VAE/invariant prior problem.

## Auxiliary clustering-coefficient prediction

The minimal spectral debug model now optionally predicts the clean graph-average
local clustering coefficient,

`C(G) = (1 / |V|) * sum_v C_v`.

This is an **auxiliary x0 target only**. The diffusion state remains the Laplacian
eigenvalue vector; clustering is not noised/diffused, and the generation refiner
still scores double-edge swaps using spectral distance only. This deliberately
separates "can the representation predict a simple structural summary?" from
"does that summary improve rewiring?".

