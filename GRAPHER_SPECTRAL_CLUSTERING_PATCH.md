# GraphER spectral-only + clustering-summary debug patch

## Generation confirmation

The spectral generation path uses degree-preserving double-edge rewiring:

1. `scripts/run_topology_grapher.py` constructs the connected HH source.
2. For `guidance_mode == spectral`, it calls `refine_graph_with_spectral_predictions`.
3. The refiner calls `propose_valid_topology_swaps` to enumerate/sample double-edge swaps.
4. Candidate graphs are ranked by spectral discrepancy to the predicted clean spectrum.
5. An accepted candidate replaces the current graph only after explicit degree and connectivity assertions.

The clustering auxiliary head added here does not change that generation path.

## New training target

The spectral-only predictor can now learn a scalar clean structural summary:

- target: NetworkX `average_clustering(clean_target_graph)`
- range: `[0, 1]`
- model output: `clean_clustering_coefficient`
- head input: masked mean pool of spectral Transformer tokens
- output constraint: sigmoid
- loss: Smooth L1, controlled by `loss_weights.clustering_coefficient`
- metrics: clustering coefficient loss, MAE, and RMSE

The clustering coefficient is **not diffused**. The only noisy continuous state remains the Laplacian eigenvalue vector.

## Debug config

`configs/experiments/grapher/community_small_topology_spectral_debug.yaml` now contains:

```yaml
structure_summary_prediction:
  clustering_coefficient: true

topology_predictor:
  loss_weights:
    spectrum: 1.0
    moment2: 0.0
    low_frequency: 0.0
    low_frequency_k: 0
    clustering_coefficient: 1.0
```

Generation remains:

```yaml
topology_refiner:
  mode: spectral
```

so clustering prediction is diagnostic/auxiliary only for now.

## Diagnostics

`scripts/diagnose_spectral_denoiser.py` additionally reports clustering-coefficient MAE when the checkpoint includes the head.

## Tests

28 targeted spectral/topology tests pass, including the new clustering-summary tests and existing rewiring/refiner tests.
