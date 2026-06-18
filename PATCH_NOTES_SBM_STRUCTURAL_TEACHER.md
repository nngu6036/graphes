# SBM structural-teacher patch

This patch targets the high SBM clustering/spectral mismatch while preserving
GraphER's hard degree and connectivity guarantees.

## Changes

- Adds an exact, graph-copy-free triangle and average-clustering delta for every
  valid double-edge swap.
- Exposes four additional action-local features when `local_feature_dim: 12`:
  removed-edge common-neighbour counts, triangle delta, and average-clustering
  delta.
- Adds a normalized structural offline-teacher discrepancy with configurable
  edge, clustering, triangle, and optional spectral terms.
- Adds a two-stage optional spectral shortlist so eigenvalue computations are
  not performed for the entire candidate set.
- Builds true local/global candidate mixtures during teacher construction,
  neural training, and generation.
- Normalizes teacher time by each realized path length (`t=s/L`) instead of the
  maximum path budget.
- Logs path-length/lower-bound, edge-symmetric-difference, clustering-gap,
  triangle-gap, top-k accuracy, rank, entropy, and teacher-margin diagnostics.
- Updates `configs/models/grapher_generic.yaml` to an SBM-64 structural-teacher
  run and keeps the prior edge-only configuration as
  `configs/models/grapher_generic_edge_teacher.yaml`.

## Train

The new model has `local_feature_dim: 12`, so retrain GraphER with a new run ID.
The DH-VAE checkpoint can be reused if it was trained on the same connected-SBM
split.

```bash
PYTHONPATH=src python scripts/train_generic_grapher_model.py \
  --dataset sbm \
  --seed 42 \
  --run-id 1 \
  --model-config configs/models/grapher_generic.yaml
```

## Generate and evaluate

```bash
PYTHONPATH=src python scripts/generate_grapher_samples.py \
  --dataset sbm --num-samples 1024 --seed 42 --run-id 1 \
  --model-config configs/models/grapher_generic.yaml --force

PYTHONPATH=src python scripts/evaluate_grapher_metrics.py \
  --dataset sbm --model grapher --run-id 1 \
  --reference-split test --max-reference-graphs 1024 \
  --max-generated-graphs 1024
```

## Validation performed

- All Python files compile.
- Connected-SBM generation returns 100% connected graphs with no isolated nodes
  in a 100-graph smoke test.
- Exact structural deltas were checked against explicit NetworkX graph
  recomputation for 950 valid swaps.
- Structural and edge-only teacher-cache smoke tests completed successfully.
