# GraphER Enriched-Source Merge (2026-09-03)

This repository combines the latest generic and attributed GraphER source-enrichment implementations.

## Generic graph path

- Joint degree-conditioned summary estimator: `D -> (spectrum, graphlet CLR)`.
- Degree-preserving source enrichment before the main dynamic refiner.
- Fixed-target best-state selection during source enrichment.
- Batched candidate eigendecomposition, fast virtual swap checks, adaptive candidate budgets, endpoint caching, and removal of redundant collate eigendecomposition.
- Config: `configs/experiments/grapher/community_small_topology_spectral_graphlet_v3_enriched_source.yaml`.

## Attributed molecular path

- Joint hard-invariant-conditioned estimator using atom categories, ordinary degrees, graph size, and global bond-type counts.
- Predicted target: topology spectrum, bond-weighted spectrum, and attributed graphlet CLR summaries.
- Cross-bond-type source enrichment followed by state-conditioned molecular GraphER refinement.
- Early virtual topology/valence rejection, batched dual-channel eigendecomposition, local attributed graphlet deltas, adaptive candidate budgets, endpoint caching, and RDKit shortlist filtering.
- Config: `configs/experiments/grapher/qm9_attributed_spectral_graphlet_v2_enriched_source.yaml`.

## Integration fix

`run_topology_grapher.py` now centralizes degree-source construction and supports the declared `learned`, `train_empirical`, `test_empirical`, and `test_oracle` modes consistently. Test-derived degree modes remain diagnostics and should not be presented as unconditional generation results.

## Verification

The focused integration suite passes:

```text
60 passed
```

Command used:

```bash
PYTHONPATH=src:. pytest -q \
  tests/test_grapher_optimized_enriched_source.py \
  tests/test_attributed_runtime_enrichment.py \
  tests/test_attributed_spectral_graphlet_diffusion.py \
  tests/test_train_attributed_grapher.py \
  tests/test_spectral_graphlet_grapher.py \
  tests/test_molecular_generation_constraints.py \
  tests/test_evaluate_generated_molecules.py \
  tests/test_graph_generation_report.py \
  tests/test_config_overrides.py \
  tests/test_topology_degree_source.py \
  tests/test_summary_diffusion_training.py
```
