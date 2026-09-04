# Held-out typed-invariant generation diagnostic

This revision adds `generation.invariant_source: test_empirical` to
`scripts/run_attributed_grapher.py`.

- `learned`: sample a typed invariant from the trained typed-degree VAE.
- `train_empirical` / `empirical`: sample a typed invariant from a random training graph.
- `test_empirical`: sample a typed invariant from a random held-out test graph.
- `test_oracle` / `oracle`: deterministically use test graph `index % len(test)`.

`test_empirical` and `test_oracle` use held-out information and are diagnostic modes,
not unconditional generation results.

A dedicated configuration is provided at:

`configs/experiments/grapher/qm9_attributed_spectral_graphlet_v2_enriched_source_test_empirical.yaml`

The focused regression suite used for this change passes 34 tests.
