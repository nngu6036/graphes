# Validation: attributed same-bond stochastic diversification

Implementation validation was run on the updated repository.

Focused tests:

```text
52 passed
```

These cover the hybrid/hierarchical attributed pipelines, masking/chemical constraints, prepared-dataset contract, and the new same-bond diversification tests.

Broader GDSM attributed/hybrid regression selection:

```text
97 passed
```

New tests verify:

- one same-bond double-edge swap preserves indexed typed degrees;
- node categories and global bond-category counts are preserved;
- connectivity is preserved;
- zero requested steps are an exact identity operation;
- hybrid configuration validation accepts the new generation-only block;
- generation writes paired `pre_refinement_graphs.pkl` and `base_graphs.pkl` artifacts;
- generation manifest records same-bond rewiring and typed-degree preservation;
- all supplied QM9 seed configs resolve with graphlet orders 3,4,5 and same-bond diversification enabled.

The full 10,000-molecule QM9/ZINC production benchmark was not run in this environment.
