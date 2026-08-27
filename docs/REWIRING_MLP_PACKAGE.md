# Rewiring MLP package boundary

All GraphER correction code is grouped under
`src/grapher/rewiring_mlp/`. This is a code-ownership refactor: it does not
change the predictor architecture, candidate construction, validity checks,
search rule, metrics, or saved checkpoint formats.

## Package layout

```text
grapher.rewiring_mlp
├── generic/       Generic structural-summary predictor and correction loop
├── attributed/    Attribute-aware predictor, selector, and correction loop
├── core/          Shared double-edge-swap actions and validity operations
├── molecular/     Molecular constraints, graph conversion, typed invariants
└── evaluation/    Raw/corrected metrics, diagnostics, and study aggregation
```

The root package is intentionally lightweight. Import the required submodule
directly; for example:

```python
from grapher.rewiring_mlp.generic.refiner import (
    refine_graph_with_topology_predictions,
)
from grapher.rewiring_mlp.evaluation.metrics import evaluate_graph_sets
```

The former top-level packages `grapher.rewiring_mlp.generic`, `grapher.rewiring_mlp.attributed`,
`grapher.rewiring_mlp.core`, `grapher.rewiring_mlp.molecular`, and `grapher.rewiring_mlp.evaluation` have been
removed. Internal scripts and tests use the new canonical imports.

Baseline generators remain isolated under `grapher.models`. They communicate
with the correction layer through completed serialized graph batches rather
than through implementation-level imports into the generic or attributed
correctors.
