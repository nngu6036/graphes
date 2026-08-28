# Attributed Spectral--Graphlet Diffusion GraphER

This implementation extends the confirmed generic GraphER design to heavy-atom
molecular graphs.

## Revised invariant and transition

The source graph may still be constructed from a typed-degree sample because
that provides a chemically plausible initialized molecule.  The typed degree is
now an **initialization state**, not the hard invariant of the rewiring process.

The revised hard rewiring invariant is

```text
(indexed atom types, indexed ordinary degrees, global bond-type counts)
```

A generation action may select two bonds of different types

```text
(u, v, type a), (x, y, type b)
```

and enumerate both topological reconnections and both assignments of the two
removed bond types:

```text
(u, x, a), (v, y, b)
(u, x, b), (v, y, a)
(u, y, a), (v, x, b)
(u, y, b), (v, x, a)
```

When `a == b`, duplicate assignments collapse to the usual two double-edge
swaps.  Every accepted move preserves each node's ordinary degree and atom
category and removes/adds exactly one edge of each selected type, so the global
bond-type histogram is preserved exactly.

Per-node typed degrees and per-node weighted valence are intentionally allowed
to change.  They are no longer asserted as invariants.  Instead, every
candidate must pass inexpensive atom-specific valence checks and the best
shortlist is checked by RDKit before acceptance.  This larger action space lets
GraphER move single/double/triple bond mass between local atom environments
while retaining exact topology-degree and global bond-count constraints.

The previous strict same-bond-type mode remains available for ablation with:

```yaml
molecular:
  require_same_edge_type_pair: true
  preserve_typed_degree: true
  preserve_weighted_valence: true
```

## Continuous training state

Training does not construct a rewiring trajectory. For a clean molecule and a
source endpoint, it computes:

- the ordinary topology Laplacian spectrum;
- the bond-order-weighted Laplacian spectrum; and
- attributed graphlet probability simplexes and their blockwise CLR logits.

A stochastic endpoint-conditioned bridge is sampled directly in these
continuous summary coordinates. The model predicts the clean endpoint from the
continuous current state, fixed source summaries, source graph, and normalized
diffusion progress.

The two Laplacian traces remain compatible with the revised rewiring kernel.
The ordinary-degree constraint fixes the topology trace, while preservation of
global bond-type counts fixes the total bond order and therefore the weighted
Laplacian trace.  In contrast, the second moment of the bond-weighted spectrum
is no longer an invariant because local weighted degrees may change.

## Generation

At generation time, the current realized graph supplies the two spectra and
attributed graphlet logits. The predictor estimates the clean summaries, the
schedulers derive the next continuous denoising target, and valid attributed
rewiring candidates are ranked by a global-to-local combination of dual-spectral
and attributed-graphlet distances.

Attributed graphlet candidates use an exact stateful local-delta cache:

```text
C_{t+1} = C_t + Delta C_t
```

A full graphlet count is required only for the source graph. RDKit sanitization
is applied only to the top combined-energy shortlist after inexpensive
invariant, connectivity, valence, spectral, and graphlet checks.

Canonical QM9 projects formal charge out of its categorical node state. The
QM9 preset therefore uses the projected-state valence envelope (including
N(+1) at valence four and O(+1) at valence three) and restores those implied
positive charges only for candidate sanitization. Raw validity reporting keeps
its existing charge-neutral behavior.

## Revised QM9 generation flags

```yaml
attributed_refiner:
  molecular:
    require_same_edge_type_pair: false
    preserve_removed_edge_type: false
    preserve_global_edge_type_counts: true
    enumerate_edge_type_permutations: true
    preserve_node_types: true
    preserve_ordinary_degree: true
    preserve_typed_degree: false
    preserve_weighted_valence: false
    enforce_molecular_valence: true
```

The generation report now distinguishes the required relaxed invariant from
strict typed-degree preservation:

```text
rewiring_invariant_preservation_rate
node_type_preservation_rate
indexed_degree_preservation_rate
edge_type_count_preservation_rate
```

must remain one, while

```text
typed_degree_preservation_rate
per_node_weighted_valence_preservation_rate
```

are diagnostics and may be below one in cross-type mode.

## Commands

Train the light QM9 development model:

```bash
PYTHONPATH=src python scripts/train_attributed_grapher.py \
  --config configs/experiments/grapher/qm9_attributed_spectral_graphlet_light.yaml \
  --output-dir outputs/attributed_grapher/qm9_spectral_graphlet_light/seed_42 \
  --seed 42 \
  --device gpu
```

Generate with empirical training source states:

```bash
PYTHONPATH=src python scripts/run_attributed_grapher.py \
  --config configs/experiments/grapher/qm9_attributed_spectral_graphlet_light.yaml \
  --output-dir outputs/attributed_generation/qm9_spectral_graphlet_light/seed_42 \
  --num-generate 256 \
  --seed 42 \
  --device gpu
```

Evaluate:

```bash
PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-dir outputs/attributed_generation/qm9_spectral_graphlet_light/seed_42 \
  --dataset-root outputs/datasets \
  --dataset qm9_attributed \
  --reference-split test \
  --train-split train \
  --output-dir outputs/attributed_grapher/qm9_spectral_graphlet_light/seed_42/evaluation
```
