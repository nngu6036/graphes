# Cycle-only attributed graphlet guidance

## Purpose

The molecular GraphER path can now restrict the predicted and scored attributed
induced-graphlet basis to ring structure. The historical behaviour remains the
default, so existing configurations and checkpoints continue to use all
connected induced graphlets.

The new configuration field is:

```yaml
graphlet_prediction:
  graphlet_topology_filter: simple_cycle
```

Supported values are:

- `all`: every graphlet allowed by `graphlet_connected_only` (historical mode);
- `cyclic`: every connected induced graphlet containing at least one cycle,
  including chorded and fused local structures;
- `simple_cycle`: only chordless induced cycles `C_k`, which is the recommended
  molecular ring-only mode.

`graphlet_cycle_only: true` is accepted as an alias for
`graphlet_topology_filter: simple_cycle`.

For a cycle-only block of order `k`, the model simplex contains the attributed
`C_k` classes observed in the training split, the existing overflow class, and
one background coordinate. The background coordinate now means “the sampled
`k`-node subset is not a selected ring graphlet”; it includes disconnected,
acyclic, and chorded subsets. The corresponding mass target is therefore the
fraction of induced `k`-node subsets that are chordless rings.

## Implementation

The filter is applied consistently to:

1. training-only attributed graphlet-vocabulary fitting;
2. clean/source target extraction and endpoint caching;
3. model output dimensions and checkpoint metadata;
4. exact candidate-local count deltas during rewiring;
5. generic graphlet utilities and optional topology-only bases; and
6. molecular graphlet MMD evaluation.

For `simple_cycle`, attributed canonicalization no longer enumerates all `k!`
node orders. A cycle has exactly `2k` dihedral symmetries, so the canonical key
is found by comparing every rotation in both traversal directions. Exact full
counts enumerate induced rings directly by growing chord-free paths from a
canonical start node, rather than scanning every `k`-node subset. Candidate
scoring uses a related bounded cycle search through both endpoints of every
changed adjacency pair. Searching both pre-swap and post-swap states captures
not only changed ring edges but also rings created or destroyed when a chord is
removed or inserted.

## Complexity

Let `n` be the number of nodes, `K` the selected graphlet orders, `Delta` the
maximum degree, `N_conn,k` the number of connected induced `k`-subsets, and
`N_ring,k` the number of induced chordless `C_k` subsets.

The previous exact Python attributed counter has worst-case cost

```text
T_all(G) = Theta(sum_k [ C(n,k) k^2 + N_conn,k k! k^2 ]).
```

The first term extracts/checks each induced subset. The second term is exact
node/edge-label canonicalization over all node permutations.

Let `Q_k(G)` be the number of chord-free simple-path prefixes visited while
enumerating induced `C_k` rings. Cycle-only exact counting has cost

```text
T_ring(G) = O(sum_k [Q_k(G) k + N_ring,k k^2]).
```

The first term is induced-path search with chord pruning; the second is exact
dihedral attributed canonicalization. A degree-based upper bound is

```text
Q_k(G) <= n Delta (Delta - 1)^(k-2),
```

so fixed-order ring counting is near-linear in `n` for bounded-valence
molecular graphs. The unrestricted dense-graph worst case remains
combinatorial, but both the `C(n,k)` all-subset scan and the factorial
canonicalization factor are removed in exact `simple_cycle` mode. Sampled
counting still samples the requested `k`-subsets so that its estimator keeps
the historical sampling semantics.

For one double-edge-swap candidate, at most four adjacency pairs change. A
naive affected-subset bound is

```text
A_k <= 4 C(n-2,k-2).
```

The historical attributed local update is bounded by

```text
T_candidate,all = O(sum_k A_k k! k^2).
```

The ring-only implementation performs bounded DFS in the pre-swap and
post-swap graphs to find induced `C_k` cycles containing both endpoints of each
changed adjacency pair. Let `P_{k-1}(G,u)` denote the number of simple
length-`k-1` paths explored from endpoint `u`. Then

```text
T_candidate,ring =
  O(sum_k sum_{(u,v) in Delta A} [P_{k-1}(G,u) + P_{k-1}(G',u)] k^2).
```

For maximum degree `Delta`, this is `O(sum_k |Delta A| Delta^(k-1) k^2)`;
the unrestricted dense-graph worst case remains exponential. The search is
small for bounded-valence molecular graphs and canonicalization is performed
only for actual rings. The pair-subset bound `A_k <= 4 C(n-2,k-2)` remains a
useful implementation-independent upper bound for an alternative direct local
scan. For candidate budget `B` and accepted-step limit `T`, multiply the
per-candidate cost by at most `B T`.

The graphlet output head also shrinks. If `V_k` is the number of retained
classes, its output/activation cost is linear in

```text
D = sum_k (V_k + 1),
```

where `+1` is the background coordinate. For unlabelled connected topology
orders `k=3,4,5,6`, the complete basis has `2, 6, 21, 112` classes, while the
simple-cycle basis has one class per order. The simplex width therefore drops
from `145` to `8` coordinates, a `94.48%` reduction. Attributed widths are
training-data dependent, but retain the same strict subset relationship.

### QM9 numerical budget (`n <= 9`, `k=3..6`)

The historical all-subset implementation considers

```text
C(9,3)+C(9,4)+C(9,5)+C(9,6) = 420
```

candidate induced subsets per graph. Ignoring cache reuse, the old generic
attributed canonicalizer can examine up to

```text
84*3! + 126*4! + 126*5! + 84*6! = 79,128
```

node orders. The cycle canonicalizer has the much smaller absolute upper bound

```text
84*(2*3) + 126*(2*4) + 126*(2*5) + 84*(2*6) = 3,780
```

orientation checks, a `95.22%` reduction even under the unrealistic assumption
that every subset is a ring. The direct enumerator actually canonicalizes only
the `N_ring,k` rings it finds. With `n=9` and molecular maximum degree
`Delta=4`, its loose terminal-path bound over `k=3..6` is

```text
9*4*(3 + 3^2 + 3^3 + 3^4) = 4,320,
```

before the canonical-start, no-repeat, and chord-pruning rules reduce the
search further.

For one candidate swap, the naive affected-subset bounds for `k=3..6` are
`28, 84, 140, 140` (392 total). Canonicalizing both before and after gives an
old orientation upper bound of `239,568`, compared with `7,840` dihedral
orientations (`96.73%` lower); the bounded cycle search normally
canonicalizes far fewer subsets than this bound.

## Recommended QM9 configuration

Use:

```text
configs/experiments/grapher/qm9_attributed_spectral_cycle_graphlet.yaml
```

It keeps the current enriched-source, dual-spectrum, and constrained rewiring
settings, changes the attributed graphlet basis to `simple_cycle`, and extends
ring orders through six nodes. The previous effective size weights are retained
(`k=3: 1.5`, `k=4: 2.5`, and the existing default `1.0` for `k=5`); `k=6` also
starts at `1.0`. Keeping the same proposal and step budgets makes the first
comparison isolate the representation change as closely as possible. A later
quality-cost sweep can reduce those budgets after measuring acceptance and
runtime.

## Commands

From the repository root:

```bash
export PYTHONPATH="$PWD/src:$PWD"
```

Train the typed-degree prior only when its checkpoint does not already exist:

```bash
python -m grapher.models.dhvae_hh.training \
  --config configs/experiments/dhvae/qm9_typed.yaml
```

Train the attributed cycle-guided predictor:

```bash
python scripts/train_attributed_grapher.py \
  --config configs/experiments/grapher/qm9_attributed_spectral_cycle_graphlet.yaml \
  --output-dir outputs/attributed_grapher/qm9_spectral_cycle_graphlet/seed_42 \
  --seed 42 \
  --device gpu
```

Generate 1,024 molecules:

```bash
python scripts/run_attributed_grapher.py \
  --config configs/experiments/grapher/qm9_attributed_spectral_cycle_graphlet.yaml \
  --checkpoint outputs/attributed_grapher/qm9_spectral_cycle_graphlet/seed_42/checkpoint.pt \
  --output-dir outputs/attributed_generation/qm9_spectral_cycle_graphlet/seed_42 \
  --num-generate 1024 \
  --seed 42 \
  --device gpu
```

Evaluate validity, uniqueness, novelty, NSPDK, FCD, and the matching cycle-only
attributed graphlet statistics:

```bash
python scripts/evaluate_generated_molecules.py \
  --generated-dir outputs/attributed_generation/qm9_spectral_cycle_graphlet/seed_42 \
  --dataset-root outputs/datasets \
  --dataset qm9_attributed \
  --reference-split test \
  --train-split train \
  --metric-molecule-source raw_valid \
  --nspdk-backend eden \
  --graphlet-mmd \
  --graphlet-k-min 3 \
  --graphlet-k-max 6 \
  --graphlet-topology-filter simple_cycle \
  --graphlet-node-attribute atomic_num \
  --graphlet-edge-attribute bond_type \
  --graphlet-attributed-backend python \
  --fcd-device auto \
  --require-fcd \
  --output-dir outputs/attributed_generation/qm9_spectral_cycle_graphlet/seed_42/evaluation
```

The additional report fields are `graphlet_histogram_mmd` and
`graphlet_selected_mass_mmd`. Remove `--require-fcd` when intentionally running
without the optional `fcd_torch` backend; the evaluator will then report
`fcd: null` rather than failing.

For the broader cycle-containing ablation, override only the filter while
using a separate checkpoint and cache path:

```bash
--set graphlet_prediction.graphlet_topology_filter=cyclic
```

A checkpoint trained with one topology filter must not be reused with another,
because the graphlet vocabulary and output dimensions differ.

## Current v2 validity and distribution tuning
### Why the first cycle-only run degraded

The first QM9 cycle-only run returned:

| Metric | Value |
|---|---:|
| Raw validity | 0.8935546875 (915/1024) |
| Corrected validity | 1.0 |
| FCD | 2.2795456549 |
| NSPDK MMD | 0.0052771484 |
| Cycle-composition MMD | 0.0242373771 |
| Cycle selected-mass MMD | 0.0182933975 |
| Uniqueness | 0.9661202186 |
| Novelty | 0.75 |

The high uniqueness and novelty indicate that the generator is not collapsing.
The main failure is instead an overly permissive refinement trajectory:

1. the old generation configuration accepted candidates using projected formal
   charges, while the evaluator's headline validity sanitizes the serialized
   graph without projected-charge inference;
2. cross-bond-type rewiring preserved ordinary degrees and global bond counts,
   but could move bond-order incidences between atoms, changing per-node typed
   degree and weighted valence;
3. up to 12 source-enrichment swaps plus 64 reverse-refinement swaps allowed
   substantial drift from the typed source;
4. the candidate energy used sparse CLR coordinates only, although the
   reported cycle diagnostics separate attributed ring composition from the
   total selected-ring mass.

The v2 update keeps cycle-only counting and the cross-type action space, but
makes the projection conservative and aligned with the evaluator.

### Revised cycle objective

For graphlet order `k`, every simplex block ends with a background coordinate.
In `simple_cycle` mode,

```text
selected_ring_mass_k = 1 - p(background_k).
```

Candidate ranking now uses

```text
D_cycle =
    w_clr  * D_clr
  + w_prob * D_probability
  + w_mass * D_selected_ring_mass.
```

The default v2 QM9 weights are:

```yaml
graphlet_guidance:
  logit_weight: 0.25
  probability_weight: 1.0
  selected_mass_weight: 1.5
  probability_distance: rmse
```

The CLR component still distinguishes attributed ring classes, but it no
longer dominates because of rare or zero-probability coordinates. The explicit
mass term directly targets `graphlet_selected_mass_mmd`.

Training also adds:

```yaml
attributed_predictor:
  loss_weights:
    graphlet_selected_mass: 1.0
```

This is an L1 loss between predicted and target selected-ring mass, averaged
over available graphlet orders. Existing checkpoints remain loadable because
the model architecture is unchanged and the new loss weight defaults to zero.

### Chemistry-drift anchor

Cross-type swaps remain available, but candidate energy now includes soft drift
from the constructed typed source:

```text
E_chem =
    lambda_typed   * normalized_L1(typed_degree, source_typed_degree)
  + lambda_valence * normalized_L1(weighted_valence, source_weighted_valence).
```

Both distances are normalized to a nominal `[0, 1]` range under the preserved
global bond-category counts. The recommended settings are:

```yaml
molecular:
  typed_degree_drift_weight: 0.75
  weighted_valence_drift_weight: 1.50
```

These are soft anchors, not new hard invariants. A cross-type move can still be
selected when its spectral/ring gain compensates for its chemistry drift.

### Exact raw-validity alignment

The new setting is:

```yaml
molecular:
  rdkit_validation_mode: raw
  rdkit_candidate_check: true
  require_rdkit_source_validity: true

generation:
  require_rdkit_source_validity: true
  require_rdkit_final_validity: true
```

`raw` uses the same `nx_to_rdkit_mol(..., sanitize=True)` path as the evaluator,
without projected formal-charge inference and without bond correction. A
returned source is raw-valid, every accepted successor is raw-valid, and the
final graph is checked once more before serialization. Consequently, the
returned batch should have raw validity 1.0 unless the generation and
evaluation environments differ.

This guarantee is conditional on *returned samples*. Rejection can bias the
sampled distribution, so `end_to_end_yield`, generation attempts, and rejection
reasons remain part of the generation report.

The evaluator now additionally reports:

```text
validity_with_projected_formal_charges
num_raw_invalid_resolved_by_projected_formal_charges
projected_formal_charge_success_rate_on_raw_invalid
```

This separates charge-representation completion from bond-order correction. If
`validity_with_projected_formal_charges` is near 1.0 while raw validity is low,
most failures are missing-charge representation issues. If it stays near the
raw value, the failures require bond modification and are genuine topology or
bond-order errors under the chosen representation.

### Conservative search budget

The v2 configuration reduces source enrichment from 12 to 4 accepted steps and
main refinement from 64 to 40 accepted steps. It also raises the relative-gain
threshold and reduces late graphlet dominance:

```yaml
attributed_refiner:
  steps: 40
  min_relative_improvement: 2.0e-5
  global_to_local:
    spectral_initial: 1.0
    graphlet_initial: 0.15
    spectral_final: 0.50
    graphlet_final: 1.25
```

The cycle-only counter still supplies the main computational reduction. The
slightly larger raw-RDKit shortlist is used only after cheap structural,
connectivity, valence, and energy filters.

### New diagnostics

Generation reports now include:

```text
rdkit_validation_mode
rdkit_valid_source_rate_raw
rdkit_valid_final_rate_raw
rdkit_valid_source_rate_configured
rdkit_valid_final_rate_configured
mean_accepted_graphlet_logit_gain
mean_accepted_graphlet_probability_gain
mean_accepted_graphlet_selected_mass_gain
mean_accepted_chemistry_drift_gain
mean_typed_degree_drift
mean_weighted_valence_drift
```

Evaluation reports the aggregate cycle MMD and a one-pass per-order breakdown:

```text
graphlet_histogram_mmd_by_order
graphlet_selected_mass_mmd_by_order
```

The per-order result is computed from the same graphlet-count pass; it does not
repeat full C3--C6 counting four times.

### Configurations

### 1. Fast generation-only A/B test

```text
configs/experiments/grapher/
qm9_attributed_spectral_cycle_graphlet_v2_reuse_checkpoint.yaml
```

This uses the existing checkpoint:

```text
outputs/attributed_grapher/qm9_spectral_cycle_graphlet/seed_42/checkpoint.pt
```

It changes only generation-time scoring, chemistry anchoring, RDKit policy, and
search budget. Run this first to determine how much of the degradation came
from projection rather than predictor training.

### 2. Full v2 retraining

```text
configs/experiments/grapher/
qm9_attributed_spectral_cycle_graphlet_v2.yaml
```

This uses lower graphlet diffusion noise, a less extreme CLR epsilon, the new
ring-mass loss, and a separate endpoint cache/checkpoint.

### 3. Validity-first same-bond ablation

```text
configs/experiments/grapher/
qm9_attributed_spectral_cycle_graphlet_v2_strict_same_bond.yaml
```

This preserves per-node typed degree and weighted valence exactly. It is the
best diagnostic for deciding whether cross-type reachability helps enough to
justify its added search and distributional risk. It reuses the first
cycle-only checkpoint by default.

### Commands

Run from the repository root:

```bash
export PYTHONPATH="$PWD/src:$PWD"
```

### Fast A/B generation without retraining

```bash
python scripts/run_attributed_grapher.py \
  --config configs/experiments/grapher/qm9_attributed_spectral_cycle_graphlet_v2_reuse_checkpoint.yaml \
  --checkpoint outputs/attributed_grapher/qm9_spectral_cycle_graphlet/seed_42/checkpoint.pt \
  --output-dir outputs/attributed_generation/qm9_spectral_cycle_graphlet_v2_reuse/seed_42 \
  --num-generate 1024 \
  --seed 42 \
  --device gpu
```

Evaluate it:

```bash
python scripts/evaluate_generated_molecules.py \
  --generated-dir outputs/attributed_generation/qm9_spectral_cycle_graphlet_v2_reuse/seed_42 \
  --dataset-root outputs/datasets \
  --dataset qm9_attributed \
  --reference-split test \
  --train-split train \
  --metric-molecule-source raw_valid \
  --nspdk-backend eden \
  --graphlet-mmd \
  --graphlet-k-min 3 \
  --graphlet-k-max 6 \
  --graphlet-topology-filter simple_cycle \
  --graphlet-node-attribute atomic_num \
  --graphlet-edge-attribute bond_type \
  --graphlet-attributed-backend python \
  --fcd-device auto \
  --require-fcd \
  --output-dir outputs/attributed_generation/qm9_spectral_cycle_graphlet_v2_reuse/seed_42/evaluation
```

### Full v2 training

The typed-degree prior can be reused when this checkpoint already exists:

```text
outputs/degree_generators/qm9_typed/seed_42/checkpoint.pt
```

Otherwise train it with:

```bash
python -m grapher.models.dhvae_hh.training \
  --config configs/experiments/dhvae/qm9_typed.yaml
```

Train the v2 predictor:

```bash
python scripts/train_attributed_grapher.py \
  --config configs/experiments/grapher/qm9_attributed_spectral_cycle_graphlet_v2.yaml \
  --output-dir outputs/attributed_grapher/qm9_spectral_cycle_graphlet_v2/seed_42 \
  --seed 42 \
  --device gpu
```

Generate:

```bash
python scripts/run_attributed_grapher.py \
  --config configs/experiments/grapher/qm9_attributed_spectral_cycle_graphlet_v2.yaml \
  --checkpoint outputs/attributed_grapher/qm9_spectral_cycle_graphlet_v2/seed_42/checkpoint.pt \
  --output-dir outputs/attributed_generation/qm9_spectral_cycle_graphlet_v2/seed_42 \
  --num-generate 1024 \
  --seed 42 \
  --device gpu
```

Evaluate:

```bash
python scripts/evaluate_generated_molecules.py \
  --generated-dir outputs/attributed_generation/qm9_spectral_cycle_graphlet_v2/seed_42 \
  --dataset-root outputs/datasets \
  --dataset qm9_attributed \
  --reference-split test \
  --train-split train \
  --metric-molecule-source raw_valid \
  --nspdk-backend eden \
  --graphlet-mmd \
  --graphlet-k-min 3 \
  --graphlet-k-max 6 \
  --graphlet-topology-filter simple_cycle \
  --graphlet-node-attribute atomic_num \
  --graphlet-edge-attribute bond_type \
  --graphlet-attributed-backend python \
  --fcd-device auto \
  --require-fcd \
  --output-dir outputs/attributed_generation/qm9_spectral_cycle_graphlet_v2/seed_42/evaluation
```

### Same-bond validity-first ablation

```bash
python scripts/run_attributed_grapher.py \
  --config configs/experiments/grapher/qm9_attributed_spectral_cycle_graphlet_v2_strict_same_bond.yaml \
  --checkpoint outputs/attributed_grapher/qm9_spectral_cycle_graphlet/seed_42/checkpoint.pt \
  --output-dir outputs/attributed_generation/qm9_spectral_cycle_graphlet_v2_strict/seed_42 \
  --num-generate 1024 \
  --seed 42 \
  --device gpu
```

Use the same evaluation command after replacing the generated directory.

### Interpreting the rerun

The first checks should be:

1. `rdkit_valid_final_rate_raw == 1.0` in `report.json`;
2. evaluator `validity == 1.0` and all 1024 molecules used by raw-valid FCD;
3. `end_to_end_yield` and rejection reasons remain acceptable;
4. `mean_typed_degree_drift` and `mean_weighted_valence_drift` are lower than
   the first cycle-only run;
5. the per-order ring-mass MMD identifies whether C3, C4, C5, or C6 needs a
   larger size weight.

FCD, NSPDK, and cycle MMD improvements cannot be guaranteed without rerunning
the trained model. The v2 changes are designed to remove the known validity
protocol mismatch, reduce local chemistry drift, and make the optimization
objective match the two cycle metrics that are actually reported.

