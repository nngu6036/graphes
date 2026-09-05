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
python scripts/train_degree_generator.py \
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

## Post-run validity/fidelity tuning

For the revised raw-validity-aligned objective, explicit selected-ring-mass
loss, chemistry-drift anchors, and conservative QM9 configurations, see
`docs/CYCLE_ONLY_GRAPHLET_V2_TUNING.md`.
