# Generic structural-summary Rewiring MLP

## Scope

The active generic GraphER component is a post-generation topology corrector.
It consumes a completed simple undirected graph from a declared base generator,
predicts permutation-invariant terminal summaries, and searches with valid
double-edge swaps. The model never receives terminal adjacency, edge/no-edge,
node-label, or edge-label targets.

The maintained target is

\[
\Phi(G)=\bigl(H_{3:5}(G), C(G), O_{0:14}(G)\bigr),
\]

with connected induced graphlet histograms, a clustering histogram, and the
standard mean orbit-count vector for connected graphlets with two to four
nodes.

## Training-source contract

`train_topology_grapher.py` requires a `training_sources` section. In
`completed_base_outputs` mode, every declaration identifies either a wrapper
manifest or a direct completed graph artifact. The loader:

1. verifies the artifact path, published count, and optional SHA-256;
2. normalizes simple NetworkX graphs and applies the declared disconnected
   policy;
3. partitions each generator's pool deterministically into train/validation;
4. creates exact node-count strata;
5. solves a one-to-one Hungarian assignment using normalized sorted-degree
   profile distance; and
6. records unmatched sources/targets, retention, costs, hashes, and strata.

Clustering, orbit, and graphlet targets are never used for matching. They remain
supervision targets, preventing target-summary leakage into the coupling rule.
Multiple declared bases are matched independently; their target graphs may be
reused across generator-specific couplings, which is reported explicitly.

The source graph used by a teacher trajectory is the completed output itself.
`source_randomization_steps` must be zero. Optional random relabelling is
allowed because all targets are graph-level invariants and the indexed source
degree sequence is relabelled consistently with its adjacency.

`legacy_havel_hakimi` remains an explicit compatibility mode only. It is not
the maintained post-correction protocol.

## Structural targets

For each graphlet size, exact connected induced counts are normalized within
that size. A 20-bin normalized clustering-coefficient histogram is used by the
maintained configs. Orbit coordinates 0--14 are derived exactly from the edge
count and cached connected three-/four-node graphlet counts and agree with the
Python ORCA-style evaluator.

The same terminal target is reused at every state in one teacher trajectory.
The predictor receives only the current source-derived state, its indexed
degrees, graph size, and `t/T`.

## Teacher trajectory

Given an explicitly coupled source/target pair `(G_base, G_target)`:

1. cache `Phi(G_target)`;
2. begin at the completed `G_base`;
3. enumerate valid degree- and connectivity-preserving double-edge swaps;
4. compute exact graphlet/orbit local deltas and candidate clustering;
5. rank candidates by the weighted structural discrepancy reduction;
6. follow the configured hard/soft distribution over positive actions; and
7. store `STOP` when no valid proposed action improves the frozen target.

The target adjacency is not queried. Source and target are required to have the
same node count but may have different degree sequences. Every visited graph
therefore stays in the source degree fibre; the degree-profile matching cost
only reduces avoidable incompatibility.

## Predictor and loss

The compatibility class name is `TopologyGraphletPredictor`, while checkpoints
use `topology_structural_predictor_v2`. Its graph-level heads are:

- graphlet Dirichlet heads;
- optional graphlet connected-mass Beta heads;
- a clustering Dirichlet head; and
- an orbit head predicting `log1p` mean counts.

The maintained loss combines graphlet cross-entropy/Dirichlet NLL, clustering
cross-entropy/Dirichlet NLL, and Smooth-L1 orbit regression in log-count space.
There is no pair cross-entropy, no terminal adjacency, no no-edge class, and no
soft degree-consistency loss.

## Correction

Prediction-horizon reuse is **generation-only**. Predictor training and teacher
trajectory construction continue to advance one accepted rewiring action per
teacher step; `topology_trajectory` must not contain `prediction_horizon`,
`refresh_prediction_every`, or `refresh_on_plateau`.

The maintained `*_topology_graphlet.yaml` training configs use the closed-loop
generation baseline `refresh_prediction_every: 1`. A separate
`community_small_topology_graphlet_adaptive_k.yaml` config records the adaptive
generation variant without changing how the predictor is trained.

During generation, an optional outer prediction loop can freeze one structural
prediction for up to `K_t` accepted swaps. `K_t` is configured by
`topology_refiner.prediction_horizon`. Fixed and annealed horizons are supported.
For accepted-swap progress `p in [0, 1]`, the exponential schedule uses
`K(p) = round(initial_k * (final_k / initial_k) ** p)`, clamped to at least one;
linear and cosine schedules are also available.

The inner loop ends when it reaches `K_t` or when no candidate passes both the
absolute and relative improvement thresholds. If a frozen target reaches a
plateau after at least one accepted swap and `refresh_on_plateau` is enabled,
the predictor is refreshed from the current graph. A plateau immediately after
a fresh prediction is a terminal STOP condition.

`run_topology_grapher.py` accepts repeatable arbitrary YAML overrides with
`--set KEY=VALUE` (alias `--override`). Dotted paths create/update nested
configuration values and values are parsed as YAML. For example:

```bash
PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config configs/experiments/grapher/community_small_topology_graphlet.yaml \
  --output-dir outputs/topology_generation/community_small/k4/seed_42 \
  --seed 42 \
  --device gpu \
  --set topology_refiner.prediction_horizon.mode=fixed \
  --set topology_refiner.prediction_horizon.k=4 \
  --set topology_refiner.prediction_horizon.refresh_on_plateau=true
```

The runner uses independent source-generation and refinement RNG streams, plus
a per-graph refiner RNG stream. Therefore changing only refiner options such as
`K` no longer shifts the later DH-VAE/HH source samples for the same run seed.
This makes K sweeps properly paired at the source-graph level.

Every accepted action preserves graph size, edge count, every indexed node
degree, simplicity, undirectedness, and connectivity. Within one frozen-target
block every accepted action monotonically reduces the same structural energy;
that guarantee resets when the predictor is refreshed.

The generation report records the effective config overrides, prediction
horizons, predictor-call count, accepted swaps per prediction, plateau
refreshes, and absolute/relative energy gains.

## Maintained commands

DH-VAE+HH source pool:

```bash
PYTHONPATH=src python scripts/run_dhvae_hh_baseline.py \
  --dataset community_small \
  --num-samples 1024 \
  --training-estimate-count 1024 \
  --seed-id 42
```

DH-VAE+HH-specific Rewiring MLP:

```bash
PYTHONPATH=src python scripts/train_topology_grapher.py \
  --config configs/experiments/grapher/community_small_topology_graphlet.yaml \
  --output-dir outputs/topology_grapher/community_small/seed_42 \
  --seed 42 \
  --device auto
```

DeFoG-specific Rewiring MLP:

```bash
PYTHONPATH=src python scripts/run_defog_baseline.py \
  --dataset community_small \
  --num-samples 1024 \
  --seed-id 42

PYTHONPATH=src python scripts/train_topology_grapher.py \
  --config configs/experiments/grapher/community_small_defog_rewiring_mlp.yaml \
  --output-dir outputs/topology_grapher/community_small_defog/seed_42 \
  --seed 42 \
  --device auto
```

## Deliberate incompatibilities

- v1 graphlet-only and endpoint checkpoints do not load into v2.
- Completed-source mode refuses nonzero source-randomization steps.
- A missing declared manifest is an error; the trainer does not silently create
  Havel--Hakimi sources.
- Only exact node-count coupling is implemented because rewiring cannot change
  graph size.
- Exact orbit supervision requires the graphlet basis to include sizes 3 and 4.
