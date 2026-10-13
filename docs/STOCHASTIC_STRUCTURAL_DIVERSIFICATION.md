# Spectral diffusion with stochastic structural diversification

This repository implements a generation-only diversification stage for the
full-spectrum Laplacian log-gap GDSM topology model.

The trained spectral generator is unchanged.  For each generated graph, the
joint topology head predicts a frozen structural target containing the degree
histogram, connected induced graphlet summaries for orders 3--5, and ORCA
orbit summaries.  After the usual spectral reconstruction and thresholding,
the final graph is diversified by a random-length sequence of valid double-edge
swaps.

## Guarantees and guidance

Every accepted swap preserves the indexed ordinary degree vector exactly.  For
a connected source, connectivity is preserved.  Self-loops and duplicate edges
are excluded by the existing valid-swap proposer.

Degree is therefore a hard invariant.  Graphlet and orbit summaries are soft
guidance targets: candidates are sampled with a softmax over their discrepancy
to the frozen predicted structural target, and every accepted candidate must
remain inside a source-anchored structural trust region.  This is not a claim
that graphlet/orbit summaries are exactly invariant.

The number of accepted-swap opportunities is sampled independently per graph.
The supplied configs use an integer uniform budget from 1 through 8.  The
refiner may terminate before that budget if no valid candidate remains in the
trust region.

The rewiring RNG is separate from the spectral generation RNG, so enabling the
diversification stage does not change the base spectral sample stream.

## Supplied configs

```
configs/experiments/gdsm_laplacian_loggap_structure_diversification_explicit/
  community_small_seed_42.yaml
  community_small_seed_43.yaml
  community_small_seed_44.yaml
  ego_small_seed_42.yaml
  ego_small_seed_43.yaml
  ego_small_seed_44.yaml
```

The main block is:

```yaml
graphlet_refinement:
  enabled: true
  mode: stochastic_diversification
  target_source: joint_structure_summary_head
  random_steps:
    enabled: true
    distribution: uniform_integer
    min_steps: 1
    max_steps: 8
  refiner:
    steps: 8
    proposal_budget: 256
    valid_candidate_budget: 128
    preserve_connectivity: true
    selection: softmax_safe
    accept_only_improving: false
    soft_selection:
      temperature_start: 0.75
      temperature_end: 0.25
      temperature_schedule: linear
      score_normalization: std
      stop_action: false
      max_target_worsening: 0.02
      max_relative_target_worsening: 0.25
    target_trust_region:
      enabled: true
      absolute_worsening: 0.02
      relative_worsening: 0.25
    graphlet_weight: 1.0
    graphlet_mass_weight: 0.10
    clustering_weight: 0.0
    orbit_weight: 0.50
    reject_revisited_states: true
```

## Community-small seed 42

```bash
SEED=42
N=1024
CFG=configs/experiments/gdsm_laplacian_loggap_structure_diversification_explicit/community_small_seed_${SEED}.yaml
RUN=seed_${SEED}_community_small_structural_diversification_g345_v1

PYTHONHASHSEED="$SEED" GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
python scripts/run_gdsm_simple_baseline.py \
  --stage train \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config "$CFG" \
  --dataset-root outputs/datasets \
  --serialized-dataset sbm \
  --output-root outputs/baselines \
  --seed-id "$SEED" \
  --run-id "$RUN" \
  --device cuda:0

PYTHONHASHSEED="$SEED" GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config "$CFG" \
  --dataset-root outputs/datasets \
  --serialized-dataset sbm \
  --output-root outputs/baselines \
  --seed-id "$SEED" \
  --generation-seed "$SEED" \
  --run-id "$RUN" \
  --generation-id seed_${SEED}_n_${N} \
  --num-samples "$N" \
  --device cuda:0

GEN=outputs/baselines/gdsm_simple/community_small/$RUN/generations/seed_${SEED}_n_${N}
ORCA_EXEC=/home/quang/orca/orca.out PYTHONPATH=src \
python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation_degree_g345.yaml \
  --reference-split val \
  --generic-mmd-protocol graphrnn \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --output-dir "$GEN/evaluation_val_diversified"
```

## Ego-small seed 42

Use the analogous config and dataset identifiers:

```bash
SEED=42
N=1024
CFG=configs/experiments/gdsm_laplacian_loggap_structure_diversification_explicit/ego_small_seed_${SEED}.yaml
RUN=seed_${SEED}_ego_small_structural_diversification_g345_v1

PYTHONHASHSEED="$SEED" GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
python scripts/run_gdsm_simple_baseline.py \
  --stage train \
  --dataset ego_small \
  --common-config configs/baselines/common_ego_small.yaml \
  --wrapper-config "$CFG" \
  --dataset-root outputs/datasets \
  --serialized-dataset ego_small \
  --output-root outputs/baselines \
  --seed-id "$SEED" \
  --run-id "$RUN" \
  --device cuda:0

PYTHONHASHSEED="$SEED" GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset ego_small \
  --common-config configs/baselines/common_ego_small.yaml \
  --wrapper-config "$CFG" \
  --dataset-root outputs/datasets \
  --serialized-dataset ego_small \
  --output-root outputs/baselines \
  --seed-id "$SEED" \
  --generation-seed "$SEED" \
  --run-id "$RUN" \
  --generation-id seed_${SEED}_n_${N} \
  --num-samples "$N" \
  --device cuda:0

GEN=outputs/baselines/gdsm_simple/ego_small/$RUN/generations/seed_${SEED}_n_${N}
ORCA_EXEC=/home/quang/orca/orca.out PYTHONPATH=src \
python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/ego_small_evaluation_degree_g345.yaml \
  --reference-split val \
  --generic-mmd-protocol graphrnn \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --output-dir "$GEN/evaluation_val_diversified"
```

Because this is generation-only relative to the base spectral model, an
existing compatible checkpoint with the same structure-summary heads can be
reused.  The new config is also safe for fresh training; the rewiring stage is
not executed during optimization.

## Artifacts and diagnostics

Generation saves both the source and diversified graphs:

- `pre_refinement_graphs.pkl`: base spectral outputs before diversification.
- `base_graphs.pkl`: final diversified outputs.
- `frozen_degree_sequences.pkl`: indexed degree vectors frozen before rewiring.
- `rewiring_diagnostics.json`: aggregate and per-graph diversification data.

The aggregate diagnostics include requested/accepted step counts, changed rate,
degree-preservation rate, trust-region preservation rate, and the mean initial
and final structural discrepancy.
