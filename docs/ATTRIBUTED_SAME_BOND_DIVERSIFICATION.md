# Attributed same-bond stochastic diversification

This extension applies the generic stochastic structural-diversification idea to the hybrid attributed GDSM after fixed-topology categorical attribute diffusion has produced a complete attributed graph.

For every rewiring move, two edges carrying the same configured edge category are selected and replaced by a valid double-edge swap. The inserted edges retain that same category. Consequently every accepted move preserves exactly:

- node attributes;
- indexed ordinary degree at every node;
- indexed per-edge-category (typed) degree at every node;
- global count of every edge/bond category;
- molecular weighted valence whenever the edge category determines bond order;
- graph simplicity; and
- connectivity for connected sources.

Optional RDKit candidate filtering is enabled by the supplied QM9 configs.

The move is not greedy repair. The topology-summary and typed-graphlet predictions from the generated endpoint are frozen and used as a structural trust region. Candidate swaps inside the trust region are sampled with a temperature-controlled softmax.

## Configuration

The supplied QM9 configs are in:

```text
configs/experiments/gdsm_laplacian_loggap_hybrid_same_bond_diversification_explicit/
  qm9_seed_42.yaml
  qm9_seed_43.yaml
  qm9_seed_44.yaml
```

The new block is:

```yaml
attributed_diversification:
  enabled: true
  mode: stochastic_same_bond_diversification
  random_steps:
    distribution: uniform_integer
    min_steps: 0
    max_steps: 4
    mean_steps: 2.0
  proposal_budget: 128
  valid_candidate_budget: 48
  preserve_connectivity: true
  strict_same_bond: true
  rdkit_candidate_filter: true
  reject_revisited_states: true
  normalization: initial
  epsilon: 1.0e-6
  guidance:
    topology_graphlet_weight: 0.50
    topology_graphlet_mass_weight: 0.05
    topology_orbit_weight: 0.25
    typed_graphlet_weight: 1.00
    typed_graphlet_mass_weight: 0.10
  soft_selection:
    temperature_start: 0.75
    temperature_end: 0.25
    temperature_schedule: linear
    score_normalization: std
    max_target_worsening: 0.02
    max_relative_target_worsening: 0.25
  target_trust_region:
    enabled: true
    absolute_worsening: 0.02
    relative_worsening: 0.25
```

`min_steps: 0` deliberately allows the original high-fidelity attributed endpoint to remain unchanged for some samples.

## Generation-only extension

No new predictor is trained. A checkpoint trained by the corresponding hybrid `g345` model can be reused as long as its topology/attribute vocabularies and summary heads match the wrapper configuration. Training with the diversification-enabled config is also valid; the diversification block does not enter the training loss.

### Example QM9 seed 42

```bash
SEED=42
N=10000
DEVICE=cuda:0
CFG=configs/experiments/gdsm_laplacian_loggap_hybrid_same_bond_diversification_explicit/qm9_seed_${SEED}.yaml
RUN=seed_${SEED}_qm9_hybrid_topology_attrdiff_g345_v1
GEN_ID=same_bond_diversified_seed_${SEED}_n_${N}

PYTHONHASHSEED="$SEED" GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset qm9 \
  --no-common-config \
  --wrapper-config "$CFG" \
  --dataset-root outputs/datasets \
  --serialized-dataset qm9_attributed \
  --output-root outputs/baselines \
  --seed-id "$SEED" \
  --generation-seed "$SEED" \
  --run-id "$RUN" \
  --generation-id "$GEN_ID" \
  --num-samples "$N" \
  --device "$DEVICE"
```

If a compatible checkpoint does not already exist under `RUN`, train it first with the same config and run id.

## Paired artifacts

When diversification is enabled generation writes:

```text
pre_refinement_graphs.pkl
base_graphs.pkl
attributed_diversification_diagnostics.json
```

- `pre_refinement_graphs.pkl`: attributed endpoint immediately after categorical attribute diffusion.
- `base_graphs.pkl`: final graph after stochastic same-bond diversification.

`topology_graphs.pkl` remains the binary topology that conditioned attribute diffusion, so the post-attribute rewiring step is auditable rather than being confused with attribute diffusion itself.

The diagnostics record requested/accepted step counts, changed rate, preservation of typed degrees/connectivity, source/final structural discrepancy, trust-region rejections, and per-step selected actions.

## Evaluation

Evaluate both populations separately using the same molecular reference split:

```bash
GEN=outputs/baselines/gdsm_simple/qm9/$RUN/generations/$GEN_ID

PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/pre_refinement_graphs.pkl" \
  --dataset-root outputs/datasets \
  --dataset qm9_attributed \
  --reference-split val \
  --train-split train \
  --output-dir "$GEN/evaluation_val_pre_rewire" \
  --strict-raw-metrics \
  --metric-molecule-source raw_valid \
  --nspdk-backend eden \
  --nspdk-complexity 4 \
  --nspdk-bond-label-mode hogdiff \
  --require-fcd \
  --fcd-device cuda:0 \
  --graphlet-mmd \
  --graphlet-k-min 3 \
  --graphlet-k-max 5 \
  --graphlet-topology-filter all \
  --graphlet-node-attribute atomic_num \
  --graphlet-edge-attribute bond_type \
  --graphlet-attributed-backend python

PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --dataset-root outputs/datasets \
  --dataset qm9_attributed \
  --reference-split val \
  --train-split train \
  --output-dir "$GEN/evaluation_val_diversified" \
  --strict-raw-metrics \
  --metric-molecule-source raw_valid \
  --nspdk-backend eden \
  --nspdk-complexity 4 \
  --nspdk-bond-label-mode hogdiff \
  --require-fcd \
  --fcd-device cuda:0 \
  --graphlet-mmd \
  --graphlet-k-min 3 \
  --graphlet-k-max 5 \
  --graphlet-topology-filter all \
  --graphlet-node-attribute atomic_num \
  --graphlet-edge-attribute bond_type \
  --graphlet-attributed-backend python
```

Use validation to choose the rewiring budget/temperature. Freeze those values before evaluating on test.
