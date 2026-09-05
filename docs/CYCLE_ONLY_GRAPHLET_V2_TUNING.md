# Cycle-only graphlet guidance v2: validity and distribution tuning

## Why the first cycle-only run degraded

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

## Revised cycle objective

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

## Chemistry-drift anchor

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

## Exact raw-validity alignment

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

## Conservative search budget

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

## New diagnostics

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

## Configurations

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

## Commands

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
python scripts/train_degree_generator.py \
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

## Interpreting the rerun

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
