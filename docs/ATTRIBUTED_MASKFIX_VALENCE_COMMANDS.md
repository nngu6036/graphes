# Explicit commands: corrected QM9 training and paired decoder comparison

Run from the extracted repository root using the existing GraphES environment.
Prepared `train.pkl`, `val.pkl`, and `test.pkl` splits must already exist under
`outputs/datasets/qm9_attributed`. No dataset split is regenerated here. Commands
use the existing `scripts/run_simple_gdsm_baseline.py` entry point; no launcher
script supplies hidden hyperparameters. The YAMLs specify the complete model
and training settings.

The optional production metric dependencies (EDeN and `fcd_torch` with its
ChemNet weights) must be available in the evaluation environment. The source
update's CPU tests do not establish production FCD/NSPDK benchmark performance.

## 1. Train one new seed-42 reference checkpoint

Use a NEW run ID; old checkpoints do not gain corrected masking retrospectively.
Training is identical for A/B/C, so train the baseline configuration once and
reuse the same managed checkpoint for all decoder modes.

```bash
SEED=42
CFGROOT=configs/experiments/gdsm_laplacian_loggap_attributed_valence_explicit
CFG="$CFGROOT/qm9_seed_${SEED}_baseline.yaml"
RUN="seed_${SEED}_qm9_loggap_maskfix_v2"

PYTHONPATH=src python scripts/run_simple_gdsm_baseline.py \
  --stage train \
  --dataset qm9 \
  --common-config configs/baselines/common_qm9.yaml \
  --wrapper-config "$CFG" \
  --seed-id "$SEED" \
  --run-id "$RUN" \
  --device gpu
```

All prepared training graphs are used. The existing 200-epoch setting and
final-configured-epoch checkpoint selection remain unchanged. The graphlet
vocabulary-discovery cap is not a training-subset cap.

## 2. Generate a paired 1,024-sample validation pilot

A=`baseline`, B=`atom_degree`, C=`atom_bond_valence`. The generation seed, batch
size, checkpoint/run ID, number of samples, topology sampler and thresholds are
identical. Atom temperature is 1.0 and bond temperature is 0.6 in all three
explicit configurations. Separate attribute RNG streams are enabled.

```bash
SEED=42
N=1024
BATCH=128
RUN="seed_${SEED}_qm9_loggap_maskfix_v2"
CFGROOT=configs/experiments/gdsm_laplacian_loggap_attributed_valence_explicit

for VARIANT in baseline atom_degree atom_bond_valence; do
  CFG="$CFGROOT/qm9_seed_${SEED}_${VARIANT}.yaml"
  GEN_ID="${VARIANT}_seed_${SEED}_n_${N}"

  PYTHONPATH=src python scripts/run_simple_gdsm_baseline.py \
    --stage generate \
    --dataset qm9 \
    --common-config configs/baselines/common_qm9.yaml \
    --wrapper-config "$CFG" \
    --seed-id "$SEED" \
    --generation-seed "$SEED" \
    --run-id "$RUN" \
    --generation-id "$GEN_ID" \
    --num-samples "$N" \
    --generation-batch-size "$BATCH" \
    --device gpu
done
```

These commands deliberately rerun diffusion for each variant. The hashes verify
that the full sampled topology state, not just its degree sequence, is paired.
There is no replay cache in this update. Leave existing artifacts untouched or
use a new generation ID when repeating a pilot.

## 3. Verify paired topology states and serialized graphs

```bash
SEED=42
N=1024
RUN="seed_${SEED}_qm9_loggap_maskfix_v2"
GENROOT="outputs/baselines/gdsm_simple/qm9/$RUN/generations"

PYTHONPATH=src python scripts/verify_gdsm_decoder_pairing.py \
  --generated-dirs \
    "$GENROOT/baseline_seed_${SEED}_n_${N}" \
    "$GENROOT/atom_degree_seed_${SEED}_n_${N}" \
    "$GENROOT/atom_bond_valence_seed_${SEED}_n_${N}" \
  --output "$GENROOT/decoder_pairing_seed_${SEED}_n_${N}.json"
```

Require `paired: true`; the command exits nonzero on a mismatch. It also checks
that all requested graph records, including infeasible ones, are present.

## 4. Evaluate on validation references without changing the primary population

```bash
SEED=42
N=1024
RUN="seed_${SEED}_qm9_loggap_maskfix_v2"
GENROOT="outputs/baselines/gdsm_simple/qm9/$RUN/generations"

for VARIANT in baseline atom_degree atom_bond_valence; do
  GEN="$GENROOT/${VARIANT}_seed_${SEED}_n_${N}"

  PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
    --generated-dir "$GEN" \
    --generated-graphs "$GEN/base_graphs.pkl" \
    --dataset-root outputs/datasets \
    --dataset qm9_attributed \
    --reference-split val \
    --train-split train \
    --output-dir "$GEN/evaluation_raw_val_g345_all" \
    --strict-raw-metrics \
    --metric-molecule-source raw_valid \
    --nspdk-backend eden \
    --nspdk-bond-label-mode hogdiff \
    --require-fcd \
    --fcd-device auto \
    --graphlet-mmd \
    --graphlet-k-min 3 \
    --graphlet-k-max 5 \
    --graphlet-topology-filter all \
    --graphlet-node-attribute atomic_num \
    --graphlet-edge-attribute bond_type \
    --graphlet-attributed-backend python
done
```

Read `attribute_prediction_diagnostics.json` in each generation directory and
`failure_diagnostics` in each `molecular_evaluation_metrics.json`. The latter
includes failure classes and atom/bond marginals over all draws. The evaluator
continues to report correction diagnostics, but raw-valid molecules remain the
population for primary metrics. Never combine `--strict-raw-metrics` with
`--hogdiff-compatible-metrics`, `--fcd-use-corrected`, or a corrected source.

## 5. Frozen final experiment: three training seeds and 10,000 molecules

Only after validation-based selection, freeze the decoder setting. The example
below selects mode C; change the explicit `VARIANT` value to the selected mode
before running the final evaluation. Do not choose it using test performance.

```bash
VARIANT=atom_bond_valence
N=10000
BATCH=128
CFGROOT=configs/experiments/gdsm_laplacian_loggap_attributed_valence_explicit

for SEED in 42 43 44; do
  RUN="seed_${SEED}_qm9_loggap_maskfix_v2"
  TRAIN_CFG="$CFGROOT/qm9_seed_${SEED}_baseline.yaml"
  GEN_CFG="$CFGROOT/qm9_seed_${SEED}_${VARIANT}.yaml"
  GEN_ID="${VARIANT}_seed_${SEED}_n_${N}"

  PYTHONPATH=src python scripts/run_simple_gdsm_baseline.py \
    --stage train \
    --dataset qm9 \
    --common-config configs/baselines/common_qm9.yaml \
    --wrapper-config "$TRAIN_CFG" \
    --seed-id "$SEED" \
    --run-id "$RUN" \
    --device gpu

  PYTHONPATH=src python scripts/run_simple_gdsm_baseline.py \
    --stage generate \
    --dataset qm9 \
    --common-config configs/baselines/common_qm9.yaml \
    --wrapper-config "$GEN_CFG" \
    --seed-id "$SEED" \
    --generation-seed "$SEED" \
    --run-id "$RUN" \
    --generation-id "$GEN_ID" \
    --num-samples "$N" \
    --generation-batch-size "$BATCH" \
    --device gpu

  GEN="outputs/baselines/gdsm_simple/qm9/$RUN/generations/$GEN_ID"
  PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
    --generated-dir "$GEN" \
    --generated-graphs "$GEN/base_graphs.pkl" \
    --dataset-root outputs/datasets \
    --dataset qm9_attributed \
    --reference-split test \
    --train-split train \
    --output-dir "$GEN/evaluation_raw_test_g345_all" \
    --strict-raw-metrics \
    --metric-molecule-source raw_valid \
    --nspdk-backend eden \
    --nspdk-bond-label-mode hogdiff \
    --require-fcd \
    --fcd-device auto \
    --graphlet-mmd \
    --graphlet-k-min 3 \
    --graphlet-k-max 5 \
    --graphlet-topology-filter all \
    --graphlet-node-attribute atomic_num \
    --graphlet-edge-attribute bond_type \
    --graphlet-attributed-backend python
done
```

A completed corrected run with an identical resolved configuration is reused
by the train stage; a different existing run is not silently overwritten.

## 6. Optional separately labeled corrected-molecule comparison

Set `GEN` explicitly to an existing generation directory. Do not use the strict
flag or an explicit raw-valid source in this secondary report.

```bash
: "${GEN:?Set GEN to the generation directory to evaluate}"
PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --dataset-root outputs/datasets \
  --dataset qm9_attributed \
  --reference-split val \
  --train-split train \
  --output-dir "$GEN/evaluation_corrected_val_g345_all" \
  --hogdiff-compatible-metrics \
  --metric-molecule-source corrected_valid \
  --require-fcd \
  --graphlet-mmd \
  --graphlet-k-min 3 \
  --graphlet-k-max 5 \
  --graphlet-topology-filter all \
  --graphlet-node-attribute atomic_num \
  --graphlet-edge-attribute bond_type \
  --graphlet-attributed-backend python
```

## Regression tests

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_gdsm_simple_attributed_loggap.py \
  tests/test_gdsm_attributed_masking_and_constraints.py \
  tests/test_molecular_raw_protocol_and_failures.py \
  tests/test_molecular_generated_inputs.py
```

Production FCD/NSPDK metrics require their real backends; do not relabel the
small proxy-based CPU evaluator test as a benchmark run.
