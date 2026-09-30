# Explicit commands: fixed training basis + degree-preserving spectral decoding

Run from the project root after applying the patch. Prepared dataset splits and
the configured matching DH-VAE checkpoint must already exist. No command below
overrides training hyperparameters inside a script.

## 1. Reuse the completed Community-small seed-42 checkpoint

This uses the completed `seed_42_spectral_topology_bond_only` training run and
writes a **new generation ID**, leaving the previous 1,024 outputs untouched.
No denoiser retraining is required to run this sampler ablation.

```bash
CFG=configs/experiments/gdsm_training_basis_degree_explicit/community_small_seed_42.yaml
RUN=seed_42_spectral_topology_bond_only
N=1024
GEN_ID=seed_42_n_${N}_training_basis

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config "$CFG" \
  --seed-id 42 \
  --generation-seed 42 \
  --run-id "$RUN" \
  --generation-id "$GEN_ID" \
  --num-samples "$N" \
  --device gpu

GEN="outputs/baselines/gdsm_simple/community_small/$RUN/generations/$GEN_ID"

PYTHONPATH=src python scripts/audit_gdsm_categorical.py \
  --generated-dir "$GEN"

export ORCA_EXEC=/home/quang/orca/orca.out
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --generated-stage gdsm_training_basis \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --output-dir "$GEN/evaluation"
```

The graph evaluator consumes all saved graphs unless `--max-graphs` is supplied.
Its legacy `--num-samples` argument does **not** set the MMD sample count.

## 2. Matched current-basis control

This control uses the same checkpoint, sample count, prior, feedback setting and
projection budgets. Its YAML differs only in `topology.basis_source`. Feedback is
zero in both arms, unlike the previous default 0.05 run. Compare this control to
section 1 to isolate the basis-source configuration change.

```bash
CFG=configs/experiments/gdsm_training_basis_degree_explicit/community_small_seed_42_current_basis_control.yaml
RUN=seed_42_spectral_topology_bond_only
N=1024
GEN_ID=seed_42_n_${N}_current_basis_no_feedback

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config "$CFG" \
  --seed-id 42 \
  --generation-seed 42 \
  --run-id "$RUN" \
  --generation-id "$GEN_ID" \
  --num-samples "$N" \
  --device gpu

GEN="outputs/baselines/gdsm_simple/community_small/$RUN/generations/$GEN_ID"

PYTHONPATH=src python scripts/audit_gdsm_categorical.py \
  --generated-dir "$GEN"

export ORCA_EXEC=/home/quang/orca/orca.out
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --generated-stage gdsm_current_basis_no_feedback \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --output-dir "$GEN/evaluation"
```

## 3. Fresh training runs, when needed

These commands train the existing joint denoiser objective and configure the
new generation rule. They do **not** implement a new fixed-basis training
corruption. A fresh run is unnecessary for an already compatible v2 checkpoint.
Do not use an old edge/no-edge v1 checkpoint for bond-only generation.

### Community-small, seed 42

```bash
CFG=configs/experiments/gdsm_training_basis_degree_explicit/community_small_seed_42.yaml
RUN=seed_42_training_basis_degree
N=1024

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage train \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config "$CFG" \
  --seed-id 42 \
  --run-id "$RUN" \
  --device gpu

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config "$CFG" \
  --seed-id 42 \
  --generation-seed 42 \
  --run-id "$RUN" \
  --num-samples "$N" \
  --device gpu

GEN="outputs/baselines/gdsm_simple/community_small/$RUN/generations/seed_42_n_${N}"

PYTHONPATH=src python scripts/audit_gdsm_categorical.py \
  --generated-dir "$GEN"

export ORCA_EXEC=/home/quang/orca/orca.out
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --generated-stage gdsm_training_basis \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --output-dir "$GEN/evaluation"
```

### Ego-small, seed 42

```bash
CFG=configs/experiments/gdsm_training_basis_degree_explicit/ego_small_seed_42.yaml
RUN=seed_42_training_basis_degree
N=1024

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage train \
  --dataset ego_small \
  --common-config configs/baselines/common_ego_small.yaml \
  --wrapper-config "$CFG" \
  --seed-id 42 \
  --run-id "$RUN" \
  --device gpu

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset ego_small \
  --common-config configs/baselines/common_ego_small.yaml \
  --wrapper-config "$CFG" \
  --seed-id 42 \
  --generation-seed 42 \
  --run-id "$RUN" \
  --num-samples "$N" \
  --device gpu

GEN="outputs/baselines/gdsm_simple/ego_small/$RUN/generations/seed_42_n_${N}"

PYTHONPATH=src python scripts/audit_gdsm_categorical.py \
  --generated-dir "$GEN"

export ORCA_EXEC=/home/quang/orca/orca.out
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/ego_small_evaluation.yaml \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --generated-stage gdsm_training_basis \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --output-dir "$GEN/evaluation"
```

### QM9, seed 42

```bash
CFG=configs/experiments/gdsm_training_basis_degree_explicit/qm9_seed_42.yaml
RUN=seed_42_training_basis_degree
N=10000

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage train \
  --dataset qm9 \
  --common-config configs/baselines/common_qm9.yaml \
  --wrapper-config "$CFG" \
  --seed-id 42 \
  --run-id "$RUN" \
  --device gpu

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset qm9 \
  --common-config configs/baselines/common_qm9.yaml \
  --wrapper-config "$CFG" \
  --seed-id 42 \
  --generation-seed 42 \
  --run-id "$RUN" \
  --num-samples "$N" \
  --device gpu

GEN="outputs/baselines/gdsm_simple/qm9/$RUN/generations/seed_42_n_${N}"

PYTHONPATH=src python scripts/audit_gdsm_categorical.py \
  --generated-dir "$GEN"

PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-graphs "$GEN/molecular_graphs.pkl" \
  --dataset qm9_attributed \
  --reference-split test \
  --train-split train \
  --output-dir "$GEN/evaluation_raw" \
  --require-fcd \
  --metric-molecule-source raw_valid
```

### ZINC, seed 42

```bash
CFG=configs/experiments/gdsm_training_basis_degree_explicit/zinc_seed_42.yaml
RUN=seed_42_training_basis_degree
N=10000

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage train \
  --dataset zinc \
  --common-config configs/baselines/common_zinc.yaml \
  --wrapper-config "$CFG" \
  --seed-id 42 \
  --run-id "$RUN" \
  --device gpu

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset zinc \
  --common-config configs/baselines/common_zinc.yaml \
  --wrapper-config "$CFG" \
  --seed-id 42 \
  --generation-seed 42 \
  --run-id "$RUN" \
  --num-samples "$N" \
  --device gpu

GEN="outputs/baselines/gdsm_simple/zinc/$RUN/generations/seed_42_n_${N}"

PYTHONPATH=src python scripts/audit_gdsm_categorical.py \
  --generated-dir "$GEN"

PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-graphs "$GEN/molecular_graphs.pkl" \
  --dataset zinc \
  --reference-split test \
  --train-split train \
  --output-dir "$GEN/evaluation_raw" \
  --require-fcd \
  --metric-molecule-source raw_valid
```

For molecules, the ordinary-degree guarantee does not establish chemical
validity. These evaluation commands keep raw validity and use the raw-valid
subset for distributional metrics; use the same metric-source convention in
any comparison. FCD and molecular evaluator dependencies must be available in
the environment used for evaluation.

## 4. Other completed runs and seeds

To reuse an existing compatible run for Ego-small, QM9 or ZINC, use its actual
run ID with `--stage generate`, the matching new dataset/seed YAML, and a new
`--generation-id`. Keep the original training manifest beside its checkpoint.
The generation CLI validates the checkpoint against that manifest.

For seeds 43 and 44, select the corresponding `*_seed_43.yaml` or
`*_seed_44.yaml`, change both CLI seeds, use a matching run ID, and use the
matching generation-directory suffix. Each YAML retains its own seed-specific
DH-VAE checkpoint path. Changing only the CLI seed is not sufficient.

The configured degree priors are unchanged. A missing prior is an explicit
error; a matching trained prior can be reused. Prior training profiles are
`configs/experiments/dhvae_final_explicit/<dataset>_seed_<seed>.yaml`, used with
`scripts/train_degree_generator.py --config ...` as in the existing workflow.

## 5. Local correctness tests

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  pytest -q tests/test_gdsm_training_basis.py

PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  pytest -q tests/test_gdsm*.py -rs
```

Full-suite results, known pre-existing failures and CPU-only scope are recorded
in `GDSM_TRAINING_BASIS_TEST_REPORT.json`. No benchmark metrics are included in
the patch because full benchmark generation/evaluation was not run here.
