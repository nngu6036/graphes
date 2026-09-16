# GraphER

GraphER is an experimental graph-generation framework built around
**constraint-preserving graph rewiring**. It supports generic graphs and
attributed molecular graphs, multiple structural guidance variants, degree or
typed-degree priors, and a common wrapper/evaluation layer for external
baselines.

The repository is organized for controlled experiments: prepared dataset splits
are frozen, generated artifacts are run-scoped, and baseline outputs are routed
through the same evaluation code whenever the graph representation permits it.

## Core idea

For generic graphs, GraphER samples or obtains a degree sequence, constructs a
simple connected realization, and refines that realization with valid rewiring
operations. Accepted operations preserve the selected invariant while learned
predictors guide higher-order structure such as adjacency targets, spectral
summaries and graphlets.

For molecular graphs, the invariant can include atom categories and bond-type
specific degrees. Generation uses typed initialization and constrained
bond/edge operations so atom/bond structure is not treated as an unconstrained
post-processing problem.

The exact experimental variant is defined by YAML rather than by separate code
forks.

## Controlled baseline comparison

`configs/baselines/common_<dataset>.yaml` is the **shared protocol/reference
layer**. DeFoG supplies the reference settings, but its raw training horizon is
**not a universal epoch budget**. One million DeFoG epochs does not mean one
million HOG-Diff iterations, GraphRNN epochs, GDSS epochs, or SPECTRE epochs.

The merge order for every maintained external-baseline runner is:

```text
upstream defaults
  -> common_<dataset>.yaml reference fallbacks
  -> <model>_<dataset>.yaml model-native/equivalent settings
  -> explicit CLI overrides
```

Therefore the common files define the comparison context and provenance, while
each model YAML is authoritative for quantities whose numerical meaning depends
on the implementation. In particular, every maintained model config now records
a `comparison:` section with the source and native training-budget unit.

The shared controls that should remain fixed across methods are the prepared
train/validation/test splits, seed policy, model selection on validation only,
requested sample count for a report, graph representation, and evaluator.
Optimizer schedules and training horizons follow the released model/dataset
profile when one exists. When the released code has no matching dataset profile,
the config is explicitly marked as an adaptation and its budget must be selected
on validation data rather than test MMD.

The DeFoG-reference profiles are:

```text
configs/baselines/common_community_small.yaml
configs/baselines/common_ego_small.yaml
configs/baselines/common_qm9.yaml
configs/baselines/common_zinc.yaml
```

`common_ego_small.yaml` remains a Comm20-derived reference because the supplied
DeFoG source does not contain the GraphER/GDSS Ego-small experiment.

Representative Community-small budgets after model-specific overrides are:

| Model | Effective training budget | Source status |
|---|---:|---|
| DeFoG | 1,000,000 epochs, batch 256 | released `comm20` profile |
| DiGress | 1,000,000 epochs, batch 256 | released `comm20` profile |
| GDSS | 5,000 epochs, batch 128 | released `community_small` profile |
| GSDM | 200 epochs, batch 128 | released `community_small` profile |
| EDGE | 50,000 epochs, batch 8 | released generic training budget; small-graph diffusion adapter |
| GraphRNN | 3,000 epochs × 32 mini-batches/epoch | released default schedule |
| HOG-Diff | 6,000 higher-order + 22,000 OU iterations | released `cs.yaml` two-stage schedule |
| SPECTRE | 12,000 epochs, batch 10 | released Community command |
| CatFlow | 10,000 epochs, batch 128 | GraphER linear-v2 generic adapter budget; no released generic CatFlow profile, validate before reporting |

These counts are deliberately **not normalized to the same integer**. They are
different optimization units and model costs. For publication, report runtime
and/or optimizer-update counts alongside quality when compute fairness matters.

See [`docs/BASELINES.md`](docs/BASELINES.md) for the full dataset-by-model budget
matrix and commands.

## Repository layout

```text
configs/
  datasets/                 Prepared-dataset protocols
  baselines/                Baseline-specific and common comparison YAMLs
  experiments/
    baselines/              Shared evaluation configs
    dhvae/                  Degree/typed-degree prior configs
    grapher/                GraphER experiment variants

docs/
  BASELINES.md              Canonical baseline/comparison guide
  DESIGN_CONTRACT.md        GraphER representation and invariant contract
  TOPOLOGY_GENERATOR.md     Generic topology generation details
  ATTRIBUTED_*.md           Attributed/molecular guidance details
  SPECTRAL_*.md             Spectral guidance and diagnostics
  degree_perturbation_*.md  Prior-perturbation experiment notes

scripts/
  prepare_*_dataset.py      Dataset preparation
  train_*_grapher.py        GraphER training
  run_*_grapher.py          GraphER generation
  run_*_baseline.py         Baseline orchestration
  evaluate_*.py             Shared evaluation
  diagnose_*.py             Focused diagnostics
  draw_*.py                 Visualization utilities

src/grapher/
  data/                     Dataset IO and sampling
  models/                   Baseline wrappers and external workers
  rewiring_mlp/             GraphER generic/attributed models and refiners
  properties/               Structural summaries
  utils/                    Shared utilities
```

## Environment

Run commands from the repository root and expose `src` on `PYTHONPATH`:

```bash
export PYTHONPATH=src
```

GraphER itself and each external baseline may require different PyTorch/CUDA or
third-party environments. Baseline wrappers are designed to launch the upstream
model with its own Python interpreter; see `docs/BASELINES.md` for environment
variables and supported datasets.

## Prepare datasets

Prepare each benchmark once, then reuse the exact split files for every model.
Do not rebuild them between baseline runs.

### Generic graphs

```bash
PYTHONPATH=src python scripts/prepare_generic_dataset.py \
  --dataset community_small \
  --root outputs/datasets

PYTHONPATH=src python scripts/prepare_generic_dataset.py \
  --dataset ego_small \
  --root outputs/datasets
```

The generic preparation script also supports the configured Grid benchmark.

### QM9

```bash
PYTHONPATH=src python scripts/prepare_qm9_dataset.py \
  --root outputs/datasets
```

The canonical molecular protocol uses the configured heavy-atom representation
and records preparation metadata with the split artifacts.

### ZINC

```bash
PYTHONPATH=src python scripts/prepare_zinc_dataset.py \
  --smiles-file /path/to/zinc250k.csv \
  --test-indices-file /path/to/valid_idx_zinc250k.json \
  --root outputs/datasets
```

Use the dataset configuration and preparation protocol that matches the intended
paper comparison.

## GraphER: train, generate and evaluate by dataset

The maintained GraphER configuration couples two learned diffusion states:

- a **soft edge state** (binary for generic graphs; categorical bond type for
  molecular graphs), and
- the **eigenvalues of the combinatorial Laplacian** (topology and, for
  molecules, bond-weighted channels).

The structural heads predict a 100-bin clustering-coefficient histogram, the
15-coordinate orbit summary, and induced graphlet distributions for
`k = 3, 4, 5`.  Molecular graphlet identities additionally include
`atomic_num` and `bond_type`.

### Structured spectral-space diffusion: eigenvalues + eigenspace projector

The maintained eigenspace experiment now predicts one structured spectral state
rather than several independent heat-kernel matrices.  For each graph, GraphER
uses

```text
Lambda(G) = ordered combinatorial-Laplacian eigenvalues
P_k(G)    = U_k U_k^T
```

where `U_k` contains the first `k` non-trivial Laplacian eigenvectors.  The
projector removes eigenvector sign ambiguity and rotations inside the selected
subspace.  The default Community-small setting uses `k = 4`.

Training constructs a connected degree-matched HH source, aligns equal-degree
nodes with a deterministic structural Hungarian assignment, and samples
endpoint-conditioned Brownian bridges for both `Lambda` and `P_k`.  The
Spectral Transformer predicts the clean eigenvalues, while a pair head predicts
the clean projector.  The projector head is projected to the rank-k manifold in
the subspace orthogonal to the constant Laplacian mode, so the prediction is
symmetric, PSD, idempotent, has trace `k`, and satisfies `P_k 1 = 0`.

Generation starts from `(Lambda_HH, P_HH)`, reverses both continuous bridges,
and realizes the predicted joint spectral target with degree-preserving,
connectivity-preserving rewiring.  Candidate spectral discrepancy is the
weighted combination of normalized eigenvalue RMSE and projector chordal
distance.

The full Community-small experiment is:

```text
configs/experiments/grapher/community_small_joint_edge_lambda_projector_graphlets345_learned.yaml
```

Train, generate and evaluate it with:

```bash
CFG=configs/experiments/grapher/community_small_joint_edge_lambda_projector_graphlets345_learned.yaml
TRAIN=outputs/topology_grapher/community_small_joint_edge_lambda_projector_graphlets345_learned/seed_42
GEN=outputs/topology_generation/community_small_joint_edge_lambda_projector_graphlets345_learned/seed_42/learned

PYTHONPATH=src python scripts/train_topology_grapher.py \
  --config "$CFG" \
  --output-dir "$TRAIN" \
  --seed 42 \
  --device gpu

PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config "$CFG" \
  --checkpoint "$TRAIN/checkpoint.pt" \
  --output-dir "$GEN" \
  --num-generate 1024 \
  --seed 42 \
  --device gpu

PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config "$CFG" \
  --generated-dir "$GEN" \
  --reference-split test \
  --output-dir "$GEN/evaluation_test"
```

For the **true structured spectral-only training ablation** (no edge-diffusion
loss and no clustering/orbit/graphlet heads), use:

```text
configs/experiments/grapher/community_small_lambda_projector_only_learned.yaml
```

The older heat-kernel configs remain available only for backward-compatible
checkpoint reproduction; they are no longer the recommended eigenspace
parameterization.  Existing eigenvalue-only configs are also unchanged for
direct ablation.  See
[`docs/LAMBDA_PROJECTOR_SPECTRAL_DIFFUSION.md`](docs/LAMBDA_PROJECTOR_SPECTRAL_DIFFUSION.md).

The degree/typed-degree encoder and decoder are trained jointly with GraphER.
At generation time the same GraphER checkpoint can be used with either of two
sequence sources:

```text
learned
    sample a degree / typed-degree sequence from the embedded jointly trained
    DH-VAE decoder

edge_relocation
    sample a training sequence -> construct a temporary connected realization
    -> relocate one edge endpoint while preserving simplicity/connectivity
    -> extract the changed degree / typed-degree sequence
    -> DISCARD the temporary adjacency
    -> construct a fresh GraphER source from the changed sequence
```

`edge_relocation` is therefore a **generation-time prior option**.  It does not
require retraining GraphER.  In the commands below each dataset is trained once,
then the exact same checkpoint is evaluated with both the learned and perturbed
sequence sources.

The standalone DH-VAE/typed-DH-VAE checkpoint is only a warm start for joint
training.  Train it once if the checkpoint referenced by the GraphER config does
not already exist.

### Community-small

One-time degree-prior warm start:

```bash
PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/dhvae/community_small.yaml
```

Train GraphER:

```bash
CFG=configs/experiments/grapher/community_small_joint_edge_laplacian_graphlets345_learned.yaml
TRAIN=outputs/topology_grapher/community_small_joint_edge_laplacian_graphlets345_learned/seed_42
GENROOT=outputs/topology_generation/community_small_joint_edge_laplacian_graphlets345/seed_42

PYTHONPATH=src python scripts/train_topology_grapher.py \
  --config "$CFG" \
  --output-dir "$TRAIN" \
  --seed 42 \
  --device gpu
```

Generate with the learned degree sequence:

```bash
GEN="$GENROOT/learned"
PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config "$CFG" \
  --checkpoint "$TRAIN/checkpoint.pt" \
  --output-dir "$GEN" \
  --num-generate 1024 \
  --seed 42 \
  --device gpu
```

Generate with the edge-relocated degree sequence:

```bash
GEN="$GENROOT/edge_relocation"
PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config "$CFG" \
  --checkpoint "$TRAIN/checkpoint.pt" \
  --output-dir "$GEN" \
  --num-generate 1024 \
  --seed 42 \
  --device gpu \
  --set generation.degree_source=edge_relocation \
  --set generation.degree_rng_mode=independent \
  --set generation.degree_perturbation.max_attempts=256 \
  --set generation.degree_perturbation.max_distance=4.0 \
  --set generation.degree_perturbation.require_novel=false
```

Evaluate both sequence sources on validation data:

```bash
for MODE in learned edge_relocation; do
  GEN="$GENROOT/$MODE"

  PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
    --config "$CFG" \
    --generated-dir "$GEN" \
    --reference-split val \
    --output-dir "$GEN/evaluation_val"

  PYTHONPATH=src python scripts/evaluate_induced_graphlets.py \
    --config "$CFG" \
    --generated-graphs "$GEN/topology_refined_graphs.pkl" \
    --reference-split val \
    --json-out "$GEN/evaluation_val/induced_graphlets345.json"
done
```

### Ego-small

One-time degree-prior warm start:

```bash
PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/dhvae/ego_small.yaml
```

Train GraphER:

```bash
CFG=configs/experiments/grapher/ego_small_joint_edge_laplacian_graphlets345_learned.yaml
TRAIN=outputs/topology_grapher/ego_small_joint_edge_laplacian_graphlets345_learned/seed_42
GENROOT=outputs/topology_generation/ego_small_joint_edge_laplacian_graphlets345/seed_42

PYTHONPATH=src python scripts/train_topology_grapher.py \
  --config "$CFG" \
  --output-dir "$TRAIN" \
  --seed 42 \
  --device gpu
```

Generate and evaluate the learned sequence:

```bash
GEN="$GENROOT/learned"
PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config "$CFG" \
  --checkpoint "$TRAIN/checkpoint.pt" \
  --output-dir "$GEN" \
  --num-generate 1024 \
  --seed 42 \
  --device gpu

PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config "$CFG" \
  --generated-dir "$GEN" \
  --reference-split val \
  --output-dir "$GEN/evaluation_val"

PYTHONPATH=src python scripts/evaluate_induced_graphlets.py \
  --config "$CFG" \
  --generated-graphs "$GEN/topology_refined_graphs.pkl" \
  --reference-split val \
  --json-out "$GEN/evaluation_val/induced_graphlets345.json"
```

Generate and evaluate the perturbed sequence:

```bash
GEN="$GENROOT/edge_relocation"
PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config "$CFG" \
  --checkpoint "$TRAIN/checkpoint.pt" \
  --output-dir "$GEN" \
  --num-generate 1024 \
  --seed 42 \
  --device gpu \
  --set generation.degree_source=edge_relocation \
  --set generation.degree_rng_mode=independent \
  --set generation.degree_perturbation.max_attempts=256 \
  --set generation.degree_perturbation.max_distance=4.0 \
  --set generation.degree_perturbation.require_novel=false

PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config "$CFG" \
  --generated-dir "$GEN" \
  --reference-split val \
  --output-dir "$GEN/evaluation_val"

PYTHONPATH=src python scripts/evaluate_induced_graphlets.py \
  --config "$CFG" \
  --generated-graphs "$GEN/topology_refined_graphs.pkl" \
  --reference-split val \
  --json-out "$GEN/evaluation_val/induced_graphlets345.json"
```

### Grid

One-time degree-prior warm start:

```bash
PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/dhvae/grid.yaml
```

The Grid graphs are much larger than Community-small/Ego-small.  The exact
`k=3,4,5` induced-graphlet target is therefore substantially more expensive;
start with a small training/generation smoke run before launching the full
configuration.

Train GraphER:

```bash
CFG=configs/experiments/grapher/grid_joint_edge_laplacian_graphlets345_learned.yaml
TRAIN=outputs/topology_grapher/grid_joint_edge_laplacian_graphlets345_learned/seed_42
GENROOT=outputs/topology_generation/grid_joint_edge_laplacian_graphlets345/seed_42

PYTHONPATH=src python scripts/train_topology_grapher.py \
  --config "$CFG" \
  --output-dir "$TRAIN" \
  --seed 42 \
  --device gpu
```

Generate both sequence sources:

```bash
PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config "$CFG" \
  --checkpoint "$TRAIN/checkpoint.pt" \
  --output-dir "$GENROOT/learned" \
  --num-generate 1024 \
  --seed 42 \
  --device gpu

PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config "$CFG" \
  --checkpoint "$TRAIN/checkpoint.pt" \
  --output-dir "$GENROOT/edge_relocation" \
  --num-generate 1024 \
  --seed 42 \
  --device gpu \
  --set generation.degree_source=edge_relocation \
  --set generation.degree_rng_mode=independent \
  --set generation.degree_perturbation.max_attempts=256 \
  --set generation.degree_perturbation.max_distance=4.0 \
  --set generation.degree_perturbation.require_novel=false
```

Evaluate both:

```bash
for MODE in learned edge_relocation; do
  GEN="$GENROOT/$MODE"

  PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
    --config "$CFG" \
    --generated-dir "$GEN" \
    --reference-split val \
    --output-dir "$GEN/evaluation_val"

  PYTHONPATH=src python scripts/evaluate_induced_graphlets.py \
    --config "$CFG" \
    --generated-graphs "$GEN/topology_refined_graphs.pkl" \
    --reference-split val \
    --json-out "$GEN/evaluation_val/induced_graphlets345.json"
done
```

### QM9

The maintained QM9 GraphER experiment uses the full prepared training split by
default (`dataset.max_train_graphs: null`).  The attributed graphlet vocabulary
is fitted on a deterministic training-only subset controlled by
`graphlet_prediction.max_basis_graphs` (20,000 in the default config); this
limits vocabulary-discovery cost only and does **not** reduce the predictor's
training set.  Unseen labeled graphlet classes map to the existing overflow bin.
Graphlet identities include topology, `atomic_num`, and `bond_type` for
`k=3,4,5`.

One-time typed-degree warm start:

```bash
PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/dhvae/qm9_typed.yaml
```

Train GraphER:

```bash
CFG=configs/experiments/grapher/qm9_joint_typed_edge_laplacian_graphlets345_learned.yaml
TRAIN=outputs/attributed_grapher/qm9_joint_typed_edge_laplacian_graphlets345_learned/seed_42
GENROOT=outputs/attributed_generation/qm9_joint_typed_edge_laplacian_graphlets345/seed_42
NGEN=1024

PYTHONPATH=src python scripts/train_attributed_grapher.py \
  --config "$CFG" \
  --output-dir "$TRAIN" \
  --seed 42 \
  --device gpu \
  --graphlet-progress-interval 100 \
  --batch-progress-interval 10 \
  --progress-interval-seconds 10
```

Generate with the learned typed sequence:

```bash
GEN="$GENROOT/learned"
PYTHONPATH=src python scripts/run_attributed_grapher.py \
  --config "$CFG" \
  --checkpoint "$TRAIN/checkpoint.pt" \
  --output-dir "$GEN" \
  --num-generate "$NGEN" \
  --seed 42 \
  --device gpu
```

Generate with the edge-relocated typed sequence:

```bash
GEN="$GENROOT/edge_relocation"
PYTHONPATH=src python scripts/run_attributed_grapher.py \
  --config "$CFG" \
  --checkpoint "$TRAIN/checkpoint.pt" \
  --output-dir "$GEN" \
  --num-generate "$NGEN" \
  --seed 42 \
  --device gpu \
  --set generation.invariant_source=edge_relocation \
  --set generation.invariant_rng_mode=independent \
  --set generation.degree_perturbation.max_attempts=256 \
  --set generation.degree_perturbation.max_distance=4.0 \
  --set generation.degree_perturbation.require_novel=false
```

Evaluate both sequence sources using the strict raw-valid molecular protocol and
the checkpoint-fitted attributed graphlet vocabulary:

```bash
for MODE in learned edge_relocation; do
  GEN="$GENROOT/$MODE"

  PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
    --generated-graphs "$GEN/molecular_graphs.pkl" \
    --dataset qm9_attributed \
    --reference-split val \
    --train-split train \
    --metric-molecule-source raw_valid \
    --nspdk-backend eden \
    --nspdk-complexity 4 \
    --nspdk-bond-label-mode hogdiff \
    --output-dir "$GEN/evaluation_val" \
    --require-fcd

  PYTHONPATH=src python scripts/evaluate_induced_graphlets.py \
    --config "$CFG" \
    --checkpoint "$TRAIN/checkpoint.pt" \
    --generated-graphs "$GEN/molecular_graphs.pkl" \
    --reference-split val \
    --max-reference-graphs 1024 \
    --max-generated-graphs 1024 \
    --seed 42 \
    --json-out "$GEN/evaluation_val/attributed_graphlets345.json"
done
```

`--hogdiff-compatible-metrics` can be run as a **separate** report when a
HOG-Diff-compatible comparison is required; it changes the molecular metric
source to corrected-valid molecules and should not be mixed with the strict
`raw_valid` report above.

### ZINC250k

One-time typed-degree warm start:

```bash
PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/dhvae/zinc_typed.yaml
```

The command below uses the full prepared ZINC training split when `NTRAIN=0`.
Set `NTRAIN=20000` for a smaller development experiment.

```bash
CFG=configs/experiments/grapher/zinc_joint_typed_edge_laplacian_graphlets345_learned.yaml
TRAIN=outputs/attributed_grapher/zinc_joint_typed_edge_laplacian_graphlets345_learned/seed_42
GENROOT=outputs/attributed_generation/zinc_joint_typed_edge_laplacian_graphlets345/seed_42
NTRAIN=100000
NGEN=1024

PYTHONPATH=src python scripts/train_attributed_grapher.py \
  --config "$CFG" \
  --output-dir "$TRAIN" \
  --num-train-graphs "$NTRAIN" \
  --seed 42 \
  --device gpu \
  --graphlet-progress-interval 100 \
  --batch-progress-interval 10 \
  --progress-interval-seconds 10 \
  --set joint_typed_degree.initialize_degree_checkpoint=null \--set joint_typed_degree.freeze_epochs=0
```

Generate learned and edge-relocated typed sequences:

```bash
PYTHONPATH=src python scripts/run_attributed_grapher.py \
  --config "$CFG" \
  --checkpoint "$TRAIN/checkpoint.pt" \
  --output-dir "$GENROOT/learned" \
  --num-generate "$NGEN" \
  --seed 42 \
  --device gpu

PYTHONPATH=src python scripts/run_attributed_grapher.py \
  --config "$CFG" \
  --checkpoint "$TRAIN/checkpoint.pt" \
  --output-dir "$GENROOT/edge_relocation" \
  --num-generate "$NGEN" \
  --seed 42 \
  --device gpu \
  --set generation.invariant_source=edge_relocation \
  --set generation.invariant_rng_mode=independent \
  --set generation.degree_perturbation.max_attempts=256 \
  --set generation.degree_perturbation.max_distance=4.0 \
  --set generation.degree_perturbation.require_novel=false
```

Evaluate both sequence sources:

```bash
for MODE in learned edge_relocation; do
  GEN="$GENROOT/$MODE"

  PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
    --generated-graphs "$GEN/molecular_graphs.pkl" \
    --dataset zinc \
    --reference-split val \
    --train-split train \
    --metric-molecule-source raw_valid \
    --nspdk-backend eden \
    --nspdk-complexity 4 \
    --nspdk-bond-label-mode hogdiff \
    --output-dir "$GEN/evaluation_val" \
    --require-fcd

  PYTHONPATH=src python scripts/evaluate_induced_graphlets.py \
    --config "$CFG" \
    --checkpoint "$TRAIN/checkpoint.pt" \
    --generated-graphs "$GEN/molecular_graphs.pkl" \
    --reference-split val \
    --max-reference-graphs 1024 \
    --max-generated-graphs 1024 \
    --seed 42 \
    --json-out "$GEN/evaluation_val/attributed_graphlets345.json"
done
```

### Final test reporting

Use `val` while selecting checkpoints, refiner weights and prior options.  After
all choices are frozen, rerun generation with the selected configuration and
change only the evaluator reference from `val` to `test`.  Do not tune against
test FCD/MMD.

## Baselines: training, generation and evaluation

All maintained baseline runners automatically resolve the applicable
`common_<dataset>.yaml` comparison profile and the model-specific
`<model>_<dataset>.yaml` profile. The model-specific profile overrides
model-dependent training quantities such as epochs/iterations, optimizer
settings, batch size and sampler settings. **Do not pass an epoch override just
to match DeFoG's raw epoch count.** Use explicit CLI overrides only for a
separately named experiment.

All ``scripts/run_<model>_baseline.py`` launchers are uniform thin shims over
``grapher.models.external_cli``.  Every maintained baseline therefore supports
``--stage train``, ``--stage generate`` and ``--stage all``; ``all`` is the
default and performs training followed by generation.  Model-specific legacy
flags remain accepted, but their translation is centralized in the shared CLI.

The examples below use Community-small, seed 42 and 1,024 generated graphs.
For all commands, run from the GraphER repository root with:

```bash
export PYTHONPATH=src
```

External implementations may use separate environments. Set the corresponding
source root and Python executable before launching a baseline:

```bash
export DEFOG=/home/quang/DeFoG
export DEFOG_PYTHON=/home/quang/miniconda3/envs/defog/bin/python

export DIGRESS=/home/quang/DiGress
export DIGRESS_PYTHON=/home/quang/miniconda3/envs/digress/bin/python

export GDSS=/home/quang/GDSS
export GDSS_PYTHON=/home/quang/miniconda3/envs/gdss/bin/python

export GRAPHRNN=/home/quang/GraphRNN
export GRAPHRNN_PYTHON=/home/quang/miniconda3/envs/graphrnn/bin/python

export HOGDIFF=/home/quang/HOG-Diff
export HOGDIFF_PYTHON=/home/quang/miniconda3/envs/defog/bin/python

export CATFLOW=/home/quang/CatFlow
export CATFLOW_PYTHON=/home/quang/miniconda3/envs/defog/bin/python

export GDSM=/home/quang/gdsm
export GDSM_PYTHON=/home/quang/miniconda3/envs/gdsm/bin/python

export EDGE=/home/quang/EDGE
export EDGE_PYTHON=/home/quang/miniconda3/envs/edge/bin/python

export SPECTRE=/home/quang/SPECTRE
export SPECTRE_PYTHON=/home/quang/miniconda3/envs/defog/bin/python
```

If a baseline is installed in the current environment, its `*_PYTHON` variable
can simply be `$(command -v python)`.

### Supported datasets

| Baseline | Community-small | Ego-small | Grid | QM9 | ZINC |
|---|:---:|:---:|:---:|:---:|:---:|
| DeFoG | yes | yes* | — | yes | yes |
| DiGress | yes | yes* | yes | yes | yes |
| GDSS | yes | yes | yes | yes | yes |
| GraphRNN | yes | yes | yes | — | — |
| HOG-Diff | yes | yes | — | yes | yes |
| CatFlow | yes* | yes* | yes* | yes | yes |
| GSDM/GDSM | yes | yes* | yes | — | — |
| EDGE | yes* | yes* | yes* | — | — |
| SPECTRE | yes | yes* | yes | yes* | — |

`*` marks a GraphER dataset adaptation rather than an exact released upstream
profile. The corresponding YAML records this status and its budget provenance.

### DeFoG

`run_defog_baseline.py` uses the shared baseline lifecycle; the command below
omits ``--stage all`` because ``all`` is the default. Community-small uses the
DeFoG reference budget from
`common_community_small.yaml`.

```bash
RUN=seed_42
N=1024

PYTHONPATH=src python scripts/run_defog_baseline.py \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --num-samples "$N" \
  --seed-id 42 \
  --run-id "$RUN" \
  --device gpu

GEN_DIR="outputs/baselines/defog/community_small/$RUN/generations/seed_42_n_${N}"

PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN_DIR" \
  --output-dir "$GEN_DIR/evaluation_report"
```

### DiGress

`run_digress_baseline.py` uses the same shared lifecycle. Its Community-small
model config retains the released `comm20` training horizon
rather than inheriting a generic cross-model epoch count.

```bash
RUN=seed_42
N=1024

PYTHONPATH=src python scripts/run_digress_baseline.py \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config configs/baselines/digress_community_small.yaml \
  --num-samples "$N" \
  --seed-id 42 \
  --run-id "$RUN" \
  --device gpu

GEN_DIR="outputs/baselines/digress/community_small/$RUN/generations/seed_42_n_${N}"

PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN_DIR" \
  --output-dir "$GEN_DIR/evaluation_report"
```

### GDSS

GDSS uses the same shared lifecycle; the example uses the default ``--stage all``.
The model-specific YAML keeps GDSS's released optimizer and training horizon.

```bash
RUN=seed_42
N=1024

PYTHONPATH=src python scripts/run_gdss_baseline.py \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config configs/baselines/gdss_community_small.yaml \
  --num-samples "$N" \
  --seed-id 42 \
  --run-id "$RUN" \
  --device gpu

GEN_DIR="outputs/baselines/gdss/community_small/$RUN/generations/seed_42_n_${N}"

PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN_DIR" \
  --output-dir "$GEN_DIR/evaluation_report"
```

### GraphRNN

GraphRNN also uses the shared lifecycle. Its native budget is expressed as
epochs times sampled mini-batches per epoch; do not compare the
raw epoch integer directly with DeFoG.

```bash
RUN=seed_42
N=1024

PYTHONPATH=src python scripts/run_graphrnn_baseline.py \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config configs/baselines/graphrnn_community_small.yaml \
  --num-samples "$N" \
  --seed-id 42 \
  --run-id "$RUN" \
  --device gpu

GEN_DIR="outputs/baselines/graphrnn/community_small/$RUN/generations/seed_42_n_${N}"

PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN_DIR" \
  --output-dir "$GEN_DIR/evaluation_report"
```

### HOG-Diff

HOG-Diff is a two-stage model. The Community-small config uses the released
6,000 higher-order iterations followed by 22,000 OU iterations. With the
default ``--stage all``, one invocation runs both training stages and then
generates the requested batch; ``--stage train`` and ``--stage generate`` are
also available.

```bash
RUN=seed_42
N=1024

PYTHONPATH=src python scripts/run_hog_diff_baseline.py \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config configs/baselines/hog_diff_community_small.yaml \
  --num-samples "$N" \
  --seed-id 42 \
  --run-id "$RUN" \
  --device gpu

GEN_DIR="outputs/baselines/hog_diff/community_small/$RUN/generations/seed_42_n_${N}"

PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN_DIR" \
  --output-dir "$GEN_DIR/evaluation_report"
```

To generate another batch from an already completed managed HOG-Diff run, use
``--stage generate`` and choose a new ``--generation-id``. ``--generation-only``
remains accepted as a compatibility alias.

### CatFlow

CatFlow uses the same shared training/generation stages. Community-small uses
the GraphER `linear_v2` adapter config; this is explicitly marked as an adapted
budget because the supplied CatFlow release does not provide a native generic
Community-small profile.

```bash
RUN=seed_42
N=1024

PYTHONPATH=src python scripts/run_catflow_baseline.py \
  --stage train \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config configs/baselines/catflow_community_small_linear_v2.yaml \
  --seed-id 42 \
  --run-id "$RUN" \
  --device gpu

PYTHONPATH=src python scripts/run_catflow_baseline.py \
  --stage generate \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config configs/baselines/catflow_community_small_linear_v2.yaml \
  --seed-id 42 \
  --run-id "$RUN" \
  --num-samples "$N" \
  --device gpu

GEN_DIR="outputs/baselines/catflow/community_small/$RUN/generations/seed_42_n_${N}"

PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN_DIR" \
  --output-dir "$GEN_DIR/evaluation_report"
```

### GSDM / GDSM

The upstream project is GSDM, while GraphER retains canonical model ID `gdsm`
for compatibility with the existing wrapper/output layout. `gsdm` is accepted
as an alias by the consolidated source runner.

```bash
RUN=seed_42
N=1024

PYTHONPATH=src python scripts/run_gdsm_baseline.py \
  --stage train \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config configs/baselines/gdsm_community_small.yaml \
  --seed-id 42 \
  --run-id "$RUN" \
  --device gpu

PYTHONPATH=src python scripts/run_gdsm_baseline.py \
  --stage generate \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config configs/baselines/gdsm_community_small.yaml \
  --seed-id 42 \
  --run-id "$RUN" \
  --num-samples "$N" \
  --device gpu

GEN_DIR="outputs/baselines/gdsm/community_small/$RUN/generations/seed_42_n_${N}"

PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN_DIR" \
  --output-dir "$GEN_DIR/evaluation_report"
```

### EDGE

EDGE uses the same shared train/generate stages. Its model YAML retains the released
generic training horizon and degree-guided diffusion settings rather than the
DeFoG epoch fallback.

```bash
RUN=seed_42
N=1024

PYTHONPATH=src python scripts/run_edge_baseline.py \
  --stage train \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config configs/baselines/edge_community_small.yaml \
  --seed-id 42 \
  --run-id "$RUN" \
  --device gpu

PYTHONPATH=src python scripts/run_edge_baseline.py \
  --stage generate \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config configs/baselines/edge_community_small.yaml \
  --seed-id 42 \
  --run-id "$RUN" \
  --num-samples "$N" \
  --device gpu

GEN_DIR="outputs/baselines/edge/community_small/$RUN/generations/seed_42_n_${N}"

PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN_DIR" \
  --output-dir "$GEN_DIR/evaluation_report"
```

### SPECTRE

SPECTRE uses the same shared stages and retains the released Community training
schedule, including the GAN-specific optimizer settings in its model YAML.

```bash
RUN=seed_42
N=1024

PYTHONPATH=src python scripts/run_spectre_baseline.py \
  --stage train \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config configs/baselines/spectre_community_small.yaml \
  --seed-id 42 \
  --run-id "$RUN" \
  --device gpu

PYTHONPATH=src python scripts/run_spectre_baseline.py \
  --stage generate \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config configs/baselines/spectre_community_small.yaml \
  --seed-id 42 \
  --run-id "$RUN" \
  --num-samples "$N" \
  --device gpu

GEN_DIR="outputs/baselines/spectre/community_small/$RUN/generations/seed_42_n_${N}"

PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN_DIR" \
  --output-dir "$GEN_DIR/evaluation_report"
```

### Ego-small and Grid

For a supported generic dataset, replace `community_small` in the runner,
output path and evaluation config. For example, GDSS on Ego-small is:

```bash
RUN=seed_42
N=1024

PYTHONPATH=src python scripts/run_gdss_baseline.py \
  --dataset ego_small \
  --common-config configs/baselines/common_ego_small.yaml \
  --wrapper-config configs/baselines/gdss_ego_small.yaml \
  --num-samples "$N" \
  --seed-id 42 \
  --run-id "$RUN" \
  --device gpu

GEN_DIR="outputs/baselines/gdss/ego_small/$RUN/generations/seed_42_n_${N}"

PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/ego_small_evaluation.yaml \
  --generated-dir "$GEN_DIR" \
  --output-dir "$GEN_DIR/evaluation_report"
```

Use `configs/experiments/baselines/grid_evaluation.yaml` analogously for Grid.

### QM9 and ZINC baseline evaluation

Molecular-capable runners use the same training command pattern with
`--dataset qm9` or `--dataset zinc`. The managed generation is still written to
`base_graphs.pkl`, and the molecular evaluator reads it directly from the
generation directory.

QM9 is supported by DeFoG, DiGress, GDSS, HOG-Diff, CatFlow and SPECTRE. ZINC
is supported by DeFoG, DiGress, GDSS, HOG-Diff and CatFlow.

For example, CatFlow/QM9 is:

```bash
RUN=seed_42
N=1024

PYTHONPATH=src python scripts/run_catflow_baseline.py \
  --stage train \
  --dataset qm9 \
  --common-config configs/baselines/common_qm9.yaml \
  --wrapper-config configs/baselines/catflow_qm9.yaml \
  --seed-id 42 \
  --run-id "$RUN" \
  --device gpu

PYTHONPATH=src python scripts/run_catflow_baseline.py \
  --stage generate \
  --dataset qm9 \
  --common-config configs/baselines/common_qm9.yaml \
  --wrapper-config configs/baselines/catflow_qm9.yaml \
  --seed-id 42 \
  --run-id "$RUN" \
  --num-samples "$N" \
  --device gpu

GEN_DIR="outputs/baselines/catflow/qm9/$RUN/generations/seed_42_n_${N}"

PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-dir "$GEN_DIR" \
  --dataset-root outputs/datasets \
  --dataset qm9_attributed \
  --reference-split test \
  --train-split train \
  --hogdiff-compatible-metrics \
  --require-fcd \
  --output-dir "$GEN_DIR/evaluation_molecules"
```

For a one-command molecular runner such as DeFoG, training and generation are:

```bash
RUN=seed_42
N=1024

PYTHONPATH=src python scripts/run_defog_baseline.py \
  --dataset qm9 \
  --common-config configs/baselines/common_qm9.yaml \
  --num-samples "$N" \
  --seed-id 42 \
  --run-id "$RUN" \
  --device gpu

GEN_DIR="outputs/baselines/defog/qm9/$RUN/generations/seed_42_n_${N}"

PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-dir "$GEN_DIR" \
  --dataset-root outputs/datasets \
  --dataset qm9_attributed \
  --reference-split test \
  --train-split train \
  --hogdiff-compatible-metrics \
  --require-fcd \
  --output-dir "$GEN_DIR/evaluation_molecules"
```

For ZINC, replace `qm9` by `zinc` in the runner and managed output path, and use
`--dataset zinc` in `evaluate_generated_molecules.py`.

### Validation versus final test reporting

Use the validation split for checkpoint/budget selection and reserve the test
split for final reporting. For generic graphs:

```bash
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --reference-split val \
  --generated-dir "$GEN_DIR" \
  --output-dir "$GEN_DIR/evaluation_val"
```

The final paper result should omit `--reference-split val` (or explicitly use
`--reference-split test`). Do not choose training horizons or checkpoints by
looking at test MMD.

Full source-derived budgets, adaptation status and artifact conventions are in
[`docs/BASELINES.md`](docs/BASELINES.md).

## Prior perturbation experiments

The maintained batch launcher covers both QM9 typed-prior experiments and
Community-small degree-prior experiments:

```bash
bash scripts/run_prior_options.sh community_small generate edge_relocation
bash scripts/run_prior_options.sh qm9 generate edge_relocation
```

Use `bash scripts/run_prior_options.sh --help` for supported stages and methods.
Detailed Community-small perturbation definitions are in
[`docs/degree_perturbation_community_small.md`](docs/degree_perturbation_community_small.md).

## Evaluation

### Generic graphs

Use the shared evaluator for both GraphER and compatible baseline outputs:

```bash
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir <generation-directory> \
  --output-dir <generation-directory>/evaluation_report
```

The report computes degree and clustering MMD and, when enabled/available,
four-node orbit MMD. Keep the generic MMD protocol and reference split fixed
across methods.

For validation-time model selection use `--reference-split val`; reserve the
test split for final reporting.

### Molecular graphs

```bash
PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-graphs <molecular_graphs.pkl> \
  --dataset qm9_attributed \
  --dataset-root outputs/datasets \
  --reference-split test \
  --train-split train \
  --require-fcd \
  --output-dir <evaluation-directory>
```

The evaluator supports validity, uniqueness, novelty, NSPDK, FCD and optional
attributed graphlet diagnostics. Use the same metric-source convention for every
model in a given table.

## Diagnostics and visualization

The `diagnose_*` scripts are intentionally narrow and are not alternate training
entry points. Examples include CatFlow path/provenance checks, spectral denoiser
analysis, typed-degree perturbation audits and joint-checkpoint diagnostics.

Useful visualization commands include:

```bash
PYTHONPATH=src python scripts/draw_dataset.py --help
PYTHONPATH=src python scripts/draw_generated_dataset.py --help
PYTHONPATH=src python scripts/draw_generated_qm9_outliers.py --help
```

## Artifact and reproducibility policy

For every reported experiment retain:

- dataset identity and split hashes;
- model/config identity;
- common baseline profile when used;
- training and generation seeds;
- checkpoint selection rule;
- exact generated sample count;
- evaluator configuration;
- source/commit or source digest for external baselines;
- runtime/hardware details.

Do not silently regenerate a missing dataset during final evaluation. Do not
select checkpoints on test MMD. Do not mix corrected and legacy CatFlow training
paths under the same result label.

## Documentation

Current canonical documents:

- [`docs/BASELINES.md`](docs/BASELINES.md) — baseline protocol, runners and common configs;
- [`docs/DESIGN_CONTRACT.md`](docs/DESIGN_CONTRACT.md) — invariants and implementation contract;
- [`docs/TOPOLOGY_GENERATOR.md`](docs/TOPOLOGY_GENERATOR.md) — generic topology generation;
- [`docs/ATTRIBUTED_SPECTRAL_GRAPHLET_DIFFUSION.md`](docs/ATTRIBUTED_SPECTRAL_GRAPHLET_DIFFUSION.md) — attributed spectral/graphlet pipeline;
- [`docs/SPECTRAL_GRAPHLET_DIFFUSION.md`](docs/SPECTRAL_GRAPHLET_DIFFUSION.md) — generic spectral/graphlet guidance;
- [`docs/LAMBDA_PROJECTOR_SPECTRAL_DIFFUSION.md`](docs/LAMBDA_PROJECTOR_SPECTRAL_DIFFUSION.md) — structured eigenvalue + eigenspace-projector diffusion;
- [`docs/SPECTRAL_ONLY_DEBUG.md`](docs/SPECTRAL_ONLY_DEBUG.md) — spectral debugging protocol;
- [`docs/CYCLE_GRAPHLET_GUIDANCE.md`](docs/CYCLE_GRAPHLET_GUIDANCE.md) — molecular cycle-only graphlet guidance and current tuning;
- [`docs/degree_perturbation_community_small.md`](docs/degree_perturbation_community_small.md) — degree-prior perturbations.

Short-lived integration audits, generated validation logs and per-wrapper setup
documents are intentionally not kept in `docs/`; current operational guidance
belongs in this README or `docs/BASELINES.md`.

### Guidance factorial ablation

For the controlled Community-small comparison **HH**, **HH + spectra**, **HH + graphlet**, and **HH + spectra + graphlet**, run:

```bash
PYTHONPATH=src python scripts/run_guidance_factorial_ablation.py \
  --profile configs/experiments/grapher/ablations/community_small_guidance_factorial.yaml \
  --degree-source learned
```

See [`docs/GUIDANCE_FACTORIAL_ABLATION.md`](docs/GUIDANCE_FACTORIAL_ABLATION.md) for the controlled degree-source variant and interpretation.
