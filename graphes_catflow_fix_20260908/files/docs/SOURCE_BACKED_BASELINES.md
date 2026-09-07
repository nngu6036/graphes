# CatFlow, GSDM, EDGE and SPECTRE in GraphER

Integration date: 6 September 2026. The basis is the five source ZIPs supplied
with the request, not a replacement implementation obtained from the web.

**CatFlow correction, 8 September 2026:** new training uses an explicit linear
probability path instead of the uploaded source's inconsistent fixed-noise path.
Read [the audit and migration guide](CATFLOW_PATH_AUDIT_20260908.md) before rerunning
CatFlow. Old checkpoints keep legacy sampling semantics and emit a warning.
The original integration tests below are historical, not validation of the v2 path.

## Scope and validation status

The four requested classes are implemented: `CatFlowWrapper`, `GDSMWrapper`,
`EDGEWrapper`, and `SPECTREWrapper`. The spectral-diffusion source calls itself
**GSDM**. `GSDMWrapper` is an alias of `GDSMWrapper`; `gsdm` is an alias of the
canonical artifact/registry ID `gdsm`. This is a separate model from the existing
`GDSSWrapper`.

| Adapter | Community-small | Ego-small | Grid | QM9 | ZINC |
|---|---|---|---|---|---|
| CatFlow | Yes | Yes | Yes | Yes | Yes |
| GSDM (`gdsm`) | Yes | Yes | Yes | No | No |
| EDGE | Yes | Yes | Yes | No | No |
| SPECTRE | Yes | Yes | Yes | Adapted; see singleton policy | No |

These are executable source-backed adapters, not placeholder classes. They
reuse the supplied neural code in subprocesses. The current validation covers
small CPU runs, not converged benchmark experiments or GPU compatibility:

- CatFlow: one-epoch generic and molecular training, checkpoint reload, exact-count
  generation, singleton/final partial batches, and relative-path invocation.
  The numerical sampling tests used **Euler with three steps**, not the default
  adaptive `dopri5` solver, whose `torchdiffeq` dependency was unavailable here.
- GSDM: one-epoch generic training and four-step spectral generation, with both
  no corrector and the native Langevin corrector; uneven generation batches.
- SPECTRE: native discriminator/generator updates and `all_fake` generation on
  generic graphs and adapted QM9, including singleton molecules. Tests used the
  complete EMA fallback, not the optional `torch_ema` package.
- EDGE: registry, configuration, data/artifact contracts and syntax checked.
  Native training was attempted but stopped before a model update because
  `torch_geometric` was unavailable. `torch_scatter` is also absent locally.
  **EDGE training/generation still require validation in its actual environment.**

The selected unit/contract/regression run passed **94 tests and five subtests**.
This is not a claim that the entire original suite passes: the supplied
`tests/test_defog_wrapper.py` imports `scripts.run_defog_grapher`, a script absent
from the original ZIP, and cannot be collected. It was excluded from that run.
The GSDM batch also passed the existing graph report evaluator, with matching
training/test split hashes. No full FCD/NSPDK or GPU benchmark runs were performed.

## Source layout and environments

The standalone ZIP includes the supplied source trees in:

```text
external/catflow/
external/gdsm/
external/edge/
external/spectre/
```

They are outside `src/grapher`, are not installed as part of the GraphER Python
package, and are not modified by training. `external/SOURCES.json` records the
original archive names and SHA-256 hashes. Runtime manifests additionally hash
the source Python/YAML files. No datasets, pretrained benchmark checkpoints,
virtual environments, or CUDA libraries are bundled.

Run from the updated GraphER repository root:

```bash
export CATFLOW="$PWD/external/catflow"
export GDSM="$PWD/external/gdsm"
export EDGE="$PWD/external/edge"
export SPECTRE="$PWD/external/spectre"

# This works when the active environment has the required worker dependencies.
export CATFLOW_PYTHON="$(command -v python)"
export GDSM_PYTHON="$(command -v python)"
export EDGE_PYTHON="$(command -v python)"
export SPECTRE_PYTHON="$(command -v python)"
```

For separate environments, replace each `*_PYTHON` with that environment's
absolute `bin/python` path. The launcher remains in the GraphER environment;
each worker uses its chosen interpreter. `--source-root` and `--python` are
explicit alternatives to these environment variables. `GSDM` and `GSDM_PYTHON`
are accepted aliases for the spectral adapter.

Install a PyTorch build appropriate for the selected CPU/CUDA environment first.
Adapter dependency lists are supplied, without asserting a universal CUDA lockfile:

```bash
"$CATFLOW_PYTHON" -m pip install -r requirements/baselines/catflow.txt
"$GDSM_PYTHON" -m pip install -r requirements/baselines/gdsm.txt
"$EDGE_PYTHON" -m pip install -r requirements/baselines/edge.txt
"$SPECTRE_PYTHON" -m pip install -r requirements/baselines/spectre.txt
```

EDGE's `torch-scatter` binary must match its PyTorch/CUDA ABI. Do not install a
random incompatible wheel. SPECTRE's managed dense path does not require the
upstream evaluation-only `graph_tool` or PyG imports. CatFlow's dense path does
not require its upstream dataset/download packages. A working `torchdiffeq`
is required for CatFlow's default adaptive sampler. EMA package fallbacks are
explicitly described below and recorded in the worker metadata.

The local numerical tests used Python 3.13, PyTorch 2.10.0+cpu, and (for SPECTRE)
PyTorch Lightning 2.6.5. These are tested CPU versions, not prescribed GPU pins.

## Reuse the prepared GraphER splits

All wrappers require existing `train.pkl`, `val.pkl`, and `test.pkl` files. They
never download, regenerate, reshuffle, or overwrite benchmark splits.

| CLI dataset | Serialized directory | Declared maximum nodes |
|---|---|---:|
| `community_small` | `outputs/datasets/sbm` | 20 |
| `ego_small` | `outputs/datasets/ego_small` | 18 |
| `grid` | `outputs/datasets/grid` | 361 |
| `qm9` | `outputs/datasets/qm9_attributed` | 9 |
| `zinc` | `outputs/datasets/zinc` | 38 |

Reuse the exact directory used by your current GraphER/DeFoG/DiGress/GDSS runs.
`--seed-id` changes training randomness, not data splits. Use `--dataset-root`
for a different location, and update the evaluation config's `dataset.root` to
match. `--serialized-dataset` is an explicit override for a custom directory.

Only when splits have not yet been prepared, use the existing preparation
workflow. For example:

```bash
PYTHONPATH=src python scripts/prepare_generic_dataset.py \
  --dataset community_small --root outputs/datasets
```

That preparation script rebuilds its output; **do not run it between baseline
experiments on already frozen splits**. The canonical community source artifact
must be available through the original preparation workflow. It is not supplied
by these source-code snapshots.

Workers receive numeric train/validation NPZ files only. They are never given a
test NPZ. Test pickle bytes are hashed by the parent for provenance, not supplied
as conditioning or neural training targets. SPECTRE does not use validation for
checkpoint selection; the other adapters record validation losses. All four
publish the final configured epoch, never the test-selected best epoch.

## Separate training, generation and evaluation: Community-small

Commands below use seed 42 and 1,024 samples. They assume that the sources,
interpreters, dependencies and prepared data described above are available.
`--device gpu` fails clearly if the selected worker environment has no CUDA.
Use `--device cpu` for a CPU run. All training budgets come from the YAML files.

### CatFlow

```bash
# Train only.
PYTHONPATH=src python scripts/run_catflow_baseline.py \
  --stage train \
  --dataset community_small \
  --wrapper-config configs/baselines/catflow_community_small.yaml \
  --seed-id 42 --run-id seed_42 --device gpu

# Generate from the managed checkpoint; no retraining.
PYTHONPATH=src python scripts/run_catflow_baseline.py \
  --stage generate \
  --dataset community_small \
  --wrapper-config configs/baselines/catflow_community_small.yaml \
  --seed-id 42 --run-id seed_42 --num-samples 1024 --device gpu

# Evaluate with the existing GraphER structural report.
GEN_DIR="outputs/baselines/catflow/community_small/seed_42/generations/seed_42_n_1024"
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN_DIR" \
  --generated-stage catflow \
  --output-dir "$GEN_DIR/evaluation_report"
```

### GSDM

```bash
# Train only.
PYTHONPATH=src python scripts/run_gdsm_baseline.py \
  --stage train \
  --dataset community_small \
  --wrapper-config configs/baselines/gdsm_community_small.yaml \
  --seed-id 42 --run-id seed_42 --device gpu

# Generate from the managed checkpoint; no retraining.
PYTHONPATH=src python scripts/run_gdsm_baseline.py \
  --stage generate \
  --dataset community_small \
  --wrapper-config configs/baselines/gdsm_community_small.yaml \
  --seed-id 42 --run-id seed_42 --num-samples 1024 --device gpu

# Evaluate with the existing GraphER structural report.
GEN_DIR="outputs/baselines/gdsm/community_small/seed_42/generations/seed_42_n_1024"
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN_DIR" \
  --generated-stage gdsm \
  --output-dir "$GEN_DIR/evaluation_report"
```

### EDGE

```bash
# Train only.
PYTHONPATH=src python scripts/run_edge_baseline.py \
  --stage train \
  --dataset community_small \
  --wrapper-config configs/baselines/edge_community_small.yaml \
  --seed-id 42 --run-id seed_42 --device gpu

# Generate from the managed checkpoint; no retraining.
PYTHONPATH=src python scripts/run_edge_baseline.py \
  --stage generate \
  --dataset community_small \
  --wrapper-config configs/baselines/edge_community_small.yaml \
  --seed-id 42 --run-id seed_42 --num-samples 1024 --device gpu

# Evaluate with the existing GraphER structural report.
GEN_DIR="outputs/baselines/edge/community_small/seed_42/generations/seed_42_n_1024"
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN_DIR" \
  --generated-stage edge \
  --output-dir "$GEN_DIR/evaluation_report"
```

### SPECTRE

```bash
# Train only.
PYTHONPATH=src python scripts/run_spectre_baseline.py \
  --stage train \
  --dataset community_small \
  --wrapper-config configs/baselines/spectre_community_small.yaml \
  --seed-id 42 --run-id seed_42 --device gpu

# Generate from the managed checkpoint; no retraining.
PYTHONPATH=src python scripts/run_spectre_baseline.py \
  --stage generate \
  --dataset community_small \
  --wrapper-config configs/baselines/spectre_community_small.yaml \
  --seed-id 42 --run-id seed_42 --num-samples 1024 --device gpu

# Evaluate with the existing GraphER structural report.
GEN_DIR="outputs/baselines/spectre/community_small/seed_42/generations/seed_42_n_1024"
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN_DIR" \
  --generated-stage spectre \
  --output-dir "$GEN_DIR/evaluation_report"
```

The report computes the existing degree, clustering and orbit MMD and writes the
report JSON, CSV and sample figures. It infers baseline graph files from
`manifest.json`/`base_graphs.pkl`; no model-specific evaluator is introduced.
The shared YAMLs explicitly choose the GraphRNN-style MMD convention. Use the
same ORCA setup as your other baselines; absent an ORCA executable, this report
can use its existing Python four-node orbit path. Large graph batches can make
that fallback expensive.

For Ego-small or Grid, replace `community_small` everywhere with `ego_small` or
`grid`, respectively. Matching configs exist for all four adapters. The generic
report configuration is shared, not one independently tuned per baseline.

## Molecular experiments and the correct evaluator

CatFlow supports QM9 and ZINC categorical graphs. SPECTRE supports adapted QM9
only. GSDM and EDGE deliberately reject molecular dataset names instead of
returning untyped adjacency graphs as molecules.

The structural report is **not** the FCD/NSPDK evaluator. Use the existing
`scripts/evaluate_generated_molecules.py` for the molecular comparison table.
For example, CatFlow on QM9:

```bash
MODEL=catflow
DATASET=qm9
REFERENCE_DATASET=qm9_attributed

PYTHONPATH=src python "scripts/run_${MODEL}_baseline.py" \
  --stage train --dataset "$DATASET" \
  --wrapper-config "configs/baselines/${MODEL}_${DATASET}.yaml" \
  --seed-id 42 --run-id seed_42 --device gpu

PYTHONPATH=src python "scripts/run_${MODEL}_baseline.py" \
  --stage generate --dataset "$DATASET" \
  --wrapper-config "configs/baselines/${MODEL}_${DATASET}.yaml" \
  --seed-id 42 --run-id seed_42 --num-samples 1024 --device gpu

GEN_DIR="outputs/baselines/${MODEL}/${DATASET}/seed_42/generations/seed_42_n_1024"
PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-graphs "$GEN_DIR/base_graphs.pkl" \
  --dataset-root outputs/datasets \
  --dataset "$REFERENCE_DATASET" \
  --reference-split test --train-split train \
  --hogdiff-compatible-metrics --require-fcd \
  --output-dir "$GEN_DIR/evaluation_molecules"
```

For SPECTRE/QM9 set `MODEL=spectre`, leaving the two dataset variables unchanged.
For CatFlow/ZINC set `MODEL=catflow`, `DATASET=zinc`, and
`REFERENCE_DATASET=zinc`. No SPECTRE/ZINC config is provided.

The example deliberately selects the existing HOG-Diff-compatible corrected-valid
molecule source for distribution metrics, matching your recent evaluator usage.
The evaluator still reports strict raw validity separately. For strict raw-valid
FCD/NSPDK/uniqueness/novelty, replace `--hogdiff-compatible-metrics` with
`--metric-molecule-source raw_valid`. Do not mix these conventions across rows.
`--require-fcd` prevents a missing FCD backend from silently becoming a missing
metric. Install the existing evaluator's RDKit/EDeN/FCD dependencies in the
parent GraphER environment. None of these evaluator corrections modifies the
raw `base_graphs.pkl` exported by the baseline worker.

QM9 atom categories are C/N/O/F; ZINC adds P/S/Cl/Br/I. The accepted bond
categories are the existing three single/double/triple states. Unsupported
aromatic category 4 is rejected; use your already-prepared declared categorical
representation, not an implicit wrapper-side kekulization. Formal charge and
stereochemistry are not learned by these adapters and are never copied from a
held-out graph. Generated formal charges are zero before the common evaluator.

## Config inventory and budgets

There are 16 wrapper YAMLs: six CatFlow (including the v2 diagnostic config), three GSDM, three EDGE, and four
SPECTRE. There are also five shared structural-report experiment YAMLs under
`configs/experiments/baselines/`. Molecular FCD/NSPDK use the separate script
above, not flags placed in a structural-report YAML.

The following are configurable rerun starting points, **not claims of published
settings or reproduced benchmark scores**:

| Model | Dataset | Epochs | Training batch | Generation batch |
|---|---|---:|---:|---:|
| CatFlow | `community_small` | 1,000 | 128 | 128 |
| CatFlow | `ego_small` | 1,000 | 128 | 128 |
| CatFlow | `grid` | 1,000 | 2 | 2 |
| CatFlow | `qm9` | 300 | 1024 | 128 |
| CatFlow | `zinc` | 500 | 1024 | 128 |
| GSDM | `community_small` | 200 | 128 | 128 |
| GSDM | `ego_small` | 200 | 128 | 128 |
| GSDM | `grid` | 1,000 | 8 | 8 |
| EDGE | `community_small` | 50,000 | 8 | 32 |
| EDGE | `ego_small` | 50,000 | 8 | 32 |
| EDGE | `grid` | 50,000 | 2 | 2 |
| SPECTRE | `community_small` | 12,000 | 10 | 32 |
| SPECTRE | `ego_small` | 12,000 | 10 | 32 |
| SPECTRE | `grid` | 12,000 | 1 | 1 |
| SPECTRE | `qm9` | 30 | 128 | 32 |

Override epochs with `--n-epochs`, training batch size with `--batch-size`, and
generation batch size with `--generation-batch-size`. Model-specific settings
remain in the YAML. Generation cannot change a trained architecture.

CatFlow v2 defaults to the explicit linear/KLD path and `dopri5` integration up to `t=0.95`;
`t_end` must be strictly less than 1 because the KLD velocity divides by `1-t`.
Its `sample.steps` controls fixed-grid solvers (Euler, RK4, etc.), not adaptive DOPRI5.
GSDM defaults to the source VP spectral configurations and native PC sampler.
EDGE uses 64 linear-schedule steps on Community-small/Ego-small, within the
small-graph range advised in the supplied README; Grid retains 128. The native
linear schedule needs more than 20 steps; use cosine for an 8/16-step experiment.
SPECTRE uses its native full spectral GAN rather than an adjacency-only ablation.

For three independently trained runs, execute training and generation for
`--seed-id 42 --run-id seed_42`, then 43/seed_43 and 44/seed_44, using the same
prepared data. Changing only `--generation-seed` is repeated sampling of one
trained model, not three training seeds.

## Managed files, reuse and failure behavior

```text
outputs/baselines/<model>/<dataset>/<run_id>/
  run.json
  train/
    manifest.json
    resolved_config.yaml
    train.log
    checkpoints/<model>.pt
    native_dataset/{train,val}.npz
    worker_manifest.json
  generations/seed_<generation_seed>_n_<count>/
    base_graphs.pkl
    samples.npz
    manifest.json
    generate.log
    worker_manifest.json
```

The defaults are `run_id=seed_<training_seed>` and equal training/generation
seeds. `--generation-id` can name an additional batch. `--checkpoint` accepts a
checkpoint matching that managed run, not an arbitrary upstream `.ckpt` file.

A matching completed training request is reused only when its options, source,
dataset fingerprint, seed and checkpoint hash match. Existing generation
outputs are not silently overwritten. Prefer a new run ID for a changed config;
`--overwrite` is explicit and destructive for the selected artifact. Training
resume is not implemented: reusable completed training is not partial resume.

Every published batch has exactly the requested count. Workers do not discard
isolates, retain only the largest component, repair molecules, or rejection-sample
until a graph is chemically valid. Nonfinite tensors, wrong dimensions, bad
categories, asymmetric export, changed source, changed NPZ data or checkpoint
hash mismatches fail before graph publication. Failed job/log directories are
retained with `.train_work_*` or `.generation_work_*` names for diagnosis.

No fake/random/empirical-graph generator is used as a fallback when a learning
dependency is missing. The legitimate empirical **conditioning priors** and
SPECTRE singleton branch below are explicit parts of the defined adapters.

These wrappers currently do not create paired post-training corrector estimates.
`training_estimates.enabled=true` raises a clear error; the compatibility flag
`--skip-training-estimates` is harmless. Existing GraphER training and refinement
code is unchanged by this integration.

## Source-specific adaptations

**CatFlow.** The worker executes the supplied `get_GT_model` and graph transformer.
Version 2 explicitly constructs `(1-t)*z + t*one_hot_target` without additional
constant noise. This corrects the probability-path mismatch in the uploaded
normal `conditional_velocity` branch; that branch is retained only through
explicit `train.path: upstream` for labelled legacy diagnostics. Source files
remain read-only. It uses dense frozen-data batches, the source CE weights
(node + 5 x edge), clipping, AdamW, cosine schedule and EMA. Validation uses fixed
noise/time draws and EMA weights on the real validation split. Both best-EMA and
final snapshots are stored, with the selected epoch recorded during sampling. The one-example time
embedding is kept two-dimensional. The managed ODE/export loop fixes the supplied
sampler's per-batch accumulator reset and floor-sized ten-batch sample loss.
The native `ema.py` is used if `torch_ema` is unavailable; the backend is saved.
Evaluation-only imports are isolated rather than importing an unrelated GraphER
`models` package into the worker.

**GSDM.** The worker uses `ScoreNetworkX`, `ScoreNetworkA_eigen`,
`get_sde_loss_fn2`, `VPSDE`, and `get_pc_sampler2`, not the existing GDSS wrapper.
Eigenvector bases and node-count conditioning are sampled from training only.
The degree-feature support is declared from the node cap for the small generic
profiles; Grid preserves the source's five-class degree support. Training or
validation isolates are explicitly rejected because the native loss infers its
active-node mask from adjacency. Generated isolates are retained using the
sampled node count. The native VP noise path, score losses, optimizers and
predictor/corrector code are retained; native download/evaluation launchers are
not invoked.

**EDGE.** The uploaded archive has no `datasets/` module. The adapter constructs
PyG records with the native full-edge, degree and batch fields directly from the
frozen split. It uses the native TGNN degree-guided model, active-node diffusion,
ELBO and sampler. The conditioning degree sequence is sampled empirically from
the training split and randomly permuted, as permitted by the supplied README;
this is not a separately trained degree prior. It starts from the empty graph,
not a copied training adjacency. Full-edge states are decoded directly instead
of relying on the native sampler's partially updated PyG `edge_index` slicing
metadata. Degree guidance is not advertised as an exact GraphER-style invariant
guarantee. Install the real PyG/scatter dependencies; there is no neural fallback.

**SPECTRE.** The adapter executes the original neural modules, `_disc_step`,
`_gen_step` and optimizer definitions, while managing epochs/backpropagation/logs
outside the obsolete Trainer CLI. Evaluation-only dataset/`graph_tool`/PyG imports
are isolated; all learned generators and discriminators remain source-native.
Generation always uses `test_type='all_fake'` with zero dummy real spectra, never
true held-out eigenvalues/eigenvectors. Molecular decoding follows the native
two-step rule: threshold total edge probability, then select the bond category.

The supplied rotation discriminator cannot operate at `k_eigval=1`; its spectral
normalization encounters a zero-width rotation output. It also assumes at least
two active vertices. The QM9 config therefore retains `k=2`, zero-pads unavailable
trailing eigenmodes as in the native loader, and explicitly sets
`train.singleton_policy: empirical`. Graphs with n>=2 train the native GAN;
n=1 has deterministic empty adjacency and an empirical training-atom categorical
model. Generation samples n from the **whole** training split and uses the
corresponding branch. Counts and singleton policy are recorded. Singleton
molecules are not silently removed from the benchmark. This is an explicit
GraphER-protocol adaptation, not a paper-identical QM9 rerun. Generic configs
use `singleton_policy=error` unless deliberately changed.

When `torch_ema` is missing, SPECTRE has a complete device-aware EMA bookkeeping
fallback with the same update-count-adjusted recurrence present in the supplied
CatFlow EMA source. Its backend is recorded. It is not a replacement GAN and
cannot compensate for missing actual learning dependencies.

## Python interface

```python
from pathlib import Path
from grapher.models import (
    CatFlowWrapper, GDSMWrapper, GSDMWrapper, EDGEWrapper, SPECTREWrapper,
    DatasetReference, GenerateRequest, RunSpec, TrainRequest,
)

wrapper = GDSMWrapper()  # GSDMWrapper is the same class.
run = RunSpec.for_seed(model_id="gdsm", dataset_id="community_small", seed=42)
training = wrapper.train(TrainRequest(
    run=run,
    dataset=DatasetReference("community_small", root=Path("outputs/datasets"), serialized_id="sbm"),
    config_path=Path("configs/baselines/gdsm_community_small.yaml"),
))
generation = wrapper.generate(GenerateRequest(
    run=run, checkpoint_path=training.checkpoint_path,
    num_graphs=1024, generation_seed=42,
))
print(generation.graphs_path)
```

## Tests and evidence

The new fast tests do not need the external learning environments:

```bash
PYTHONPATH=src:. python -m pytest -q \
  tests/test_source_backed_baselines.py \
  tests/test_baseline_model_wrappers.py
```

That pair contains 37 tests plus five subtests. The broader selected run contains
94 tests plus five subtests, excluding the pre-existing missing DeFoG-script
collection failure noted above. `validation/source_baselines/VALIDATION.json`
and accompanying logs/configs record the local scope. Toy-run losses and
three-graph MMD values are smoke diagnostics, not benchmark results.
