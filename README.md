# GraphER: Post-Generation Structural Correction for Graphs

GraphER is a post-generation correction model for graph generators. It takes a
completed graph from a frozen base generator and applies valid double-edge
swaps to improve selected structural statistics while preserving inherited
invariants.

The base generator and GraphER have separate responsibilities:

```text
frozen base generator -> completed graph G0 -> GraphER -> corrected graph GT
```

- The base generator determines the initial graph, including its node count,
  degree profile, and attributes.
- The GraphER predictor estimates a target structural summary from the current
  graph.
- Randomized greedy search samples valid rewiring candidates and accepts an
  improving swap.
- If no admissible improving move is found, GraphER returns the current graph
  unchanged.

This repositioning changes how the system is trained, evaluated, and reported;
it does not replace the degree-preserving rewiring operation itself.

## Current implementation scope

The codebase is organized around two independent packages:

| Package | Responsibility |
| --- | --- |
| `grapher.models` | Train or run base graph generators and serialize completed raw graphs |
| `grapher.rewiring_mlp` | Predict structural targets, construct valid swaps, correct graphs, and evaluate raw/corrected batches |

The generic GraphER implementation currently uses graphlet summaries as its
maintained prediction and candidate-scoring target. The structural-summary
interface is intended to support additional components such as clustering,
orbit, mixing, and spectral statistics after their predictor heads and
candidate-side computations are implemented and validated.

The attributed implementation and molecular constraint utilities remain in the
repository, but the generic post-correction route is the primary maintained
workflow.

## Rewiring guarantees

For an admissible simple connected input graph, every accepted generic
double-edge swap preserves:

- the node and edge counts;
- the degree of every indexed node;
- the degree multiset;
- simplicity and undirectedness;
- the absence of self-loops and parallel edges; and
- connectivity.

For attributed graphs, the code also provides typed-invariant and molecular
validity utilities. The exact preserved signature depends on the enabled
candidate constraints. Degree preservation alone does not guarantee chemical
validity, so molecular experiments must separately report RDKit sanitization
and valence checks.

GraphER cannot repair an invariant that is already incorrect in the raw base
sample. Its purpose is to improve complementary higher-order structure inside
the invariant fibre of that sample.

## Base-generator wrappers

All base generators use the same interface:

```python
train(request: TrainRequest) -> TrainingArtifacts
generate(request: GenerateRequest) -> GenerationArtifacts
```

Wrappers live under `src/grapher/models/`. Third-party implementations remain
in their own repositories and environments; GraphER stores only their adapters
and normalized output artifacts.

| Wrapper ID | Model | Status |
| --- | --- | --- |
| `defog` | DeFoG | Generic `train()` and `generate()` implemented |
| `dhvae_hh` | DH-VAE + randomized Havel--Hakimi | Core implementation isolated; common wrapper orchestration pending |
| `digress` | DiGress | Placeholder |
| `catflow` | CatFlow | Placeholder |
| `hog_diff` | HOG-Diff | Placeholder |
| `flagg` | FLAGG | Placeholder |

Placeholder methods raise `BaselineNotImplementedError` before creating a
partial run directory.

### Dataset identity

`DatasetReference` keeps three names separate:

- `benchmark_id`: the dataset name used in reports and artifact paths;
- `serialized_id`: the directory containing GraphER's prepared split files;
  and
- `native_id`: the dataset key used by the upstream model.

For example, Community-small currently uses:

```python
DatasetReference(
    benchmark_id="community_small",
    root=Path("outputs/datasets"),
    serialized_id="sbm",
    native_id="comm20",
)
```

## Artifact layout

Training runs and generated batches have separate identities:

```text
outputs/baselines/
└── <model_id>/
    └── <benchmark_id>/
        └── <training_run_id>/
            ├── run.json
            ├── train/
            │   ├── checkpoints/
            │   ├── manifest.json
            │   ├── resolved_config.yaml
            │   └── train.log
            └── generations/
                └── <generation_id>/
                    ├── base_graphs.pkl
                    ├── manifest.json
                    ├── generate.log
                    └── native/
```

The default identifiers are:

```text
training_run_id = seed_<training-seed>
generation_id   = seed_<generation-seed>_n_<sample-count>
```

`base_graphs.pkl` is the common ordered batch consumed by GraphER. Native
exports are retained for provenance but are not substituted for the normalized
batch during paired evaluation.

Corrected graphs must be stored separately, for example:

```text
outputs/corrections/<model>/<dataset>/<training-run>/<generation>/
```

The correction manifest should reference the raw batch hash, sample count, and
order. Failed or unsupported inputs must be returned unchanged rather than
dropped, because paired raw-versus-corrected evaluation requires one output for
every input.

## Repository layout

```text
configs/
├── datasets/                     Dataset preparation configurations
└── experiments/                  DH-VAE and GraphER experiment configurations
docs/
├── BASELINE_MODEL_WRAPPERS.md    Shared wrapper and artifact contract
├── DEFOG_WRAPPER.md              DeFoG setup and usage
├── DHVAE_HH_PACKAGE.md           Optional degree-sequence baseline
└── REWIRING_MLP_PACKAGE.md       GraphER package boundary
scripts/
├── prepare_generic_dataset.py
├── prepare_qm9_dataset.py
├── prepare_zinc_dataset.py
├── defog_prepare_dataset_worker.py
├── defog_export_worker.py
├── run_defog_grapher.py
├── train_topology_grapher.py
├── run_topology_grapher.py
└── evaluate_graph_generation_report.py
src/grapher/
├── data/                          Shared dataset loading and preparation
├── models/                        Base-generator wrappers and DH-VAE+HH
├── rewiring_mlp/
│   ├── core/                      Degree-preserving swap operations
│   ├── generic/                   Generic predictor and refiner
│   ├── attributed/                Attribute-aware predictor and refiner
│   ├── molecular/                 Molecular constraints and typed invariants
│   ├── properties/                Structural-summary calculations
│   └── evaluation/                Metrics and experiment utilities
└── utils/                         Shared utilities
tests/                             Unit and regression tests
```

The legacy topology and attributed scripts remain available while the unified
baseline-to-correction experiment runner is completed. They should not be
interpreted as evidence that GraphER is still positioned as a standalone graph
generator.

## Environment setup

Python 3.10 or newer is recommended. This archive uses a `src` layout without
packaging metadata, so commands should be run from the repository root with
`PYTHONPATH=src`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Select the PyTorch build appropriate for the local CUDA environment.
python -m pip install torch
python -m pip install numpy networkx pyyaml matplotlib

# Development and verification.
python -m pip install pytest ruff
```

Optional dependencies include ORCA for orbit statistics and RDKit, PyG, and
`fcd-torch` for molecular workflows.

## Prepare a dataset

Run all preparation commands from the repository root with `PYTHONPATH=src`.

### Community-small

```bash
PYTHONPATH=src python scripts/prepare_generic_dataset.py \
  --dataset community_small \
  --root outputs/datasets
```

The current builder stores Community-small under the historical serialized ID
`outputs/datasets/sbm/`. The wrapper manifest records both this storage alias
and the report-facing benchmark name.

### Ego-small

```bash
PYTHONPATH=src python scripts/prepare_generic_dataset.py \
  --dataset ego_small \
  --root outputs/datasets
```

This command constructs the Ego-small benchmark from the Citeseer source
specified in `configs/datasets/ego_small.yaml` and writes its splits to
`outputs/datasets/ego_small/`.

### Grid

```bash
PYTHONPATH=src python scripts/prepare_generic_dataset.py \
  --dataset grid \
  --root outputs/datasets
```

Grid graphs are generated from `configs/datasets/grid.yaml` and saved under
`outputs/datasets/grid/`.

### QM9

QM9 preparation requires RDKit and writes both a topology-only dataset and an
attributed molecular dataset. To obtain the source through PyTorch Geometric:

```bash
PYTHONPATH=src python scripts/prepare_qm9_dataset.py \
  --source pyg \
  --pyg-root data/pyg_qm9 \
  --root outputs/datasets \
  --topology-name qm9_topology \
  --attributed-name qm9_attributed
```

If `gdb9.sdf` is already available locally, prepare the same datasets without
the PyTorch Geometric download path:

```bash
PYTHONPATH=src python scripts/prepare_qm9_dataset.py \
  --source sdf \
  --sdf-file /path/to/gdb9.sdf \
  --root outputs/datasets \
  --topology-name qm9_topology \
  --attributed-name qm9_attributed
```

The outputs are stored in `outputs/datasets/qm9_topology/` and
`outputs/datasets/qm9_attributed/`. The attributed representation stores
`atomic_num` and `atom_type` on nodes and `bond_type` and `bond_order` on
edges.

### ZINC

ZINC preparation requires RDKit and a local `.smi`, `.txt`, `.csv`, or `.tsv`
source. For a CSV file whose SMILES column is named `smiles`, run:

```bash
PYTHONPATH=src python scripts/prepare_zinc_dataset.py \
  --config configs/datasets/zinc.yaml \
  --smiles-file /path/to/zinc250k.csv \
  --smiles-column smiles \
  --root outputs/datasets
```

For a plain `.smi` or `.txt` file, omit `--smiles-column`. The configured fixed
subset and its train, validation, and test splits are written to
`outputs/datasets/zinc/`. Automatic ZINC downloading is not currently
implemented, so the local source file is required.

## DeFoG wrapper

The DeFoG integration is the first completed third-party wrapper. It runs the
upstream source in a subprocess to avoid dependency and import-name conflicts.

Set the source root and isolated interpreter:

```bash
export DEFOG=/absolute/path/to/DeFoG
export DEFOG_PYTHON=/absolute/path/to/defog-env/bin/python
```

The attached DeFoG implementation supports the generic native configurations
`comm20`, `sbm`, `planar`, and `tree`. It does not provide ready-made
Ego-small or Grid configurations. Molecular DeFoG support is intentionally not
advertised until the wrapper preserves atom and bond attributes in its neutral
export.

### Train DeFoG

```python
from pathlib import Path

from grapher.models import DatasetReference, RunSpec, TrainRequest, create_baseline

run = RunSpec.for_seed(
    model_id="defog",
    dataset_id="community_small",
    seed=42,
    output_root=Path("outputs/baselines"),
)

dataset = DatasetReference(
    benchmark_id="community_small",
    root=Path("outputs/datasets"),
    serialized_id="sbm",
    native_id="comm20",
)

wrapper = create_baseline("defog")
training = wrapper.train(
    TrainRequest(
        run=run,
        dataset=dataset,
        options={
            "experiment": "comm20",
            "runtime": {"gpus": 1},
        },
    )
)
```

The training worker converts the trusted GraphER split pickles into DeFoG's
adjacency-tensor format, invokes the upstream Hydra entry point, and publishes
the selected checkpoint, resolved configuration, logs, checksums, source
revision, and dataset-conversion manifest.

### Generate a raw batch

```python
from grapher.models import GenerateRequest

generation = wrapper.generate(
    GenerateRequest(
        run=run,
        checkpoint_path=training.checkpoint_path,
        num_graphs=1024,
        generation_seed=7,
        options={
            "native_dataset": "comm20",
            "sampling": {
                "sample_steps": 1000,
                "batch_size": 64,
            },
            "runtime": {"device": "cuda"},
        },
    )
)
```

Generation validates the sample count and order, checks generic graph shape and
adjacency constraints, and saves both `base_graphs.pkl` and a numeric NPZ
export. It fails rather than silently filtering a malformed or incomplete
batch.

See [`docs/DEFOG_WRAPPER.md`](docs/DEFOG_WRAPPER.md) for the complete option
reference.

## Optional DH-VAE + Havel--Hakimi baseline

The project-owned DH-VAE, degree samplers, and randomized Havel--Hakimi
constructors are isolated under `src/grapher/models/dhvae_hh/`. This pipeline
is retained as an optional weak, degree-exact base for correction experiments;
it is not part of the general GraphER corrector.

The common DH-VAE+HH wrapper orchestration is still pending. Its existing
training, evaluation, and construction scripts are retained as compatibility
entry points. New experiment orchestration should treat their completed graphs
exactly like outputs from any other frozen base and should pass them through
the common correction interface.

## Run the current GraphER corrector

The generic predictor and randomized greedy refiner are implemented under
`src/grapher/rewiring_mlp/generic/`. The current DeFoG correction entry point is:

```bash
PYTHONPATH=src python scripts/run_defog_grapher.py \
  --config configs/experiments/grapher/community_small_defog_corrector.yaml \
  --defog-checkpoint /absolute/path/to/comm20.ckpt \
  --checkpoint /absolute/path/to/grapher-checkpoint.pt \
  --output-dir outputs/corrections/defog/community_small/seed_42 \
  --num-generate 1024 \
  --seed 42
```

This compatibility script combines base sampling and correction. The unified
experiment runner should instead consume an already serialized
`base_graphs.pkl`, ensuring that all ablations use exactly the same raw batch.

## Paired evaluation protocol

GraphER must be evaluated on the exact same serialized graphs before and after
correction:

1. Generate and save one ordered raw batch for each dataset, base, and seed.
2. Evaluate the raw batch against the held-out reference set.
3. Apply GraphER to every raw graph, using identity fallback where correction
   is unsupported or no move is accepted.
4. Evaluate the corrected batch with the same reference graphs and evaluator
   configuration.
5. Report raw and corrected values, paired seed-level differences, correction
   coverage, invariant checks, diversity, candidate budget, and runtime.

Published standalone baseline values must not be used as the before-value in a
paired GraphER comparison.

Degree MMD is a preservation control for generic degree-preserving correction:
raw and corrected batches should have identical degree statistics, up to
numerical precision. Improvements are expected in complementary higher-order
statistics such as clustering, orbit, and graphlet distributions.

## Verification

Run the wrapper and artifact tests:

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_baseline_model_wrappers.py \
  tests/test_defog_common_wrapper.py \
  tests/test_defog_wrapper.py
```

Run static checks:

```bash
ruff check src scripts tests
ruff format --check src scripts tests
python -m compileall -q src scripts tests
```

The full test suite requires the optional dependencies used by molecular and
legacy compatibility paths:

```bash
PYTHONPATH=src python -m pytest -q
```

## Known limitations

- The maintained generic corrector currently predicts graphlet summaries; the
  full multi-component structural-summary model is not yet implemented.
- The common wrappers for DiGress, CatFlow, HOG-Diff, FLAGG, and DH-VAE+HH are
  incomplete.
- The DeFoG wrapper currently supports only generic graph configurations.
- A degree-preserving corrector cannot fix the base model's degree error.
- Finite candidate budgets can miss valid improving swaps.
- Graphlet summaries do not uniquely determine a graph.
- Exact graphlet counting and dense pair features limit scalability.
- Returning a prediction after every accepted move prevents a global
  monotonic-energy or convergence guarantee.

## Additional documentation

- [Baseline wrapper contract](docs/BASELINE_MODEL_WRAPPERS.md)
- [DeFoG wrapper](docs/DEFOG_WRAPPER.md)
- [DH-VAE+HH package boundary](docs/DHVAE_HH_PACKAGE.md)
- [Rewiring MLP package boundary](docs/REWIRING_MLP_PACKAGE.md)
- [Implementation audit](docs/IMPLEMENTATION_AUDIT.md)
