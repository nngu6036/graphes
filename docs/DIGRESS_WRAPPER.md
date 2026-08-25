# DiGress baseline wrapper

`grapher.models.digress.DiGressWrapper` implements the common GraphER
`train()` and `generate()` contract for the attached external DiGress source
snapshot. The upstream repository remains external and executes in its own
Python environment. GraphER communicates with it through command-line worker
processes and a validated numeric NPZ export rather than importing DiGress into
the GraphER process.

## Environment

```bash
export DIGRESS=/absolute/path/to/DiGress
export DIGRESS_PYTHON=/absolute/path/to/digress-env/bin/python
export PYTHONPATH=src
```

The DiGress environment must contain the dependencies required by the attached
source tree, including PyTorch, PyTorch Lightning, Hydra/OmegaConf,
PyTorch-Geometric, NetworkX, and the model-specific scientific packages. QM9
additionally requires RDKit.

The wrapper does not require the DiGress and GraphER environments to use the
same package versions. All upstream imports occur in the isolated interpreter.

## Supported datasets

| GraphER benchmark | Prepared directory | Native DiGress profile | Scope |
| --- | --- | --- | --- |
| `community_small` | `outputs/datasets/sbm` | dataset/experiment `comm20` | Generic topology |
| `ego_small` | `outputs/datasets/ego_small` | `comm20` compatibility architecture | Generic topology |
| `grid` | `outputs/datasets/grid` | `planar` compatibility architecture | Generic topology |
| `qm9` | `outputs/datasets/qm9_attributed` | dataset `qm9`, experiment `qm9_no_h` | Heavy-atom attributed molecules |

The attached DiGress snapshot has no ZINC dataset or experiment configuration,
so `DiGressWrapper` deliberately rejects ZINC instead of silently substituting
another molecular benchmark.

Ego-small and Grid use the attached source's data-driven generic loader with a
declared architecture profile. The wrapper still trains exclusively on the
immutable GraphER split and records the report-facing benchmark separately
from the native compatibility profile.

## Package layout

All DiGress-specific integration code is under the model package, following the
DeFoG package convention:

```text
src/grapher/models/digress/
├── __init__.py
├── wrapper.py
├── backend.py
├── codec.py
├── runtime.py
└── workers/
    ├── common.py
    ├── prepare_dataset.py
    ├── prepare_molecular_dataset.py
    ├── train.py
    └── export.py
```

The public orchestration entrypoint remains model-specific, while evaluation
uses the repository's common artifact-driven evaluators:

```text
scripts/run_digress_baseline.py
scripts/evaluate_graph_generation_report.py
scripts/evaluate_generated_molecules.py
```

## Design and source-specific safeguards

The attached upstream entrypoint is not invoked directly. Its `main.py`
imports optional packages unconditionally and configures DDP even for a
single-device run. The isolated training worker instead composes the same
Hydra experiment, constructs the upstream data module and
`DiscreteDenoisingDiffusion` model, and launches Lightning with a single-device
strategy.

The wrapper also addresses these source-specific issues:

- GraphER split files are converted without modifying the original
  `train.pkl`, `val.pkl`, or `test.pkl` artifacts.
- Generic processed PyG files are written directly because the attached
  `SpectreGraphDataset.process()` duplicates each graph in this source
  snapshot.
- Expensive validation-time graph sampling, visualization, and final
  `trainer.test()` generation are disabled during managed training. The
  validation likelihood loop remains available at the configured cadence.
- QM9 node-count, atom-type, edge-type, and valency priors are recomputed from
  the exact converted GraphER splits rather than copied from bundled DiGress
  data.
- Generation calls the upstream model's `sample_batch()` method, exports only
  numeric arrays, then decodes and validates them in GraphER.
- Generated samples are not filtered, repaired, or silently replaced. Invalid
  or disconnected raw outputs remain evaluation outcomes.

## Configuration files

Maintained wrapper configurations are available under `configs/baselines/`:

```text
digress_default.yaml
digress_community_small.yaml
digress_ego_small.yaml
digress_grid.yaml
digress_qm9.yaml
```

A wrapper configuration has a top-level `digress` section. For example:

```yaml
digress:
  native_dataset: comm20
  experiment: comm20
  n_epochs: 200000
  batch_size: 256
  num_workers: 0
  check_val_every_n_epochs: 1000
  save_every_n_epochs: 10000
  generation_batch_size: 64

  training_estimates:
    enabled: true

  runtime:
    gpus: 1
    cuda_visible_devices: "0"
    device: auto
    progress:
      enabled: true
      stream_output: true
      interval_seconds: 15
      epoch_interval: 2000
      generation_batch_interval: 1
```

CLI values such as `--n-epochs` and `--batch-size` override the corresponding
YAML values. `--n-epochs` is the total Lightning horizon; when resuming, it is
not interpreted as an additional number of epochs.

The isolated workers install in-memory compatibility replacements for
DiGress's CUDA boolean-index assignments in `src.utils.encode_no_edge()`,
`src.diffusion.diffusion_utils.sample_discrete_features()`,
`src.diffusion.diffusion_utils.mask_distributions()`, and the discrete model's
`reconstruction_logp()`. The replacements use device-local broadcast masks and
leave the external DiGress checkout unchanged. This avoids PyTorch 2.0.x
`Indexing.cu` size assertions during training, validation, and generation.

## Train and generate

### Community-small

```bash
PYTHONPATH=src python scripts/run_digress_baseline.py \
  --dataset community_small \
  --num-samples 1024 \
  --seed-id 42 \
  --run-id seed_42_e200k \
  --wrapper-config configs/baselines/digress_community_small.yaml
```

### Ego-small

```bash
PYTHONPATH=src python scripts/run_digress_baseline.py \
  --dataset ego_small \
  --num-samples 1024 \
  --seed-id 42 \
  --run-id seed_42_e200k \
  --wrapper-config configs/baselines/digress_ego_small.yaml
```

### Grid

```bash
PYTHONPATH=src python scripts/run_digress_baseline.py \
  --dataset grid \
  --num-samples 1024 \
  --seed-id 42 \
  --run-id seed_42_e100k \
  --wrapper-config configs/baselines/digress_grid.yaml
```

### QM9

```bash
PYTHONPATH=src python scripts/run_digress_baseline.py \
  --dataset qm9 \
  --num-samples 1024 \
  --seed-id 42 \
  --run-id seed_42_e1000 \
  --wrapper-config configs/baselines/digress_qm9.yaml
```

The runner prints human-readable progress to stderr and one final JSON artifact
summary to stdout. Useful CLI overrides include:

```text
--n-epochs
--batch-size
--generation-batch-size
--num-workers
--check-val-every-n-epochs
--save-every-n-epochs
--resume-from
--skip-training-estimates
--training-estimate-count
--epoch-progress-interval
```

## Evaluation

DiGress publishes the same managed `base_graphs.pkl` and `manifest.json`
contract as the other baseline wrappers. Evaluation therefore uses the common
model-agnostic scripts directly; there is no DiGress-specific evaluator.

### Generic graphs

```bash
GEN_DIR="outputs/baselines/digress/community_small/seed_42_e200k/generations/seed_42_n_1024"

PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/grapher/community_small_topology_graphlet.yaml \
  --generated-dir "$GEN_DIR" \
  --output-dir "$GEN_DIR/evaluation_report"
```

The evaluator auto-detects `base_graphs.pkl` and reads `model_id: digress` from
the generation manifest. For Ego-small or Grid, change the generation path and
use the matching `ego_small_topology_graphlet.yaml` or
`grid_topology_graphlet.yaml` configuration. Use `--max-graphs` only for a
smoke evaluation.

### QM9

```bash
GEN_DIR="outputs/baselines/digress/qm9/seed_42_e1000/generations/seed_42_n_1024"

PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-graphs "$GEN_DIR/base_graphs.pkl" \
  --dataset-root outputs/datasets \
  --dataset qm9_attributed \
  --reference-split test \
  --train-split train \
  --output-dir "$GEN_DIR/evaluation" \
  --fcd-device auto
```

This reports RDKit validity, uniqueness, novelty, NSPDK, and FCD when its
optional dependency is available. Add `--skip-fcd` when the environment lacks
a compatible FCD package.

## Artifact layout

```text
outputs/baselines/digress/<benchmark>/<run_id>/
├── run.json
├── train/
│   ├── checkpoints/model.ckpt
│   ├── dataset_conversion.json
│   ├── manifest.json
│   ├── molecular_statistics.json       # QM9 only
│   ├── native_dataset/
│   ├── resolved_config.yaml
│   ├── train.log
│   └── training_estimates/
│       ├── estimated_graphs.pkl
│       ├── ground_truth_graphs.pkl
│       ├── ground_truth_model_view.pkl  # QM9 only
│       ├── manifest.json
│       └── native/
├── failures/                            # created only after failed attempts
│   └── attempt-<id>/
└── generations/<generation_id>/
    ├── base_graphs.pkl
    ├── generate.log
    ├── manifest.json
    └── native/
        ├── digress_samples.npz
        └── digress_manifest.json
```

The post-training estimate pool is an independent unconditional sample, not an
index-aligned reconstruction of the training set. Its manifest therefore
records `pairing.status: unpaired`. A separate declared matching operation must
create Rewiring-MLP supervision pairs.

## Neutral export schema

The isolated exporter writes format `digress_graph_batch_v1` with numeric
arrays:

```text
node_offsets
node_types
edge_offsets
edge_endpoints
edge_types
```

`allow_pickle=False` is used when GraphER loads this file. The decoder verifies
graph counts, offsets, node/edge vocabularies, endpoint ranges, duplicate
edges, and self-loops before constructing the ordered `base_graphs.pkl` batch.

## Limitations

- Only the attached discrete DiGress model is wrapped; the continuous model is
  outside this integration.
- ZINC, MOSES, GuacaMol, and explicit-hydrogen QM9 are not exposed by the
  current GraphER runner.
- Grid and Ego-small use declared compatibility architectures because this
  source snapshot has no native profiles for those benchmark names.
- A complete GPU train-and-generate run still depends on a working external
  DiGress environment; repository unit tests validate the wrapper boundary and
  neutral codec without reproducing a full long-running training job.
