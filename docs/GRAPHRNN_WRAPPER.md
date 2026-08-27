# GraphRNN baseline wrapper

`grapher.models.graphrnn.GraphRNNWrapper` implements the common GraphES
`train()` and `generate()` interface for the attached GraphRNN source. It
supports generic, unlabelled, simple undirected graphs. It does not claim
support for QM9 or ZINC because the attached GraphRNN formulation generates
only topology and has no joint categorical atom--bond decoder.

## Why a compatibility worker is used

The attached GraphRNN snapshot is the original legacy implementation. Its
entry point assumes historical PyTorch/NetworkX APIs, invokes `.cuda()`
directly, and exchanges Python graph pickles. Forcing GraphES onto those old
versions would conflict with the maintained project environment.

The wrapper therefore uses an isolated subprocess and imports only the
upstream neural modules from `model.py`:

- `GRU_plain` for the graph-level recurrent state;
- `GRU_plain` for the dependent edge-sequence decoder in `GraphRNN_RNN`; and
- `MLP_plain` for the independent edge decoder in `GraphRNN_MLP`.

The GraphES worker preserves the attached implementation's random node
permutation, BFS ordering, truncated adjacency-sequence encoding, weighted
sampling with replacement, optimizer schedule, autoregressive Bernoulli
sampling, and all-zero-row trimming. It replaces only obsolete runtime APIs and
hard-coded device operations. The legacy loop's periodic sample-file dumping
is omitted because it is not part of the optimization objective; requested
evaluation batches are generated explicitly through the managed `generate()`
operation instead.

No NetworkX pickle crosses the subprocess boundary. Prepared data and generated
samples use validated, topology-only numeric NPZ files with `allow_pickle=False`.

## Environment

Extract the attached `GraphRNN.zip`, then identify its root—the directory that
contains `model.py`, `data.py`, `args.py`, and `README.md`:

```bash
unzip GraphRNN.zip -d external/GraphRNN
export GRAPHRNN="$PWD/external/GraphRNN"
export GRAPHRNN_PYTHON="$(command -v python)"
export PYTHONPATH=src
```

`GRAPHRNN_PYTHON` needs current `torch` and `numpy`; it does not need the
legacy versions listed by the attached repository because the old training
entry point is not executed. The source root and hashes of key upstream files
are recorded in the training manifest and verified again before generation.

Equivalent command-line options are `--graphrnn-root` and
`--graphrnn-python`.

## Supported datasets and configurations

| GraphES benchmark | Prepared directory | Configuration |
| --- | --- | --- |
| `community_small` | `outputs/datasets/sbm` | `configs/baselines/graphrnn_community_small.yaml` |
| `ego_small` | `outputs/datasets/ego_small` | `configs/baselines/graphrnn_ego_small.yaml` |
| `grid` | `outputs/datasets/grid` | `configs/baselines/graphrnn_grid.yaml` |

The configurations also contain the `dataset`, `protocol`, and `evaluation`
sections required by `evaluate_graph_generation_report.py`, so the same file
is used for training and evaluation.

For Community-small and Ego-small, the configured look-back width covers the
entire strict lower triangle. Grid uses a bounded width of 40, preserving the
intended GraphRNN BFS compression. Changing `max_prev_node` changes the model
representation and must be treated as a separate run.

## Prepare a dataset

Prepared split files are immutable inputs to the wrapper:

```bash
PYTHONPATH=src python scripts/prepare_generic_dataset.py \
  --dataset community_small \
  --root outputs/datasets
```

Use `ego_small` or `grid` for the other supported benchmarks.

## Train and generate

The baseline runner follows the same train-then-generate convention as the
DeFoG and DiGress runners:

```bash
PYTHONPATH=src python scripts/run_graphrnn_baseline.py \
  --dataset community_small \
  --num-samples 1024 \
  --seed-id 42 \
  --run-id seed_42 \
  --wrapper-config configs/baselines/graphrnn_community_small.yaml
```

The paper-default dependent-output variant is `GraphRNN_RNN`. A controlled
`GraphRNN_MLP` run can be launched without editing YAML:

```bash
PYTHONPATH=src python scripts/run_graphrnn_baseline.py \
  --dataset community_small \
  --num-samples 1024 \
  --seed-id 42 \
  --run-id seed_42_mlp \
  --wrapper-config configs/baselines/graphrnn_community_small.yaml \
  --variant GraphRNN_MLP
```

Useful CLI overrides include:

```text
--n-epochs
--batch-size
--batch-ratio
--max-prev-node
--num-workers
--generation-batch-size
--sample-time
--device
--cuda-visible-devices
--timeout-seconds
--resume-from
--skip-training-estimates
```

`--n-epochs` is the total horizon, including when resuming. The legacy source
steps `MultiStepLR` after every mini-batch; the supplied configs retain that
behaviour with `scheduler_step_unit: batch`. Setting it to `epoch` is a new
ablation rather than reproduction of the attached training loop.

The optional post-training estimate pool is an independent unconditional
sample and is explicitly recorded as `pairing.status: unpaired`. Disable it for
baseline-only experiments with `--skip-training-estimates`.

## Generate another batch from a managed checkpoint

The common wrapper API can generate additional immutable batches without
retraining:

```python
from pathlib import Path

from grapher.models import GenerateRequest, RunSpec, create_baseline

run = RunSpec.for_seed(
    model_id="graphrnn",
    dataset_id="community_small",
    seed=42,
    run_id="seed_42",
)

generation = create_baseline("graphrnn").generate(
    GenerateRequest(
        run=run,
        checkpoint_path=Path(
            "outputs/baselines/graphrnn/community_small/seed_42/"
            "train/checkpoints/graphrnn.pt"
        ),
        num_graphs=1024,
        generation_seed=43,
    )
)
print(generation.graphs_path)
```

The generation ID defaults to `seed_43_n_1024`, so it coexists with the seed-42
batch under the same trained run.

## Evaluate

```bash
GEN_DIR="outputs/baselines/graphrnn/community_small/seed_42/generations/seed_42_n_1024"

PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/baselines/graphrnn_community_small.yaml \
  --generated-dir "$GEN_DIR" \
  --output-dir "$GEN_DIR/evaluation_report"
```

The evaluator reads `manifest.json`, finds `base_graphs.pkl`, and labels the row
`graphrnn_to_test`. It reports degree, clustering, and four-node orbit MMD using
the same prepared held-out split as the other generic baselines.

## Artifacts

```text
outputs/baselines/graphrnn/<benchmark>/<run_id>/
├── run.json
├── train/
│   ├── checkpoints/
│   │   ├── graphrnn.pt
│   │   └── graphrnn_epoch_<epoch>.pt
│   ├── loss_history.jsonl
│   ├── manifest.json
│   ├── native_dataset/
│   │   ├── graphrnn_dataset.npz
│   │   └── manifest.json
│   ├── resolved_config.yaml
│   ├── train.log
│   ├── worker_manifest.json
│   └── training_estimates/          # optional, explicitly unpaired
└── generations/<generation_id>/
    ├── base_graphs.pkl
    ├── generate.log
    ├── manifest.json
    └── native/
        ├── graphrnn_samples.npz
        ├── graphrnn_manifest.json
        └── graphrnn.log
```

The common batch preserves sample order and retains raw disconnected or empty
samples rather than filtering or replacing them. The manifest reports these
counts as diagnostics. GraphRNN's original decoder can temporarily represent
`max_num_node + 1` rows; the official all-zero-row/column trimming is applied
before publication, and the retained raw node count is stored on each graph.

## Reproducibility notes

- Use the same frozen `train.pkl`, `val.pkl`, and `test.pkl` files for every
  baseline seed.
- Keep `GraphRNN_RNN` and `GraphRNN_MLP` in separate run IDs.
- Record any change to BFS width, scheduler step unit, sampling retry count, or
  architecture as a distinct configuration.
- The wrapper does not copy or claim results from the original GraphRNN data
  split; it retrains on the GraphES split and records its SHA-256 fingerprint.
- Because GraphRNN is node-order autoregressive, generation cost grows with the
  maximum node count and, for `GraphRNN_RNN`, with the edge-sequence look-back
  width.
