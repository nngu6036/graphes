# HOG-Diff baseline wrapper

GraphER integrates the external HOG-Diff implementation as an isolated baseline
wrapper. The upstream checkout is not vendored or modified. Training and
sampling run under the Python interpreter declared by `HOGDIFF_PYTHON`, while
the GraphER process owns dataset identity, immutable splits, run artifacts, and
the final NetworkX serialization.

## Supported GraphER benchmarks

| GraphER benchmark | Serialized GraphER data | HOG-Diff native id | Upstream config |
|---|---|---|---|
| `community_small` | `sbm` | `community_small` | `cs.yaml` |
| `ego_small` | `ego_small` | `ego_small` | `ego.yaml` |
| `qm9` | `qm9_attributed` | `qm9` | `qm9.yaml` |
| `zinc` | `zinc` | `zinc250k` | `zinc250k.yaml` |

QM9 and ZINC use GraphER's heavy-atom, implicit-hydrogen representation. The
adapter supports the atom vocabularies in the supplied HOG-Diff release and
single/double/triple Kekule bond categories.

## External environment

Set the source root and, preferably, a dedicated Python environment containing
HOG-Diff's dependencies:

```bash
export HOGDIFF=/path/to/HOG-Diff
export HOGDIFF_PYTHON=/path/to/hogdiff-env/bin/python
```

The interpreter is checked before a run and must import PyTorch,
PyTorch-Geometric, RDKit, PyYAML, `easydict`, and `wandb`. The wrapper sets
`WANDB_MODE=disabled`; no online Weights & Biases service is required.

The supplied HOG-Diff archive imports `_GENERIC_DATASETS` and `_MOL_DATASETS`
from `data.py` but does not include that module. Isolated GraphER workers contain
a narrow compatibility shim defining only those two constants. If a future
HOG-Diff checkout provides its own `data.py`, that upstream file takes
precedence.

## Training lifecycle

HOG-Diff is a two-model baseline and must not be flattened into one training
stage. GraphER therefore executes:

1. higher-order / VPSDE score-model training (`mode=higher-order`);
2. conditional OU-bridge score-model training (`mode=OU`);
3. final checkpoint publication containing both HOG-Diff states.

The upstream molecular YAMLs can enable `snapshot_sampling`, which performs
sampling/evaluation while training and can select snapshots using held-out
metrics. The GraphER wrapper forces `snapshot_sampling: false` for both stages
and saves the final configured states explicitly. Test evaluation is performed
only after training by GraphER's common evaluator.

## Split adaptation

For generic graphs, the supplied HOG-Diff loader interprets the first
`int(test_split * N)` graphs in `<dataset>.pkl` as test and all remaining graphs
as training. The wrapper writes the native file as:

```text
GraphER test split + GraphER training split
```

and computes a `test_split` value that selects exactly the GraphER test prefix.
GraphER validation graphs are retained for provenance but are excluded from the
HOG-Diff optimizer.

For QM9 and ZINC, the wrapper bypasses the upstream raw CSV/index split path. It
converts only the frozen GraphER training split to HOG-Diff atom/bond tensors,
then lets the isolated worker materialize `processed/atom_bond.pt`. Validation
and test graphs are stored beside the projection only for provenance and are
not supplied to the optimizer.

## Raw generation protocol

Sampling preserves HOG-Diff's native two-stage sequence: the higher-order
sampler produces the conditioning state used by the OU sampler. The GraphER
worker intercepts the raw OU outputs before the upstream evaluation callback.
This is important for molecular benchmarking because HOG-Diff's normal
molecular evaluation path can correct molecules with MoFlow/RDKit utilities.
GraphER exports the uncorrected atom/bond tensors as the baseline sample, then
converts them to NetworkX for the common evaluator.

Generation is batched but always truncated and checked to return the exact
requested sample count. The wrapper records the neutral NPZ hash, checkpoint
hash, source-code fingerprint, Python identity, resolved HOG-Diff YAML,
dataset-projection manifest, and GraphER generation manifest.

## Commands

Community-small:

```bash
PYTHONPATH=src:. python scripts/run_hog_diff_baseline.py \
  --dataset community_small \
  --num-samples 1024 \
  --seed-id 42 \
  --wrapper-config configs/baselines/hog_diff_community_small.yaml \
  --skip-training-estimates
```

Ego-small:

```bash
PYTHONPATH=src:. python scripts/run_hog_diff_baseline.py \
  --dataset ego_small \
  --num-samples 1024 \
  --seed-id 42 \
  --wrapper-config configs/baselines/hog_diff_ego_small.yaml \
  --skip-training-estimates
```

QM9:

```bash
PYTHONPATH=src:. python scripts/run_hog_diff_baseline.py \
  --dataset qm9 \
  --num-samples 1024 \
  --seed-id 42 \
  --wrapper-config configs/baselines/hog_diff_qm9.yaml \
  --skip-training-estimates
```

ZINC:

```bash
PYTHONPATH=src:. python scripts/run_hog_diff_baseline.py \
  --dataset zinc \
  --num-samples 1024 \
  --seed-id 42 \
  --wrapper-config configs/baselines/hog_diff_zinc.yaml \
  --skip-training-estimates
```

Use `--ho-iters` / `--ou-iters` for controlled smoke runs, and
`--ho-batch-size` / `--ou-batch-size` if the upstream defaults do not fit the
available GPU. HOG-Diff's training loop includes both endpoints of
`range(initial_step, n_iters + 1)`, so `--*-iters 0` still performs the initial
optimization iteration in a fresh run.

## Artifact layout

A completed run follows the same contract as other GraphER baselines:

```text
outputs/baselines/hog_diff/<dataset>/<run_id>/
├── train/
│   ├── checkpoints/hog_diff.pth
│   ├── manifest.json
│   ├── resolved_config.yaml
│   ├── native_dataset/
│   ├── higher_order_worker_manifest.json
│   ├── ou_worker_manifest.json
│   └── train.log
└── generations/<generation_id>/
    ├── base_graphs.pkl
    ├── manifest.json
    └── native/
        ├── hog_diff_samples.npz
        ├── hog_diff_manifest.json
        └── generate.log
```

Generation verifies the checkpoint, source-code fingerprint, dataset identity,
and external Python identity recorded at training time. A source/environment
change therefore requires a new run rather than silently mixing artifacts.
