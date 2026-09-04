# GDSS baseline wrapper

GraphER integrates the attached **GDSS** implementation as an isolated external
baseline. The wrapper preserves GDSS's joint score-based diffusion over node
features and adjacency while enforcing GraphER's common split, artifact, and
raw-generation contracts.

## Supported datasets

| GraphER benchmark | Prepared directory | GDSS native id | Training config | Sampling config |
|---|---|---|---|---|
| Community-small | `sbm` | `community_small` | `community_small.yaml` | embedded |
| Ego-small | `ego_small` | `ego_small` | `ego_small.yaml` | embedded |
| Grid | `grid` | `grid` | `grid.yaml` | embedded |
| QM9 | `qm9_attributed` | `QM9` | `qm9.yaml` | `sample_qm9.yaml` |
| ZINC | `zinc` | `ZINC250k` | `zinc250k.yaml` | `sample_zinc250k.yaml` |

QM9 uses heavy atoms C/N/O/F. ZINC uses C/N/O/F/P/S/Cl/Br/I. The supplied
release represents bonds as single, double, or triple categories.

## Environment

```bash
export GDSS=/path/to/GDSS
export GDSS_PYTHON=/path/to/gdss-env/bin/python
```

`GDSS_PYTHON` must provide the numerical runtime used by the released model
(PyTorch, NumPy, SciPy, NetworkX, PyYAML and tqdm). GraphER does **not** invoke
GDSS's native evaluator inside training or generation workers, so evaluation-only
`pyemd`/`dill` imports are bypassed. Final benchmark metrics are computed by the
common GraphER evaluator instead.

## Split protocol

The released GDSS loaders normally define their own train/test split. The wrapper
replaces only that loading boundary:

- GraphER `train.pkl` is the optimizer split;
- GraphER `val.pkl` is the training-monitor split;
- GraphER `test.pkl` is serialized for provenance but never loaded by the GDSS
  training worker.

GDSS's model architecture, SDEs, losses, optimizers, EMA, and score-network
training loop are unchanged. The managed checkpoint is the **final configured
epoch**. The attached checkpoint format does not save optimizer/scheduler state,
so managed resume is deliberately unsupported rather than emulated incorrectly.

## Raw molecular generation

The native `Sampler_mol` calls `gen_mol`, which applies `correct_mol` and can
replace a disconnected molecule by its largest connected component. That is not
appropriate for GraphER's paired raw-baseline evaluation. The GraphER worker
therefore stops earlier:

1. threshold GDSS atom logits exactly as the native sampler does;
2. append the virtual-node category and take the categorical argmax;
3. quantize adjacency to no/single/double/triple;
4. export that raw categorical graph directly.

No valence correction, RDKit repair, molecule filtering, or largest-component
rewrite is applied. Invalid outputs, including an empty molecular graph, remain
in the returned batch so validity is measured by the common evaluator rather
than hidden by sampler-side filtering.

## Commands

Community-small:

```bash
PYTHONPATH=src:. python scripts/run_gdss_baseline.py \
  --dataset community_small \
  --num-samples 1024 \
  --seed-id 42 \
  --wrapper-config configs/baselines/gdss_community_small.yaml
```

Ego-small:

```bash
PYTHONPATH=src:. python scripts/run_gdss_baseline.py \
  --dataset ego_small \
  --num-samples 1024 \
  --seed-id 42 \
  --wrapper-config configs/baselines/gdss_ego_small.yaml
```

QM9:

```bash
PYTHONPATH=src:. python scripts/run_gdss_baseline.py \
  --dataset qm9 \
  --num-samples 1024 \
  --seed-id 42 \
  --wrapper-config configs/baselines/gdss_qm9.yaml
```

ZINC:

```bash
PYTHONPATH=src:. python scripts/run_gdss_baseline.py \
  --dataset zinc \
  --num-samples 1024 \
  --seed-id 42 \
  --wrapper-config configs/baselines/gdss_zinc.yaml
```

Useful smoke-test overrides are `--num-epochs`, `--batch-size`,
`--generation-batch-size`, and `--device cpu`.

## Artifact layout

```text
outputs/baselines/gdss/<dataset>/<run_id>/
├── run.json
├── train/
│   ├── manifest.json
│   ├── resolved_config.yaml
│   ├── train.log
│   ├── training_worker_manifest.json
│   ├── native_dataset/
│   │   ├── manifest.json
│   │   ├── train.npz
│   │   ├── val.npz
│   │   └── test.npz
│   └── checkpoints/gdss.pth
└── generations/<generation_id>/
    ├── base_graphs.pkl
    ├── manifest.json
    ├── generate.log
    └── native/
        ├── gdss_samples.npz
        └── gdss_manifest.json
```

The training and generation manifests record source fingerprints, Python
identity, resolved configuration hashes, dataset hashes, sample order, and
checkpoint/export SHA-256 values.
