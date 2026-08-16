# DeFoG baseline wrapper

`grapher.models.defog.DeFoGWrapper` implements the common baseline `train()`
and `generate()` contract for generic graphs. The upstream repository remains
external and runs in its own Python environment.

## Environment

```bash
export DEFOG=/absolute/path/to/DeFoG
export DEFOG_PYTHON=/absolute/path/to/defog-env/bin/python
export PYTHONPATH=src
```

The attached upstream revision provides the generic native configurations
`comm20`, `sbm`, `planar`, and `tree`. Community-small should use
`native_id="comm20"`. Grid and attributed molecular datasets are rejected
until matching upstream configuration and an atom/bond-preserving neutral
export are implemented.

## Training

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
training = create_baseline("defog").train(
    TrainRequest(
        run=run,
        dataset=dataset,
        options={
            "experiment": "comm20",
            "n_epochs": 1_000_000,
            "batch_size": 256,
            "runtime": {"gpus": 1},
            "training_estimates": {
                "enabled": True,
                "seed": 42,
                # Defaults to the number of training graphs.
                "sampling": {"sample_steps": 1000, "batch_size": 64},
                "runtime": {"device": "cuda"},
            },
        },
    )
)
```

The wrapper converts the trusted GraphER split pickles to DeFoG adjacency
tensors inside the isolated environment, invokes Hydra with direct argument
lists and `shell=False`, and publishes the selected checkpoint, resolved
configuration, converted dataset, provenance, and log atomically. By default,
it then draws one independent sample per training graph from the final
checkpoint and saves both the estimate pool and an exact copy of `train.pkl`.

DeFoG is an unconditional generator: these samples are not reconstructions of
the correspondingly indexed training graphs. The estimates manifest therefore
records `pairing.status: unpaired` and `pair_count: 0`. A separate, declared
training-only matching procedure is required before these pools are used as
samplewise GraphER supervision. The wrapper never omits the ground-truth copy
for the attached DeFoG implementation because it exposes no source-training
indices.

## Generation

```python
from grapher.models import GenerateRequest

generation = create_baseline("defog").generate(
    GenerateRequest(
        run=run,
        checkpoint_path=training.checkpoint_path,
        num_graphs=1024,
        generation_seed=7,
        options={
            "native_dataset": "comm20",
            "sampling": {"sample_steps": 1000, "batch_size": 64},
            "runtime": {"device": "cuda"},
        },
    )
)
```

Generation produces an ordered `base_graphs.pkl`, a neutral numeric DeFoG NPZ
export, logs, hashes, and a manifest below:

```text
outputs/baselines/defog/community_small/seed_42/
├── run.json
├── train/
│   ├── checkpoints/model.ckpt
│   ├── dataset_conversion.json
│   ├── manifest.json
│   ├── native_dataset/
│   ├── resolved_config.yaml
│   ├── train.log
│   └── training_estimates/
│       ├── estimated_graphs.pkl
│       ├── ground_truth_graphs.pkl
│       ├── generate.log
│       ├── manifest.json
│       └── native/
│           ├── defog_manifest.json
│           └── defog_samples.npz
└── generations/seed_7_n_1024/
    ├── base_graphs.pkl
    ├── generate.log
    ├── manifest.json
    └── native/
        ├── defog_manifest.json
        └── defog_samples.npz
```

The wrapper fails if DeFoG does not return exactly the requested sample count.
It never filters raw outputs, so downstream raw-versus-GraphER evaluation can
use the exact same ordered batch.
