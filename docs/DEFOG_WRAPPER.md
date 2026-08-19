# DeFoG baseline wrapper

`grapher.models.defog.DeFoGWrapper` implements the common baseline `train()`
and `generate()` interface for generic graphs and attributed QM9/ZINC
molecules. The upstream DeFoG repository remains external and runs in its own
Python environment.

## Environment

```bash
export DEFOG=/absolute/path/to/DeFoG
export DEFOG_PYTHON=/absolute/path/to/defog-env/bin/python
export PYTHONPATH=src
```

The DeFoG environment must contain the dependencies required by the selected
upstream experiment. Molecular conversion additionally requires RDKit, Torch,
and PyTorch Geometric in that environment.

## Supported datasets

| GraphER benchmark | Prepared directory | DeFoG profile | Scope |
| --- | --- | --- | --- |
| `community_small` | `sbm` in the legacy layout | `comm20` | Generic topology |
| `ego_small` | `ego_small` | `comm20` compatibility profile | Generic topology |
| `qm9` | `qm9_attributed` | dataset `qm9`, experiment `qm9_no_h` | Attributed molecules |
| `zinc` | `zinc` | dataset/experiment `zinc` | Attributed molecules |

Community-small resolves to `comm20` automatically. Ego-small also uses
DeFoG's data-driven generic loader through the `comm20` profile because the
attached source has no complete Ego-specific Hydra profile. The manifest keeps
`benchmark_id="ego_small"` and `native_id="comm20"` separate. This is a
loader compatibility choice, not a claim that the datasets are equivalent;
final metrics must use GraphER's declared Ego-small evaluator. Grid has no
declared compatibility profile.

## Training

```python
from pathlib import Path

from grapher.models import DatasetReference, RunSpec, TrainRequest, create_baseline

run = RunSpec.for_seed(
    model_id="defog",
    dataset_id="community_small",
    seed=42,
    run_id="replicate_a",
    output_root=Path("outputs/baselines"),
)
dataset = DatasetReference(
    benchmark_id="community_small",
    root=Path("outputs/datasets"),
    serialized_id="sbm",
)
training = create_baseline("defog").train(
    TrainRequest(
        run=run,
        dataset=dataset,
        options={
            "runtime": {
                "gpus": 1,
                "cuda_visible_devices": "0",
                "progress": {
                    "enabled": True,
                    "stream_output": True,
                    "interval_seconds": 15,
                    "epoch_interval": 1000,
                    "generation_batch_interval": 1,
                },
            },
            "training_estimates": {
                "enabled": True,
                "seed": 43,
                "num_graphs": 1024,
                "sampling": {"batch_size": 64},
                "runtime": {"device": "cuda:0"},
            },
        },
    )
)
```

The wrapper converts the immutable GraphER splits into DeFoG-native tensors,
launches Hydra with a direct argument list and `shell=False`, and atomically
publishes the final checkpoint, resolved configuration, converted data,
provenance, and log. Sampling settings omitted from the request are inherited
from the saved DeFoG configuration rather than replaced by generic defaults.

When `runtime.progress.enabled` is true, the wrapper mirrors the persisted
subprocess log to stderr and emits a heartbeat after
`runtime.progress.interval_seconds`. The training worker additionally prints a
stable epoch summary at the first, final, and every configured
`epoch_interval`; when that value is omitted, it automatically emits roughly
one hundred summaries over the full training horizon. Generation reports the
completed sample count every `generation_batch_interval` batches. These
messages do not alter stdout or the artifact manifests.

The reference DeFoG entrypoint declares a DDP strategy even for one device.
The isolated GraphER training worker replaces that strategy with Lightning's
single-device strategy whenever `devices=1`; this avoids creating an
unnecessary NCCL process group while leaving DeFoG's model, optimizer, data,
and training horizon unchanged. Multi-GPU training remains unsupported and is
rejected by the wrapper.

Before training, the worker prints and stores `runtime_diagnostics.json` with
the selected interpreter, PyTorch/CUDA versions, visible devices, device
properties, NVIDIA driver query, and the effective single-device policy. If a
subprocess fails, GraphER preserves `train.log`, the runtime diagnostics, and a
machine-readable failure record under
`<run>/failures/attempt-<id>/` before deleting its temporary workspace.

Post-training estimates are independent unconditional samples, not
reconstructions of same-index training graphs. The wrapper therefore saves the
estimate pool and the exact source training pool, records
`pairing.status: unpaired`, and never invents index alignment. Generic runs
default to one estimate per training graph; molecular runs default to at most
1,024 estimates to avoid an unexpectedly large post-training job. Set
`training_estimates.num_graphs` explicitly for the final protocol. For generic GraphER
correction, these completed estimates are consumed by
`configs/experiments/grapher/community_small_defog_rewiring_mlp.yaml`; the
trainer constructs and reports its own same-size degree-profile coupling before
building structural-summary teacher trajectories.

### QM9 and ZINC

Only the dataset reference and run identity change:

```python
qm9_run = RunSpec.for_seed(
    model_id="defog", dataset_id="qm9", seed=42, run_id="seed_42"
)
qm9_data = DatasetReference(
    benchmark_id="qm9",
    serialized_id="qm9_attributed",
    root=Path("outputs/datasets"),
)
qm9_training = create_baseline("defog").train(
    TrainRequest(run=qm9_run, dataset=qm9_data)
)

zinc_run = RunSpec.for_seed(
    model_id="defog", dataset_id="zinc", seed=42, run_id="seed_42"
)
zinc_data = DatasetReference(
    benchmark_id="zinc",
    serialized_id="zinc",
    root=Path("outputs/datasets"),
)
zinc_training = create_baseline("defog").train(
    TrainRequest(run=zinc_run, dataset=zinc_data)
)
```

QM9 preserves atomic numbers and bond types 1--4. Its report-facing preparer
requires the official uncharacterized list and verifies the canonical
130,831-molecule pool before splitting; development inputs must be explicitly
marked noncanonical. Because DeFoG's QM9 state contains only atom and bond
categories, preparation keeps every canonical molecule, records the exact
number of source formal-charge/stereo annotations, and explicitly projects
those unsupported channels rather than silently filtering molecules. The
DeFoG converter requires this versioned audit marker on every graph, so older
QM9 pickles must be regenerated instead of being mislabeled as canonical. The
attached DeFoG ZINC
model uses a Kekule representation with bond types 1--3, whereas the prepared
GraphER source may use aromatic type 4. Conversion verifies every ZINC graph
against its recorded `source_smiles` and performs deterministic RDKit
kekulization; it never guesses a bond assignment. The wrapper saves both:

- `ground_truth_graphs.pkl`: the exact GraphER source representation; and
- `ground_truth_model_view.pkl`: the identity-ordered DeFoG representation.

The fixed ZINC preparation protocol rejects every molecule containing a
formally charged atom, including net-neutral zwitterions, because DeFoG's
categorical state has no formal-charge channel. Existing ZINC splits prepared
before this rule was introduced must be regenerated. Preparation also rejects
any unsupported stereo state present in the serialized graph; the declared
GraphER benchmark representation itself omits stereochemistry. Generated
molecules are saved without filtering or repair, so validity remains an
evaluation result. Molecular node-count, atom, bond, and valency statistics are
recomputed from the converted GraphER splits and applied consistently during
training and checkpoint sampling.

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
            "sampling": {"batch_size": 64},
            "runtime": {"device": "cuda:0"},
        },
    )
)
```

The wrapper recovers the dataset profile, converted dataset, and resolved
configuration from the matching training run. It requires exactly the
requested number of outputs and preserves their raw order. The common
manifest records both explicit sampling overrides and the effective values
after composing the saved DeFoG configuration. The common `base_graphs.pkl`,
neutral numeric NPZ, log, checksums, and manifest are written under:

```text
outputs/baselines/defog/<benchmark>/<run_id>/
├── run.json
├── train/
│   ├── checkpoints/model.ckpt
│   ├── dataset_conversion.json
│   ├── manifest.json
│   ├── native_dataset/
│   ├── resolved_config.yaml
│   ├── runtime_diagnostics.json
│   ├── train.log
│   └── training_estimates/
│       ├── estimated_graphs.pkl
│       ├── ground_truth_graphs.pkl
│       ├── ground_truth_model_view.pkl  # molecular runs only
│       ├── manifest.json
│       └── native/
├── failures/                         # present only after failed attempts
│   └── attempt-<id>/
│       ├── failure.json
│       ├── runtime_diagnostics.json
│       └── train.log
└── generations/<generation_id>/
    ├── base_graphs.pkl
    ├── generate.log
    ├── manifest.json
    └── native/
```

## Run identifiers

`run_id` is independent of `train_seed`. Two runs with the same seed but
different safe identifiers, such as `replicate_a` and `replicate_b`, publish to
separate directories. Every manifest records both values. Generation IDs are
also scoped by run, so the same `seed_7_n_1024` generation name can be used in
both runs without collision.

Training overwrite is rejected when a run already contains generated batches;
use a new `run_id` to keep every raw batch tied to the checkpoint that produced
it.

## GPU runtime troubleshooting

For a normal one-GPU run, `train.log` must contain both messages below before
the first training batch:

```text
[GraphER/DeFoG] Disabled one-device DDP: ... -> 'auto', devices=1.
[GraphER/DeFoG] Effective Lightning runtime: strategy=SingleDeviceStrategy, ...
```

If an error contains `Driver/library version mismatch`, run `nvidia-smi` in the
same shell. If that command also fails, the loaded NVIDIA kernel module and the
user-space NVML library are inconsistent; the server driver installation must
be reconciled, often by an administrator and a reboot after a driver update.
This is independent of the graph dataset and Hydra configuration. The wrapper
reports this classification explicitly and records the complete evidence in
the failure directory.
