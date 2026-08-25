# Baseline model wrapper contract

## Purpose

GraphER is evaluated as a post-generation corrector. Every upstream model must
therefore expose the same two operations:

1. train from the datasets prepared by this repository; and
2. generate and serialize an ordered batch of completed raw graphs.

This refactor standardizes how those operations are invoked and reported. It
does **not** change the GraphER predictor, candidate construction, swap validity
rules, or randomized greedy correction procedure.

## Package location

Wrappers live under `src/grapher/models/` rather than top-level `src/models/`.
The repository uses the `grapher.*` namespace, while some upstream projects
(notably DeFoG) use bare imports such as `models.*`. A top-level `models`
package could therefore shadow an upstream package inside an isolated worker.

The package contains adapters for third-party models and the implementation of
the project-owned DH-VAE+HH baseline. It must not vendor or duplicate the full
source of a third-party baseline.

```text
src/grapher/models/
├── base.py
├── artifacts.py
├── errors.py
├── registry.py
├── dhvae_hh/
│   ├── wrapper.py
│   ├── degree_vae.py
│   ├── typed_degree_vae.py
│   ├── degree_sampler.py
│   ├── havel_hakimi.py
│   ├── typed_constructor.py
│   ├── training.py
│   └── evaluation.py
├── digress.py
├── catflow.py
├── defog/
│   ├── __init__.py
│   ├── wrapper.py
│   ├── backend.py
│   ├── runtime.py
│   ├── molecular_codec.py
│   └── workers/
│       ├── export.py
│       ├── train.py
│       ├── prepare_dataset.py
│       ├── prepare_molecular_dataset.py
│       └── molecular_runtime.py
├── hog_diff.py
└── flagg.py
```

The project-owned DH-VAE+HH implementation is colocated with its wrapper. Thin
compatibility modules preserve its former import and CLI paths. Third-party
implementations remain external; for example, `DeFoGWrapper` delegates to
`grapher.models.defog.backend` rather than replace its validated subprocess and NPZ
export code.

## Public interface

Each wrapper subclasses `BaseGeneratorWrapper` and implements:

```python
def train(self, request: TrainRequest) -> TrainingArtifacts:
    ...

def generate(self, request: GenerateRequest) -> GenerationArtifacts:
    ...
```

`TrainRequest` contains a `DatasetReference` and a `RunSpec`.
`GenerateRequest` identifies the trained run, checkpoint, requested sample
count, and a generation seed. Model-specific fields belong in `options`; they
must not alter the common method signatures.

The canonical registry identifiers are:

- `dhvae_hh`
- `digress`
- `catflow`
- `defog`
- `hog_diff`
- `flagg`

The registry is lazy. Importing `grapher.models` must never import an upstream
project or require its environment.

## Dataset identity

`DatasetReference` distinguishes three names that must not be conflated:

- `benchmark_id`: paper/report name and output-path component;
- `serialized_id`: directory below `outputs/datasets`; and
- `native_id`: optional dataset key expected by an upstream project.

For example:

```python
DatasetReference(
    benchmark_id="community_small",
    serialized_id="sbm",
    native_id="comm20",
)
```

Training wrappers consume `train.pkl`, `val.pkl`, and `test.pkl` from the
prepared project dataset directory. They do not mutate these files. Their
manifest must include the combined dataset fingerprint returned by
`DatasetReference.fingerprint()`.

## Artifact layout

Training identity and generation identity are separate. Several raw batches may
be sampled from one checkpoint without overwriting one another.

```text
outputs/baselines/
└── <model_id>/
    └── <benchmark_id>/
        └── <run_id>/
            ├── run.json
            ├── train/
            │   ├── manifest.json
            │   ├── resolved_config.yaml
            │   ├── train.log
            │   ├── native_dataset/
            │   ├── training_estimates/
            │   │   ├── estimated_graphs.pkl
            │   │   ├── ground_truth_graphs.pkl
            │   │   ├── ground_truth_model_view.pkl  # molecular, when needed
            │   │   ├── manifest.json
            │   │   └── native/
            │   └── checkpoints/
            └── generations/
                └── <generation_id>/
                    ├── base_graphs.pkl
                    ├── manifest.json
                    ├── generate.log
                    └── native/
```

The default identifiers are:

```text
run_id        = seed_<training-seed>
generation_id = seed_<generation-seed>_n_<requested-count>
```

`base_graphs.pkl` is the common GraphER-facing representation. Native exports
may also be retained below `native/`, but downstream correction must use the
validated common batch.

Corrected outputs must be written separately, for example:

```text
outputs/corrections/<model>/<dataset>/<run>/<generation>/
```

The correction manifest must reference the raw `base_graphs.pkl` SHA-256,
sample count, and order. A raw batch is never silently filtered, replaced, or
overwritten because paired raw-versus-GraphER evaluation depends on using the
exact same inputs.

## Implementation requirements

Every completed wrapper must:

1. require the prepared GraphER dataset splits for training;
2. propagate separate training and generation seeds;
3. isolate incompatible upstream dependencies in a subprocess;
4. invoke subprocesses with an argument list and `shell=False`;
5. save the resolved configuration, logs, checkpoint, and provenance;
6. validate that exactly the requested number of graphs was returned;
7. preserve graph order, isolated nodes, and node/edge attributes;
8. retain disconnected or invalid raw samples rather than silently dropping
   them;
9. compute checkpoint, dataset, and graph-batch hashes; and
10. publish final artifacts atomically after validation.

When a base does not implement graph-specific reconstruction, generated
post-training samples must not be presented as naturally aligned with the
training graphs. Such a wrapper saves both pools and records
`pairing.status: unpaired`; a separate declared matching step must construct
the supervision pairs. Equal pool sizes alone do not establish correspondence. The maintained generic
matching step is implemented in `grapher.rewiring_mlp.generic.training_sources`:
it validates the completed-output manifest and checksum, partitions each base
pool deterministically, then performs one-to-one Hungarian matching within
exact node-count strata using normalized sorted-degree profiles. Structural
prediction targets are excluded from that cost and the complete coupling audit
is saved by `train_topology_grapher.py`.

DH-VAE + HH is a composite base. Its `train()` operation trains the DH-VAE;
randomized Havel--Hakimi is a stateless constructor invoked by `generate()`.

FLAGG manifests must additionally identify the FLAGG variant, insertion policy,
and one-shot filler model/configuration.

## Current status

The DH-VAE+HH wrapper implements training and exact-count generation for
generic graphs and attributed QM9/ZINC molecular graphs using the existing
project-owned trainer, invariant samplers, and constructors. The DeFoG wrapper
implements isolated training, post-training estimate export, and generation
for the same domains. Its neutral backend validates the dataset-specific
schema before publishing GraphER-facing NetworkX batches. DiGress, CatFlow,
HOG-Diff, and FLAGG remain explicit placeholders. Unimplemented methods raise
`BaselineNotImplementedError` before creating any directories; an incomplete
adapter must not leave a partial run that looks like evidence.

`scripts/run_dhvae_hh_baseline.py` provides a thin train-then-generate command
over the common wrapper. The former DH-VAE+HH compatibility CLIs remain
available. Internal isolated DeFoG workers live under
`grapher.models.defog.workers`; molecular preparation records the exact source
representation and, for ZINC, the verified Kekule model view.
