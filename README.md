# GraphES / Graph-ER codebase

This revision keeps the MS-VAE and GraphER model implementations essentially intact, but cleans the code layout around the experiment pipeline:

1. prepare dataset splits;
2. train the MS-VAE degree-sequence prior;
3. generate/check degree sequences;
4. train the GraphER rewiring scorer from Havel-Hakimi-to-data teacher steps;
5. generate graph samples from MS-VAE + GraphER;
6. evaluate generated samples.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

`torch-geometric` is required for PyG-backed downloads (`ego_citeseer` through Planetoid, QM9, and ZINC). The generic SBM/planar pipeline, and `ego_citeseer` from a local `ind.citeseer.graph` pickle, can run without PyG.

## Dataset preparation

```bash
PYTHONPATH=src python scripts/prepare_dataset.py --dataset sbm --force
PYTHONPATH=src python scripts/prepare_dataset.py --dataset planar --force
PYTHONPATH=src python scripts/prepare_dataset.py --dataset ego_citeseer --download-root outputs/raw_datasets/citeseer --force
```

`ego_citeseer` builds GraphRNN-style small ego graphs from CiteSeer. It first looks for a local `ind.citeseer.graph` pickle and otherwise uses PyG Planetoid CiteSeer. To use an existing local pickle:

```bash
PYTHONPATH=src python scripts/prepare_dataset.py --dataset ego_citeseer --raw-graph-path dataset/EGO/ind.citeseer.graph --force
```

Molecular datasets use PyG:

```bash
PYTHONPATH=src python scripts/prepare_dataset.py --dataset qm9 --download-root outputs/raw_datasets/qm9 --force
PYTHONPATH=src python scripts/prepare_dataset.py --dataset zinc --download-root outputs/raw_datasets/zinc --force
```

Persisted outputs are written to `outputs/datasets/<dataset>/`:

```text
train.pkl
val.pkl
test.pkl
metadata.json
resolved_dataset_config.yaml
```

## Training and sampling

Train the degree prior:

```bash
PYTHONPATH=src python scripts/train_msvae_model.py --dataset sbm --seed 42 --run-id 0
# use --dataset ego_citeseer for the CiteSeer ego-graph benchmark
```

Generate degree sequences for diagnostics:

```bash
PYTHONPATH=src python scripts/generate_msvae_samples.py --dataset sbm --num-samples 1024 --seed 42 --run-id 0 --force
```

Train the rewiring scorer:

```bash
PYTHONPATH=src python scripts/train_grapher_model.py --dataset sbm --seed 42 --run-id 0
```

Generate graph samples:

```bash
PYTHONPATH=src python scripts/generate_grapher_samples.py --dataset sbm --num-samples 1024 --seed 42 --run-id 0 --force
```

For repeated runs:

```bash
for run_id in 0 1 2; do
  seed=$((42 + run_id))
  PYTHONPATH=src python scripts/train_msvae_model.py --dataset sbm --seed "$seed" --run-id "$run_id"
  PYTHONPATH=src python scripts/generate_msvae_samples.py --dataset sbm --num-samples 1024 --seed "$seed" --run-id "$run_id" --force
  PYTHONPATH=src python scripts/train_grapher_model.py --dataset sbm --seed "$seed" --run-id "$run_id"
  PYTHONPATH=src python scripts/generate_grapher_samples.py --dataset sbm --num-samples 1024 --seed "$seed" --run-id "$run_id" --force
done
```

## Evaluation

Evaluate GraphER graph samples:

```bash
PYTHONPATH=src python scripts/evaluate_grapher_metrics.py \
  --dataset sbm \
  --model grapher \
  --run-id 0 \
  --reference-split test \
  --max-reference-graphs 1024 \
  --max-generated-graphs 1024
```

Evaluate MS-VAE degree sequences:

```bash
PYTHONPATH=src python scripts/evaluate_grapher_metrics.py \
  --dataset sbm \
  --model msvae \
  --run-id 0 \
  --reference-split test \
  --max-reference-graphs 1024 \
  --max-generated-graphs 1024
```

Compatibility aliases are available:

```bash
PYTHONPATH=src python scripts/evaluate_generic_metrics.py --dataset sbm --model grapher --run-id 0
PYTHONPATH=src python scripts/evaluate_degree_sequence.py --dataset sbm --run-id 0
PYTHONPATH=src python scripts/eval.py --dataset sbm --model grapher --run-id 0
```

Metric files are written to `outputs/metrics/<dataset>/<model>/`.

## Main output locations

```text
outputs/checkpoints/<dataset>/msvae/msvae.pt
outputs/checkpoints/<dataset>/grapher/grapher.pt
outputs/runs/<dataset>/<model>/run_000/train_metadata.json
outputs/samples/<dataset>/msvae_degree_sequences.pkl                 # single-run/no --run-id
outputs/samples/<dataset>/msvae/run_000.pkl                    # with --run-id 0
outputs/samples/<dataset>/grapher.pkl                          # single-run/no --run-id
outputs/samples/<dataset>/grapher/run_000.pkl                  # with --run-id 0
outputs/metrics/<dataset>/<model>/run_000/grapher_metrics.json
```

## Notes on model scope

The rewritten scripts implement the paper's topology-first pipeline: sampled degree sequence, connected Havel-Hakimi source, degree-preserving rewiring, and target-free candidate sets at generation time. The MS-VAE and GraphER internals are not replaced; only minimal dependency/import and checkpoint adaptations were added so the scripts can interoperate consistently.
