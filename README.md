# GraphES / Graph-ER codebase

This revision aligns the degree-prior implementation with the paper's size-conditioned DH-VAE. The code layout supports the full experiment pipeline:

1. prepare dataset splits;
2. train the size-conditioned DH-VAE degree-sequence prior;
3. generate/check degree sequences sampled from the empirical graph-size distribution;
4. train the GraphER rewiring scorer from Havel-Hakimi-to-data teacher steps;
5. generate graph samples from DH-VAE + GraphER;
6. evaluate generated samples.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

`torch-geometric` is required for PyG-backed downloads (`ego_citeseer` through Planetoid and QM9). ZINC is intentionally prepared from SMILES with RDKit, not from the PyG ZINC loader, because the molecular pipeline needs explicit atomic numbers. The generic SBM/planar pipeline, and `ego_citeseer` from a local `ind.citeseer.graph` pickle, can run without PyG.

## Dataset preparation

```bash
PYTHONPATH=src python scripts/prepare_dataset.py --dataset sbm --force
PYTHONPATH=src python scripts/prepare_dataset.py --dataset planar --force
PYTHONPATH=src python scripts/prepare_dataset.py --dataset ego_citeseer --raw-graph-path dataset/EGO/ind.citeseer.graph --force
```

`ego_citeseer` builds GraphRNN-style small ego graphs from CiteSeer. It first looks for a local `ind.citeseer.graph` pickle and otherwise uses PyG Planetoid CiteSeer. CiteSeer is a finite source graph: after radius and node-count filtering, the builder uses `min(num_graphs, available_candidates)` by default. This prevents failures when a large synthetic-dataset value such as `num_graphs: 10240` is accidentally reused. Add `--strict-num-graphs` or set `strict_num_graphs: true` to fail on shortfall instead. To use an existing local pickle:

Molecular datasets use different preparation paths. QM9 can use the generic dataset script through PyG:

```bash
PYTHONPATH=src python scripts/prepare_dataset.py --dataset qm9 --download-root outputs/raw_datasets/qm9 --force
```

ZINC must be prepared from a SMILES table with RDKit so every node keeps `atomic_number` and `z` attributes. Do not use `prepare_dataset.py --dataset zinc`; it now fails intentionally to avoid persisting PyG categorical atom-type ids as if they were atomic numbers.

```bash
PYTHONPATH=src python scripts/prepare_zinc_from_smiles.py \
  --csv data/zinc_smiles.csv \
  --smiles-col smiles \
  --split-col split \
  --output-root outputs/datasets \
  --force
```

If the CSV does not contain a split column, omit `--split-col` and use either fractional splitting or fixed counts, for example:

```bash
PYTHONPATH=src python scripts/prepare_zinc_from_smiles.py \
  --csv data/zinc_smiles.csv \
  --smiles-col smiles \
  --train-count 10000 --val-count 1000 --test-count 1000 \
  --force
```

Optional flags include `--target-col <column>` for molecular regression targets and `--keep-hs` if explicit hydrogens should be retained.

Persisted outputs are written to `outputs/datasets/<dataset>/`:

```text
train.pkl
val.pkl
test.pkl
metadata.json
resolved_dataset_config.yaml
```


## Dataset statistics

After preparing a dataset, print aggregate and split-level statistics for the paper appendix tables:

```bash
PYTHONPATH=src python scripts/print_dataset_statistics.py --dataset sbm
PYTHONPATH=src python scripts/print_dataset_statistics.py --dataset ego_citeseer
```

The default terminal table reports the core appendix statistics quickly: graph counts, node and edge ranges, maximum degree, average node/edge counts, average degree, density, and connectedness. The full JSON payload also includes degree histograms, component statistics, graphicality/connectivity-feasibility checks for degree sequences, and attributed-graph schema summaries. Use `--full` to add clustering, transitivity, and triangle counts; use `--include-wl-hashes` for fast duplicate diagnostics; use `--include-planarity` only when planarity checks are needed.

Save machine-readable outputs with:

```bash
PYTHONPATH=src python scripts/print_dataset_statistics.py \
  --dataset ego_citeseer \
  --json-out outputs/datasets/ego_citeseer/statistics.json \
  --csv-out outputs/datasets/ego_citeseer/statistics.csv \
  --force
```

For a dataset that has not been prepared yet, the statistics script can build it in memory from the dataset config, except for ZINC. ZINC must be prepared first with `scripts/prepare_zinc_from_smiles.py` because atomic numbers are extracted during RDKit SMILES parsing. For other datasets, the same raw-data overrides as `prepare_dataset.py` are supported:

```bash
PYTHONPATH=src python scripts/print_dataset_statistics.py \
  --dataset ego_citeseer \
  --raw-graph-path dataset/EGO/ind.citeseer.graph \
  --rebuild
```

Use `--save-built` to persist splits built by the statistics script. Use `--max-graphs-per-split` for a quick subset report, and `--include-path-stats` or `--include-exact-isomorphism` only when the extra cost is acceptable. The same script can summarize sample pickles too:

```bash
PYTHONPATH=src python scripts/print_dataset_statistics.py \
  --input-pkl outputs/samples/ego_citeseer/grapher/run_000.pkl \
  --skip-planarity

PYTHONPATH=src python scripts/print_dataset_statistics.py \
  --input-pkl outputs/samples/ego_citeseer/dhvae/run_000.pkl
```

## Training and sampling

Train the paper-aligned size-conditioned DH-VAE degree prior:

```bash
PYTHONPATH=src python scripts/train_dhvae_model.py --dataset sbm --seed 42 --run-id 0
PYTHONPATH=src python scripts/train_dhvae_model.py --dataset ego_citeseer --seed 42 --run-id 0
# use --dataset ego_citeseer for the CiteSeer ego-graph benchmark
```

Generate degree sequences for diagnostics. The sampler first draws a graph size from the empirical training-size distribution, then samples a degree histogram with a multinomial decoder:

```bash
PYTHONPATH=src python scripts/generate_dhvae_samples.py --dataset sbm --num-samples 1024 --seed 42 --run-id 0 --temperature 1.0 --force
```

Train the rewiring scorer:

```bash
PYTHONPATH=src python scripts/train_generic_grapher_model.py --dataset sbm --seed 42 --run-id 0
```

Generate graph samples:

```bash
PYTHONPATH=src python scripts/generate_grapher_samples.py --dataset sbm --num-samples 1024 --seed 42 --run-id 0 --force
```

For repeated runs:

```bash
for run_id in 0 1 2; do
  seed=$((42 + run_id))
  PYTHONPATH=src python scripts/train_dhvae_model.py --dataset sbm --seed "$seed" --run-id "$run_id"
  PYTHONPATH=src python scripts/generate_dhvae_samples.py --dataset sbm --num-samples 1024 --seed "$seed" --run-id "$run_id" --force
  PYTHONPATH=src python scripts/train_generic_grapher_model.py --dataset sbm --seed "$seed" --run-id "$run_id"
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

Evaluate DH-VAE degree sequences:

```bash
PYTHONPATH=src python scripts/evaluate_grapher_metrics.py \
  --dataset sbm \
  --model dhvae \
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
outputs/checkpoints/<dataset>/dhvae/dhvae.pt                 # DH-VAE checkpoint
outputs/checkpoints/<dataset>/grapher/grapher.pt
outputs/runs/<dataset>/<model>/run_000/train_metadata.json
outputs/samples/<dataset>/dhvae_degree_sequences.pkl                 # DH-VAE degree sequences, single-run/no --run-id
outputs/samples/<dataset>/dhvae/run_000.pkl                    # with --run-id 0
outputs/samples/<dataset>/grapher.pkl                          # single-run/no --run-id
outputs/samples/<dataset>/grapher/run_000.pkl                  # with --run-id 0
outputs/metrics/<dataset>/<model>/run_000/grapher_metrics.json
```

## Notes on model scope

The rewritten scripts implement the paper's topology-first pipeline: sampled degree sequence, connected Havel-Hakimi source, degree-preserving rewiring, and target-free candidate sets at generation time. GraphER remains structurally unchanged except for integration fixes. Legacy checkpoints from the previous independent-count degree prior must be retrained because the encoder/decoder parameterization changed.

### Degree-prior implementation

The old implementation predicted a categorical count independently for each degree bin and then rescaled the sampled counts to a fixed node count. The revised implementation follows the paper's DH-VAE design:

- degree histograms include bin 0, so isolated vertices are representable;
- the encoder receives both the histogram and a learned embedding of graph size `n`;
- the decoder predicts `pi(k | z, n)` over valid degree values `k = 0, ..., n - 1`;
- the reconstruction objective is the multinomial negative log-likelihood `-sum_k m_k log pi(k | z,n)` plus beta-KL;
- sampling draws `n` from the empirical training-size distribution and then draws `h ~ Multinomial(n, pi(. | z,n))`, so every sampled histogram has exactly `n` nodes before graphicality/connectedness filtering.

### Generic GraphER complete-action scorer

The generic GraphER path now scores complete rewiring actions:

```text
a = (e1, e2, r)
```

rather than scoring only the second edge after choosing an anchor edge. This matches the revised rewiring-flow layout: offline teacher construction may use target-aware candidates, but neural training and generation use target-free valid local/random candidates. The default generic config is:

```text
configs/models/grapher_generic.yaml
```

Train and sample:

```bash
PYTHONPATH=src python scripts/train_generic_grapher_model.py \
  --dataset ego_citeseer \
  --model-config configs/models/grapher_generic.yaml

PYTHONPATH=src python scripts/generate_grapher_samples.py \
  --dataset ego_citeseer \
  --model-config configs/models/grapher_generic.yaml \
  --num-samples 128 \
  --force
```

Important config fields:

```yaml
candidate_budget: 64
offline_candidate_budget: 256
k_hop: 2
action_temperature: 1.0
sample_actions: true
teacher_discrepancy: edge_symmetric_difference
```

Existing GraphER checkpoints from the older second-edge scorer must be retrained.
