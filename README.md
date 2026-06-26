# GraphES / Graph-ER

GraphES implements the Graph-ER pipeline for degree-constrained graph generation. The codebase supports generic featureless graphs and attributed molecular graphs:

1. prepare dataset splits;
2. train a size-conditioned DH-VAE degree prior;
3. generate and diagnose degree sequences;
4. train generic GraphER or attributed MolecularGraphER;
5. generate graph or molecule samples;
6. evaluate with the metrics used in the paper.

GraphER is topology-first: it samples a degree sequence, constructs a connected Havel-Hakimi source graph, and applies valid double-edge swaps. Molecular GraphER keeps node types fixed, proposes bond types from endpoint-conditioned empirical priors, and rejects valence-invalid actions.

---

## 1. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

Molecular preparation/generation/evaluation additionally requires RDKit. Paper molecular metrics also need `fcd_torch`, and EDeN is optional for the reference NSPDK backend:

```bash
pip install -r requirements-molecular.txt
```

Notes:

- `torch-geometric` is needed for PyG-backed downloads such as QM9 and Planetoid CiteSeer.
- ZINC is prepared from SMILES with RDKit. Do not use the PyG ZINC loader, because its atom-type ids are not atomic numbers.
- Old CUDA drivers may require an older PyTorch/PyG stack. Keep `numpy<2` for old PyTorch binary wheels.

---

## 2. Dataset preparation

### Generic datasets

```bash
PYTHONPATH=src python scripts/prepare_dataset.py --dataset sbm --force
PYTHONPATH=src python scripts/prepare_dataset.py --dataset planar --force
PYTHONPATH=src python scripts/prepare_dataset.py --dataset grid --force
PYTHONPATH=src python scripts/prepare_dataset.py \
  --dataset ego_citeseer \
  --raw-graph-path dataset/EGO/ind.citeseer.graph \
  --force
```

`ego_citeseer` builds GraphRNN-style ego graphs from CiteSeer. It first checks for a local `ind.citeseer.graph` pickle and otherwise uses PyG Planetoid CiteSeer.

### Molecular datasets

QM9 can be prepared through the generic dataset script. Rebuild QM9 after code updates that affect molecular attribute extraction, because atomic numbers and bond categories are persisted in the split files.

```bash
rm -rf outputs/datasets/qm9
PYTHONPATH=src python scripts/prepare_dataset.py \
  --dataset qm9 \
  --download-root outputs/raw_datasets/qm9 \
  --force
```

ZINC must be prepared from a SMILES table:

```bash
PYTHONPATH=src python scripts/prepare_zinc_from_smiles.py \
  --csv data/zinc_smiles.csv \
  --smiles-col smiles \
  --split-col split \
  --output-root outputs/datasets \
  --force
```

If there is no split column, use fixed counts or fractions:

```bash
PYTHONPATH=src python scripts/prepare_zinc_from_smiles.py \
  --csv data/zinc_smiles.csv \
  --smiles-col smiles \
  --train-count 10000 --val-count 1000 --test-count 1000 \
  --output-root outputs/datasets \
  --force
```

Prepared outputs are written to:

```text
outputs/datasets/<dataset>/train.pkl
outputs/datasets/<dataset>/val.pkl
outputs/datasets/<dataset>/test.pkl
outputs/datasets/<dataset>/metadata.json
outputs/datasets/<dataset>/resolved_dataset_config.yaml
```

---

## 3. Dataset statistics

```bash
PYTHONPATH=src python scripts/print_dataset_statistics.py --dataset sbm
PYTHONPATH=src python scripts/print_dataset_statistics.py --dataset ego_citeseer
PYTHONPATH=src python scripts/print_dataset_statistics.py --dataset qm9
PYTHONPATH=src python scripts/print_dataset_statistics.py --dataset zinc
```

Save appendix-ready JSON/CSV reports:

```bash
PYTHONPATH=src python scripts/print_dataset_statistics.py \
  --dataset qm9 \
  --json-out outputs/datasets/qm9/statistics.json \
  --csv-out outputs/datasets/qm9/statistics.csv \
  --force
```

Useful optional flags:

```text
--full                    add clustering/transitivity/triangle statistics
--include-wl-hashes       add fast duplicate diagnostics
--include-planarity       add planarity checks
--max-graphs-per-split N  quick subset statistics
```

---

## 4. DH-VAE degree-prior training

DH-VAE models **only the topological degree sequence**. For QM9 and ZINC, atom types and bond types are handled later by MolecularGraphER.

### Generic DH-VAE

```bash
PYTHONPATH=src python scripts/train_dhvae_model.py \
  --dataset sbm \
  --seed 42 \
  --run-id 0

PYTHONPATH=src python scripts/train_dhvae_model.py \
  --dataset grid \
  --seed 42 \
  --run-id 0

PYTHONPATH=src python scripts/train_dhvae_model.py \
  --dataset ego_citeseer \
  --seed 42 \
  --run-id 0
```

### Molecular DH-VAE

Use the molecular configs so the model capacity and training limits match the molecule datasets.

```bash
PYTHONPATH=src python scripts/train_dhvae_model.py \
  --dataset qm9 \
  --seed 42 \
  --run-id 0 \
  --model-config configs/models/dhvae_qm9.yaml

PYTHONPATH=src python scripts/train_dhvae_model.py \
  --dataset zinc \
  --seed 42 \
  --run-id 0 \
  --model-config configs/models/dhvae_zinc.yaml
```

For a quick molecular smoke test:

```bash
PYTHONPATH=src python scripts/train_dhvae_model.py \
  --dataset zinc \
  --seed 42 \
  --run-id 99 \
  --model-config configs/models/dhvae_zinc.yaml \
  --max-train-graphs 1024
```

The diagnostic flags `--allow-disconnected-graphs` and `--allow-disconnected-degree-sequences` are for debugging only. Do not use them for the connected GraphER protocol.

---

## 5. DH-VAE degree-sequence generation and compatibility checks

`generate_dhvae_samples.py` is dataset-aware:

- `qm9` defaults to `configs/models/dhvae_qm9.yaml`;
- `zinc` defaults to `configs/models/dhvae_zinc.yaml`;
- generic datasets default to `configs/models/dhvae.yaml`.

The output payload is always:

```text
list[list[int]]
```

where each inner list is a sampled degree sequence. For molecular datasets, this is intentionally only the **topological degree prior** used by MolecularGraphER.

### Generic examples

```bash
PYTHONPATH=src python scripts/generate_dhvae_samples.py \
  --dataset sbm \
  --num-samples 1024 \
  --seed 42 \
  --run-id 0 \
  --temperature 1.0 \
  --force

PYTHONPATH=src python scripts/generate_dhvae_samples.py \
  --dataset ego_citeseer \
  --num-samples 1024 \
  --seed 42 \
  --run-id 0 \
  --force
```

### Molecular examples

```bash
PYTHONPATH=src python scripts/generate_dhvae_samples.py \
  --dataset qm9 \
  --num-samples 10000 \
  --seed 42 \
  --run-id 0 \
  --force

PYTHONPATH=src python scripts/generate_dhvae_samples.py \
  --dataset zinc \
  --num-samples 10000 \
  --seed 42 \
  --run-id 0 \
  --force
```

The metadata file records compatibility diagnostics, including graphicality rate, connected-feasible rate, zero-degree multi-node rate, rejection reasons, and whether the sampled degree prior is ready for generic or molecular GraphER:

```text
outputs/samples/<dataset>/dhvae/run_000.metadata.json
```

Useful flags:

```text
--allow-disconnected-degree-sequences  diagnostic only; disables connected-feasible filtering
--max-attempts N                       increase if many sampled sequences are rejected
--temperature T                        override sample_temperature from the config
--reference-split train|val|test        include prepared-split degree diagnostics in metadata
--skip-reference-diagnostics            skip loading dataset splits during degree sampling
```

Evaluate DH-VAE degree samples against the prepared train/test degree-sequence distributions with:

```bash
PYTHONPATH=src python scripts/evaluate_dhvae_metrics.py \
  --dataset sbm \
  --run-id 0 \
  --max-train-sequences 1024 \
  --max-test-sequences 1024 \
  --max-generated-sequences 1024
```

The DH-VAE evaluator reports KL and Gaussian-EMD MMD between test degree sequences and both train/generated degree sequences. It writes:

```text
outputs/metrics/<dataset>/dhvae/run_000/dhvae_metrics.json
```

Useful evaluation flags:

```text
--sample-path PATH              evaluate an explicit generated degree-sequence pickle
--run-ids 0 1 2                 evaluate several run ids and write an aggregate JSON
--degree-bins N                 minimum number of degree histogram bins
--sigma S                       Gaussian kernel width for MMD
--max-*-sequences N             subsample train, test, or generated sequences
```

---

## 6. Generic GraphER training, generation, and evaluation

### Train

```bash
PYTHONPATH=src python scripts/train_generic_grapher_model.py \
  --dataset sbm \
  --seed 42 \
  --run-id 0 \
  --model-config configs/models/grapher_generic.yaml

PYTHONPATH=src python scripts/train_generic_grapher_model.py \
  --dataset ego_citeseer \
  --seed 42 \
  --run-id 0 \
  --model-config configs/models/grapher_generic_ego_citeseer.yaml
```

The generic scorer predicts complete actions:

```text
a = (e1, e2, r)
```

Offline teacher construction may use target-aware candidates. Neural training and generation use target-free valid local/random candidates.

### Resume generic training after an interruption

Generic teacher-cache construction can also be expensive, especially for SBM-64. The generic training script saves:

```text
outputs/runs/<dataset>/grapher/run_000/teacher_cache.pt
outputs/runs/<dataset>/grapher/run_000/training_state.pt
```

By default, rerunning the same command with the same `--run-id` reuses a complete teacher cache, resumes a partial teacher-cache build from the last checkpointed graph, and resumes neural optimization from the last completed epoch.

To force cache reconstruction from the command line:

```bash
PYTHONPATH=src python scripts/train_generic_grapher_model.py \
  --dataset sbm \
  --seed 42 \
  --run-id 0 \
  --model-config configs/models/grapher_generic.yaml \
  --force-rebuild-cache
```

To disable all resume behavior:

```bash
PYTHONPATH=src python scripts/train_generic_grapher_model.py \
  --dataset sbm \
  --seed 42 \
  --run-id 0 \
  --model-config configs/models/grapher_generic.yaml \
  --no-resume
```

The cache checkpoint interval is controlled by:

```yaml
teacher_cache_checkpoint_interval: 25
```

Use a smaller value such as `1` or `5` if interruptions are frequent. Smaller values reduce lost cache work but write the large cache file more often.

### Generate

```bash
PYTHONPATH=src python scripts/generate_grapher_samples.py \
  --dataset sbm \
  --num-samples 1024 \
  --seed 42 \
  --run-id 0 \
  --force
```

### Evaluate

```bash
PYTHONPATH=src python scripts/evaluate_grapher_metrics.py \
  --dataset sbm \
  --model grapher \
  --run-id 0 \
  --reference-split test \
  --max-reference-graphs 1024 \
  --max-generated-graphs 1024
```

The generic evaluator reports degree MMD, clustering MMD, spectral MMD, motif proxy MMD, optional ORCA orbit-count MMD, connectedness, validity, uniqueness, novelty, and runtime.

---

## 7. Molecular GraphER training, generation, and evaluation

Molecular GraphER uses hard attributed graph states. Node types are fixed during rewiring; bond labels for new edges are proposed from:

```text
p(edge_type | unordered endpoint atomic numbers)
```

The model scores complete typed actions:

```text
a = (e1, e2, r, c1, c2)
```

Every candidate is rejected if it creates a self-loop, duplicate edge, disconnected graph, or violates the fitted/overridden valence constraint.

### Train QM9

```bash
PYTHONPATH=src python scripts/train_molecular_grapher_model.py \
  --dataset qm9 \
  --seed 42 \
  --run-id 0 \
  --model-config configs/models/grapher_molecular_qm9.yaml
```

### Train ZINC

```bash
PYTHONPATH=src python scripts/train_molecular_grapher_model.py \
  --dataset zinc \
  --seed 42 \
  --run-id 0 \
  --model-config configs/models/grapher_molecular_zinc.yaml
```

### Resume molecular training after an interruption

Molecular teacher-cache construction can be expensive. The training script now saves:

```text
outputs/runs/<dataset>/grapher_molecular/run_000/teacher_cache.pt
outputs/runs/<dataset>/grapher_molecular/run_000/training_state.pt
```

By default, rerunning the same command with the same `--run-id` resumes from the last completed epoch and reuses the saved teacher cache. To force cache reconstruction, add the following setting to the model config:

```yaml
force_rebuild_teacher_cache: true
```

To disable resume behavior:

```yaml
resume: false
```

### Generate molecules

```bash
PYTHONPATH=src python scripts/generate_molecular_grapher_samples.py \
  --dataset qm9 \
  --num-samples 10000 \
  --seed 42 \
  --run-id 0 \
  --write-sdf \
  --force

PYTHONPATH=src python scripts/generate_molecular_grapher_samples.py \
  --dataset zinc \
  --num-samples 10000 \
  --seed 42 \
  --run-id 0 \
  --write-sdf \
  --force
```

### Molecular sample representation

The molecular sampler writes:

```text
outputs/samples/<dataset>/grapher_molecular/run_000.pkl
outputs/samples/<dataset>/grapher_molecular/run_000.jsonl
outputs/samples/<dataset>/grapher_molecular/run_000.smi
outputs/samples/<dataset>/grapher_molecular/run_000.sdf   # only with --write-sdf
```

The pickle is a dictionary containing:

```text
graphs                       list[nx.Graph]
canonical_smiles             list[str]
valid_without_correction     list[bool]
degree_sequences             list[list[int]]
generation_records           list[dict]
```

Each molecular graph is a hard attributed `networkx.Graph` with:

```text
node.atomic_number / node.z
edge.edge_type      # 1=single, 2=double, 3=triple, 4=aromatic
edge.bond_order
```

JSONL preserves all generated hard attributed graphs, including chemically invalid outputs. SMILES and SDF contain only molecules that convert and sanitize directly in RDKit.

### Evaluate molecular samples

The paper molecular metrics are validity without correction, NSPDK MMD, and FCD.

```bash
PYTHONPATH=src python scripts/evaluate_molecular_grapher_metrics.py \
  --dataset qm9 \
  --run-id 0 \
  --reference-split test \
  --max-generated-molecules 10000 \
  --nspdk-backend auto \
  --require-fcd

PYTHONPATH=src python scripts/evaluate_molecular_grapher_metrics.py \
  --dataset zinc \
  --run-id 0 \
  --reference-split test \
  --max-generated-molecules 10000 \
  --nspdk-backend auto \
  --require-fcd
```

Evaluation details:

- `validity_without_correction` directly converts each hard attributed graph to RDKit and sanitizes it without repair, valence correction, bond resampling, or fragment selection.
- `nspdk_mmd` uses EDeN when available. Otherwise the evaluator records a fallback to a deterministic attributed neighborhood-pair feature map.
- `fcd` uses canonical SMILES from directly valid molecules through `fcd_torch`; report it together with `validity_without_correction` because invalid generated molecules are excluded from the FCD population. The evaluator also records `fcd_validity_without_correction`, `fcd_num_valid_generated_molecules`, and `fcd_num_generated_molecules`.
- `nspdk_mmd_valid_only`, uniqueness, novelty, connectedness, and self-loop rates are also reported as diagnostics.

Use `--skip-fcd` for dependency-light smoke tests and `--nspdk-backend eden` to require the reference EDeN backend.

---

## 8. Repeated runs

```bash
for run_id in 0 1 2; do
  seed=$((42 + run_id))
  PYTHONPATH=src python scripts/train_dhvae_model.py --dataset sbm --seed "$seed" --run-id "$run_id"
  PYTHONPATH=src python scripts/generate_dhvae_samples.py --dataset sbm --num-samples 1024 --seed "$seed" --run-id "$run_id" --force
  PYTHONPATH=src python scripts/evaluate_dhvae_metrics.py --dataset sbm --seed "$seed" --run-id "$run_id"
  PYTHONPATH=src python scripts/train_generic_grapher_model.py --dataset sbm --seed "$seed" --run-id "$run_id"
  PYTHONPATH=src python scripts/generate_grapher_samples.py --dataset sbm --num-samples 1024 --seed "$seed" --run-id "$run_id" --force
  PYTHONPATH=src python scripts/evaluate_grapher_metrics.py --dataset sbm --model grapher --seed "$seed" --run-id "$run_id"
done
```

Aggregate results with:

```bash
PYTHONPATH=src python scripts/aggregate_results.py --datasets sbm --models grapher
```

Average DH-VAE degree-prior metrics from existing per-run metric JSONs with:

```bash
PYTHONPATH=src python scripts/aggregate_dhvae_results.py --dataset sbm --run-ids 0 1 2
```

If `--run-ids` is omitted, `aggregate_dhvae_results.py` discovers all `outputs/metrics/<dataset>/dhvae/run_*/dhvae_metrics.json` files and writes:

```text
outputs/metrics/<dataset>/dhvae/dhvae_metrics.aggregate.json
```

---

## 9. Main output locations

```text
outputs/checkpoints/<dataset>/dhvae/run_000/dhvae.pt
outputs/checkpoints/<dataset>/grapher/run_000/grapher.pt
outputs/checkpoints/<dataset>/grapher_molecular/run_000/grapher_molecular.pt

outputs/runs/<dataset>/<model>/run_000/train_metadata.json

outputs/samples/<dataset>/dhvae/run_000.pkl
outputs/samples/<dataset>/dhvae/run_000.metadata.json
outputs/samples/<dataset>/grapher/run_000.pkl
outputs/samples/<dataset>/grapher_molecular/run_000.pkl
outputs/samples/<dataset>/grapher_molecular/run_000.jsonl
outputs/samples/<dataset>/grapher_molecular/run_000.smi
outputs/samples/<dataset>/grapher_molecular/run_000.sdf

outputs/metrics/<dataset>/dhvae/run_000/dhvae_metrics.json
outputs/metrics/<dataset>/dhvae/dhvae_metrics.aggregate.json
outputs/metrics/<dataset>/grapher/run_000/grapher_metrics.json
outputs/metrics/<dataset>/grapher_molecular/run_000/molecular_grapher_metrics.json
```

---

## 10. Reproducibility notes

- Retrain DH-VAE when dataset preprocessing changes.
- Retrain GraphER checkpoints when model config changes `hidden_dim`, `num_layer`, `local_feature_dim`, or action representation.
- For connected GraphER benchmarks, use connected-feasible degree sequences and connected reference datasets.
- For molecular paper runs, report the prepared split statistics, molecular attribute schema, valence settings, FCD backend, NSPDK backend, random seeds, and the number of generated samples.
