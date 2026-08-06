# Graph-ER

This repository contains the maintained degree-constrained Graph-ER pipeline:

1. DH-VAE samples a degree multiset.
2. Connected Havel-Hakimi realizes that multiset as an initial graph.
3. An endpoint/graphlet predictor estimates the terminal all-pairs edge law and
   connected induced graphlet distributions.
4. Valid double-edge swaps refine the graph while preserving its degree
   sequence and connectedness.

The maintained selector is energy-guided. A learned policy/`STOP` selector and
the paper's full typed-signature molecular constructor are not implemented in
this archive.

## TODO: Research gaps

The current code supports an energy-guided generic pipeline and a
topology-first QM9 baseline. The following work is required before the
repository implements and evaluates the complete method described in the
manuscript. A task should be checked only after its implementation, tests, and
reported diagnostics are complete.

### P0 - Method-completeness blockers

- [ ] **Implement the joint typed-invariant prior for QM9.** Model the
  graph-size-conditioned histogram of atom--typed-degree signatures
  `(atom, single degree, double degree, triple degree)` and report sample
  feasibility before construction.
- [ ] **Implement an exact typed molecular constructor.** Realize sampled
  signatures directly while enforcing typed-degree balance, atom/bond endpoint
  compatibility, valence, simplicity, and connectedness. Validate it first on
  oracle invariants extracted from held-out molecules.
- [ ] **Implement attributed graphlet targets.** Canonicalize connected induced
  graphlets of sizes 3, 4, and 5 jointly by topology, atom type, and bond type;
  build the training vocabulary from the training split only and reserve an
  overflow class.
- [ ] **Implement the learned candidate selector and explicit `STOP` action.**
  Train it against cached teacher-action distributions and support comparable
  energy-only, policy-only, and hybrid inference modes.
- [ ] **Complete teacher-trajectory construction.** Add randomized feasible
  sources, hard and soft teachers, target-aware candidate proposals, multiple
  paths or top-k action sampling, and explicit target-reach and `STOP`
  diagnostics.
- [ ] **Make full-QM9 training memory bounded.** Replace the in-memory
  trajectory materialization with a streaming or sharded trajectory dataset.

### P1 - Open scientific questions

- [ ] **Separate the sources of generation error.** Use learned-versus-oracle
  invariants and constructor-only-versus-refined outputs to attribute error to
  the invariant prior, feasibility rejection, constructor, predictor, and
  rewiring search.
- [ ] **Measure constructor bias.** Compare canonical Havel--Hakimi,
  randomized Havel--Hakimi, and random feasible construction, and quantify how
  rejection and restart policies bias the effective invariant distribution.
- [ ] **Study reachability under constrained rewiring.** Measure target-reach
  rate and connected realization-space coverage under connectivity,
  same-bond-type, locality, compatibility, and RDKit masks.
- [ ] **Resolve prediction inconsistency.** Quantify conflicts among predicted
  pair marginals, graphlet histograms, and the fixed invariant, then evaluate
  consistency regularization or projection onto a feasible target set.
- [ ] **Test whether local summaries are sufficient.** Determine when
  graphlets fail to capture community or hierarchical structure and evaluate
  hierarchical summaries for larger generic graphs.
- [ ] **Characterize molecular validity limits.** Typed-degree preservation
  fixes bond counts and weighted valence but not aromaticity, charge,
  stereochemistry, stability, or synthetic accessibility; report these limits
  without post-hoc molecule repair.

### P2 - Research-grade evaluation

- [ ] **Standardize all evaluators.** Use the same Gaussian Earth-Mover MMD,
  graphlet conventions, ORCA setup, reference counts, and preprocessing for
  Graph-ER and every baseline.
- [ ] **Complete the fixed three-seed protocol.** Freeze dataset splits and
  budgets, tune only on validation data, generate the declared number of
  samples, and report mean and standard deviation for seeds 42, 43, and 44.
- [ ] **Add complete pipeline diagnostics.** Report predictor NLL and
  macro-F1, graphlet error, consistency residual, invariant feasibility,
  constructor success, candidate pass rate, proposals per accepted swap,
  accepted swaps, `STOP` rate, rejection reasons, runtime, and end-to-end
  generation yield. No silent fallback should enter the final metrics.
- [ ] **Run controlled ablations and protocol-compatible baselines.** Include
  source constructor, oracle invariant, random rewiring, pair-only,
  graphlet-only, static-versus-dynamic prediction, energy-only,
  policy-only, hybrid, and molecular-filter ablations using identical sampled
  invariants and initial graphs where applicable.
- [ ] **Report quality--cost trade-offs.** Sweep candidate budgets, valid
  candidate counts, locality radius, and maximum accepted steps, and publish
  per-seed configurations, checkpoints, generated graphs, and failure logs.

## Environment

Use Python 3.10 or newer. Install a PyTorch build that matches the local CUDA
runtime first, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install PyTorch separately for the machine's CUDA/CPU environment.
python -m pip install numpy networkx pyyaml
python -m pip install matplotlib rdkit pytest ruff
```

Optional dependencies:

```bash
# PyG is needed only for --source pyg; direct SDF loading does not require it.
python -m pip install torch-geometric

# Required only for FCD in molecular evaluation.
python -m pip install fcd-torch
```

The archive does not contain packaging metadata, so run commands from the
repository root with `PYTHONPATH=src`.

ORCA is required for four-node orbit evaluation. Put `orca` on `PATH`, set an
absolute executable path in `ORCA_EXEC`, or set `evaluation.orca_exec` in the
experiment YAML:

```bash
export ORCA_EXEC=/absolute/path/to/orca
```

## SBM/Community-small workflow

Despite its historical `sbm` filename, `configs/datasets/sbm.yaml` implements
the paper's Community-small protocol: 500 two-community graphs, 30--80 nodes
per community, within-community probability 0.30, and approximately
`0.05 |V|` inter-community edges.

### 1. Prepare and freeze the dataset

```bash
PYTHONPATH=src python scripts/prepare_generic_dataset.py \
  --dataset community_small \
  --root outputs/datasets

PYTHONPATH=src python scripts/prepare_generic_dataset.py \
  --dataset ego_small --root outputs/datasets

PYTHONPATH=src python scripts/prepare_generic_dataset.py \
  --dataset grid --root outputs/datasets
```

The command writes `train.pkl`, `val.pkl`, `test.pkl`,
`resolved_dataset_config.yaml`, `metadata.json`, and `prep_report.json` under
`outputs/datasets/sbm`. Keep these files fixed across models and random seeds.
The configured split contains 400/50/50 graphs.

### 2. Train and evaluate DH-VAE

```bash
PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/dhvae/community_small.yaml

PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/dhvae/ego_small.yaml

PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/dhvae/grid.yaml

PYTHONPATH=src python scripts/evaluate_degree_generator.py \
  --config configs/experiments/community_small.yaml \
  --num-samples 1024 \
  --max-reference-sequences 1024

PYTHONPATH=src python scripts/evaluate_degree_generator.py \
  --config configs/experiments/ego_small.yaml \
  --num-samples 1024 \
  --max-reference-sequences 1024

  PYTHONPATH=src python scripts/evaluate_degree_generator.py \
  --config configs/experiments/grid.yaml \
  --num-samples 1024 \
  --max-reference-sequences 1024
```

Check graphicality, connected-feasibility, repair use, sampling trials, and
prior-to-test degree-distribution distances before using the checkpoint in
end-to-end generation. The research configuration uses `fallback: error`, so a
failed model draw raises instead of being silently replaced with a training
degree sequence. The current runner still needs a batch-level failure counter
to convert such exceptions into an end-to-end yield statistic.

### 3. Train the endpoint/graphlet predictor

```bash
PYTHONPATH=src python scripts/train_hybrid_endpoint_grapher.py \
  --config configs/experiments/grapher/community_small_hybrid_endpoint_graphlet.yaml \
  --output-dir outputs/hybrid_endpoint/sbm/seed_42 \
  --seed 42
```

The best validation checkpoint is saved as
`outputs/hybrid_endpoint/sbm/seed_42/checkpoint.pt`; the complete epoch and
teacher diagnostics are saved in `training_report.json`.

### 4. Generate 1,024 graphs

```bash
PYTHONPATH=src python scripts/run_hybrid_endpoint_grapher.py \
  --config configs/experiments/sbm_hybrid_endpoint_graphlet.yaml \
  --checkpoint outputs/hybrid_endpoint/sbm/seed_42/checkpoint.pt \
  --output-dir outputs/hybrid_endpoint/sbm/seed_42/generated \
  --num-generate 1024 \
  --seed 42
```

The run saves the Havel-Hakimi sources, refined graphs, step traces, structural
metrics, degree-preservation diagnostics, and connectedness diagnostics.

For a compact diagnostic table and sample figure:

```bash
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/sbm_hybrid_endpoint_graphlet.yaml \
  --generated-dir outputs/hybrid_endpoint/sbm/seed_42/generated \
  --output-dir outputs/hybrid_endpoint/sbm/seed_42/evaluation_report
```

The Community-small test split has 50 graphs. The generation `report.json`
compares all 1,024 generated graphs with the held-out test set, whereas the
paper-facing report utility currently forms an equal-size 50-versus-50 subset.
Record the sample counts beside every reported metric.

The current degree/clustering implementation uses a median-bandwidth Euclidean
RBF MMD, while the manuscript specifies a common Gaussian Earth-Mover kernel.
Treat these values as internal diagnostics until the evaluator is standardized
across Graph-ER and every baseline.

### 5. Three-seed protocol

Use seeds 42, 43, and 44 and independent output directories. For a fully
independent end-to-end rerun, copy both experiment YAML files per seed and
change all of the following together:

- top-level `seed`;
- `degree_generator.checkpoint_path`;
- `degree_evaluation.output_dir`;
- endpoint predictor output directory;
- generation output directory.

Report mean and standard deviation over the three complete runs. Do not select
hyperparameters using the test split.

## QM9 attributed baseline

### Scope warning

`configs/experiments/qm9_attributed_hybrid_endpoint_graphlet.yaml` is a
non-conformant molecular baseline. It constructs an ordinary-degree topology,
then samples atom and bond attributes. Same-bond-type swaps and RDKit candidate
checks preserve validity of accepted states, but this route does **not**
implement the paper's joint atom--typed-degree prior, typed constructor, or
atom/bond-aware graphlet vocabulary. Do not report it as the full attributed
Graph-ER model.

### 1. Prepare QM9

Use the direct SDF loader because PyG preprocessing can stop at an RDKit-invalid
record:

```bash
PYTHONPATH=src python scripts/prepare_qm9_topology_dataset.py \
  --source sdf \
  --sdf-file data/qm9_deepchem/qm9.sdf \
  --root outputs/datasets \
  --seed 42
```

The command writes aligned topology and attributed splits under
`outputs/datasets/qm9_topology` and `outputs/datasets/qm9_attributed`. With the
current DeepChem SDF and RDKit preprocessing, 131,887 of 133,885 records are
retained. Freeze the resulting files and preprocessing report before training.

For the full split, set the following in the QM9 experiment config:

```yaml
dataset:
  max_train_graphs: null
  max_val_graphs: null
```

The existing `10000/1000` limits are suitable only for development runs. The
trainer materializes every trajectory example in memory before optimization;
therefore, full-split training requires sufficient host RAM. A streaming or
sharded trajectory dataset is still needed for memory-bounded full-QM9 runs.



### 2. Train and evaluate DH-VAE

```bash
PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/dhvae/qm9_typed.yaml

PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/dhvae/zinc_typed.yaml

PYTHONPATH=src python scripts/evaluate_degree_generator.py \
  --config configs/experiments/qm9_typed.yaml \
  --num-samples 1024 \
  --max-reference-sequences 1024

  PYTHONPATH=src python scripts/evaluate_degree_generator.py \
  --config configs/experiments/zinc_typed.yaml \
  --num-samples 1024 \
  --max-reference-sequences 1024
```

### 3. Train QM9 on the full prepared split

After setting both dataset limits to `null`:

```bash
PYTHONPATH=src python scripts/train_hybrid_endpoint_grapher.py \
  --config configs/experiments/qm9_attributed_hybrid_endpoint_graphlet.yaml \
  --output-dir outputs/hybrid_endpoint/qm9_attributed/seed_42 \
  --seed 42
```

The configuration uses the validation split to retain the lowest-loss
checkpoint. Inspect `present_edge_recall` and graphlet error in addition to raw
edge accuracy, because the no-edge class dominates all-pairs prediction.

### 4. Generate QM9 baseline molecules

The supplied QM9 config uses oracle test degree multisets, so this is a
refiner/attribute-initialization experiment rather than unconditional molecular
generation:

```bash
PYTHONPATH=src python scripts/run_hybrid_endpoint_grapher.py \
  --config configs/experiments/qm9_attributed_hybrid_endpoint_graphlet.yaml \
  --checkpoint outputs/hybrid_endpoint/qm9_attributed/seed_42/checkpoint.pt \
  --output-dir outputs/hybrid_endpoint/qm9_attributed/seed_42/generated \
  --num-generate 1024 \
  --seed 42
```

Use `--num-generate 10000` for the final molecular protocol after the complete
run has been validated at 1,024 samples.

### 5. Evaluate QM9 outputs

Pass the serialized graph file explicitly so invalid generated graphs remain in
the validity denominator:

```bash
PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-graphs outputs/hybrid_endpoint/qm9_attributed/seed_42/generated/hybrid_refined_graphs.pkl \
  --dataset-root outputs/datasets \
  --dataset qm9_attributed \
  --reference-split test \
  --train-split train \
  --output-dir outputs/hybrid_endpoint/qm9_attributed/seed_42/molecular_evaluation
```

This reports RDKit validity without correction, uniqueness, novelty, the
built-in NSPDK-style proxy, and FCD when `fcd-torch` is available. The saved
`valid_generated.smi` contains only valid molecules; the serialized graph file
contains every generated output.

## Required experiment records

For every final run, retain:

- dataset configuration, prepared split files, and preparation report;
- exact experiment YAML and seed;
- best checkpoint and training report;
- DH-VAE sampling and feasibility diagnostics;
- all generated graphs, sources, and rewiring traces;
- generated/reference counts for every metric;
- degree preservation, connectedness, constructor success, and generation
  yield;
- dependency versions, ORCA path/version, commit identifier, and hardware.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m compileall -q src scripts
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src pytest -q
ruff check src scripts tests
ruff format --check src scripts tests

PYTHONPATH=src python scripts/train_degree_generator.py --help
PYTHONPATH=src python scripts/train_hybrid_endpoint_grapher.py --help
PYTHONPATH=src python scripts/run_hybrid_endpoint_grapher.py --help
PYTHONPATH=src python scripts/evaluate_generated_molecules.py --help
```
