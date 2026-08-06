# Graph-ER

Graph-ER is a hybrid graph generator based on target-summary prediction and
degree-constrained rewiring. This repository implements both the generic
ordinary-degree route and the attributed molecular route described in
*Hybrid Target-Summary Prediction and Degree-Constrained Rewiring*.

The implementation contains:

- ordinary and atom-typed degree-histogram VAEs conditioned on graph size;
- connected Havel-Hakimi and exact typed residual-demand constructors;
- hierarchical node, pair, and attributed graphlet prediction;
- invariant-preserving teacher trajectories with hard or soft action targets;
- energy-only, policy-only, and hybrid selectors with an explicit learned
  `STOP` action;
- connected, degree-preserving double-edge swaps and same-bond-type molecular
  swaps;
- generic, QM9, and ZINC preparation/configuration paths; and
- fixed-seed research diagnostics, ablations, cost sweeps, and report
  aggregation.

## Method

Generation has four stages:

1. Sample an invariant: an ordinary degree multiset for generic graphs, or an
   atom category plus per-bond-type degree signature for molecular graphs.
2. Construct a connected source graph that realizes the sampled invariant.
3. Predict terminal pair and connected induced graphlet summaries from the
   current state, invariant, graph size, and normalized time.
4. Rewire with valid double-edge swaps. Energy mode scores every valid
   candidate; policy mode uses the learned selector; hybrid mode uses a policy
   shortlist followed by summary energy and a positive-improvement gate.

Molecular candidates are screened for invariant, valence, and optional RDKit
validity before attributed graphlet scoring. Accepted same-type swaps preserve
every indexed atom's typed-degree vector, ordinary degree, global bond-type
counts, and weighted valence.

Training builds one or more randomized invariant-matched sources for each
target. A terminal-summary teacher assigns hard or temperature-controlled soft
distributions over valid actions and `STOP`. Pair prediction is conditioned on
fixed or predicted node categories; graphlet prediction is conditioned on soft
pair predictions. Molecular trajectories can be streamed rather than
materialized in memory.

## Implementation status

| Component | Status |
| --- | --- |
| Size-conditioned ordinary degree prior | Implemented |
| Joint atom-typed-degree prior and feasibility checks | Implemented |
| Connected generic constructor | Implemented |
| Exact typed molecular constructor | Implemented |
| Train-only attributed graphlet vocabulary with overflow | Implemented |
| Hierarchical endpoint/graphlet predictor | Implemented |
| Typed-invariant teacher with hard/soft targets | Implemented |
| Streaming trajectory dataset | Implemented |
| Energy, policy, and hybrid selectors with `STOP` | Implemented |
| Typed QM9 and ZINC generation | Implemented |
| Standardized EMD/MMD and molecular evaluation | Implemented |
| Three-seed studies, ablations, and cost reports | Implemented |

No research result is bundled: checkpoints, generated samples, external
baseline outputs, and aggregate tables must be produced from fixed dataset
splits.

## Installation

Python 3.10 or newer is required. Install the PyTorch build appropriate for the
machine, then install this project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,evaluation]'
```

Optional packages:

```bash
# PyG is used only by the optional QM9 PyG source loader.
python -m pip install torch-geometric

# Optional molecular FCD metric.
python -m pip install fcd-torch
```

ORCA is required only for exact four-node orbit evaluation. Put `orca` on
`PATH`, set `ORCA_EXEC`, or set `evaluation.orca_exec` in the experiment
configuration. All commands below are run from the repository root.

## Dataset preparation

### Generic benchmarks

```bash
PYTHONPATH=src python scripts/prepare_generic_dataset.py --dataset community_small --root outputs/datasets
PYTHONPATH=src python scripts/prepare_generic_dataset.py --dataset ego_small --root outputs/datasets
PYTHONPATH=src python scripts/prepare_generic_dataset.py --dataset grid --root outputs/datasets
```

`configs/datasets/community_small.yaml` follows the 100-graph
Community-small protocol with a fixed 70/10/20 train/validation/test split. Its
serialized dataset name remains `sbm` for compatibility. Keep the generated
split files, resolved configuration, metadata, and preparation report fixed
across model seeds.

### QM9

```bash
PYTHONPATH=src python scripts/prepare_qm9_dataset.py \
  --source sdf \
  --sdf-file data/qm9/qm9.sdf \
  --root outputs/datasets \
  --seed 42
```

The direct SDF path is recommended. Preparation writes aligned topology and
attributed splits. Invalid or unsupported records are counted in the report
rather than silently accepted.

### ZINC

```bash
PYTHONPATH=src python scripts/prepare_zinc_dataset.py \
  --smiles-file data/zinc/zinc.csv \
  --smiles-column smiles \
  --root outputs/datasets
```

The ZINC preparer accepts a local SMILES or delimited file, uses deterministic
selection/splitting, preserves aromatic bonds, and applies strict RDKit
sanitization. A source download is intentionally not bundled; see
[Explicit placeholders](#explicit-placeholders).

## Degree priors

Train a prior:

```bash
PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/dhvae/community_small.yaml
```

Other maintained configurations are:

- `configs/experiments/dhvae/ego_small.yaml`
- `configs/experiments/dhvae/grid.yaml`
- `configs/experiments/dhvae/qm9_typed.yaml`
- `configs/experiments/dhvae/zinc_typed.yaml`

Evaluate the sampled invariant distribution and constructor yield:

```bash
PYTHONPATH=src python scripts/evaluate_degree_generator.py \
  --config configs/experiments/dhvae/community_small.yaml \
  --num-samples 1024 \
  --max-reference-sequences 1024
```

Main configurations use `fallback: error`: exhausted model draws are reported
instead of being silently replaced by training examples.

## Predictor and selector training

```bash
PYTHONPATH=src python scripts/train_hybrid_endpoint_grapher.py \
  --config configs/experiments/grapher/community_small_hybrid_endpoint_graphlet.yaml \
  --output-dir outputs/hybrid_endpoint/sbm/seed_42 \
  --seed 42
```

The command trains the target-summary predictor, then trains the configured
candidate selector from cached teacher action distributions. It writes the best
predictor checkpoint, selector checkpoint, training history, and teacher
diagnostics. Molecular configurations default to streaming trajectories.

Maintained graph-refiner configurations are:

- `configs/experiments/grapher/community_small_hybrid_endpoint_graphlet.yaml`
- `configs/experiments/grapher/ego_small_hybrid_endpoint_graphlet.yaml`
- `configs/experiments/grapher/grid_hybrid_endpoint_graphlet.yaml`
- `configs/experiments/grapher/qm9_attributed_hybrid_endpoint_graphlet.yaml`
- `configs/experiments/grapher/zinc_attributed_hybrid_endpoint_graphlet.yaml`

## Generation and evaluation

```bash
PYTHONPATH=src python scripts/run_hybrid_endpoint_grapher.py \
  --config configs/experiments/grapher/community_small_hybrid_endpoint_graphlet.yaml \
  --checkpoint outputs/hybrid_endpoint/sbm/seed_42/checkpoint.pt \
  --output-dir outputs/hybrid_endpoint/sbm/seed_42/generated \
  --num-generate 1024 \
  --seed 42

PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/grapher/community_small_hybrid_endpoint_graphlet.yaml \
  --generated-dir outputs/hybrid_endpoint/sbm/seed_42/generated \
  --output-dir outputs/hybrid_endpoint/sbm/seed_42/evaluation
```

For molecules, use the QM9 or ZINC graph-refiner configuration and run:

```bash
PYTHONPATH=src python scripts/evaluate_generated_molecules.py --help
```

Generation reports proposal/pass rates, valid-candidate counts, selector and
energy decisions, `STOP` behavior, accepted swaps, construction failures,
degree/typed-invariant preservation, connectivity, and molecular validity.

## Research protocol

`scripts/run_research_protocol.py` materializes a reproducible manifest for
ablations, cost sweeps, and external baselines over exactly seeds 42, 43, and
44. It is dry-run by default:

```bash
PYTHONPATH=src python scripts/run_research_protocol.py \
  --protocol path/to/protocol.yaml \
  --output-dir outputs/research_protocol

# Execute only after reviewing manifest.json.
PYTHONPATH=src python scripts/run_research_protocol.py \
  --protocol path/to/protocol.yaml \
  --output-dir outputs/research_protocol \
  --execute
```

Protocol commands must be YAML token lists, never shell strings. The harness
records expanded argv, environment, working directory, configuration
content/hash, timing, return codes, stdout/stderr, failures, and aggregate
reports. It rejects fallback-contaminated runs.

The study APIs in `src/grapher/evaluation/studies.py` provide stagewise error
decomposition, constructor bias/yield, exact small-graph switch-state coverage,
pair/invariant/graphlet consistency, local-summary collision tests, molecular
limitation audits, paired ablations, fixed-seed aggregation, and quality-cost
Pareto frontiers.

## Explicit placeholders

The manuscript leaves several research choices undefined or requires external
systems/data. These boundaries are callable placeholders that raise
`NotImplementedError` rather than returning a misleading approximation:

- projecting an infeasible predicted target onto the realizable summary set;
- hierarchical/coarsened summaries;
- exact constrained switch-state reachability above eight nodes;
- chemical stability, synthesizability, or 3D-quality oracles;
- external baseline runners without a pinned executable adapter; and
- an automatic/bundled ZINC source downloader.

The implemented refiner can still use an infeasible prediction as guidance when
`infeasible_target_policy: guidance_only`; that is distinct from claiming an
exact projection.

## Verification

```bash
python -m pytest -q
ruff check src scripts tests
ruff format --check src scripts tests
python -m compileall -q src scripts tests
```

The tests cover invariants and constructor postconditions, randomized property
cases, attributed canonicalization, singleton graphs, equivariance, teacher
distributions, selector modes and `STOP`, molecular validity, evaluation
metrics, research studies, protocol safety, configuration paths, and checkpoint
round trips.

## Reproducibility rules

- Prepare and freeze one dataset split before running model seeds.
- Use seeds 42, 43, and 44 in that order for paper-facing aggregation.
- Tune on validation data and evaluate the final choice once on test data.
- Report feasibility, rejection, restart, and fallback counts with quality
  metrics.
- Keep proposal budget, valid-candidate budget, policy shortlist, and graphlet
  scoring cost separate.
- Label energy-only, policy-only, hybrid, empirical-invariant, and
  oracle-invariant runs explicitly.
- Do not mix topology-only and attributed graphlet results.
