## Installation

Use Python 3.10 or newer. Install a PyTorch build appropriate for the machine,
then install this package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Optional features:

```bash
python -m pip install -e '.[molecular,reports]'
python -m pip install -e '.[qm9-pyg]'
python -m pip install -e '.[fcd]'
```

ORCA is optional for exact orbit/graphlet evaluation. Put `orca` on `PATH`, set
`ORCA_EXEC`, or set `evaluation.orca_exec` in the experiment configuration.

## Generic workflow

Prepare a dataset:

```bash
python scripts/prepare_generic_dataset.py \
  --dataset sbm \
  --root outputs/datasets
```

Train the endpoint/graphlet predictor:

```bash
python scripts/train_hybrid_endpoint_grapher.py \
  --config configs/experiments/sbm_hybrid_endpoint_graphlet.yaml
```

Generate and evaluate graphs:

```bash
python scripts/run_hybrid_endpoint_grapher.py \
  --config configs/experiments/sbm_hybrid_endpoint_graphlet.yaml \
  --output-dir outputs/hybrid_endpoint/sbm/generated
```

Train or evaluate the optional degree VAE:

```bash
python scripts/train_degree_generator.py \
  --config configs/experiments/degreevae.yaml
python scripts/evaluate_degree_generator.py \
  --config configs/experiments/degreevae.yaml
```

## Molecular baseline

Prepare attributed QM9 graphs from PyG, SDF, or SMILES with
`scripts/prepare_qm9_topology_dataset.py`, then use
`configs/experiments/qm9_attributed_hybrid_endpoint_graphlet.yaml`. Both train
and generation commands print a warning because the current teacher,
constructor, and graphlet targets do not preserve the paper's typed invariant.
Do not report this route as the full molecular method.

Molecular evaluation utilities are:

```bash
python scripts/evaluate_generated_molecules.py --help
python scripts/evaluate_graph_generation_report.py --help
```

## Maintained layout

| Path | Purpose |
| --- | --- |
| `src/grapher/construction/` | Ordinary-degree source construction |
| `src/grapher/generators/` | Empirical and learned degree sampling |
| `src/grapher/hybrid/` | Endpoint model, teacher data, and energy refiner |
| `src/grapher/refinement/` | Valid double-edge-swap primitives |
| `src/grapher/properties/` | Graph summaries and graphlet extraction |
| `src/grapher/molecular/` | Attribute initialization and chemical checks |
| `src/grapher/evaluation/` | Graph-set metrics |

Legacy summary-only generation, disconnected learned selectors, alternative
CatFlow attribute models, forwarding scripts, plotting scripts, and generated
bytecode were removed from this refactor. Reusable correctness boundaries such
as action validation, molecular checks, and graphlet canonicalization remain
factored rather than being inlined into callers.

## Verification

```bash
ruff check src scripts tests
ruff format --check src scripts tests
PYTHONDONTWRITEBYTECODE=1 pytest -q
python scripts/train_hybrid_endpoint_grapher.py --help
python scripts/run_hybrid_endpoint_grapher.py --help
```
