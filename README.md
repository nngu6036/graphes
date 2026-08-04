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
PYTHONPATH=src python scripts/train_hybrid_endpoint_grapher.py \
  --config configs/experiments/sbm_hybrid_endpoint_graphlet.yaml
```

Generate and evaluate graphs:

```bash
PYTHONPATH=src python scripts/run_hybrid_endpoint_grapher.py \
  --config configs/experiments/sbm_hybrid_endpoint_graphlet.yaml \
  --output-dir outputs/hybrid_endpoint/sbm/generated
```

Train or evaluate the optional degree VAE:

```bash
PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/degreevae.yaml
PYTHONPATH=src python scripts/evaluate_degree_generator.py \
  --config configs/experiments/degreevae.yaml
```

## Molecular baseline

Prepare the QM9 topology and attributed dataset splits directly from the QM9
SDF file (the PyG download places it at the path below):

```bash
PYTHONPATH=src python scripts/prepare_qm9_topology_dataset.py \
  --source sdf \
  --sdf-file data/pyg_qm9/raw/gdb9.sdf \
  --root outputs/datasets
```

The direct SDF loader skips individual records that RDKit cannot parse, unlike
PyG's QM9 preprocessing, which may stop at the first invalid record. The script
can also read SMILES input; run it with `--help` for all options. It writes both
`qm9_topology` and `qm9_attributed` splits. Then use
`configs/experiments/qm9_attributed_hybrid_endpoint_graphlet.yaml`. Both train
and generation commands print a warning because the current teacher,
constructor, and graphlet targets do not preserve the paper's typed invariant.
Do not report this route as the full molecular method.

Molecular evaluation utilities are:

```bash
python scripts/evaluate_generated_molecules.py --help
python scripts/evaluate_graph_generation_report.py --help
```

## Verification

```bash
ruff check src scripts tests
ruff format --check src scripts tests
PYTHONDONTWRITEBYTECODE=1 pytest -q
python scripts/train_hybrid_endpoint_grapher.py --help
python scripts/run_hybrid_endpoint_grapher.py --help
```
