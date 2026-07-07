# GraphES Fresh Start: Coarse-to-Fine Graph Generation

This branch is a clean implementation scaffold for the new proposal:

> Generate global/local graph properties, build a coarse topology, then refine it with training-free GraphER-Opt rewiring.

The old DH-VAE, learned GraphER, molecular GraphER, and DiGress post-processing scripts have been removed from this fresh branch. The focus is now a small, verifiable pipeline:

```text
training graphs
  -> permutation-invariant property summaries
  -> empirical or learned summary generator
  -> coarse graph constructor
  -> GraphER-Opt energy-guided rewiring refiner
  -> evaluation
```

## Current scope

The current branch intentionally starts simple:

- generic featureless graphs only;
- SPECTRE-style SBM as the first dataset;
- empirical summary sampler as the first property generator;
- Havel-Hakimi coarse constructor;
- training-free GraphER-Opt refiner;
- placeholders for learned summary generation and attributed graph extension.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

## Prepare dataset
```bash
export PYTHONPATH=src

PYTHONPATH=src python scripts/prepare_sbm_spectre_dataset.py \
  --config configs/datasets/sbm_spectre.yaml \
  --root outputs/datasets
```

## Run a smoke experiment

```bash
PYTHONPATH=src python scripts/run_coarse_to_fine.py \
  --config configs/experiments/sbm_v0.yaml \
  --num-generate 20 \
  --output-dir outputs/coarse_to_fine/sbm_v0_smoke
```

## Verification-first workflow

Run these checks before adding more complexity:

```bash
PYTHONPATH=src python scripts/verify_stage.py --stage summary --config configs/experiments/sbm_v0.yaml
PYTHONPATH=src python scripts/verify_stage.py --stage constructor --config configs/experiments/sbm_v0.yaml
PYTHONPATH=src python scripts/verify_stage.py --stage refiner --config configs/experiments/sbm_v0.yaml
PYTHONPATH=src python scripts/verify_stage.py --stage equivariance --config configs/experiments/sbm_v0.yaml
```

Each stage is meant to be pass/fail. Do not proceed to a learned property generator or attributed graphs until these checks pass.

## Key modules

```text
src/grapher/properties/summary.py       invariant graph summaries and energy
src/grapher/properties/sampler.py       empirical summary sampler
src/grapher/construction/coarse.py      coarse topology constructors
src/grapher/refinement/rewiring.py      valid double-edge swap actions
src/grapher/refinement/grapher_opt.py   training-free rewiring optimizer
src/grapher/evaluation/metrics.py       generic graph metrics
src/grapher/pipeline/coarse_to_fine.py  end-to-end pipeline
```

## Future placeholders

```text
src/grapher/generators/summary_vae.py          learned summary generator placeholder
src/grapher/attributes/conditional_generator.py attributed graph extension placeholder
```


## Degree generator workflow

To improve degree MMD, train a dedicated degree-histogram VAE and use the hybrid summary sampler:

```bash
PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/sbm_report_degreevae.yaml \
  --output-dir outputs/degree_generators/sbm_report \
  --epochs 300 \
  --batch-size 32 \
  --beta 0.005 \
  --degree-weight 5.0 \
  --edge-moment-weight 0.1 \
  --seed 42

PYTHONPATH=src python scripts/verify_degree_generator.py \
  --config configs/experiments/sbm_report_degreevae.yaml \
  --num-samples 1000

PYTHONPATH=src python scripts/build_rewiring_teacher.py \
  --config configs/experiments/sbm_report_degreevae.yaml \
  --output-dir outputs/teachers/sbm_report_degreevae \
  --num-trajectories 512 \
  --seed 42 \
  --debug

PYTHONPATH=src python scripts/train_rewiring_selector.py \
  --config configs/experiments/sbm_report_degreevae.yaml \
  --teacher-dir outputs/teachers/sbm_report_degreevae \
  --output-dir outputs/selectors/sbm_report_degreevae \
  --epochs 100 \
  --batch-size 64 \
  --seed 42

ORCA_EXEC=/path/to/orca PYTHONPATH=src python scripts/run_coarse_to_fine.py \
  --config configs/experiments/sbm_report_degreevae.yaml \
  --num-generate 40 \
  --output-dir outputs/coarse_to_fine/sbm_report_degreevae_seed42 \
  --debug
```

## QM9 topology-first mixture CatFlow

Prepare QM9 topology-only and attributed molecular graph splits. By default this uses `torch_geometric.datasets.QM9`, so a separate SMILES file is not required if PyTorch Geometric is installed:

```bash
PYTHONPATH=src python scripts/prepare_qm9_topology_dataset.py \
  --source sdf \
  --sdf-file data/pyg_qm9/raw/gdb9.sdf \
  --root outputs/datasets
```

For custom SMILES files, use `--source smiles`:

```bash
PYTHONPATH=src python scripts/prepare_qm9_topology_dataset.py \
  --source smiles \
  --smiles-file data/qm9/qm9.smi \
  --root outputs/datasets
```

For custom CSV/TSV inputs, pass the SMILES column name:

```bash
PYTHONPATH=src python scripts/prepare_qm9_topology_dataset.py \
  --source smiles \
  --smiles-file data/qm9/qm9.csv \
  --smiles-column smiles \
  --root outputs/datasets
```

Useful options include `--max-molecules` for a small subset, `--keep-hydrogens` to keep explicit hydrogens, and `--no-kekulize` to skip kekulization for SMILES/RDKit input.

Train a topology-conditioned mixture CatFlow model for atom and bond labels:

```bash
PYTHONPATH=src python scripts/train_qm9_mixture_catflow.py \
  --config configs/experiments/qm9_topology_mixture_catflow.yaml \
  --output-dir outputs/attribute_flows/qm9_mixture_catflow \
  --epochs 100 \
  --batch-size 64 \
  --seed 42
```

Sample molecular graphs from generated topologies:

```bash
PYTHONPATH=src python scripts/sample_qm9_mixture_catflow.py \
  --checkpoint outputs/attribute_flows/qm9_mixture_catflow/checkpoint.pt \
  --topology-graphs outputs/coarse_to_fine/qm9_topology_seed42/learned_selector_graphs.pkl \
  --output-dir outputs/molecular/qm9_topology_first_mixture_catflow \
  --steps 64 \
  --temperature 1.0 \
  --seed 42
```

A CatFlow-style Stage-2 ablation is obtained by setting `num_mixtures: 1` in `configs/experiments/qm9_topology_catflow_stage2_ablation.yaml`.

Attribute-related topology summaries can be extracted with:

```bash
PYTHONPATH=src python scripts/extract_qm9_molecular_summaries.py \
  --dataset qm9_attributed \
  --root outputs/datasets \
  --output-dir outputs/molecular_summaries/qm9
```
