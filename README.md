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

