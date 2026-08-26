# Attributed Spectral--Graphlet Diffusion GraphER

This implementation extends the confirmed generic GraphER design to heavy-atom
molecular graphs.

## Invariant and transition

The source is an exact realization of the indexed typed-degree invariant

```text
(atom type, degree in every bond category)
```

and every generation transition is a same-bond-type double-edge swap. Therefore
atom categories, indexed typed degrees, ordinary degrees, global bond counts,
and weighted valence are preserved exactly. Simplicity and connectedness are
checked for every candidate.

## Continuous training state

Training does not construct a rewiring trajectory. For a clean molecule and a
same-invariant typed source, it computes:

- the ordinary topology Laplacian spectrum;
- the bond-order-weighted Laplacian spectrum; and
- attributed graphlet probability simplexes and their blockwise CLR logits.

A stochastic endpoint-conditioned bridge is sampled directly in these
continuous summary coordinates. The model predicts the clean endpoint from the
continuous current state, fixed source summaries, typed source graph, and
normalized diffusion progress.

## Generation

At generation time, the current realized graph supplies the two spectra and
attributed graphlet logits. The predictor estimates the clean summaries, the
schedulers derive the next continuous denoising target, and valid rewiring
candidates are ranked by a global-to-local combination of dual-spectral and
attributed-graphlet distances.

Attributed graphlet candidates use an exact stateful local-delta cache:

```text
C_{t+1} = C_t + Delta C_t
```

A full graphlet count is required only for the source graph. RDKit sanitization
is applied only to the top combined-energy shortlist after inexpensive
invariant, connectivity, valence, spectral, and graphlet checks.

## Commands

Train the light QM9 development model:

```bash
PYTHONPATH=src python scripts/train_attributed_grapher.py \
  --config configs/experiments/grapher/qm9_attributed_spectral_graphlet_light.yaml \
  --output-dir outputs/attributed_grapher/qm9_spectral_graphlet_light/seed_42 \
  --seed 42 \
  --device gpu
```

Generate with empirical training typed invariants:

```bash
PYTHONPATH=src python scripts/run_attributed_grapher.py \
  --config configs/experiments/grapher/qm9_attributed_spectral_graphlet_light.yaml \
  --output-dir outputs/attributed_generation/qm9_spectral_graphlet_light/seed_42 \
  --num-generate 256 \
  --seed 42 \
  --device gpu
```

Evaluate:

```bash
PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-dir outputs/attributed_generation/qm9_spectral_graphlet_light/seed_42 \
  --dataset-root outputs/datasets \
  --dataset qm9_attributed \
  --reference-split test \
  --train-split train \
  --output-dir outputs/attributed_grapher/qm9_spectral_graphlet_light/seed_42/evaluation
```

The molecular report includes `validity_without_correction`, corrected
`validity` (also exposed as `validity_with_correction`), and FCD.  By default,
FCD uses the raw valid molecules to preserve the historical evaluator protocol.
Pass `--fcd-use-corrected` to use the corrected-valid set and `--require-fcd`
to make a missing or failed `fcd_torch` backend an error rather than returning
`fcd: null`.

The default config uses empirical typed invariants to isolate the new attributed
refiner. For end-to-end learned invariant generation, provide the typed-degree
VAE checkpoint and use:

```bash
--set generation.invariant_source=learned \
--set degree_generator.enabled=true
```
