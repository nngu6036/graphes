# QM9 typed priors and Community-small graphlets 3–5

Base archive audited: `graphes(20260912-163051).zip`.

## What the original QM9 command does

`configs/experiments/grapher/qm9_attributed_joint_typed_edge.yaml` specifies:

```yaml
generation:
  invariant_source: learned
  sample_num_nodes: empirical
```

`run_attributed_grapher.py` dispatches to `generate_joint_typed_edge` because
`joint_typed_degree.enabled` is true. The learned branch calls
`model.degree_model.sample_outputs(...)`, then converts its output to feasible typed
invariants. The empirical choice above applies only to node count. Generation uses
the degree model embedded in the selected joint checkpoint, not a separate degree
checkpoint. `TRAIN/checkpoint.pt` is an alias/copy of the `best_joint` snapshot in
this training family.

The supplied base config does not enable an induced-graphlet head for QM9. Its
auxiliary structural heads are clustering and four-node orbits. A progress-report
flag does not turn on graphlet prediction. These model settings are unchanged by
this compatibility patch.

## Compatibility correction required in this archive

The original joint generation source dispatcher accepted only `learned` and
`train_empirical`. The previously delivered typed perturbation module and integration
were absent from this upload. This patch restores those pieces specifically for
the joint typed generation path. It does NOT overwrite the newer adjacency diffusion
implementation, checkpoint semantics, or saved-random-training-subset restoration.
Do not apply the earlier full typed patch over this archive: it predates those changes.

`generation.invariant_source=train_empirical_perturbed` now supports
`unit_transfer`, `moment_preserving`, `edge_relocation`, and `interpolation`.
`generation.invariant_rng_mode=independent` provides the corresponding matched-parent
empirical control. Accepted changes preserve node categories, node count and each
edge-type incidence total; feasibility, generation limits and the checkpoint's fixed
signature vocabulary are checked. Unavailable changes are recorded rather than
silently selecting another method or parent.

Generation restores the training subset saved in the joint checkpoint. For the
20,000-graph training example, that is the saved random subset, not the first
20,000 graphs and not validation/test graphs. The empirical sampler also reports
training parents excluded by configured generation limits.

The patch additionally records the checkpoint SHA256 for the non-joint multi-size
topology predictor so the existing perturbation comparison script can audit it.
No topology prediction or candidate-scoring code was changed.

## Install

Extract `graphes_prior_options_163051_patch.zip` at the repository root, over the
files from the audited archive. Existing datasets and checkpoints are retained.
The patch manifest lists original and updated hashes of each included file.

## Unified runner

```bash
bash scripts/run_prior_options.sh DATASET STAGE METHOD
```

- DATASET: `qm9` or `community_small`.
- STAGE: `all` (train if needed, then generate/evaluate), `train`, or `generate`
  (generation and evaluation only).
- METHOD: `all`, `empirical`, `unit_transfer`, `moment_preserving`,
  `edge_relocation`, `interpolation`, or `learned`.
- `all` means the empirical control and four perturbations, not learned sampling.

Training is shared by all priors. An existing `CKPT` is reused; to train a new
model, choose a fresh `TRAIN`. Generation refuses every nonempty output directory.
Use either the all-method command or the individual-method commands for a given
output root. Set a new `OUT_ROOT` for a rerun.

### QM9

Defaults match the user's paths:

```bash
CFG=configs/experiments/grapher/qm9_attributed_joint_typed_edge.yaml
TRAIN=outputs/attributed_grapher/qm9_joint_typed_edge/seed_42
CKPT="$TRAIN/checkpoint.pt"
```

Existing model; generate/evaluate all empirical-prior variants:

```bash
NGEN=1024 SEEDS="42 43 44" \
  bash scripts/run_prior_options.sh qm9 generate all
```

Train once, if absent, and generate/evaluate:

```bash
NTRAIN=20000 NGEN=1024 SEEDS="42 43 44" \
  bash scripts/run_prior_options.sh qm9 all all
```

Select a single variant (one generation seed by default):

```bash
bash scripts/run_prior_options.sh qm9 generate empirical
bash scripts/run_prior_options.sh qm9 generate unit_transfer
bash scripts/run_prior_options.sh qm9 generate moment_preserving
bash scripts/run_prior_options.sh qm9 generate edge_relocation
bash scripts/run_prior_options.sh qm9 generate interpolation
```

`qm9 generate learned` explicitly tests the original embedded-DH-VAE source.
The runner uses `evaluate_generated_molecules.py` with `--dataset qm9_attributed`,
`--reference-split val`, `--train-split train`, `--metric-molecule-source raw_valid`,
`--nspdk-backend eden`, and `--require-fcd`. There is no comma after `--require-fcd`.
Production evaluation requires the installed FCD/EDeN dependencies.
The base QM9 configuration retains its positive proposal and valid-candidate budgets
(1024 and 256). Its joint typed refiner does not accept a `-1` candidate cap.

The configured typed prior warm start is still required if training a new QM9 model:
`outputs/degree_generators/qm9_typed/seed_42/checkpoint.pt`.

### Community-small: true simultaneous graphlet sizes 3, 4, 5

Use the supplied config:
`configs/experiments/grapher/community_small_spectral_graphlet35_prior_options.yaml`.
It is based on the existing `community_small_topology_spectral_graphlet_v2.yaml`
family (`spectral_graphlet_transformer`), not the previous joint-degree
clustering/orbit-only predictor. The single-k `...joint_degree_graphlets5.yaml`
config is not equivalent: its `induced_graphlet_k` accepts one integer only.
Changing legacy `graphlet_k_min/max` fields alone does not make that single-k head
multi-scale.

The multi-size family trains a connected-induced-graphlet block for each of
k=3,4,5, each with one additional aggregate disconnected-subset bin. The resulting
block widths are 3,7,22. Both its graphlet-logit and graphlet-probability losses
are enabled. The config retains the V2 model/bridge/guidance settings, with a
shared 32 accepted-step limit, 1024 proposals, uncapped valid candidates, no source
enrichment, and independent empirical parent sampling for this prior comparison.
Those are experimental search settings, not a claim of optimality.

Train a new checkpoint once and generate/evaluate all variants:

```bash
EPOCHS=200 NGEN=64 SEEDS="42 43 44" \
  bash scripts/run_prior_options.sh community_small all all
```

Here 200 epochs is an explicit pilot override; without `EPOCHS` the base V2
configuration retains 800 epochs. Community-small uses its existing full training
split. Do not pass QM9's 20,000-graph limit as a dataset-size claim.

Generation and evaluation only, after training:

```bash
NGEN=64 SEEDS="42 43 44" \
  bash scripts/run_prior_options.sh community_small generate all
```

Replace `all` with one of the same five prior names to test a single option.
The default training output is
`outputs/topology_grapher/community_small_spectral_graphlet35_prior_options/seed_42`.
No separate DH-VAE training is needed for these empirical Community-small variants.
The runner uses `evaluate_graph_generation_report.py` and then the existing
`summarize_degree_perturbation_evaluations.py` with matching-parent checks.

## Common controls and outputs

Defaults: `DEVICE=gpu`, training seed 42, `SEEDS=42`, `REFERENCE_SPLIT=val`,
`PERTURB_PROBABILITY=0.25`, one accepted perturbation attempt target,
`max_attempts=256`, `max_distance=4.0`, `failure_policy=keep_original`, and
`require_novel=false`.

The 0.25 probability requests a perturbation for approximately 25% of samples; it
is not a guarantee of a 25% changed or novel fraction. Impossible requests retain
the same parent and are recorded. Use a separate output root when changing this
probability.

Examples:

```bash
PERTURB_PROBABILITY=1.0 OUT_ROOT=outputs/attributed_generation/qm9_prior_rho100 \
  bash scripts/run_prior_options.sh qm9 generate edge_relocation
```

QM9 outputs default to
`outputs/attributed_generation/qm9_typed_prior_options/ckpt_seed_42/generation_seed_SEED/METHOD`.
Read `typed_degree_prior_report.json`, `report.json`, and
`evaluation_val/molecular_evaluation_metrics.json`.

Community-small outputs default to
`outputs/topology_generation/community_small_graphlet35_prior_options/ckpt_seed_42/generation_seed_SEED/METHOD`.
Read `degree_prior_report.json` and `evaluation_val/graph_mmd_metrics.csv`.
For an all-method run, the summary is `summary_val_all/aggregates.csv` under that root.

The evaluation train split used for molecular novelty is the full prepared training
split; this does not expand the empirical sampling bank beyond the model's saved
training subset.

## Validation scope

Tests exercise ordinary and typed perturbations, the existing new adjacency
implementation, restoration of a random saved training subset, toy training and
CLI generation/evaluation for every prior with graphlet sizes 3–5, and runner
argument routing. Toy molecular evaluation explicitly skips FCD and uses the proxy
NSPDK solely as an interface test; production commands require real FCD and EDeN.
The user's trained checkpoints and prepared benchmark splits were not present, so
no new QM9/Community-small benchmark metrics are claimed.
