# Baseline experiments

This document is the canonical external-baseline guide for GraphER.

## Fair-comparison rule

A fair comparison does **not** require every model to receive the same numeric
`epochs` value. The uploaded implementations use incompatible optimization
units: full-data epochs, a fixed number of mini-batches per epoch, or two
separate iteration schedules. Setting all models to DeFoG's one-million epochs
would therefore change model budgets by orders of magnitude without creating a
meaningful compute or convergence control.

GraphER uses four layers, in this order:

1. upstream implementation defaults;
2. `common_<dataset>.yaml` DeFoG-reference fallbacks and protocol provenance;
3. `<model>_<dataset>.yaml` native/equivalent model settings;
4. explicit CLI overrides.

The model-specific YAML always wins over common training values. CLI overrides
win over both. Use a new run ID whenever a training control changes.

The common layer is authoritative for the *comparison protocol*: frozen data
representation and split hashes, validation-only model selection, final test
evaluation, declared seeds, requested sample count in a report, and evaluator.
Training horizons and optimizer schedules are authoritative in the model config
when their semantics are model-specific.

For a model/dataset pair with no released upstream profile, the YAML is marked
as an adaptation. Such budgets may be tuned on the validation split only. Test
MMD must not be used to choose epochs, checkpoint cadence, architecture, or
sampling hyperparameters.

## DeFoG-reference common profiles

```text
configs/baselines/common_community_small.yaml
configs/baselines/common_ego_small.yaml
configs/baselines/common_qm9.yaml
configs/baselines/common_zinc.yaml
```

The `training:` block remains the DeFoG reference/fallback. The `policy:` block
explicitly records that raw epoch counts are not cross-model equivalent.
`reference_only:` stores DeFoG-specific sampler/model values without applying
them to other methods.

Ego-small is special: the supplied DeFoG source has no profile matching the
GraphER/GDSS Ego-small benchmark. Its common profile is therefore explicitly
adapted from DeFoG `comm20`.

## Source-reviewed training budgets

The following values were extracted from the uploaded baseline implementations
or their released command/config files. "Adapted" means the released source has
no exact GraphER dataset profile; the closest architecture/profile is used while
training on the frozen GraphER split.

### Community-small

| Model | Model-specific budget | Status/source |
|---|---:|---|
| DeFoG | 1,000,000 epochs; batch 256; AdamW 2e-4 | exact `comm20` profile |
| DiGress | 1,000,000 epochs; batch 256; AdamW 2e-4 | exact `configs/experiment/comm20.yaml` |
| GDSS | 5,000 epochs; batch 128; Adam 1e-2 | exact `config/community_small.yaml` |
| GSDM | 200 epochs; batch 128; Adam 1e-2 | exact `config/community_small.yaml` |
| EDGE | 50,000 epochs; batch 8; Adam 1e-4 | released README generic training budget; 64 diffusion steps for small graphs |
| GraphRNN | 3,000 epochs × 32 mini-batches/epoch; batch 32; lr 3e-3 | released `args.py` default schedule |
| HOG-Diff | 6,000 HO iterations (batch 64) + 22,000 OU iterations (batch 32) | exact `configs/cs.yaml` |
| SPECTRE | 12,000 epochs; batch 10; Adam G/D 1e-4 | released Community command |
| CatFlow | 10,000 epochs; batch 128; AdamW 2e-4 | GraphER linear-v2 adapter; uploaded source has no generic graph profile |

### Ego-small

| Model | Model-specific budget | Status/source |
|---|---:|---|
| DeFoG | 1,000,000 epochs; batch 256 | Comm20-derived reference; no native GraphER Ego-small profile |
| DiGress | 1,000,000 epochs; batch 256 | adapted from released Comm20 profile |
| GDSS | 5,000 epochs; batch 128; Adam 1e-2 | exact `config/ego_small.yaml` |
| GSDM | 200 epochs; batch 128 | adapted from GSDM Community-small; no native Ego profile |
| EDGE | 50,000 epochs; batch 8 | released generic budget with small-graph adapter |
| GraphRNN | 3,000 × 32 mini-batches; batch 32 | released default budget; `small` width adaptation |
| HOG-Diff | 14,000 HO (batch 32) + 4,000 OU (batch 64) | exact `configs/ego.yaml` |
| SPECTRE | 12,000 epochs; batch 10 | adapted from Community profile; no native Ego profile |
| CatFlow | 1,000 epochs; batch 128 | adapter budget; no released Ego profile, validate before reporting |

### QM9

| Model | Model-specific budget | Status/source |
|---|---:|---|
| DeFoG | 1,000 epochs; batch 1024; AdamW 2e-4 | exact `qm9_no_h` profile |
| DiGress | 1,000 epochs; batch 1024; AdamW 2e-4 | exact `qm9_no_h` profile |
| GDSS | 300 epochs; batch 1024; Adam 5e-3 | exact `config/qm9.yaml` |
| HOG-Diff | 92,000 HO (batch 32) + 102,000 OU (batch 512) | exact `configs/qm9.yaml` |
| SPECTRE | 30 epochs; batch 128; 3 G / 3 D updates | released QM9 command; GraphER wrapper handles singleton molecules explicitly |
| CatFlow | 300 epochs; batch 1024; AdamW 2e-4 | released CatFlow QM9 command |

EDGE, GraphRNN and the uploaded GSDM release do not provide the categorical
molecular generation required by this QM9 protocol.

For CatFlow, the molecular epoch/batch/LR budgets above are taken from the
released README, while the GraphER integration uses the explicit `linear_v2`
probability path recorded in its checkpoint manifest. This is therefore not a
bit-for-bit execution of the uploaded fixed-endpoint-noise branch; the
integration choice must be reported with the result.

### ZINC

| Model | Model-specific budget | Status/source |
|---|---:|---|
| DeFoG | 300 epochs; batch 256; AdamW 2e-4 | exact ZINC profile |
| DiGress | 1,000 epochs; batch 256 | GraphER adapter; uploaded DiGress has no ZINC experiment profile |
| GDSS | 500 epochs; batch 1024; Adam 5e-3 | exact `config/zinc250k.yaml` |
| HOG-Diff | 12,000 HO (batch 1024) + 1,000,000 OU (batch 64) | exact `configs/zinc250k.yaml` |
| CatFlow | 500 epochs; batch 1024; AdamW 2e-4 | released CatFlow ZINC command |

The uploaded SPECTRE, EDGE, GraphRNN and GSDM releases are not used as ZINC
categorical-molecule baselines here.

## Why these budgets are not “equal compute”

The table above is a source-faithful/native-or-adapted **quality budget**. It is
not an equal-FLOP benchmark. Examples:

- GraphRNN defines one epoch as 32 sampled mini-batches, not one full pass over
  the frozen split.
- HOG-Diff has two separately optimized models, with different batch sizes and
  learning rates.
- SPECTRE trains adversarial generator/discriminator updates.
- DeFoG/DiGress use very large epoch counts on small datasets.

If compute-normalized comparison is required, report wall-clock/GPU-hours,
optimizer steps, and generated-sample runtime separately. Do not replace the
native schedules with a common integer and call that fair training.

## Frozen datasets

| CLI dataset | Serialized directory | Type |
|---|---|---|
| `community_small` | `outputs/datasets/sbm` | generic |
| `ego_small` | `outputs/datasets/ego_small` | generic |
| `grid` | `outputs/datasets/grid` | generic |
| `qm9` | `outputs/datasets/qm9_attributed` | molecular |
| `zinc` | `outputs/datasets/zinc` | molecular |

Every baseline must reuse the same `train.pkl`, `val.pkl`, and `test.pkl` bytes.
Do not regenerate or reshuffle splits between methods.

## Runner behavior

Every ``scripts/run_<model>_baseline.py`` file is now the same thin shim over
``grapher.models.external_cli``.  Baseline-specific parsing and option
translation live in that shared module rather than in individual scripts.

All maintained runners share the same lifecycle:

```text
--stage train
--stage generate
--stage all        # default: train then generate
```

They also share the common comparison controls:

```text
--common-config PATH
--no-common-config
--wrapper-config PATH
--seed-id N
--generation-seed N
--run-id ID
--generation-id ID
--num-samples N
--device DEVICE
```

Model-specific compatibility flags such as ``--n-epochs`` (DeFoG/DiGress),
``--batch-ratio`` (GraphRNN), and ``--ho-iters``/``--ou-iters`` (HOG-Diff) are
accepted by the same shared parser and translated centrally into wrapper
options.

For Community-small/Ego-small/QM9/ZINC the common config is auto-discovered when
not supplied. Model YAMLs are also auto-discovered. This means the normal command
is intentionally short:

```bash
PYTHONPATH=src python scripts/run_hog_diff_baseline.py \
  --dataset community_small \
  --num-samples 1024 \
  --seed-id 42 \
  --run-id seed_42 \
  --device gpu
```

That command applies `common_community_small.yaml` as fallback/provenance and
then `hog_diff_community_small.yaml`, so the effective HOG-Diff budget is 6k +
22k iterations, **not** one million iterations.

Use `--no-common-config` only for a native-only diagnostic. The model YAML still
loads unless explicitly replaced with `--wrapper-config`.

## Entry points

| Baseline | Runner | Supported GraphER datasets |
|---|---|---|
| DeFoG | `scripts/run_defog_baseline.py` | Community-small, Ego-small, QM9, ZINC |
| DiGress | `scripts/run_digress_baseline.py` | Community-small, Ego-small, Grid, QM9, ZINC |
| GDSS | `scripts/run_gdss_baseline.py` | Community-small, Ego-small, Grid, QM9, ZINC |
| GraphRNN | `scripts/run_graphrnn_baseline.py` | Community-small, Ego-small, Grid |
| HOG-Diff | `scripts/run_hog_diff_baseline.py` | Community-small, Ego-small, QM9, ZINC |
| CatFlow | `scripts/run_catflow_baseline.py` | Community-small, Ego-small, Grid, QM9, ZINC |
| GSDM/GDSM | `scripts/run_gdsm_baseline.py` (`run_gsdm_baseline.py` alias) | Community-small, Ego-small, Grid |
| EDGE | `scripts/run_edge_baseline.py` | Community-small, Ego-small, Grid |
| SPECTRE | `scripts/run_spectre_baseline.py` | Community-small, Ego-small, Grid, adapted QM9 |
| DH-VAE + HH | `scripts/run_dhvae_hh_baseline.py` | GraphER internal prior baseline; not governed by the external-source budget table above |

`run_source_baseline.py --model {catflow,gdsm,edge,spectre}` is an equivalent
consolidated entry point for the four source-backed wrappers.

## External environments

```bash
export DEFOG=/absolute/path/to/DeFoG
export DEFOG_PYTHON=/absolute/path/to/defog-env/bin/python
export DIGRESS=/absolute/path/to/DiGress
export DIGRESS_PYTHON=/absolute/path/to/digress-env/bin/python
export GDSS=/absolute/path/to/GDSS
export GDSS_PYTHON=/absolute/path/to/gdss-env/bin/python
export GRAPHRNN=/absolute/path/to/GraphRNN
export GRAPHRNN_PYTHON=/absolute/path/to/graphrnn-env/bin/python
export HOGDIFF=/absolute/path/to/HOG-Diff
export HOGDIFF_PYTHON=/absolute/path/to/hogdiff-env/bin/python
export CATFLOW=/absolute/path/to/CatFlow
export CATFLOW_PYTHON=/absolute/path/to/catflow-env/bin/python
export GDSM=/absolute/path/to/Fast_Graph_Generation_via_Spectral_Diffusion
export GDSM_PYTHON=/absolute/path/to/gdsm-env/bin/python
export EDGE=/absolute/path/to/graph-generation-EDGE
export EDGE_PYTHON=/absolute/path/to/edge-env/bin/python
export SPECTRE=/absolute/path/to/SPECTRE
export SPECTRE_PYTHON=/absolute/path/to/spectre-env/bin/python
export PYTHONPATH=src
```

## Separate train/generate example

Every maintained model exposes the same explicit stages.  For example:

```bash
PYTHONPATH=src python scripts/run_catflow_baseline.py \
  --stage train --dataset qm9 --seed-id 42 --run-id seed_42 --device gpu

PYTHONPATH=src python scripts/run_catflow_baseline.py \
  --stage generate --dataset qm9 --seed-id 42 --run-id seed_42 \
  --num-samples 1024 --device gpu
```

Omitting ``--stage`` is equivalent to ``--stage all`` for every model.  A
``--stage generate`` command reuses the managed checkpoint recorded under the
selected ``--run-id``.  HOG-Diff retains ``--generation-only`` as a backward-
compatible alias for ``--stage generate``.

## Evaluation

Generic graph outputs use the shared evaluator:

```bash
GEN_DIR=outputs/baselines/<model>/community_small/<run>/generations/<generation>

PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN_DIR" \
  --generated-stage <model> \
  --output-dir "$GEN_DIR/evaluation_report"
```

Molecular outputs use `scripts/evaluate_generated_molecules.py` with the same
reference split and molecule representation for every compatible model.

## Reproducibility checklist

For each reported result retain:

- upstream repository/source digest;
- frozen split hashes;
- common-profile path/hash;
- model-specific YAML path and `comparison.budget` provenance;
- training and generation seeds;
- selected checkpoint rule;
- optimizer steps or stage iterations where meaningful;
- exact requested/generated sample count;
- evaluator configuration;
- wall-clock/hardware metadata for compute comparisons.
