# Comparable baseline training budgets

Default comparison runs match approximate **training graph exposure** per dataset.
One pass presents each graph in the prepared training split once. These are project
comparison budgets, not claims of convergence or published settings.

| Dataset | Target train-set passes | Graph exposures |
| --- | ---: | ---: |
| Community-small | 10,000 | 640,000 with 64 training graphs |
| Ego-small | 10,000 | 1,280,000 with 128 training graphs |
| Grid | 10,000 | 700,000 with 70 training graphs |
| QM9 | 200 | 20,933,000 with 104,665 training graphs |
| ZINC | 200 | 200 times the actual prepared training count |

The targets follow the user-specified GDSM-simple reference budget: 10,000 passes
for generic datasets and 200 for molecular datasets. Neither target was selected
using test metrics.

## Audit and changes

| Baseline | Previous generic horizon (Community / Ego / Grid) | New default |
| --- | --- | --- |
| DiGress | 1,000,000 / 1,000,000 / 100,000 epochs | 10,000 epochs |
| DeFoG | 1,000,000 / 1,000,000 / unsupported | 10,000 epochs |
| GDSM (GSDM alias) | 200 / 200 / 5,000 epochs | 10,000 epochs |
| GDSM-simple | 5,000 / 200 / 200 epochs | 10,000 epochs |
| CatFlow | 10,000 / 1,000 / 1,000 epochs | 10,000 epochs |
| GDSS | 5,000 epochs | 10,000 epochs |
| EDGE | 50,000 epochs | 10,000 epochs |
| SPECTRE | 12,000 epochs | 10,000 epochs |
| GraphRNN | 3,000 epochs of 32 batches of 32 | approximately 10,000 passes |
| HOG-Diff | separate higher-order and OU iteration budgets | approximately 10,000 passes summed across stages |
| DH-VAE+HH | 5,000 / 5,000 / 300 epochs | 10,000 epochs |

Supported default molecular profiles use 200 passes: DiGress, DeFoG, CatFlow,
GDSS, GDSM-simple, DH-VAE+HH, HOG-Diff, and SPECTRE (QM9 only). HOG-Diff shares
these passes across its stages instead of granting 200 independently to each.

Previous model budget declarations remain under `comparison.native_budget` for
audit purposes. Some were already adaptations; that field does not establish an
official upstream configuration. Explicit GDSM-simple categorical, structure3,
option-A and other ablation configs retain their experiment-specific budgets;
selecting one opts out of the default matched comparison.

## Exposure calculation

For a complete DataLoader traversal, exposure is `epochs * N_train`, including
the final partial batch. Batch size 256 with 64 training graphs therefore consumes
64 graphs per epoch. Batch size changes update counts, not pass counts.

GraphRNN samples with replacement. Its wrapper resolves
`epochs = round(target_passes * N_train / (batch_size * batch_ratio))`, with a
minimum of one epoch. Nominal Community/Ego/Grid horizons are 625/1250/684 epochs.
The wrapper recalculates them using the actual prepared split and batch settings.

HOG-Diff resolves each stage's iterations using actual training count, effective
stage batch size, and `comparison.budget.stage_exposure_weights`. The weights
preserve the previous relative allocation of graph exposure between stages.
Counts round to the nearest iteration, with a minimum of one. The wrapper supplies
live `n_iters` rather than inheriting the million-iteration upstream ZINC budget.

Both sampled-loop wrappers record target exposure, estimated realized exposure,
training count, horizons and explicit overrides in
`comparison_reference.resolved_exposure_budget` in the training options/manifest.
The HOG-Diff estimate assumes one configured minibatch per iteration; upstream
loop endpoints or partial batches can cause small differences.

Exposure parity does not equalize GPU hours, FLOPs, optimizer updates, parameter
counts, or convergence. GDSM/GDSS train multiple components per batch; SPECTRE
trains generator/discriminator components. SPECTRE's empirical singleton branch
can exclude singleton graphs from its GAN loader. Report these differences where
applicable.

## Schedules and precedence

GDSM/GDSS exponential learning-rate decays preserve their original terminal decay:
`new_gamma = old_gamma ** (old_epochs / new_epochs)`. CatFlow's cosine schedule
already uses the configured horizon. SPECTRE's explicit warmup/temperature-decay
durations and GraphRNN's learning-rate milestones scale with their horizons.
DiGress and CatFlow validation/checkpoint cadences are adjusted. Model-specific
learning rates, batch sizes, architectures, EMA settings and samplers remain
separate choices.

Common configs now declare 10,000/200-epoch fallback horizons, including a new Grid
common file. DeFoG uses these directly; the other default model files agree with
their dataset target. Merge precedence is:

1. Wrapper/upstream defaults.
2. Common dataset fallback controls.
3. Model-specific wrapper YAML.
4. Explicit CLI or `TrainRequest.options` overrides.

GraphRNN/HOG-Diff conversion applies only to configs declaring
`comparison.budget.unit: train_set_passes`. Explicit epoch/iteration overrides
remain authoritative and are flagged in exposure metadata; they can break parity.
Batch overrides trigger conversion with the new batch size. Custom configs
without this declaration retain their existing behavior.

Normal `scripts/run_<model>_baseline.py --dataset <dataset>` commands need no new
flag. `--no-common-config` disables common fallback controls but does not undo a
budget declared in the selected model YAML. Use an explicit horizon or custom
wrapper config for other budgets.

Use identical prepared splits, seeds and evaluation; tune and select on validation
only. Config edits do not retrain existing checkpoints. New matched comparisons
require new training runs, retaining their resolved configuration hashes.

## Generate the dataset statistics table

Read all five prepared datasets and write the LaTeX table with exact node/edge
extrema over train, validation and test combined:

```bash
PYTHONPATH=src python scripts/create_dataset_statistics_table.py \
  --root outputs/datasets \
  --output outputs/reports/dataset_statistics.tex \
  --json-out outputs/reports/dataset_statistics.json
```

The script resolves Community-small to `sbm` and QM9 to `qm9_attributed` when
needed. Training epochs come from the common dataset exposure targets. Generated
sample counts are 1,024 for generic and 10,000 for molecular datasets. The table
includes its caption and label and requires `\usepackage{booktabs}` in the paper.

It reads one split at a time, never downloads/prepares data, and never plots or
generates graphs. Missing or malformed datasets fail without writing a partial
table. Use `--datasets community_small ego_small grid` to select a subset,
`--root` for another dataset location, and `--force` to replace existing reports.
