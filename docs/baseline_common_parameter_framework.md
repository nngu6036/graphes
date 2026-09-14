# DeFoG-referenced common baseline parameter framework

## Purpose

The common files under `configs/baselines/common_<dataset>.yaml` define a
**controlled comparison layer** shared by baseline wrappers. DeFoG is the
reference implementation for this layer.

This is intentionally different from forcing every baseline to emulate
DeFoG's internal algorithm. Parameters are split into two categories:

1. **Portable comparison controls**: training horizon, batch size, optimizer,
   learning rate, weight decay, gradient clipping, EMA decay, data-loader
   worker count, validation cadence, and representation flags that have the
   same meaning across implementations.
2. **Reference-only / model-specific settings**: DeFoG transition type,
   transformer depth, RRWP settings, time distortion, balanced-rate-matrix
   parameters (`eta`, `omega`), sampler steps, validation-time sample counts,
   and DeFoG visualization/checkpoint conveniences. These are recorded for
   provenance but are not imposed on other algorithms.

The controlled profile is useful for an equal-budget experiment. It should not
replace an **official/recommended-hyperparameter** comparison, because forcing a
single optimizer or learning rate across different architectures can favor one
model and handicap another. A strong paper can report both:

- **Protocol-controlled**: common data, split, seeds, evaluation, hardware, and
  DeFoG-reference portable training controls.
- **Native/recommended**: each baseline's published training hyperparameters,
  while retaining the same GraphER data/evaluation protocol.

## Source-derived profiles

### Community-small

GraphER Community-small corresponds to DeFoG `comm20`. The portable DeFoG
settings are:

- epochs: 1,000,000
- batch size: 256
- optimizer: AdamW
- learning rate: 2e-4 (inherited from `train_default.yaml`)
- weight decay: 1e-12 (inherited)
- gradient clipping: disabled (inherited)
- EMA decay: 0 (inherited)
- data-loader workers: 0 (inherited)
- validation every 1,000 epochs

### Ego-small

The attached DeFoG code has no Hydra experiment for GraphER's Ego-small. Its
`SpectreGraphDataset` contains an `ego` loader for the separate 757-graph EDGE
Ego benchmark, and `src/main.py` does not expose that dataset in the generic
training branch. Therefore `common_ego_small.yaml` explicitly reuses the
Comm20 small-generic-graph budget as a compatibility reference. It is an
adaptation, not an official DeFoG Ego-small configuration.

### QM9

The profile is DeFoG `qm9_no_h`:

- epochs: 1,000
- batch size: 1,024
- optimizer: AdamW (inherited)
- learning rate: 2e-4 (inherited)
- weight decay: 1e-12 (inherited)
- gradient clipping: disabled (inherited)
- EMA decay: 0 (inherited)
- validation every 50 epochs
- heavy-atom / remove-H representation
- pin memory: true

`qm9_no_h.yaml` contains `dataset.num_workers: 16`, but DeFoG's
`AbstractDataModule` reads `cfg.train.num_workers`. The effective upstream
value is therefore `0` from `train_default.yaml`; the common config records
that effective value and preserves the misplaced declaration under
`reference_only`.

### ZINC

The profile is DeFoG `zinc`:

- epochs: 300
- batch size: 256
- optimizer: AdamW
- learning rate: 2e-4
- weight decay: 1e-12 (inherited)
- gradient clipping: disabled (inherited)
- EMA decay: 0 (inherited)
- data-loader workers: 4
- validation every 4 epochs
- remove H: true
- aromatic categorical edge class: false

## DeFoG runner integration

`run_defog_baseline.py` now accepts:

```bash
--common-config configs/baselines/common_<dataset>.yaml
```

Precedence is:

1. upstream DeFoG defaults / experiment config
2. DeFoG-specific `--wrapper-config`
3. portable values from `--common-config`
4. explicit CLI controls such as `--n-epochs`

The common-config path, SHA256 digest, reference identity, and applied values
are retained in the training options and therefore in DeFoG's GraphER training
manifest.

Example:

```bash
PYTHONPATH=src python scripts/run_defog_baseline.py \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --num-samples 1024 \
  --seed-id 42 \
  --device gpu
```

`--num-samples` remains an explicit evaluation-protocol control. DeFoG's native
`final_model_samples_to_generate` is recorded under `reference_only`, because
copying `20` samples from Comm20 into a generic MMD benchmark would create a
high-variance comparison. Sampling-step counts and DeFoG-specific rate-matrix
parameters likewise remain method-specific.
