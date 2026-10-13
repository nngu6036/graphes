# Prepared dataset contract fix

This patch replaces environment-dependent raw pickle SHA-256 validation for the pinned SPECTRE Community-small split with a semantic source-index contract.

Apply from the GraphES repository root:

```bash
unzip -o /path/to/graphes_prepared_dataset_source_index_fix_20261013.zip -d .
```

Then rebuild:

```bash
PYTHONPATH=src python scripts/prepare_generic_dataset.py \
  --dataset community_small \
  --root outputs/datasets
```

Expected split sizes: 64 / 16 / 20.

Validate:

```bash
PYTHONPATH=src pytest -q tests/test_prepared_dataset_contract.py
```

Expected: 4 passed.
