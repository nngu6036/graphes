#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHONPATH=src python scripts/prepare_generic_dataset.py --dataset community_small --root outputs/datasets
PYTHONPATH=src python scripts/prepare_generic_dataset.py --dataset ego_small --root outputs/datasets

: "${QM9_SDF:?Set QM9_SDF to gdb9.sdf}"
: "${QM9_UNCHARACTERIZED:?Set QM9_UNCHARACTERIZED to uncharacterized.txt}"
PYTHONPATH=src python scripts/prepare_qm9_dataset.py \
  --config configs/datasets/qm9.yaml \
  --source sdf \
  --sdf-file "$QM9_SDF" \
  --uncharacterized-file "$QM9_UNCHARACTERIZED" \
  --root outputs/datasets

: "${ZINC_SMILES:?Set ZINC_SMILES to the ZINC250k CSV/SMILES file}"
: "${ZINC_TEST_INDICES:?Set ZINC_TEST_INDICES to valid_idx_zinc250k.json}"
PYTHONPATH=src python scripts/prepare_zinc_dataset.py \
  --config configs/datasets/zinc.yaml \
  --smiles-file "$ZINC_SMILES" \
  --test-indices-file "$ZINC_TEST_INDICES" \
  --root outputs/datasets
