#!/usr/bin/env bash
# Prepare the Ego-small, Grid, QM9, and ZINC datasets used by GraphER.
#
# Usage:
#   bash scripts/prepare_benchmark_datasets.sh /path/to/zinc250k.csv [smiles_column]
#
# The ZINC source may also be supplied through ZINC_SMILES_FILE. For a CSV or
# TSV source, pass its SMILES column as the second argument or set
# ZINC_SMILES_COLUMN. A column is unnecessary for .smi/.txt files.
#
# QM9 defaults to the PyTorch Geometric source. Alternative local sources:
#
#   QM9_SOURCE=sdf QM9_SDF_FILE=/path/to/gdb9.sdf \
#     bash scripts/prepare_benchmark_datasets.sh /path/to/zinc250k.csv smiles
#
#   QM9_SOURCE=smiles QM9_SMILES_FILE=/path/to/qm9.csv \
#     QM9_SMILES_COLUMN=smiles \
#     bash scripts/prepare_benchmark_datasets.sh /path/to/zinc250k.csv smiles
#
# Optional environment variables:
#   PYTHON_BIN          Python executable (default: python)
#   DATASET_ROOT        Prepared-data root (default: outputs/datasets)
#   QM9_SOURCE          pyg, sdf, or smiles (default: pyg)
#   QM9_PYG_ROOT        PyG QM9 cache/source root (default: data/pyg_qm9)
#   QM9_SDF_FILE        Local gdb9.sdf path when QM9_SOURCE=sdf
#   QM9_SMILES_FILE     Local SMILES/CSV path when QM9_SOURCE=smiles
#   QM9_SMILES_COLUMN   Optional QM9 CSV/TSV SMILES column
#   ZINC_SMILES_FILE    Local ZINC .smi/.txt/.csv/.tsv source
#   ZINC_SMILES_COLUMN  Optional ZINC CSV/TSV SMILES column

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-outputs/datasets}"
QM9_SOURCE="${QM9_SOURCE:-pyg}"
QM9_PYG_ROOT="${QM9_PYG_ROOT:-data/pyg_qm9}"
ZINC_SMILES_FILE="${1:-${ZINC_SMILES_FILE:-}}"
ZINC_SMILES_COLUMN="${2:-${ZINC_SMILES_COLUMN:-}}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${ZINC_SMILES_FILE}" ]]; then
  echo "Error: a local ZINC SMILES/CSV source is required." >&2
  echo "Usage: bash scripts/prepare_benchmark_datasets.sh /path/to/zinc250k.csv [smiles_column]" >&2
  exit 2
fi

if [[ ! -f "${ZINC_SMILES_FILE}" ]]; then
  echo "Error: ZINC source not found: ${ZINC_SMILES_FILE}" >&2
  exit 2
fi

echo "[1/4] Preparing Ego-small"
"${PYTHON_BIN}" scripts/prepare_generic_dataset.py \
  --dataset ego_small \
  --root "${DATASET_ROOT}"

echo "[2/4] Preparing Grid"
"${PYTHON_BIN}" scripts/prepare_generic_dataset.py \
  --dataset grid \
  --root "${DATASET_ROOT}"

echo "[3/4] Preparing QM9 from ${QM9_SOURCE}"
qm9_args=(
  scripts/prepare_qm9_dataset.py
  --source "${QM9_SOURCE}"
  --root "${DATASET_ROOT}"
  --topology-name qm9_topology
  --attributed-name qm9_attributed
)

case "${QM9_SOURCE}" in
  pyg)
    qm9_args+=(--pyg-root "${QM9_PYG_ROOT}")
    ;;
  sdf)
    if [[ -z "${QM9_SDF_FILE:-}" || ! -f "${QM9_SDF_FILE}" ]]; then
      echo "Error: QM9_SOURCE=sdf requires an existing QM9_SDF_FILE." >&2
      exit 2
    fi
    qm9_args+=(--sdf-file "${QM9_SDF_FILE}")
    ;;
  smiles)
    if [[ -z "${QM9_SMILES_FILE:-}" || ! -f "${QM9_SMILES_FILE}" ]]; then
      echo "Error: QM9_SOURCE=smiles requires an existing QM9_SMILES_FILE." >&2
      exit 2
    fi
    qm9_args+=(--smiles-file "${QM9_SMILES_FILE}")
    if [[ -n "${QM9_SMILES_COLUMN:-}" ]]; then
      qm9_args+=(--smiles-column "${QM9_SMILES_COLUMN}")
    fi
    ;;
  *)
    echo "Error: QM9_SOURCE must be one of: pyg, sdf, smiles." >&2
    exit 2
    ;;
esac

"${PYTHON_BIN}" "${qm9_args[@]}"

echo "[4/4] Preparing ZINC"
zinc_args=(
  scripts/prepare_zinc_dataset.py
  --config configs/datasets/zinc.yaml
  --smiles-file "${ZINC_SMILES_FILE}"
  --root "${DATASET_ROOT}"
)
if [[ -n "${ZINC_SMILES_COLUMN}" ]]; then
  zinc_args+=(--smiles-column "${ZINC_SMILES_COLUMN}")
fi
"${PYTHON_BIN}" "${zinc_args[@]}"

required_datasets=(
  ego_small
  grid
  qm9_topology
  qm9_attributed
  zinc
)
required_splits=(train val test)

for dataset in "${required_datasets[@]}"; do
  for split in "${required_splits[@]}"; do
    artifact="${DATASET_ROOT}/${dataset}/${split}.pkl"
    if [[ ! -f "${artifact}" ]]; then
      echo "Error: expected dataset artifact was not produced: ${artifact}" >&2
      exit 1
    fi
  done
done

echo "Dataset preparation completed successfully."
echo "Prepared datasets are under: ${DATASET_ROOT}"
