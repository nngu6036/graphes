#!/usr/bin/env bash
# Usage: bash scripts/run_gdsm_final.sh {all|DATASET} {all|STAGE}
# Defaults to ALL THREE SEEDS, not one sampling repeat of a shared checkpoint.
set -euo pipefail
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export GDSM_EIGH_BACKEND="${GDSM_EIGH_BACKEND:-cpu}"
PYTHON="${PYTHON:-python}"
read -r -a seeds <<< "${SEEDS:-${SEED:-42 43 44}}"
exec "$PYTHON" scripts/gdsm_final_suite.py \
  --dataset "${1:-all}" --stage "${2:-all}" --seeds "${seeds[@]}" \
  --device "${DEVICE:-cuda:0}" --run-tag "${RUN_TAG:-gdsm_final_g345}" "${@:3}"
