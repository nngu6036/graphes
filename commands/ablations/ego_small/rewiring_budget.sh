#!/usr/bin/env bash
# ego_small: rewiring budget study; evaluates D, R1, R12, DR.
# Run from any directory. Set DRY_RUN=1 to print validation and creation plans.
# Primary training YAML: configs/experiments/grapher_final/ego_small_seed_${SEED}.yaml
# Primary generation YAML: configs/experiments/grapher_paper_ablation/ego_small_seed_${SEED}_matched.yaml
# The first study prepares ONE shared seven-branch population (T,D,TR,DR,D0,R1,R12).
# Other studies verify/reuse those exact saved samples and evaluate only their listed branches.
# REUSE=1 rejects incompatible saved artifacts; it never silently retrains or regenerates.
# Study evaluations and summaries use separate subfolders, so other study reports are retained.
set -euo pipefail
[[ $# -eq 0 ]] || { echo "This dataset-specific command takes no positional arguments; use environment overrides." >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ABLATION_STUDY="rewiring_budget"
export SEEDS="${SEEDS:-41 42 43 44 45}"
export N="${N:-1024}"
export DEVICE="${DEVICE:-cuda:0}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export DO_TRAIN="${DO_TRAIN:-0}"
export DO_GENERATE="${DO_GENERATE:-1}"
export DO_EVALUATE="${DO_EVALUATE:-1}"
export REUSE="${REUSE:-1}"
export REFERENCE_SPLIT="${REFERENCE_SPLIT:-test}"
export GENERATION_TAG="${GENERATION_TAG:-paper_ablation_v1}"
export OVERWRITE="${OVERWRITE:-0}"
export DRY_RUN="${DRY_RUN:-0}"
export RUN_S0=0
export DO_S0_GENERATE=0
export DO_S0_TRAIN=0
exec bash "$SCRIPT_DIR/../../ablation.sh" ego_small
