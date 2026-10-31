#!/usr/bin/env bash
# community_small: summary supervision study; evaluates D, S0.
# Run from any directory. Set DRY_RUN=1 to print validation and creation plans.
# Primary training YAML: configs/experiments/grapher_final/community_small_seed_${SEED}.yaml
# Primary generation YAML: configs/experiments/grapher_paper_ablation/community_small_seed_${SEED}_matched.yaml
# The first study prepares ONE shared seven-branch population (T,D,TR,DR,D0,R1,R12).
# Other studies verify/reuse those exact saved samples and evaluate only their listed branches.
# S0 training/generation YAML: configs/experiments/grapher_paper_ablation/community_small_seed_${SEED}_s0.yaml
# S0 uses a separate summary-off checkpoint; train/generate only if verified artifacts are absent.
# REUSE=1 rejects incompatible saved artifacts; it never silently retrains or regenerates.
# Study evaluations and summaries use separate subfolders, so other study reports are retained.
set -euo pipefail
[[ $# -eq 0 ]] || { echo "This dataset-specific command takes no positional arguments; use environment overrides." >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ABLATION_STUDY="summary_supervision"
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
export RUN_S0=1
export DO_S0_GENERATE="${DO_S0_GENERATE:-$DO_GENERATE}"
export DO_S0_TRAIN="${DO_S0_TRAIN:-$DO_S0_GENERATE}"
export S0_GENERATION_TAG="${S0_GENERATION_TAG:-paper_S0_v1}"
exec bash "$SCRIPT_DIR/../../ablation.sh" community_small
