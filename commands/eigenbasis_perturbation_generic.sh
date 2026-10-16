#!/usr/bin/env bash
set -euo pipefail

# Generation-only eigenbasis-perturbation sweep for Community-small or Ego-small.
# The trained run must already exist. Example:
#   DATASET=community_small SEED=42 RUN=seed_42_gdsm_final_g345 \
#     bash commands/eigenbasis_perturbation_generic.sh

DATASET="${DATASET:-community_small}"
SEED="${SEED:-42}"
RUN="${RUN:-seed_${SEED}_gdsm_final_g345}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/baselines}"
DATASET_ROOT="${DATASET_ROOT:-outputs/datasets}"

case "$DATASET" in
  community_small|ego_small) N="${N:-1024}" ;;
  *) echo "DATASET must be community_small or ego_small" >&2; exit 2 ;;
esac

EVAL_CFG="configs/experiments/baselines/${DATASET}_evaluation.yaml"

for STAGE in reverse decode; do
  for TAG in 0p005 0p01 0p025 0p05 0p1; do
    CFG="configs/experiments/grapher_eigenbasis_perturbation/generic_${STAGE}_sigma_${TAG}.yaml"
    GEN_ID="basis_${STAGE}_sigma_${TAG}_seed_${SEED}_n_${N}"

    PYTHONHASHSEED="$SEED" GRAPHER_EIGH_BACKEND=cpu PYTHONPATH=src \
    python scripts/run_grapher_baseline.py \
      --stage generate \
      --dataset "$DATASET" \
      --no-common-config \
      --wrapper-config "$CFG" \
      --dataset-root "$DATASET_ROOT" \
      --output-root "$OUTPUT_ROOT" \
      --seed-id "$SEED" \
      --generation-seed "$SEED" \
      --run-id "$RUN" \
      --generation-id "$GEN_ID" \
      --num-samples "$N" \
      --device "$DEVICE"

    GEN_DIR="$OUTPUT_ROOT/grapher/$DATASET/$RUN/generations/$GEN_ID"

    PYTHONHASHSEED="$SEED" PYTHONPATH=src \
    python scripts/evaluate_graph_generation_report.py \
      --config "$EVAL_CFG" \
      --generated-dir "$GEN_DIR" \
      --output-dir "$GEN_DIR/evaluation_report"
  done
done
