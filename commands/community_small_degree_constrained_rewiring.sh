#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

N="${N:-1024}"
DEVICE="${DEVICE:-cuda:0}"
REFERENCE_SPLIT="${REFERENCE_SPLIT:-val}"
DO_TRAIN="${DO_TRAIN:-1}"
REUSE_FINAL_CHECKPOINT="${REUSE_FINAL_CHECKPOINT:-0}"
ORCA_EXEC="${ORCA_EXEC:-/home/quang/orca/orca.out}"
SEEDS=(41 42 43 44 45)

for SEED in "${SEEDS[@]}"; do
  CFG="configs/experiments/grapher_degree_constrained_rewiring/community_small_seed_${SEED}.yaml"
  if [[ "$REUSE_FINAL_CHECKPOINT" == "1" ]]; then
    if [[ "$DO_TRAIN" == "1" ]]; then
      echo "ERROR: REUSE_FINAL_CHECKPOINT=1 requires DO_TRAIN=0."
      exit 1
    fi
    RUN="seed_${SEED}_community_small_grapher_final_g345_v1"
  else
    RUN="seed_${SEED}_community_small_dcr_tau0p4_rewire1to4_g345_v1"
  fi
  GEN_ID="dcr_tau0p4_rewire1to4_seed_${SEED}_n_${N}"

  echo "============================================================"
  echo "Community-small: degree-constrained realization + rewiring"
  echo "seed=${SEED}, tau=0.4, rewiring=Uniform{1,2,3,4}"
  echo "split=${REFERENCE_SPLIT}, train=${DO_TRAIN}, reuse_final=${REUSE_FINAL_CHECKPOINT}"
  echo "============================================================"

  if [[ "$DO_TRAIN" == "1" ]]; then
    PYTHONHASHSEED="$SEED" \
    GRAPHER_EIGH_BACKEND=cpu \
    PYTHONPATH=src \
    python scripts/run_grapher_baseline.py \
      --stage train \
      --dataset community_small \
      --no-common-config \
      --wrapper-config "$CFG" \
      --dataset-root outputs/datasets \
      --serialized-dataset sbm \
      --output-root outputs/baselines \
      --seed-id "$SEED" \
      --run-id "$RUN" \
      --device "$DEVICE"
  fi

  PYTHONHASHSEED="$SEED" \
  GRAPHER_EIGH_BACKEND=cpu \
  PYTHONPATH=src \
  python scripts/run_grapher_baseline.py \
    --stage generate \
    --dataset community_small \
    --no-common-config \
    --wrapper-config "$CFG" \
    --dataset-root outputs/datasets \
    --serialized-dataset sbm \
    --output-root outputs/baselines \
    --seed-id "$SEED" \
    --generation-seed "$SEED" \
    --run-id "$RUN" \
    --generation-id "$GEN_ID" \
    --num-samples "$N" \
    --device "$DEVICE"

  GEN="outputs/baselines/grapher/community_small/$RUN/generations/$GEN_ID"

  test -f "$GEN/pre_realization_graphs.pkl"
  test -f "$GEN/pre_refinement_graphs.pkl"
  test -f "$GEN/base_graphs.pkl"

  PYTHONHASHSEED="$SEED" \
  ORCA_EXEC="$ORCA_EXEC" \
  PYTHONPATH=src \
  python scripts/evaluate_graph_generation_report.py \
    --config configs/evaluation/community_small.yaml \
    --reference-split "$REFERENCE_SPLIT" \
    --generic-mmd-protocol graphrnn \
    --generated-dir "$GEN" \
    --generated-graphs "$GEN/base_graphs.pkl" \
    --output-dir "$GEN/evaluation_${REFERENCE_SPLIT}_paired"
done
