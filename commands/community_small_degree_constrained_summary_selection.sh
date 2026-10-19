#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

N="${N:-1024}"
DEVICE="${DEVICE:-cuda:0}"
REFERENCE_SPLIT="${REFERENCE_SPLIT:-val}"
ORCA_EXEC="${ORCA_EXEC:-/home/quang/orca/orca.out}"
DO_TRAIN="${DO_TRAIN:-1}"
SEEDS=(41 42 43 44 45)

for SEED in "${SEEDS[@]}"; do
  CFG="configs/experiments/grapher_degree_constrained_summary_selection/community_small_seed_${SEED}.yaml"
  RUN="seed_${SEED}_community_small_degree_constrained_summary_selection_g345_v1"
  GEN_ID="summary_selected_k8_tau_0p50_seed_${SEED}_n_${N}"

  echo "============================================================"
  echo "Community-small: degree-constrained summary candidate selection"
  echo "seed=${SEED}, K=8, tau=0.50, split=${REFERENCE_SPLIT}"
  echo "============================================================"

  if [[ ! -f "$CFG" ]]; then
    echo "ERROR: missing config: $CFG" >&2
    exit 1
  fi

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
  if [[ ! -f "$GEN/base_graphs.pkl" ]]; then
    echo "ERROR: expected generated graphs not found: $GEN/base_graphs.pkl" >&2
    exit 1
  fi

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
