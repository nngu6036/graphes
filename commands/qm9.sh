#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

N="${N:-10000}"
DEVICE="${DEVICE:-cuda:0}"
REFERENCE_SPLIT="${REFERENCE_SPLIT:-test}"
SEEDS=(41 42 43 44 45)

for SEED in "${SEEDS[@]}"; do
  CFG="configs/experiments/grapher_final/qm9_seed_${SEED}.yaml"
  RUN="seed_${SEED}_qm9_grapher_final_g345_v1"
  GEN_ID="seed_${SEED}_n_${N}"

  PYTHONHASHSEED="$SEED" GRAPHER_EIGH_BACKEND=cpu PYTHONPATH=src \
  python scripts/run_grapher_baseline.py \
    --stage train \
    --dataset qm9 \
    --no-common-config \
    --wrapper-config "$CFG" \
    --dataset-root outputs/datasets \
    --serialized-dataset qm9_attributed \
    --output-root outputs/baselines \
    --seed-id "$SEED" \
    --run-id "$RUN" \
    --device "$DEVICE"

  PYTHONHASHSEED="$SEED" GRAPHER_EIGH_BACKEND=cpu PYTHONPATH=src \
  python scripts/run_grapher_baseline.py \
    --stage generate \
    --dataset qm9 \
    --no-common-config \
    --wrapper-config "$CFG" \
    --dataset-root outputs/datasets \
    --serialized-dataset qm9_attributed \
    --output-root outputs/baselines \
    --seed-id "$SEED" \
    --generation-seed "$SEED" \
    --run-id "$RUN" \
    --generation-id "$GEN_ID" \
    --num-samples "$N" \
    --device "$DEVICE"

  GEN="outputs/baselines/grapher/qm9/$RUN/generations/$GEN_ID"

  for STAGE in diversified pre_rewire; do
    if [[ "$STAGE" == "diversified" ]]; then
      GRAPHS="$GEN/base_graphs.pkl"
    else
      GRAPHS="$GEN/pre_refinement_graphs.pkl"
    fi
    OUT="$GEN/evaluation_raw_${REFERENCE_SPLIT}_${STAGE}_g345"

    PYTHONHASHSEED="$SEED" PYTHONPATH=src \
    python scripts/evaluate_generated_molecules.py \
      --generated-dir "$GEN" \
      --generated-graphs "$GRAPHS" \
      --dataset-root outputs/datasets \
      --dataset qm9_attributed \
      --reference-split "$REFERENCE_SPLIT" \
      --train-split train \
      --output-dir "$OUT" \
      --strict-raw-metrics \
      --metric-molecule-source raw_valid \
      --nspdk-backend eden \
      --nspdk-complexity 4 \
      --nspdk-bond-label-mode hogdiff \
      --require-fcd \
      --fcd-device "$DEVICE" \
      --graphlet-mmd \
      --graphlet-k-min 3 \
      --graphlet-k-max 5 \
      --graphlet-topology-filter all \
      --graphlet-node-attribute atomic_num \
      --graphlet-edge-attribute bond_type \
      --graphlet-attributed-backend python

    PYTHONPATH=src python scripts/evaluate_qed.py \
      --generated-graphs "$GRAPHS" \
      --reference-graphs "outputs/datasets/qm9_attributed/${REFERENCE_SPLIT}.pkl" \
      --molecular-report "$OUT/molecular_evaluation_metrics.json" \
      --expected-num-generated "$N" \
      --output-dir "$OUT/qed"
  done

done
