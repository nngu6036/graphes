N="${N:-1024}"
DEVICE="${DEVICE:-cuda:0}"
REFERENCE_SPLIT="${REFERENCE_SPLIT:-test}"
SEEDS=(41 42 43 44 45)

for SEED in "${SEEDS[@]}"; do
  CFG="configs/experiments/grapher_final/community_small_seed_${SEED}.yaml"
  RUN="seed_${SEED}_community_small_grapher_final_g345_v1"
  GEN_ID="seed_${SEED}_n_${N}"

  PYTHONHASHSEED="$SEED" GRAPHER_EIGH_BACKEND=cpu PYTHONPATH=src \
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

  PYTHONHASHSEED="$SEED" GRAPHER_EIGH_BACKEND=cpu PYTHONPATH=src \
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
  PYTHONHASHSEED="$SEED" ORCA_EXEC="${ORCA_EXEC:-/home/quang/orca/orca.out}" PYTHONPATH=src \
  python scripts/evaluate_graph_generation_report.py \
    --config configs/evaluation/community_small.yaml \
    --reference-split "$REFERENCE_SPLIT" \
    --generic-mmd-protocol graphrnn \
    --generated-dir "$GEN" \
    --generated-graphs "$GEN/base_graphs.pkl" \
    --output-dir "$GEN/evaluation_${REFERENCE_SPLIT}_paired"
done
