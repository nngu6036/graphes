N="${N:-1024}"
DEVICE="${DEVICE:-cuda:0}"
REFERENCE_SPLIT="${REFERENCE_SPLIT:-val}"
ORCA_EXEC="${ORCA_EXEC:-/home/quang/orca/orca.out}"

SEEDS=(41 42 43 44 45)

for SEED in "${SEEDS[@]}"; do
  CFG="configs/experiments/grapher_degree_constrained_realization/community_small_seed_${SEED}.yaml"

  RUN="seed_${SEED}_community_small_degree_constrained_realization_g345_v1"
  GEN_ID="seed_${SEED}_n_${N}"

  echo "============================================================"
  echo "Community-small degree-constrained realization"
  echo "seed=${SEED}, N=${N}, split=${REFERENCE_SPLIT}"
  echo "============================================================"

  # ----------------------------------------------------------
  # 1. Train
  # ----------------------------------------------------------
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

  # ----------------------------------------------------------
  # 2. Generate 1,024 graphs
  # ----------------------------------------------------------
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

  # ----------------------------------------------------------
  # 3. Paired evaluation
  #
  # The evaluator automatically compares:
  #   pre_realization_graphs.pkl = deterministic threshold decoder
  #   base_graphs.pkl            = stochastic exact-degree decoder
  # ----------------------------------------------------------
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