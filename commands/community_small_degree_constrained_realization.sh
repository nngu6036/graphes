

N="${N:-1024}"
DEVICE="${DEVICE:-cuda:0}"
REFERENCE_SPLIT="val"
SEED=42

TAUS=(0p20 0p30 0p50)

# Use the checkpoint you already trained.
# If you trained with the new degree-constrained experiment command:
RUN="seed_${SEED}_community_small_degree_constrained_realization_g345_v1"

# If instead you are reusing the old final GraphES checkpoint, use:
# RUN="seed_${SEED}_community_small_grapher_final_g345_v1"

for TAG in "${TAUS[@]}"; do

  CFG="configs/experiments/grapher_degree_constrained_realization/community_small_seed_${SEED}_tau_${TAG}.yaml"

  GEN_ID="degree_constrained_tau_${TAG}_seed_${SEED}_n_${N}"

  echo "============================================================"
  echo "Degree-constrained realization"
  echo "seed=${SEED}, tau=${TAG}"
  echo "============================================================"

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

  PYTHONHASHSEED="$SEED" \
  ORCA_EXEC="${ORCA_EXEC:-/home/quang/orca/orca.out}" \
  PYTHONPATH=src \
  python scripts/evaluate_graph_generation_report.py \
    --config configs/evaluation/community_small.yaml \
    --reference-split "$REFERENCE_SPLIT" \
    --generic-mmd-protocol graphrnn \
    --generated-dir "$GEN" \
    --generated-graphs "$GEN/base_graphs.pkl" \
    --output-dir "$GEN/evaluation_${REFERENCE_SPLIT}_paired"

done