cd /home/quang/graphes

N="${N:-1024}"
DEVICE="${DEVICE:-cuda:0}"
REFERENCE_SPLIT="${REFERENCE_SPLIT:-val}"
ORCA_EXEC="${ORCA_EXEC:-/home/quang/orca/orca.out}"

SEED=42
DATASETS=(community_small ego_small)
BSETTINGS=(B1 B1to2 B1to4)

for DATASET in "${DATASETS[@]}"; do

  # ------------------------------------------------------------
  # Dataset-specific serialization name
  # ------------------------------------------------------------
  if [[ "$DATASET" == "community_small" ]]; then
    SERIALIZED="sbm"
  elif [[ "$DATASET" == "ego_small" ]]; then
    SERIALIZED="ego"
  else
    echo "Unsupported dataset: $DATASET"
    exit 1
  fi

  # ------------------------------------------------------------
  # Train the spectral GraphES model ONCE
  # ------------------------------------------------------------
  TRAIN_CFG="configs/experiments/grapher_final/${DATASET}_seed_${SEED}.yaml"

  RUN="seed_${SEED}_${DATASET}_grapher_final_g345_v1"

  echo
  echo "============================================================"
  echo "TRAINING"
  echo "dataset=${DATASET}"
  echo "seed=${SEED}"
  echo "run=${RUN}"
  echo "============================================================"

  PYTHONHASHSEED="$SEED" \
  GRAPHER_EIGH_BACKEND=cpu \
  PYTHONPATH=src \
  python scripts/run_grapher_baseline.py \
    --stage train \
    --dataset "$DATASET" \
    --no-common-config \
    --wrapper-config "$TRAIN_CFG" \
    --dataset-root outputs/datasets \
    --serialized-dataset "$SERIALIZED" \
    --output-root outputs/baselines \
    --seed-id "$SEED" \
    --run-id "$RUN" \
    --device "$DEVICE"

  # ============================================================
  # Generation-time ablations
  # ============================================================
  for BSETTING in "${BSETTINGS[@]}"; do

    CFG="configs/experiments/grapher_degree_constrained_rewiring/${DATASET}_seed_${SEED}_${BSETTING}.yaml"

    GEN_ID="dcr_tau0p4_${BSETTING}_seed_${SEED}_n_${N}"

    echo
    echo "============================================================"
    echo "GENERATION + EVALUATION"
    echo "dataset=${DATASET}"
    echo "seed=${SEED}"
    echo "DCR tau=0.4"
    echo "rewiring=${BSETTING}"
    echo "split=${REFERENCE_SPLIT}"
    echo "============================================================"

    # ----------------------------------------------------------
    # Generate
    # ----------------------------------------------------------
    PYTHONHASHSEED="$SEED" \
    GRAPHER_EIGH_BACKEND=cpu \
    PYTHONPATH=src \
    python scripts/run_grapher_baseline.py \
      --stage generate \
      --dataset "$DATASET" \
      --no-common-config \
      --wrapper-config "$CFG" \
      --dataset-root outputs/datasets \
      --serialized-dataset "$SERIALIZED" \
      --output-root outputs/baselines \
      --seed-id "$SEED" \
      --generation-seed "$SEED" \
      --run-id "$RUN" \
      --generation-id "$GEN_ID" \
      --num-samples "$N" \
      --device "$DEVICE"

    GEN="outputs/baselines/grapher/${DATASET}/${RUN}/generations/${GEN_ID}"

    if [[ ! -f "$GEN/base_graphs.pkl" ]]; then
      echo "ERROR: generation failed:"
      echo "  $GEN/base_graphs.pkl"
      exit 1
    fi

    # ----------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------
    PYTHONHASHSEED="$SEED" \
    ORCA_EXEC="$ORCA_EXEC" \
    PYTHONPATH=src \
    python scripts/evaluate_graph_generation_report.py \
      --config "configs/evaluation/${DATASET}.yaml" \
      --reference-split "$REFERENCE_SPLIT" \
      --generic-mmd-protocol graphrnn \
      --generated-dir "$GEN" \
      --generated-graphs "$GEN/base_graphs.pkl" \
      --output-dir "$GEN/evaluation_${REFERENCE_SPLIT}_paired"

  done
done