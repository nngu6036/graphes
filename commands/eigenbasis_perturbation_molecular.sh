#!/usr/bin/env bash
set -euo pipefail

# Generation-only eigenbasis perturbation for the hybrid QM9/ZINC model.
# Rewiring is explicitly disabled so this isolates basis perturbation.
# Example:
#   DATASET=qm9 SEED=42 RUN=<existing-trained-run> \
#     bash commands/eigenbasis_perturbation_molecular.sh

DATASET="${DATASET:-qm9}"
SEED="${SEED:-42}"
RUN="${RUN:?Set RUN to an existing trained GraphES molecular run id}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/baselines}"
DATASET_ROOT="${DATASET_ROOT:-outputs/datasets}"
REFERENCE_SPLIT="${REFERENCE_SPLIT:-test}"
N="${N:-10000}"

case "$DATASET" in
  qm9) SERIALIZED="qm9_attributed" ;;
  zinc) SERIALIZED="zinc" ;;
  *) echo "DATASET must be qm9 or zinc" >&2; exit 2 ;;
esac

for STAGE in reverse decode; do
  CFG="configs/experiments/grapher_eigenbasis_perturbation/molecular_${STAGE}_sigma_0p025.yaml"
  GEN_ID="basis_${STAGE}_sigma_0p025_seed_${SEED}_n_${N}"

  PYTHONHASHSEED="$SEED" GRAPHER_EIGH_BACKEND=cpu PYTHONPATH=src \
  python scripts/run_grapher_baseline.py \
    --stage generate \
    --dataset "$DATASET" \
    --no-common-config \
    --wrapper-config "$CFG" \
    --dataset-root "$DATASET_ROOT" \
    --serialized-dataset "$SERIALIZED" \
    --output-root "$OUTPUT_ROOT" \
    --seed-id "$SEED" \
    --generation-seed "$SEED" \
    --run-id "$RUN" \
    --generation-id "$GEN_ID" \
    --num-samples "$N" \
    --device "$DEVICE"

  GEN_DIR="$OUTPUT_ROOT/grapher/$DATASET/$RUN/generations/$GEN_ID"

  PYTHONHASHSEED="$SEED" PYTHONPATH=src \
  python scripts/evaluate_generated_molecules.py \
    --generated-dir "$GEN_DIR" \
    --generated-graphs "$GEN_DIR/base_graphs.pkl" \
    --dataset-root "$DATASET_ROOT" \
    --dataset "$SERIALIZED" \
    --reference-split "$REFERENCE_SPLIT" \
    --train-split train \
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
    --graphlet-attributed-backend python \
    --output-dir "$GEN_DIR/evaluation_molecules_${REFERENCE_SPLIT}"

  PYTHONHASHSEED="$SEED" PYTHONPATH=src \
  python scripts/evaluate_qed_research.py \
    --generated-graphs "$GEN_DIR/base_graphs.pkl" \
    --reference-graphs "$DATASET_ROOT/$SERIALIZED/$REFERENCE_SPLIT.pkl" \
    --molecular-report "$GEN_DIR/evaluation_molecules_${REFERENCE_SPLIT}/molecular_evaluation_metrics.json" \
    --evaluator scripts/evaluate_generated_molecules.py \
    --expected-num-generated "$N" \
    --thresholds 0.5 0.7 \
    --output-dir "$GEN_DIR/evaluation_qed_${REFERENCE_SPLIT}"

done
