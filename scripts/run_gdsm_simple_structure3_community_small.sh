#!/usr/bin/env bash
# Usage: bash scripts/run_gdsm_simple_structure3_community_small.sh [degree|train|generate|evaluate|all]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
STAGE="${1:-all}"
CFG="${CFG:-configs/baselines/gdsm_simple_community_small_structure3.yaml}"
DEGREE_CFG="${DEGREE_CFG:-configs/experiments/dhvae/community_small.yaml}"
RUN="${RUN:-seed_42_structure3_degree_basis}"
SEED="${SEED:-42}"
N="${N:-1024}"
DEVICE="${DEVICE:-gpu}"
GEN_ID="${GEN_ID:-seed_${SEED}_n_${N}}"
GEN="outputs/baselines/gdsm_simple/community_small/${RUN}/generations/${GEN_ID}"
case "$STAGE" in degree|train|generate|evaluate|all) ;; *) echo "Unknown stage: $STAGE" >&2; exit 2;; esac
if [[ "$STAGE" == degree || "$STAGE" == all ]]; then
  # Degree trainer reads its own seed, device and checkpoint path from DEGREE_CFG.
  # Set those consistently when changing SEED/DEVICE from their default values.
  python scripts/train_degree_generator.py --config "$DEGREE_CFG"
fi
if [[ "$STAGE" == train || "$STAGE" == all ]]; then
  python scripts/run_gdsm_simple_baseline.py --stage train --dataset community_small \
    --common-config configs/baselines/common_community_small.yaml --wrapper-config "$CFG" \
    --seed-id "$SEED" --run-id "$RUN" --device "$DEVICE"
fi
if [[ "$STAGE" == generate || "$STAGE" == all ]]; then
  python scripts/run_gdsm_simple_baseline.py --stage generate --dataset community_small \
    --common-config configs/baselines/common_community_small.yaml --wrapper-config "$CFG" \
    --seed-id "$SEED" --generation-seed "$SEED" --run-id "$RUN" \
    --generation-id "$GEN_ID" --num-samples "$N" --device "$DEVICE"
fi
if [[ "$STAGE" == evaluate || "$STAGE" == all ]]; then
  python scripts/evaluate_graph_generation_report.py \
    --config configs/experiments/baselines/community_small_evaluation.yaml \
    --generated-dir "$GEN" --generated-stage gdsm_simple_structure3 \
    --base-graphs "$GEN/initial_graphs.pkl" --reference-split test \
    --output-dir "$GEN/evaluation_test"
fi
