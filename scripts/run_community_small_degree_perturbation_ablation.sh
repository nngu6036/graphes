#!/usr/bin/env bash
# Run from the GraphES repository root. Existing checkpoints/data are required.
set -euo pipefail
MODE="${1:-all}"
PYTHON="${PYTHON:-python}"
CKPT="${CKPT:-outputs/topology_grapher/community_small_joint_degree_multicheckpoint/seed_42/checkpoints/best_joint/checkpoint.pt}"
SEEDS="${SEEDS:-42 43 44}"
NUM_GENERATE="${NUM_GENERATE:-64}"
DEVICE="${DEVICE:-gpu}"
REFERENCE_SPLIT="${REFERENCE_SPLIT:-val}"
OUT_ROOT="${OUT_ROOT:-outputs/topology_generation/community_small_degree_perturbation/ckpt_seed_42}"
PERTURB_PROBABILITY="${PERTURB_PROBABILITY:-0.25}"
if [[ "$MODE" == all ]]; then
  METHODS=(empirical unit_transfer moment_preserving edge_relocation interpolation)
else
  case "$MODE" in
    empirical|unit_transfer|moment_preserving|edge_relocation|interpolation) METHODS=("$MODE");;
    *) echo "Unknown method: $MODE" >&2; exit 2;;
  esac
fi
if [[ ! -f "$CKPT" ]]; then echo "Checkpoint not found: $CKPT" >&2; exit 2; fi
read -r -a SEED_ARRAY <<< "$SEEDS"
for METHOD in "${METHODS[@]}"; do
  CFG="configs/experiments/grapher/community_small_degree_perturb_${METHOD}.yaml"
  for SEED in "${SEED_ARRAY[@]}"; do
    OUT="$OUT_ROOT/generation_seed_${SEED}/$METHOD"
    if [[ -e "$OUT/report.json" ]]; then
      echo "Refusing to overwrite an existing run: $OUT (use a new OUT_ROOT)." >&2
      exit 2
    fi
    EXTRA=()
    if [[ "$METHOD" != empirical ]]; then
      EXTRA+=(--set "generation.degree_perturbation.probability=$PERTURB_PROBABILITY")
    fi
    PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" scripts/run_topology_grapher.py \
      --config "$CFG" --checkpoint "$CKPT" --output-dir "$OUT" \
      --num-generate "$NUM_GENERATE" --seed "$SEED" --device "$DEVICE" "${EXTRA[@]}"
    PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" scripts/evaluate_graph_generation_report.py \
      --config "$CFG" --generated-dir "$OUT" --reference-split "$REFERENCE_SPLIT" \
      --output-dir "$OUT/evaluation_${REFERENCE_SPLIT}"
  done
done
