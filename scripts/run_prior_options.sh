#!/usr/bin/env bash
# Run from the repository root. No datasets are downloaded or rebuilt.
# Usage: bash scripts/run_prior_options.sh {qm9|community_small} {all|train|generate} [all|empirical|METHOD|learned]
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "Usage: $0 {qm9|community_small} {all|train|generate} [all|empirical|unit_transfer|moment_preserving|edge_relocation|interpolation|learned]"
  echo "Environment: CFG TRAIN CKPT OUT_ROOT PYTHON DEVICE TRAIN_SEED SEEDS NTRAIN NGEN EPOCHS BATCH_SIZE REFERENCE_SPLIT PERTURB_PROBABILITY DATASET_ROOT"
  echo "all trains only when the shared checkpoint is missing, then generates/evaluates. Existing outputs are never overwritten."
  exit 0
fi
DATASET="${1:-}"
STAGE="${2:-all}"
METHOD="${3:-all}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-gpu}"
TRAIN_SEED="${TRAIN_SEED:-42}"
SEEDS="${SEEDS:-42}"
REFERENCE_SPLIT="${REFERENCE_SPLIT:-val}"
PERTURB_PROBABILITY="${PERTURB_PROBABILITY:-0.25}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
case "$STAGE" in all|train|generate) ;; *) echo "Stage must be all, train, or generate." >&2; exit 2;; esac
case "$REFERENCE_SPLIT" in val|test) ;; *) echo "Reference must be val or test." >&2; exit 2;; esac
case "$DATASET" in
  qm9)
    CFG="${CFG:-configs/experiments/grapher/qm9_attributed_joint_typed_edge.yaml}"
    TRAIN="${TRAIN:-outputs/attributed_grapher/qm9_joint_typed_edge/seed_${TRAIN_SEED}}"
    OUT_ROOT="${OUT_ROOT:-outputs/attributed_generation/qm9_typed_prior_options/ckpt_seed_${TRAIN_SEED}}"
    NGEN="${NGEN:-1024}"
    NTRAIN="${NTRAIN:-20000}"
    ;;
  community_small)
    CFG="${CFG:-configs/experiments/grapher/community_small_spectral_graphlet35_prior_options.yaml}"
    TRAIN="${TRAIN:-outputs/topology_grapher/community_small_spectral_graphlet35_prior_options/seed_${TRAIN_SEED}}"
    OUT_ROOT="${OUT_ROOT:-outputs/topology_generation/community_small_graphlet35_prior_options/ckpt_seed_${TRAIN_SEED}}"
    NGEN="${NGEN:-64}"
    ;;
  *) echo "Usage: $0 {qm9|community_small} {all|train|generate} [all|empirical|unit_transfer|moment_preserving|edge_relocation|interpolation|learned]" >&2; exit 2;;
esac
CKPT="${CKPT:-$TRAIN/checkpoint.pt}"
test -f "$CFG" || { echo "Missing config: $CFG" >&2; exit 2; }
case "$METHOD" in
  all) METHODS=(empirical unit_transfer moment_preserving edge_relocation interpolation);;
  empirical|unit_transfer|moment_preserving|edge_relocation|interpolation|learned) METHODS=("$METHOD");;
  *) echo "Unknown method: $METHOD" >&2; exit 2;;
esac
read -r -a SEED_ARRAY <<< "$SEEDS"
if [[ ${#SEED_ARRAY[@]} -eq 0 ]]; then echo "SEEDS must not be empty." >&2; exit 2; fi
if [[ "$STAGE" != generate ]]; then
  if [[ -f "$CKPT" ]]; then
    echo "Reusing the shared checkpoint: $CKPT (choose a new TRAIN to train a new model)."
  else
    EXTRA_TRAIN=()
    if [[ -n "${EPOCHS:-}" ]]; then EXTRA_TRAIN+=(--epochs "$EPOCHS"); fi
    if [[ -n "${BATCH_SIZE:-}" ]]; then EXTRA_TRAIN+=(--batch-size "$BATCH_SIZE"); fi
    if [[ "$DATASET" == qm9 ]]; then
      "$PYTHON" scripts/train_attributed_grapher.py \
        --config "$CFG" --output-dir "$TRAIN" --num-train-graphs "$NTRAIN" \
        --seed "$TRAIN_SEED" --device "$DEVICE" \
        --graphlet-progress-interval 100 --batch-progress-interval 10 \
        --progress-interval-seconds 10 "${EXTRA_TRAIN[@]}"
    else
      if [[ -n "${NTRAIN:-}" ]]; then EXTRA_TRAIN+=(--max-train-graphs "$NTRAIN"); fi
      "$PYTHON" scripts/train_topology_grapher.py \
        --config "$CFG" --output-dir "$TRAIN" --seed "$TRAIN_SEED" \
        --device "$DEVICE" "${EXTRA_TRAIN[@]}"
    fi
  fi
fi
test -f "$CKPT" || { echo "Checkpoint not found: $CKPT" >&2; exit 2; }
if [[ "$STAGE" == train ]]; then exit 0; fi

# Preflight every output so a partly completed batch cannot be overwritten.
for M in "${METHODS[@]}"; do
  for SEED in "${SEED_ARRAY[@]}"; do
    OUT="$OUT_ROOT/generation_seed_${SEED}/$M"
    if [[ -d "$OUT" ]] && [[ -n "$(ls -A "$OUT")" ]]; then
      echo "Refusing to overwrite nonempty output: $OUT. Set a fresh OUT_ROOT." >&2
      exit 2
    fi
  done
done
for M in "${METHODS[@]}"; do
  EXTRA=(--set 'generation.degree_perturbation={}')
  if [[ "$DATASET" == qm9 ]]; then SOURCE_KEY=invariant_source; RNG_KEY=invariant_rng_mode;
  else SOURCE_KEY=degree_source; RNG_KEY=degree_rng_mode; fi
  if [[ "$M" == learned ]]; then SOURCE=learned;
  elif [[ "$M" == empirical ]]; then SOURCE=train_empirical;
  else
    SOURCE=train_empirical_perturbed
    EXTRA+=(--set "generation.degree_perturbation.method=$M"
            --set "generation.degree_perturbation.probability=$PERTURB_PROBABILITY"
            --set generation.degree_perturbation.steps=1
            --set generation.degree_perturbation.max_attempts=256
            --set generation.degree_perturbation.failure_policy=keep_original
            --set generation.degree_perturbation.max_distance=4.0
            --set generation.degree_perturbation.require_novel=false)
  fi
  EXTRA+=(--set "generation.$SOURCE_KEY=$SOURCE" --set "generation.$RNG_KEY=independent")
  for SEED in "${SEED_ARRAY[@]}"; do
    OUT="$OUT_ROOT/generation_seed_${SEED}/$M"
    echo "Dataset=$DATASET prior=$M generation_seed=$SEED checkpoint=$CKPT"
    if [[ "$DATASET" == qm9 ]]; then
      "$PYTHON" scripts/run_attributed_grapher.py \
        --config "$CFG" --checkpoint "$CKPT" --output-dir "$OUT" \
        --num-generate "$NGEN" --seed "$SEED" --device "$DEVICE" "${EXTRA[@]}"
      "$PYTHON" scripts/evaluate_generated_molecules.py \
        --generated-graphs "$OUT/molecular_graphs.pkl" --dataset qm9_attributed \
        --dataset-root "${DATASET_ROOT:-outputs/datasets}" \
        --reference-split "$REFERENCE_SPLIT" --train-split train \
        --metric-molecule-source raw_valid --nspdk-backend eden \
        --output-dir "$OUT/evaluation_${REFERENCE_SPLIT}" --require-fcd
    else
      "$PYTHON" scripts/run_topology_grapher.py \
        --config "$CFG" --checkpoint "$CKPT" --output-dir "$OUT" \
        --num-generate "$NGEN" --seed "$SEED" --device "$DEVICE" "${EXTRA[@]}"
      "$PYTHON" scripts/evaluate_graph_generation_report.py \
        --config "$CFG" --generated-dir "$OUT" --reference-split "$REFERENCE_SPLIT" \
        --output-dir "$OUT/evaluation_${REFERENCE_SPLIT}"
    fi
  done
done
if [[ "$DATASET" == community_small && "$METHOD" != learned ]]; then
  "$PYTHON" scripts/summarize_degree_perturbation_evaluations.py \
    --generation-root "$OUT_ROOT" --reference-split "$REFERENCE_SPLIT" \
    --seeds "${SEED_ARRAY[@]}" --methods "${METHODS[@]}" \
    --output-dir "$OUT_ROOT/summary_${REFERENCE_SPLIT}_${METHOD}"
fi
