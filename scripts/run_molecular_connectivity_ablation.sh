#!/usr/bin/env bash
# Paired GraphER ablation: current sampler vs final connected-sample rejection/resampling.
# Requires the main GraphER checkpoints for the requested seeds to already exist.
set -euo pipefail

DATASET="${1:-zinc}"
case "$DATASET" in qm9|zinc) ;; *) echo 'Dataset must be qm9 or zinc' >&2; exit 2 ;; esac
SEEDS="${SEEDS:-42 43 44}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"
N="${N:-10000}"
OUT="${OUT:-outputs/ablations/${DATASET}_final_connectivity}"

baseline_args=()
fixed_args=()
for seed in $SEEDS; do
  run="seed_${seed}_grapher_g345"
  baseline_gid="seed_${seed}_n_${N}_connectivity_baseline"
  fixed_gid="seed_${seed}_n_${N}_connectivity_filter"

  SEED="$seed" GEN_SEED="$seed" N="$N" DEVICE="$DEVICE" VARIANT=main GID="$baseline_gid" \
    bash scripts/run_grapher_research.sh "$DATASET" generate
  SEED="$seed" GEN_SEED="$seed" N="$N" DEVICE="$DEVICE" VARIANT=main GID="$baseline_gid" \
    bash scripts/run_grapher_research.sh "$DATASET" evaluate

  SEED="$seed" GEN_SEED="$seed" N="$N" DEVICE="$DEVICE" VARIANT=connectivity_filter GID="$fixed_gid" \
    bash scripts/run_grapher_research.sh "$DATASET" generate
  SEED="$seed" GEN_SEED="$seed" N="$N" DEVICE="$DEVICE" VARIANT=connectivity_filter GID="$fixed_gid" \
    bash scripts/run_grapher_research.sh "$DATASET" evaluate

  root="outputs/baselines/gdsm_simple/$DATASET/$run/generations"
  baseline_args+=(--baseline-dir "$root/$baseline_gid")
  fixed_args+=(--fixed-dir "$root/$fixed_gid")
done

"$PYTHON" scripts/summarize_molecular_connectivity_ablation.py \
  "${baseline_args[@]}" "${fixed_args[@]}" --output-dir "$OUT"
