#!/usr/bin/env bash
# Paired GraphER ablation: ordinary final sampler vs connected-final rejection/resampling.
#
# IMPORTANT: this script reuses the *frozen gdsm_final_g345 training run* and its
# exact resolved model config.  It does not call run_grapher_research.sh, because
# that launcher has a different default RUN/config family and can therefore point
# generation at the wrong managed training manifest / degree-prior checkpoint.
set -euo pipefail

DATASET="${1:-zinc}"
case "$DATASET" in
  qm9)  SERIAL=qm9_attributed ;;
  zinc) SERIAL=zinc ;;
  *) echo 'Dataset must be qm9 or zinc' >&2; exit 2 ;;
esac

SEEDS="${SEEDS:-42 43 44}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"
N="${N:-10000}"
RUN_TAG="${RUN_TAG:-gdsm_final_g345}"
DATASET_ROOT="${DATASET_ROOT:-outputs/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/baselines}"
OUT="${OUT:-outputs/ablations/${DATASET}_final_connectivity}"
MAX_ATTEMPT_MULTIPLIER="${MAX_ATTEMPT_MULTIPLIER:-10.0}"

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export GDSM_EIGH_BACKEND="${GDSM_EIGH_BACKEND:-cpu}"

[[ "$N" =~ ^[1-9][0-9]*$ ]] || { echo 'N must be a positive integer' >&2; exit 2; }

baseline_args=()
fixed_args=()

for seed in $SEEDS; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }

  # This is the run-id produced by scripts/run_gdsm_final.sh with the default
  # RUN_TAG=gdsm_final_g345.
  run="seed_${seed}_${RUN_TAG}"
  run_root="${OUTPUT_ROOT}/gdsm_simple/${DATASET}/${run}"
  train_manifest="${run_root}/train/manifest.json"
  main_cfg="outputs/final_configs/${DATASET}/${run}/model.yaml"

  if [[ ! -f "$train_manifest" ]]; then
    echo "Missing trained GraphER run: $train_manifest" >&2
    echo "Expected a completed final run with RUN_TAG=$RUN_TAG for seed $seed." >&2
    echo "Available runs under ${OUTPUT_ROOT}/gdsm_simple/${DATASET}:" >&2
    find "${OUTPUT_ROOT}/gdsm_simple/${DATASET}" -maxdepth 2 -path '*/train/manifest.json' -print 2>/dev/null >&2 || true
    exit 1
  fi
  if [[ ! -f "$main_cfg" ]]; then
    echo "Missing exact resolved final config: $main_cfg" >&2
    echo "Do not substitute configs/experiments/grapher_research/* here; the final" >&2
    echo "experiment resolves seed-specific degree-prior checkpoint paths." >&2
    exit 1
  fi

  resolved_dir="${OUT}/resolved/${run}"
  mkdir -p "$resolved_dir"
  fixed_cfg="${resolved_dir}/connectivity_filter_model.yaml"

  # Create a generation-only overlay from the exact config used by the frozen
  # final run.  The ONLY model/sampler change is final connectedness acceptance.
  "$PYTHON" - "$main_cfg" "$fixed_cfg" "$MAX_ATTEMPT_MULTIPLIER" <<'PY'
import math
import sys
from pathlib import Path
import yaml

src, dst, multiplier = sys.argv[1], sys.argv[2], float(sys.argv[3])
if not math.isfinite(multiplier) or multiplier < 1:
    raise SystemExit('MAX_ATTEMPT_MULTIPLIER must be finite and >= 1')
obj = yaml.safe_load(Path(src).read_text())
cat = obj['gdsm_simple']['extensions']['attributed_categorical']
cat['final_acceptance'] = {
    'require_connected': True,
    'max_attempt_multiplier': multiplier,
}
text = yaml.safe_dump(obj, sort_keys=False)
p = Path(dst)
if p.exists() and p.read_text() != text:
    raise SystemExit(f'Connectivity ablation config collision at {p}; use a new OUT directory')
p.write_text(text)
PY

  baseline_gid="seed_${seed}_n_${N}_connectivity_baseline"
  fixed_gid="seed_${seed}_n_${N}_connectivity_filter"
  baseline_dir="${run_root}/generations/${baseline_gid}"
  fixed_dir="${run_root}/generations/${fixed_gid}"

  echo "=== ${DATASET} seed=${seed}: baseline ==="
  "$PYTHON" scripts/run_gdsm_simple_baseline.py \
    --stage generate \
    --dataset "$DATASET" \
    --no-common-config \
    --wrapper-config "$main_cfg" \
    --dataset-root "$DATASET_ROOT" \
    --serialized-dataset "$SERIAL" \
    --output-root "$OUTPUT_ROOT" \
    --seed-id "$seed" \
    --generation-seed "$seed" \
    --run-id "$run" \
    --generation-id "$baseline_gid" \
    --num-samples "$N" \
    --device "$DEVICE"

  "$PYTHON" scripts/evaluate_gdsm_categorical.py \
    --generated-dir "$baseline_dir" \
    --reference-graphs "$DATASET_ROOT/$SERIAL/test.pkl" \
    --seed "$seed"
  "$PYTHON" scripts/evaluate_generated_molecules.py \
    --generated-graphs "$baseline_dir/molecular_graphs.pkl" \
    --dataset-root "$DATASET_ROOT" \
    --dataset "$SERIAL" \
    --reference-split test \
    --train-split train \
    --require-fcd \
    --metric-molecule-source raw_valid \
    --output-dir "$baseline_dir/evaluation_molecules"

  echo "=== ${DATASET} seed=${seed}: connectivity filter ==="
  "$PYTHON" scripts/run_gdsm_simple_baseline.py \
    --stage generate \
    --dataset "$DATASET" \
    --no-common-config \
    --wrapper-config "$fixed_cfg" \
    --dataset-root "$DATASET_ROOT" \
    --serialized-dataset "$SERIAL" \
    --output-root "$OUTPUT_ROOT" \
    --seed-id "$seed" \
    --generation-seed "$seed" \
    --run-id "$run" \
    --generation-id "$fixed_gid" \
    --num-samples "$N" \
    --device "$DEVICE"

  "$PYTHON" scripts/evaluate_gdsm_categorical.py \
    --generated-dir "$fixed_dir" \
    --reference-graphs "$DATASET_ROOT/$SERIAL/test.pkl" \
    --seed "$seed"
  "$PYTHON" scripts/evaluate_generated_molecules.py \
    --generated-graphs "$fixed_dir/molecular_graphs.pkl" \
    --dataset-root "$DATASET_ROOT" \
    --dataset "$SERIAL" \
    --reference-split test \
    --train-split train \
    --require-fcd \
    --metric-molecule-source raw_valid \
    --output-dir "$fixed_dir/evaluation_molecules"

  baseline_args+=(--baseline-dir "$baseline_dir")
  fixed_args+=(--fixed-dir "$fixed_dir")
done

"$PYTHON" scripts/summarize_molecular_connectivity_ablation.py \
  "${baseline_args[@]}" "${fixed_args[@]}" --output-dir "$OUT"
