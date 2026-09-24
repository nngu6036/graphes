#!/usr/bin/env bash
# Paired GraphER ablation: ordinary final sampler vs connected-final rejection/resampling.
#
# This launcher binds generation to the MANAGED TRAINING MANIFEST.  The gdsm_simple
# wrapper always reconstructs the trained options from train/manifest.json and then
# applies generation-only overrides.  Therefore no outputs/final_configs/... YAML is
# required and there is no risk of selecting a different degree-prior checkpoint.
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

  run="seed_${seed}_${RUN_TAG}"
  run_root="${OUTPUT_ROOT}/gdsm_simple/${DATASET}/${run}"
  train_manifest="${run_root}/train/manifest.json"

  if [[ ! -f "$train_manifest" ]]; then
    echo "Missing trained GraphER run: $train_manifest" >&2
    echo "Expected a completed managed run with RUN_TAG=$RUN_TAG for seed $seed." >&2
    echo "Available managed runs under ${OUTPUT_ROOT}/gdsm_simple/${DATASET}:" >&2
    find "${OUTPUT_ROOT}/gdsm_simple/${DATASET}" -maxdepth 3 -path '*/train/manifest.json' -print 2>/dev/null >&2 || true
    exit 1
  fi

  resolved_dir="${OUT}/resolved/${run}"
  mkdir -p "$resolved_dir"
  baseline_cfg="${resolved_dir}/baseline_generation_overlay.yaml"
  fixed_cfg="${resolved_dir}/connectivity_filter_generation_overlay.yaml"
  train_snapshot="${resolved_dir}/managed_training_options.yaml"

  # Validate the managed run and write two GENERATION-ONLY overlays.
  # Baseline: no model/sampler override at all (runtime is supplied by CLI).
  # Fixed: only final_acceptance is changed.  The wrapper recursively merges this
  # overlay into manifest['options'], so all trained settings and the exact
  # seed-specific degree-prior checkpoint remain those recorded at training time.
  "$PYTHON" - "$train_manifest" "$baseline_cfg" "$fixed_cfg" "$train_snapshot" \
    "$MAX_ATTEMPT_MULTIPLIER" "$run" "$seed" <<'PY'
import json
import math
import sys
from pathlib import Path
import yaml

manifest_path, baseline_path, fixed_path, snapshot_path, multiplier, expected_run, expected_seed = sys.argv[1:]
multiplier = float(multiplier)
expected_seed = int(expected_seed)
if not math.isfinite(multiplier) or multiplier < 1:
    raise SystemExit('MAX_ATTEMPT_MULTIPLIER must be finite and >= 1')

manifest = json.loads(Path(manifest_path).read_text())
if manifest.get('model_id') != 'gdsm_simple':
    raise SystemExit(f"{manifest_path}: expected model_id=gdsm_simple, got {manifest.get('model_id')!r}")
if manifest.get('run_id') not in (None, expected_run):
    raise SystemExit(f"{manifest_path}: run_id mismatch: {manifest.get('run_id')!r} != {expected_run!r}")
if int(manifest.get('train_seed', expected_seed)) != expected_seed:
    raise SystemExit(f"{manifest_path}: train_seed mismatch")
options = manifest.get('options')
if not isinstance(options, dict):
    raise SystemExit(f"{manifest_path}: managed manifest has no options mapping")
cat = options.get('extensions', {}).get('attributed_categorical', {})
if not isinstance(cat, dict) or not cat.get('enabled', False):
    raise SystemExit(f"{manifest_path}: managed run is not attributed_categorical")

# Provenance snapshot for the ablation record only; generation reads the same
# options directly from the managed training manifest.
Path(snapshot_path).write_text(yaml.safe_dump({'gdsm_simple': options}, sort_keys=False))

baseline = {'gdsm_simple': {}}
fixed = {
    'gdsm_simple': {
        'extensions': {
            'attributed_categorical': {
                'final_acceptance': {
                    'require_connected': True,
                    'max_attempt_multiplier': multiplier,
                }
            }
        }
    }
}

def locked_write(path, obj):
    p = Path(path)
    text = yaml.safe_dump(obj, sort_keys=False)
    if p.exists() and yaml.safe_load(p.read_text()) != obj:
        raise SystemExit(f'Ablation overlay collision at {p}; use a new OUT directory')
    if not p.exists():
        p.write_text(text)

locked_write(baseline_path, baseline)
locked_write(fixed_path, fixed)

prior = cat.get('initialization', {}).get('degree_generator', {}).get('checkpoint_path')
checkpoint = manifest.get('checkpoint', {}).get('path')
print(f'managed training manifest: {manifest_path}')
print(f'managed model checkpoint: {checkpoint}')
print(f'managed degree-prior checkpoint: {prior}')
print(f'baseline overlay: {baseline_path} (no generation override)')
print(f'connectivity overlay: {fixed_path}')
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
    --wrapper-config "$baseline_cfg" \
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
