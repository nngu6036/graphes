#!/usr/bin/env bash
# Paired GraphER molecular connectivity ablation.
#
# For each requested seed, this launcher resolves a managed *training* run by
# manifest provenance rather than trusting the directory name.  It requires:
#   - gdsm_simple
#   - matching dataset + train_seed
#   - attributed_categorical enabled
#   - graphlet sizes exactly [3,4,5]
#   - the final-suite degree-prior checkpoint for the same seed
#
# Baseline and connectivity-filter generation then reuse that exact checkpoint;
# only the generation-only final_acceptance overlay differs.
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
EVAL_SEED="${EVAL_SEED:-42}"

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export GDSM_EIGH_BACKEND="${GDSM_EIGH_BACKEND:-cpu}"

[[ "$N" =~ ^[1-9][0-9]*$ ]] || { echo 'N must be a positive integer' >&2; exit 2; }
[[ "$EVAL_SEED" =~ ^[0-9]+$ ]] || { echo 'EVAL_SEED must be an integer' >&2; exit 2; }

baseline_args=()
fixed_args=()

for seed in $SEEDS; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }

  # Resolve the actual managed training run from its manifest.  This prevents a
  # mislabeled/stale directory (e.g. seed_42_* containing train_seed != 42)
  # from being used in a final ablation.
  resolution="$($PYTHON - "$OUTPUT_ROOT" "$DATASET" "$seed" "$RUN_TAG" <<'PY'
import json
import sys
from pathlib import Path

output_root, dataset, seed_s, run_tag = sys.argv[1:]
seed = int(seed_s)
root = Path(output_root) / 'gdsm_simple' / dataset
expected_prior = Path('outputs') / 'degree_generators' / run_tag / dataset / f'seed_{seed}' / 'checkpoint.pt'
expected_run = f'seed_{seed}_{run_tag}'

def get_nested(obj, *keys, default=None):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def prior_matches(value):
    if not value:
        return False
    p = Path(str(value))
    # Configs may store an absolute path or a repository-relative path.
    return p == expected_prior or str(p).replace('\\', '/').endswith('/' + str(expected_prior).replace('\\', '/')) or str(p).replace('\\', '/') == str(expected_prior).replace('\\', '/')

records = []
if root.is_dir():
    for manifest_path in sorted(root.glob('*/train/manifest.json')):
        try:
            m = json.loads(manifest_path.read_text())
        except Exception as exc:
            records.append({'path': str(manifest_path), 'error': f'unreadable: {exc}'})
            continue
        options = m.get('options') if isinstance(m.get('options'), dict) else {}
        cat = get_nested(options, 'extensions', 'attributed_categorical', default={})
        sizes = cat.get('graphlets', {}).get('sizes') if isinstance(cat, dict) else None
        prior = cat.get('initialization', {}).get('degree_generator', {}).get('checkpoint_path') if isinstance(cat, dict) else None
        try:
            train_seed = int(m.get('train_seed'))
        except Exception:
            train_seed = None
        rec = {
            'path': str(manifest_path),
            'run_id': m.get('run_id') or manifest_path.parents[1].name,
            'train_seed': train_seed,
            'model_id': m.get('model_id'),
            'dataset': get_nested(m, 'dataset', 'benchmark_id'),
            'categorical': bool(cat.get('enabled', False)) if isinstance(cat, dict) else False,
            'sizes': sizes,
            'prior': prior,
            'prior_match': prior_matches(prior),
        }
        records.append(rec)

matches = [r for r in records if not r.get('error')
           and r['model_id'] == 'gdsm_simple'
           and r['dataset'] == dataset
           and r['train_seed'] == seed
           and r['categorical']
           and list(r['sizes'] or []) == [3, 4, 5]
           and r['prior_match']]

# Prefer the exact final-suite run id if it is internally consistent.
exact = [r for r in matches if r['run_id'] == expected_run]
if len(exact) == 1:
    chosen = exact[0]
elif len(matches) == 1:
    chosen = matches[0]
else:
    print(f'Could not resolve one provenance-matched final GraphER checkpoint for {dataset} seed={seed}.', file=sys.stderr)
    requested = root / expected_run / 'train' / 'manifest.json'
    if requested.is_file():
        try:
            rm = json.loads(requested.read_text())
            print(f'Requested path exists but reports train_seed={rm.get("train_seed")!r}: {requested}', file=sys.stderr)
        except Exception:
            pass
    print(f'Expected degree prior: {expected_prior}', file=sys.stderr)
    print('Available managed training manifests:', file=sys.stderr)
    if not records:
        print('  (none)', file=sys.stderr)
    for r in records:
        if r.get('error'):
            print(f"  {r['path']}: {r['error']}", file=sys.stderr)
        else:
            print('  '
                  f"run_id={r['run_id']!r} train_seed={r['train_seed']!r} "
                  f"dataset={r['dataset']!r} g345={list(r['sizes'] or []) == [3,4,5]} "
                  f"prior_match={r['prior_match']} prior={r['prior']!r}", file=sys.stderr)
    if len(matches) > 1:
        print('Multiple matching runs were found; set RUN_TAG to select the intended experiment.', file=sys.stderr)
    else:
        print('Do not bypass this check for a final three-seed ablation: the checkpoint provenance would no longer match the requested seed.', file=sys.stderr)
    raise SystemExit(3)

print(json.dumps(chosen))
PY
)" || exit $?

  run="$($PYTHON -c 'import json,sys; print(json.loads(sys.stdin.read())["run_id"])' <<<"$resolution")"
  train_seed="$($PYTHON -c 'import json,sys; print(json.loads(sys.stdin.read())["train_seed"])' <<<"$resolution")"
  train_manifest="$($PYTHON -c 'import json,sys; print(json.loads(sys.stdin.read())["path"])' <<<"$resolution")"
  run_root="${OUTPUT_ROOT}/gdsm_simple/${DATASET}/${run}"

  resolved_dir="${OUT}/resolved/${run}"
  mkdir -p "$resolved_dir"
  baseline_cfg="${resolved_dir}/baseline_generation_overlay.yaml"
  fixed_cfg="${resolved_dir}/connectivity_filter_generation_overlay.yaml"
  train_snapshot="${resolved_dir}/managed_training_options.yaml"

  "$PYTHON" - "$train_manifest" "$baseline_cfg" "$fixed_cfg" "$train_snapshot" \
    "$MAX_ATTEMPT_MULTIPLIER" "$run" "$train_seed" <<'PY'
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
if int(manifest.get('train_seed')) != expected_seed:
    raise SystemExit(f"{manifest_path}: internal train_seed changed during resolution")
options = manifest.get('options')
if not isinstance(options, dict):
    raise SystemExit(f"{manifest_path}: managed manifest has no options mapping")
cat = options.get('extensions', {}).get('attributed_categorical', {})
if not isinstance(cat, dict) or not cat.get('enabled', False):
    raise SystemExit(f"{manifest_path}: managed run is not attributed_categorical")
if list(cat.get('graphlets', {}).get('sizes', [])) != [3, 4, 5]:
    raise SystemExit(f"{manifest_path}: managed run is not the g345 profile")

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
print(f'managed run_id: {expected_run}')
print(f'managed train_seed: {expected_seed}')
print(f'managed model checkpoint: {checkpoint}')
print(f'managed degree-prior checkpoint: {prior}')
print(f'baseline overlay: {baseline_path} (no generation override)')
print(f'connectivity overlay: {fixed_path}')
PY

  baseline_gid="seed_${seed}_n_${N}_connectivity_baseline"
  fixed_gid="seed_${seed}_n_${N}_connectivity_filter"
  baseline_dir="${run_root}/generations/${baseline_gid}"
  fixed_dir="${run_root}/generations/${fixed_gid}"

  echo "=== ${DATASET} requested_seed=${seed} train_seed=${train_seed}: baseline ==="
  "$PYTHON" scripts/run_gdsm_simple_baseline.py \
    --stage generate \
    --dataset "$DATASET" \
    --no-common-config \
    --wrapper-config "$baseline_cfg" \
    --dataset-root "$DATASET_ROOT" \
    --serialized-dataset "$SERIAL" \
    --output-root "$OUTPUT_ROOT" \
    --seed-id "$train_seed" \
    --generation-seed "$seed" \
    --run-id "$run" \
    --generation-id "$baseline_gid" \
    --num-samples "$N" \
    --device "$DEVICE"

  "$PYTHON" scripts/evaluate_gdsm_categorical.py \
    --generated-dir "$baseline_dir" \
    --reference-graphs "$DATASET_ROOT/$SERIAL/test.pkl" \
    --seed "$EVAL_SEED"
  "$PYTHON" scripts/evaluate_generated_molecules.py \
    --generated-graphs "$baseline_dir/molecular_graphs.pkl" \
    --dataset-root "$DATASET_ROOT" \
    --dataset "$SERIAL" \
    --reference-split test \
    --train-split train \
    --require-fcd \
    --nspdk-backend eden \
    --metric-molecule-source raw_valid \
    --output-dir "$baseline_dir/evaluation_molecules"

  echo "=== ${DATASET} requested_seed=${seed} train_seed=${train_seed}: connectivity filter ==="
  "$PYTHON" scripts/run_gdsm_simple_baseline.py \
    --stage generate \
    --dataset "$DATASET" \
    --no-common-config \
    --wrapper-config "$fixed_cfg" \
    --dataset-root "$DATASET_ROOT" \
    --serialized-dataset "$SERIAL" \
    --output-root "$OUTPUT_ROOT" \
    --seed-id "$train_seed" \
    --generation-seed "$seed" \
    --run-id "$run" \
    --generation-id "$fixed_gid" \
    --num-samples "$N" \
    --device "$DEVICE"

  "$PYTHON" scripts/evaluate_gdsm_categorical.py \
    --generated-dir "$fixed_dir" \
    --reference-graphs "$DATASET_ROOT/$SERIAL/test.pkl" \
    --seed "$EVAL_SEED"
  "$PYTHON" scripts/evaluate_generated_molecules.py \
    --generated-graphs "$fixed_dir/molecular_graphs.pkl" \
    --dataset-root "$DATASET_ROOT" \
    --dataset "$SERIAL" \
    --reference-split test \
    --train-split train \
    --require-fcd \
    --nspdk-backend eden \
    --metric-molecule-source raw_valid \
    --output-dir "$fixed_dir/evaluation_molecules"

  baseline_args+=(--baseline-dir "$baseline_dir")
  fixed_args+=(--fixed-dir "$fixed_dir")
done

"$PYTHON" scripts/summarize_molecular_connectivity_ablation.py \
  "${baseline_args[@]}" "${fixed_args[@]}" --output-dir "$OUT"
