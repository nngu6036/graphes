#!/usr/bin/env bash
# Run from the project root. Does not rebuild data or overwrite an existing run.
set -euo pipefail
DATASET="${1:-qm9}"
STAGE="${2:-all}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
N="${N:-1024}"
SEED="${SEED:-42}"
GEN_SEED="${GEN_SEED:-$SEED}"
VARIANT="${VARIANT:-main}"
SUFFIX=""
case "$VARIANT" in
  main) ;;
  no_guidance|categorical_only|empirical_prior) SUFFIX="_${VARIANT}" ;;
  *) echo "Unknown VARIANT: $VARIANT" >&2; exit 2 ;;
esac
RUN_SUFFIX=""
if [[ "$VARIANT" == categorical_only || "$VARIANT" == empirical_prior ]]; then RUN_SUFFIX="_${VARIANT}"; fi
RUN="${RUN:-seed_${SEED}_spectral_categorical${RUN_SUFFIX}}"
CFG="${CFG:-configs/baselines/gdsm_simple_${DATASET}_categorical${SUFFIX}.yaml}"
ROOT="${DATASET_ROOT:-outputs/datasets}"
OUT="${OUTPUT_ROOT:-outputs/baselines}"
GID="${GID:-seed_${GEN_SEED}_n_${N}${SUFFIX}}"
GEN="${OUT}/gdsm_simple/${DATASET}/${RUN}/generations/${GID}"
case "$DATASET" in
  qm9) SERIAL=qm9_attributed; DEGREE_CFG=configs/experiments/dhvae/qm9_categorical_spectral_prior.yaml ;;
  zinc) SERIAL=zinc; DEGREE_CFG=configs/experiments/dhvae/zinc_categorical_spectral_prior.yaml ;;
  community_small) SERIAL=sbm; DEGREE_CFG=configs/experiments/dhvae/community_small.yaml ;;
  ego_small) SERIAL=ego_small; DEGREE_CFG=configs/experiments/dhvae/ego_small.yaml ;;
  attributed) SERIAL=attributed; DEGREE_CFG="" ;;
  *) echo "Dataset must be qm9, zinc, community_small, ego_small, or attributed" >&2; exit 2 ;;
esac
DEFAULT_SERIAL="$SERIAL"
SERIAL="${SERIALIZED_DATASET:-$SERIAL}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

check() {
  "$PYTHON" scripts/check_gdsm_categorical_data.py --wrapper-config "$CFG" --dataset-root "$ROOT" --serialized-dataset "$SERIAL"
}
degree() {
  local checkpoint type mode
  mapfile -t info < <("$PYTHON" - "$CFG" <<'PY'
import sys,yaml
c=yaml.safe_load(open(sys.argv[1]))['gdsm_simple']['extensions']['attributed_categorical']
p=c['initialization']['degree_generator']
print(c['initialization']['mode']);print(p['type']);print(p.get('checkpoint_path',''))
PY
)
  mode="${info[0]}"; type="${info[1]}"; checkpoint="${info[2]}"
  if [[ "$mode" == gaussian || "$type" == empirical ]]; then echo "No trained degree prior required by this config."; return; fi
  if [[ -f "$checkpoint" ]]; then echo "Reuse $checkpoint; generation verifies its training-degree multiset."; return; fi
  if [[ "$ROOT" != outputs/datasets || "$SERIAL" != "$DEFAULT_SERIAL" ]]; then
    echo "Train the matching ordinary degree prior explicitly with your custom dataset configuration." >&2; exit 2
  fi
  if [[ -z "$DEGREE_CFG" ]]; then echo "Train your custom DH-VAE explicitly before generation." >&2; exit 2; fi
  "$PYTHON" - "$DEGREE_CFG" "$checkpoint" <<'PRIORCHECK'
import sys,yaml
from pathlib import Path
configured=yaml.safe_load(Path(sys.argv[1]).read_text())['degree_generator']['checkpoint_path']
if Path(configured).resolve()!=Path(sys.argv[2]).resolve():
    raise SystemExit('Custom prior checkpoint path: train its matching degree config explicitly.')
PRIORCHECK
  "$PYTHON" scripts/train_degree_generator.py --config "$DEGREE_CFG"
}
train() {
  "$PYTHON" scripts/run_gdsm_simple_baseline.py --stage train --dataset "$DATASET" --no-common-config \
    --wrapper-config "$CFG" --dataset-root "$ROOT" --serialized-dataset "$SERIAL" \
    --output-root "$OUT" --seed-id "$SEED" --run-id "$RUN" --device "$DEVICE"
}
generate() {
  "$PYTHON" scripts/run_gdsm_simple_baseline.py --stage generate --dataset "$DATASET" --no-common-config \
    --wrapper-config "$CFG" --dataset-root "$ROOT" --serialized-dataset "$SERIAL" \
    --output-root "$OUT" --seed-id "$SEED" --generation-seed "$GEN_SEED" --run-id "$RUN" \
    --generation-id "$GID" --num-samples "$N" --device "$DEVICE"
}
audit() { "$PYTHON" scripts/audit_gdsm_categorical.py --generated-dir "$GEN"; }
evaluate() {
  "$PYTHON" scripts/evaluate_gdsm_categorical.py --generated-dir "$GEN" --reference-graphs "$ROOT/$SERIAL/test.pkl"
  if [[ "$DATASET" == qm9 || "$DATASET" == zinc ]]; then
    "$PYTHON" scripts/evaluate_generated_molecules.py --generated-graphs "$GEN/molecular_graphs.pkl" \
      --dataset-root "$ROOT" --dataset "$SERIAL" --reference-split test --train-split train \
      --output-dir "$GEN/evaluation_molecules" --require-fcd --metric-molecule-source raw_valid
  elif [[ "$DATASET" == community_small || "$DATASET" == ego_small ]]; then
    "$PYTHON" scripts/evaluate_graph_generation_report.py \
      --config "configs/experiments/baselines/${DATASET}_evaluation.yaml" \
      --generated-dir "$GEN" --generated-graphs "$GEN/base_graphs.pkl" --base-graphs "$GEN/initial_graphs.pkl" \
      --generated-stage gdsm_categorical --reference-split test --generic-mmd-protocol graphrnn \
      --num-samples 16 --output-dir "$GEN/evaluation_topology"
  fi
}
case "$STAGE" in
 check|degree|train|generate|audit|evaluate) "$STAGE" ;;
 all) check; degree; train; generate; audit; evaluate ;;
 *) echo "Stage must be check, degree, train, generate, audit, evaluate or all" >&2; exit 2 ;;
esac
