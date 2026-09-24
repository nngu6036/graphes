#!/usr/bin/env bash
# Run from the project root. Never rebuild datasets or overwrite artifacts.
set -euo pipefail
DATASET="${1:-qm9}"; STAGE="${2:-all}"
VARIANT="${VARIANT:-main}"; SEED="${SEED:-42}"; GEN_SEED="${GEN_SEED:-$SEED}"
DEVICE="${DEVICE:-cuda:0}"; PYTHON="${PYTHON:-python}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
# Retain the validated fix for the reported small-matrix CUDA solver failure.
export GDSM_EIGH_BACKEND="${GDSM_EIGH_BACKEND:-cpu}"
[[ "$SEED" =~ ^[0-9]+$ && "$GEN_SEED" =~ ^[0-9]+$ ]] || { echo 'Seeds must be nonnegative integers' >&2; exit 2; }
case "$DATASET" in
 community_small) SERIAL=sbm; BASE_DEGREE=configs/experiments/dhvae/community_small.yaml ;;
 ego_small) SERIAL=ego_small; BASE_DEGREE=configs/experiments/dhvae/ego_small.yaml ;;
 qm9) SERIAL=qm9_attributed; BASE_DEGREE=configs/experiments/dhvae/qm9_categorical_spectral_prior.yaml ;;
 zinc) SERIAL=zinc; BASE_DEGREE=configs/experiments/dhvae/zinc_categorical_spectral_prior.yaml ;;
 *) echo 'Dataset must be community_small, ego_small, qm9 or zinc' >&2; exit 2 ;;
esac
SUFFIX='';TRAIN_SUFFIX=''
case "$VARIANT" in
 main) ;;
 no_guidance|final_only) SUFFIX="_$VARIANT" ;;
 connectivity_filter)
  if [[ "$DATASET" != qm9 && "$DATASET" != zinc ]]; then
   echo 'connectivity_filter is a molecular-only sampling ablation/fix (qm9 or zinc)' >&2; exit 2
  fi
  SUFFIX="_$VARIANT" ;;
 backbone_only|categorical_only|k3|k34) SUFFIX="_$VARIANT";TRAIN_SUFFIX="$SUFFIX" ;;
 *) echo 'Unsupported ablation' >&2; exit 2 ;;
esac
TEMPLATE="configs/experiments/grapher_research/${DATASET}_g345${SUFFIX}.yaml"
RUN="${RUN:-seed_${SEED}_grapher_g345${TRAIN_SUFFIX}}"
ROOT=outputs/datasets
OUT=outputs/baselines
if [[ -z "${N:-}" ]]; then
 N="$("$PYTHON" - "$DATASET" <<'PY'
import yaml,sys
p=yaml.safe_load(open('configs/experiments/grapher_research/protocol.yaml'))
print(p['datasets'][sys.argv[1]]['num_generate'])
PY
)"
fi
[[ "$N" =~ ^[1-9][0-9]*$ ]] || { echo 'N must be a positive integer' >&2; exit 2; }
GID="${GID:-seed_${GEN_SEED}_n_${N}${SUFFIX}}"
GEN="$OUT/gdsm_simple/$DATASET/$RUN/generations/$GID"
# Persist the exact seed-resolved model AND degree-prior YAMLs used by each run.
RESOLVED="outputs/research_configs/$DATASET/$RUN/$VARIANT"
mkdir -p "$RESOLVED"
CFG="$RESOLVED/model.yaml";DEGREE_CFG="$RESOLVED/degree_prior.yaml"
"$PYTHON" - "$TEMPLATE" "$BASE_DEGREE" "$CFG" "$DEGREE_CFG" "$SEED" <<'PY'
import sys,yaml
from pathlib import Path
src,ds,out,dout,seed=sys.argv[1:];seed=int(seed)
def rewrite(x):
    if isinstance(x,dict):return {k:rewrite(v) for k,v in x.items()}
    if isinstance(x,list):return [rewrite(v) for v in x]
    if isinstance(x,str):return x.replace('/seed_42/',f'/seed_{seed}/')
    return x
model=rewrite(yaml.safe_load(Path(src).read_text()))
degree=rewrite(yaml.safe_load(Path(ds).read_text()));degree['seed']=seed
# No data-seed mutation: all training seeds use the same prepared split.
for path,value in ((out,model),(dout,degree)):
    p=Path(path)
    if p.exists() and yaml.safe_load(p.read_text())!=value:
        raise SystemExit(f'Resolved configuration collision at {p}; use a new RUN')
    if not p.exists():p.write_text(yaml.safe_dump(value,sort_keys=False))
PY
export DATASET SERIAL CFG DEGREE_CFG RUN N SEED GEN_SEED DEVICE GID GEN
printf 'dataset=%s variant=%s seed=%s run=%s N=%s solver=%s\n' "$DATASET" "$VARIANT" "$SEED" "$RUN" "$N" "$GDSM_EIGH_BACKEND"
check() { "$PYTHON" scripts/check_gdsm_categorical_data.py --wrapper-config "$CFG" --dataset-root "$ROOT" --serialized-dataset "$SERIAL"; }
profile() { "$PYTHON" scripts/profile_grapher_graphlets.py --wrapper-config "$CFG" --graphs "$ROOT/$SERIAL/train.pkl" --max-graphs "${PROFILE_N:-32}"; }
degree() {
 "$PYTHON" - "$CFG" "$DEGREE_CFG" <<'PY'
import sys,yaml,subprocess
from pathlib import Path
c=yaml.safe_load(Path(sys.argv[1]).read_text())['gdsm_simple']['extensions']['attributed_categorical']
p=c['initialization']['degree_generator']
if c['initialization']['mode']=='gaussian' or p['type']=='empirical':raise SystemExit(0)
checkpoint=Path(p['checkpoint_path'])
if checkpoint.exists():
    print(f'Reusing {checkpoint}; generation verifies the training-degree provenance.');raise SystemExit(0)
d=yaml.safe_load(Path(sys.argv[2]).read_text())
if Path(d['degree_generator']['checkpoint_path']).resolve()!=checkpoint.resolve():
    raise SystemExit('Degree-prior config/checkpoint mismatch')
subprocess.run([sys.executable,'scripts/train_degree_generator.py','--config',sys.argv[2]],check=True)
PY
}
train() {
 if [[ "$VARIANT" == no_guidance || "$VARIANT" == final_only || "$VARIANT" == connectivity_filter ]]; then
  echo 'This is a sampling-only ablation. Train main first; run this variant with generate/audit/evaluate.' >&2; return 2
 fi
 "$PYTHON" scripts/run_gdsm_simple_baseline.py --stage train --dataset "$DATASET" --no-common-config \
  --wrapper-config "$CFG" --dataset-root "$ROOT" --serialized-dataset "$SERIAL" --output-root "$OUT" \
  --seed-id "$SEED" --run-id "$RUN" --device "$DEVICE"
}
generate() {
 "$PYTHON" scripts/run_gdsm_simple_baseline.py --stage generate --dataset "$DATASET" --no-common-config \
  --wrapper-config "$CFG" --dataset-root "$ROOT" --serialized-dataset "$SERIAL" --output-root "$OUT" \
  --seed-id "$SEED" --generation-seed "$GEN_SEED" --run-id "$RUN" \
  --generation-id "$GID" --num-samples "$N" --device "$DEVICE"
}
audit() { "$PYTHON" scripts/audit_gdsm_categorical.py --generated-dir "$GEN"; }
evaluate() {
 "$PYTHON" scripts/evaluate_gdsm_categorical.py --generated-dir "$GEN" --reference-graphs "$ROOT/$SERIAL/test.pkl" --seed "$GEN_SEED"
 if [[ "$DATASET" == qm9 || "$DATASET" == zinc ]]; then
  "$PYTHON" scripts/evaluate_generated_molecules.py --generated-graphs "$GEN/molecular_graphs.pkl" \
   --dataset-root "$ROOT" --dataset "$SERIAL" --reference-split test --train-split train \
   --require-fcd --metric-molecule-source raw_valid --output-dir "$GEN/evaluation_molecules"
 else
  "$PYTHON" scripts/evaluate_graph_generation_report.py --config "configs/experiments/baselines/${DATASET}_evaluation.yaml" \
   --generated-dir "$GEN" --generated-graphs "$GEN/base_graphs.pkl" --base-graphs "$GEN/initial_graphs.pkl" \
   --generated-stage grapher_g345 --reference-split test --generic-mmd-protocol graphrnn \
   --num-samples 16 --output-dir "$GEN/evaluation_topology"
 fi
}
case "$STAGE" in
 resolve) echo "$CFG" ;;
 check|profile|degree|train|generate|audit|evaluate) "$STAGE" ;;
 all) check; profile; degree; train; generate; audit; evaluate ;;
 *) echo 'Stage must be resolve, check, profile, degree, train, generate, audit, evaluate or all' >&2;exit 2 ;;
esac
