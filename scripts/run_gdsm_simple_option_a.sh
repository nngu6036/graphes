#!/usr/bin/env bash
# Run from any directory. Dataset and stage are positional arguments.
# VARIANT=initialization_only (default) or legacy_conditioning (existing checkpoint).
# Example: DEVICE=cuda:0 N=1024 bash scripts/run_gdsm_simple_option_a.sh community_small all
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
DATASET="${1:-community_small}"
STAGE="${2:-check}"
VARIANT="${VARIANT:-initialization_only}"
SEED=42
N="${N:-1024}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"
case "$DATASET" in
  community_small) SERIALIZED=sbm ;;
  ego_small) SERIALIZED=ego_small ;;
  *) echo 'Dataset must be community_small or ego_small.' >&2; exit 2 ;;
esac
case "$VARIANT" in
  initialization_only) SUFFIX=""; DEFAULT_RUN=seed_42_structure3_option_a ;;
  legacy_conditioning) SUFFIX=_legacy_conditioning; DEFAULT_RUN=seed_42_structure3_degree_basis ;;
  *) echo 'VARIANT must be initialization_only or legacy_conditioning.' >&2; exit 2 ;;
esac
case "$STAGE" in check|degree|train|generate|audit|evaluate|evaluate-local|all) ;;
  *) echo 'Stage must be check, degree, train, generate, audit, evaluate, evaluate-local or all.' >&2; exit 2 ;;
esac
if [[ "$VARIANT" == legacy_conditioning && ( "$STAGE" == train || "$STAGE" == all ) ]]; then
  echo 'legacy_conditioning is for existing checkpoints; use generate, audit or evaluate. Use the default variant for new strict training.' >&2
  exit 2
fi
if ! [[ "$N" =~ ^[1-9][0-9]*$ ]]; then echo 'N must be a positive integer.' >&2; exit 2; fi
CFG="${CFG:-configs/baselines/gdsm_simple_${DATASET}_structure3_option_a${SUFFIX}.yaml}"
COMMON="configs/baselines/common_${DATASET}.yaml"
DEGREE_CFG="configs/experiments/dhvae/${DATASET}.yaml"
DEGREE_CKPT="outputs/degree_generators/${SERIALIZED}/seed_42/checkpoint.pt"
EVAL_CFG="configs/experiments/baselines/${DATASET}_evaluation.yaml"
RUN="${RUN:-$DEFAULT_RUN}"
GEN_ID="${GEN_ID:-seed_${SEED}_n_${N}_option_a}"
GEN="outputs/baselines/gdsm_simple/${DATASET}/${RUN}/generations/${GEN_ID}"
for FILE in "$CFG" "$COMMON" "$DEGREE_CFG" "$EVAL_CFG"; do
  if [[ ! -f "$FILE" ]]; then echo "Missing configuration: $FILE" >&2; exit 1; fi
done
if [[ "$STAGE" == check || "$STAGE" == degree || "$STAGE" == train || "$STAGE" == all ]]; then
  "$PYTHON" - "$CFG" "$SERIALIZED" <<'PY'
from pathlib import Path
import math
import sys
import yaml
from grapher.models.gdsm_simple.wrapper import _graphs
from grapher.models.gdsm_simple.structure3 import validate_structure_options
cfg=yaml.safe_load(Path(sys.argv[1]).read_text())['gdsm_simple']
validate_structure_options(cfg)
if cfg['extensions'].get('generation_mode') != 'spectral_decode':
    raise SystemExit('Expected an Option-A configuration.')
root=Path('outputs/datasets')/sys.argv[2]
for split in ('train','val','test'):
    if not (root/f'{split}.pkl').is_file():
        raise SystemExit(f'Missing prepared split {root / (split+".pkl")}. Restore the shared splits; this runner never rebuilds data.')
train,val=(_graphs(root/f'{s}.pkl') for s in ('train','val'))
maximum=cfg['model']['max_nodes']
if any(len(g)<2 or len(g)>maximum for g in train+val):
    raise SystemExit(f'A train/validation graph is outside size range [2,{maximum}].')
missing=set(map(len,val))-set(map(len,train))
if missing:
    raise SystemExit(f'Validation sizes absent from the training-only basis bank: {sorted(missing)}. Do not add held-out bases.')
pairings=cfg['extensions']['structural_summary']['basis_pairings_per_graph']
updates=cfg['train']['epochs']*math.ceil(len(train)*pairings/cfg['train']['batch_size'])
print(f'Prepared data: train={len(train)}, val={len(val)}. Configured spectral optimizer updates: {updates:,}.')
print('Conditioning:',cfg['extensions']['initialization'].get('conditioning','degree_basis'))
print('Option-A configuration and prepared-size support check passed.')
PY
fi
if [[ "$STAGE" == degree || "$STAGE" == all ]]; then
  if [[ "$STAGE" == all && -f "$DEGREE_CKPT" ]]; then
    echo "Reusing $DEGREE_CKPT; generation verifies training-degree provenance."
  else
    "$PYTHON" scripts/train_degree_generator.py --config "$DEGREE_CFG"
  fi
fi
if [[ "$STAGE" == train || "$STAGE" == all ]]; then
  "$PYTHON" scripts/run_gdsm_simple_baseline.py --stage train --dataset "$DATASET" \
    --common-config "$COMMON" --wrapper-config "$CFG" \
    --seed-id "$SEED" --run-id "$RUN" --device "$DEVICE"
fi
if [[ "$STAGE" == generate || "$STAGE" == all ]]; then
  if [[ ! -f "$DEGREE_CKPT" ]]; then echo "Missing $DEGREE_CKPT; run the degree stage first." >&2; exit 1; fi
  "$PYTHON" scripts/run_gdsm_simple_baseline.py --stage generate --dataset "$DATASET" \
    --common-config "$COMMON" --wrapper-config "$CFG" \
    --seed-id "$SEED" --generation-seed "$SEED" --run-id "$RUN" \
    --generation-id "$GEN_ID" --num-samples "$N" --device "$DEVICE"
fi
if [[ "$STAGE" == audit || "$STAGE" == all ]]; then
  "$PYTHON" scripts/audit_gdsm_option_a.py --generated-dir "$GEN"
fi
if [[ "$STAGE" == evaluate || "$STAGE" == evaluate-local || "$STAGE" == all ]]; then
  BASE="$GEN/initial_graphs.pkl"; REPORT="$GEN/evaluation_test"
  if [[ "$STAGE" == evaluate-local ]]; then
    BASE="$GEN/final_pre_rewire_graphs.pkl"; REPORT="$GEN/evaluation_final_event"
  fi
  "$PYTHON" scripts/evaluate_graph_generation_report.py \
    --config "$EVAL_CFG" --generated-dir "$GEN" --generated-graphs "$GEN/base_graphs.pkl" \
    --generated-stage gdsm_simple_option_a --base-graphs "$BASE" --reference-split test \
    --generic-mmd-protocol graphrnn --num-samples 16 --output-dir "$REPORT"
fi
