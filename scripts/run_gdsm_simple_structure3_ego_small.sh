#!/usr/bin/env bash
# Run from the project or any directory:
# bash scripts/run_gdsm_simple_structure3_ego_small.sh [check|degree|train|generate|evaluate|all]
# DEVICE overrides the spectral model device; DH-VAE reads device from DEGREE_CFG.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
STAGE="${1:-all}"
CFG="configs/baselines/gdsm_simple_ego_small_structure3.yaml"
DEGREE_CFG="configs/experiments/dhvae/ego_small.yaml"
COMMON="configs/baselines/common_ego_small.yaml"
EVAL_CFG="configs/experiments/baselines/ego_small_evaluation.yaml"
DEGREE_CKPT="outputs/degree_generators/ego_small/seed_42/checkpoint.pt"
RUN="${RUN:-seed_42_structure3_degree_basis}"
SEED=42
N="${N:-1024}"
DEVICE="${DEVICE:-cuda:0}"
GEN_ID="seed_${SEED}_n_${N}"
GEN="outputs/baselines/gdsm_simple/ego_small/${RUN}/generations/${GEN_ID}"
case "$STAGE" in check|degree|train|generate|evaluate|all) ;; *) echo "Unknown stage: $STAGE" >&2; exit 2;; esac
if ! [[ "$N" =~ ^[1-9][0-9]*$ ]]; then echo "N must be a positive integer." >&2; exit 2; fi
for FILE in "$CFG" "$DEGREE_CFG" "$COMMON" "$EVAL_CFG"; do
  if [[ ! -f "$FILE" ]]; then echo "Missing configuration: $FILE" >&2; exit 1; fi
done
# Fail early without rebuilding or modifying prepared benchmark splits.
if [[ "$STAGE" == check || "$STAGE" == degree || "$STAGE" == train || "$STAGE" == all ]]; then
  python - <<'PY'
from pathlib import Path
import math
import networkx as nx
import yaml
from grapher.models.gdsm_simple.wrapper import _graphs
from grapher.models.gdsm_simple.structure3 import validate_structure_options

cfg = yaml.safe_load(Path('configs/baselines/gdsm_simple_ego_small_structure3.yaml').read_text())['gdsm_simple']
validate_structure_options(cfg)
root = Path('outputs/datasets/ego_small')
missing = [str(root / f'{s}.pkl') for s in ('train', 'val', 'test') if not (root / f'{s}.pkl').is_file()]
if missing:
    raise SystemExit('Missing prepared split(s): ' + ', '.join(missing) + '\nPrepare or restore the common Ego-small dataset before training. Do not rebuild splits between stages.')
splits = {s: _graphs(root / f'{s}.pkl') for s in ('train', 'val')}
maximum = cfg['model']['max_nodes']
for split, graphs in splits.items():
    sizes = sorted(set(map(len, graphs)))
    print(f'{split}: {len(graphs)} graphs; node counts {sizes}')
    if any(len(g) < 2 or len(g) > maximum for g in graphs):
        raise SystemExit(f'{split} contains graphs outside the configured node range [2, {maximum}].')
    if any(nx.number_of_selfloops(g) or not nx.is_connected(g) for g in graphs):
        raise SystemExit(f'{split} contains self-loops or disconnected graphs; check benchmark preparation.')
unsupported = set(map(len, splits['val'])) - set(map(len, splits['train']))
if unsupported:
    raise SystemExit('This Structure3 implementation requires every validation node count in the training eigenbasis bank. Unsupported validation sizes: ' + str(sorted(unsupported)) + '. Do not add held-out eigenvectors or silently change the benchmark split.')
pairings = cfg['extensions']['structural_summary']['basis_pairings_per_graph']
updates = cfg['train']['epochs'] * math.ceil(len(splits['train']) * pairings / cfg['train']['batch_size'])
print(f'Configured spectral training budget: {updates:,} optimizer updates.')
print('Structure3 configuration and prepared-split compatibility check passed.')
PY
fi
if [[ "$STAGE" == degree || "$STAGE" == all ]]; then
  if [[ "$STAGE" == all && -f "$DEGREE_CKPT" ]]; then
    echo "Reusing $DEGREE_CKPT (generation verifies training-degree provenance)."
  else
    python scripts/train_degree_generator.py --config "$DEGREE_CFG"
  fi
fi
if [[ "$STAGE" == train || "$STAGE" == all ]]; then
  python scripts/run_gdsm_simple_baseline.py --stage train --dataset ego_small \
    --common-config "$COMMON" --wrapper-config "$CFG" \
    --seed-id "$SEED" --run-id "$RUN" --device "$DEVICE"
fi
if [[ "$STAGE" == generate || "$STAGE" == all ]]; then
  if [[ ! -f "$DEGREE_CKPT" ]]; then
    echo "Missing $DEGREE_CKPT; run this script with the degree stage first." >&2; exit 1
  fi
  python scripts/run_gdsm_simple_baseline.py --stage generate --dataset ego_small \
    --common-config "$COMMON" --wrapper-config "$CFG" \
    --seed-id "$SEED" --generation-seed "$SEED" --run-id "$RUN" \
    --generation-id "$GEN_ID" --num-samples "$N" --device "$DEVICE"
fi
if [[ "$STAGE" == evaluate || "$STAGE" == all ]]; then
  # Explicit files distinguish the pre-refinement source from the final output.
  python scripts/evaluate_graph_generation_report.py \
    --config "$EVAL_CFG" --generated-dir "$GEN" \
    --generated-graphs "$GEN/base_graphs.pkl" \
    --generated-stage gdsm_simple_structure3 \
    --base-graphs "$GEN/initial_graphs.pkl" --reference-split test \
    --generic-mmd-protocol graphrnn --num-samples 16 \
    --output-dir "$GEN/evaluation_test"
fi
