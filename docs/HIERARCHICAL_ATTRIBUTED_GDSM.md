# Hierarchical attributed Laplacian log-gap GDSM

This variant removes post-generation rewiring from the molecular GDSM path and
aligns auxiliary supervision with the factorization

\[
p(A,X,R)=p(A)\,p(X,R\mid A).
\]

## Architecture

The topology branch receives no atom or bond categories. It performs Laplacian
log-gap diffusion and is jointly supervised by connected induced **unattributed**
graphlets of orders 3, 4, and 5 and the 15-dimensional ORCA orbit summary.

The attribute branch has a separate PPGN encoder. It conditions on the generated
topology and masked categorical inputs, predicts atoms and bond types on present
edges only, and receives connected induced **typed** graphlet supervision of
orders 3, 4, and 5. The final decoder retains degree-compatible atom sampling
and valence-budgeted bond sampling.

The two PPGN branches have disjoint parameters. Typed-graphlet, atom, and bond
losses therefore cannot update the topology score network. There is no rewiring,
post-hoc repair, filtering, or replacement sampling.

## Configs

```text
configs/experiments/gdsm_laplacian_loggap_hierarchical_attributed_explicit/
  qm9_seed_42.yaml
  qm9_seed_43.yaml
  qm9_seed_44.yaml
```

## Train, generate, and evaluate QM9

```bash
SEED=42
RUN="seed_${SEED}_qm9_hierarchical_topology_typed_v1"
CFG="configs/experiments/gdsm_laplacian_loggap_hierarchical_attributed_explicit/qm9_seed_${SEED}.yaml"
N=10000

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage train \
  --dataset qm9 \
  --common-config configs/baselines/common_qm9.yaml \
  --wrapper-config "$CFG" \
  --seed-id "$SEED" \
  --run-id "$RUN" \
  --device gpu

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset qm9 \
  --common-config configs/baselines/common_qm9.yaml \
  --wrapper-config "$CFG" \
  --seed-id "$SEED" \
  --generation-seed "$SEED" \
  --run-id "$RUN" \
  --generation-id "hierarchical_seed_${SEED}_n_${N}" \
  --num-samples "$N" \
  --device gpu

GEN="outputs/baselines/gdsm_simple/qm9/$RUN/generations/hierarchical_seed_${SEED}_n_${N}"

PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --dataset-root outputs/datasets \
  --dataset qm9_attributed \
  --reference-split val \
  --train-split train \
  --output-dir "$GEN/evaluation_raw_val_g345_all" \
  --strict-raw-metrics \
  --metric-molecule-source raw_valid \
  --nspdk-backend eden \
  --nspdk-bond-label-mode hogdiff \
  --require-fcd \
  --fcd-device auto \
  --graphlet-mmd \
  --graphlet-k-min 3 \
  --graphlet-k-max 5 \
  --graphlet-topology-filter all \
  --graphlet-node-attribute atomic_num \
  --graphlet-edge-attribute bond_type \
  --graphlet-attributed-backend python
```

## Evaluate the auxiliary heads

```bash
PYTHONPATH=src python scripts/evaluate_gdsm_structure_summary_predictions.py \
  --generated-dir "$GEN"

PYTHONPATH=src python scripts/evaluate_gdsm_typed_graphlet_summary_predictions.py \
  --generated-dir "$GEN"
```

The first command evaluates topology-only graphlets and orbit summaries. The
second evaluates typed graphlet predictions using the training-only typed basis
stored in the checkpoint.

## Final three-seed run

Repeat the same commands for seeds 42, 43, and 44 without changing the decoder,
loss weights, or evaluation protocol. Select the architecture using validation
results and use the held-out test split only after all choices are frozen.
