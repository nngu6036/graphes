#!/usr/bin/env bash
# Run from the GraphES repository root after extracting the Option-C patch.
# Every dataset-specific choice is visible in its YAML and the commands below.
# These are explicit command examples, not a replacement baseline runner.
set -euo pipefail

# community_small: unchanged shared generic evaluator.
for SEED in 42 43 44; do
  PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
    --config configs/experiments/baselines/community_small_evaluation.yaml \
    --generated-dir "outputs/option_c/community_small/seed_${SEED}/generation" \
    --generated-stage option_c \
    --reference-split test \
    --output-dir "outputs/option_c/community_small/seed_${SEED}/generation/evaluation"
done

# ego_small: unchanged shared generic evaluator.
for SEED in 42 43 44; do
  PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
    --config configs/experiments/baselines/ego_small_evaluation.yaml \
    --generated-dir "outputs/option_c/ego_small/seed_${SEED}/generation" \
    --generated-stage option_c \
    --reference-split test \
    --output-dir "outputs/option_c/ego_small/seed_${SEED}/generation/evaluation"
done

# qm9: strict raw-valid population; no hidden HoG-Diff correction mode.
for SEED in 42 43 44; do
  PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
    --generated-graphs "outputs/option_c/qm9/seed_${SEED}/generation/molecular_graphs.pkl" \
    --dataset qm9_attributed \
    --dataset-root outputs/datasets \
    --reference-split test \
    --train-split train \
    --metric-molecule-source raw_valid \
    --nspdk-backend eden \
    --nspdk-bond-label-mode hogdiff \
    --require-fcd \
    --fcd-device auto \
    --output-dir "outputs/option_c/qm9/seed_${SEED}/generation/evaluation"
done

# zinc: strict raw-valid population; no hidden HoG-Diff correction mode.
for SEED in 42 43 44; do
  PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
    --generated-graphs "outputs/option_c/zinc/seed_${SEED}/generation/molecular_graphs.pkl" \
    --dataset zinc \
    --dataset-root outputs/datasets \
    --reference-split test \
    --train-split train \
    --metric-molecule-source raw_valid \
    --nspdk-backend eden \
    --nspdk-bond-label-mode hogdiff \
    --require-fcd \
    --fcd-device auto \
    --output-dir "outputs/option_c/zinc/seed_${SEED}/generation/evaluation"
done

