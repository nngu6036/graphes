#!/usr/bin/env bash
# Option C v2: soft-degree + normalized-Laplacian consistency.
# Run from GraphES repository root. These commands use separate output roots,
# so the original Option-C runs/checkpoints are not overwritten.
set -euo pipefail

# Community-small: first run this dataset to test the proposed correction.
for SEED in 42 43 44; do
  CFG=configs/experiments/option_c_soft_consistency/community_small.yaml
  TRAIN="outputs/option_c_soft_consistency/community_small/seed_${SEED}/train"
  GEN="outputs/option_c_soft_consistency/community_small/seed_${SEED}/generation"

  PYTHONPATH=src python scripts/train_option_c.py \
    --config "$CFG" \
    --output-dir "$TRAIN" \
    --seed "$SEED" \
    --device gpu

  PYTHONPATH=src python scripts/generate_option_c.py \
    --config "$CFG" \
    --checkpoint "$TRAIN/checkpoints/best.pt" \
    --output-dir "$GEN" \
    --num-generate 1024 \
    --steps 500 \
    --batch-size 32 \
    --sampler ddpm \
    --seed "$SEED" \
    --device gpu
done

# Ego-small.
for SEED in 42 43 44; do
  CFG=configs/experiments/option_c_soft_consistency/ego_small.yaml
  TRAIN="outputs/option_c_soft_consistency/ego_small/seed_${SEED}/train"
  GEN="outputs/option_c_soft_consistency/ego_small/seed_${SEED}/generation"

  PYTHONPATH=src python scripts/train_option_c.py \
    --config "$CFG" --output-dir "$TRAIN" --seed "$SEED" --device gpu

  PYTHONPATH=src python scripts/generate_option_c.py \
    --config "$CFG" --checkpoint "$TRAIN/checkpoints/best.pt" --output-dir "$GEN" \
    --num-generate 1024 --steps 500 --batch-size 32 --sampler ddpm --seed "$SEED" --device gpu
done

# QM9.
for SEED in 42 43 44; do
  CFG=configs/experiments/option_c_soft_consistency/qm9.yaml
  TRAIN="outputs/option_c_soft_consistency/qm9/seed_${SEED}/train"
  GEN="outputs/option_c_soft_consistency/qm9/seed_${SEED}/generation"

  PYTHONPATH=src python scripts/train_option_c.py \
    --config "$CFG" --output-dir "$TRAIN" --seed "$SEED" --device gpu

  PYTHONPATH=src python scripts/generate_option_c.py \
    --config "$CFG" --checkpoint "$TRAIN/checkpoints/best.pt" --output-dir "$GEN" \
    --num-generate 10000 --steps 500 --batch-size 32 --sampler ddpm --seed "$SEED" --device gpu
done

# ZINC.
for SEED in 42 43 44; do
  CFG=configs/experiments/option_c_soft_consistency/zinc.yaml
  TRAIN="outputs/option_c_soft_consistency/zinc/seed_${SEED}/train"
  GEN="outputs/option_c_soft_consistency/zinc/seed_${SEED}/generation"

  PYTHONPATH=src python scripts/train_option_c.py \
    --config "$CFG" --output-dir "$TRAIN" --seed "$SEED" --device gpu

  PYTHONPATH=src python scripts/generate_option_c.py \
    --config "$CFG" --checkpoint "$TRAIN/checkpoints/best.pt" --output-dir "$GEN" \
    --num-generate 10000 --steps 500 --batch-size 32 --sampler ddpm --seed "$SEED" --device gpu
done
