#!/usr/bin/env bash
# Run from the GraphES repository root after extracting the Option-C patch.
# Every dataset-specific choice is visible in its YAML and the commands below.
# These are explicit command examples, not a replacement baseline runner.
set -euo pipefail

# community_small: all training graphs, then 1024 outputs for each independent seed.
for SEED in 42 43 44; do
  PYTHONPATH=src python scripts/train_option_c.py \
    --config configs/experiments/option_c/community_small.yaml \
    --output-dir "outputs/option_c/community_small/seed_${SEED}/train" \
    --seed "$SEED" \
    --device gpu

  PYTHONPATH=src python scripts/generate_option_c.py \
    --config configs/experiments/option_c/community_small.yaml \
    --checkpoint "outputs/option_c/community_small/seed_${SEED}/train/checkpoints/best.pt" \
    --output-dir "outputs/option_c/community_small/seed_${SEED}/generation" \
    --num-generate 1024 \
    --steps 500 \
    --batch-size 32 \
    --sampler ddpm \
    --seed "$SEED" \
    --device gpu
done

# ego_small: all training graphs, then 1024 outputs for each independent seed.
for SEED in 42 43 44; do
  PYTHONPATH=src python scripts/train_option_c.py \
    --config configs/experiments/option_c/ego_small.yaml \
    --output-dir "outputs/option_c/ego_small/seed_${SEED}/train" \
    --seed "$SEED" \
    --device gpu

  PYTHONPATH=src python scripts/generate_option_c.py \
    --config configs/experiments/option_c/ego_small.yaml \
    --checkpoint "outputs/option_c/ego_small/seed_${SEED}/train/checkpoints/best.pt" \
    --output-dir "outputs/option_c/ego_small/seed_${SEED}/generation" \
    --num-generate 1024 \
    --steps 500 \
    --batch-size 32 \
    --sampler ddpm \
    --seed "$SEED" \
    --device gpu
done

# qm9: all training graphs, then 10000 outputs for each independent seed.
for SEED in 42 43 44; do
  PYTHONPATH=src python scripts/train_option_c.py \
    --config configs/experiments/option_c/qm9.yaml \
    --output-dir "outputs/option_c/qm9/seed_${SEED}/train" \
    --seed "$SEED" \
    --device gpu

  PYTHONPATH=src python scripts/generate_option_c.py \
    --config configs/experiments/option_c/qm9.yaml \
    --checkpoint "outputs/option_c/qm9/seed_${SEED}/train/checkpoints/best.pt" \
    --output-dir "outputs/option_c/qm9/seed_${SEED}/generation" \
    --num-generate 10000 \
    --steps 500 \
    --batch-size 32 \
    --sampler ddpm \
    --seed "$SEED" \
    --device gpu
done

# zinc: all training graphs, then 10000 outputs for each independent seed.
for SEED in 42 43 44; do
  PYTHONPATH=src python scripts/train_option_c.py \
    --config configs/experiments/option_c/zinc.yaml \
    --output-dir "outputs/option_c/zinc/seed_${SEED}/train" \
    --seed "$SEED" \
    --device gpu

  PYTHONPATH=src python scripts/generate_option_c.py \
    --config configs/experiments/option_c/zinc.yaml \
    --checkpoint "outputs/option_c/zinc/seed_${SEED}/train/checkpoints/best.pt" \
    --output-dir "outputs/option_c/zinc/seed_${SEED}/generation" \
    --num-generate 10000 \
    --steps 500 \
    --batch-size 32 \
    --sampler ddpm \
    --seed "$SEED" \
    --device gpu
done

