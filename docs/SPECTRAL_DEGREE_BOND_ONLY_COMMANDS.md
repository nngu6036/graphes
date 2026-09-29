# Explicit commands: spectral-degree topology + bond-only labels
Run from the repository root in your existing project environment. The seed-42
examples below retain the frozen common dataset configurations. The seed-specific
ordinary-degree-prior checkpoints named in the YAMLs must already exist.
Use a **new run ID**; do not overwrite a trained joint edge/no-edge run.
For another seed, change all three of the YAML filename, `--seed-id`, and the
run/generation IDs together. Standalone YAMLs for seeds 41–45 are included.

## community_small — seed 42

```bash
PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage train \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config configs/experiments/gdsm_spectral_degree_explicit/community_small_seed_42.yaml \
  --seed-id 42 \
  --run-id spectral_degree_bond_only_seed_42 \
  --device gpu

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config configs/experiments/gdsm_spectral_degree_explicit/community_small_seed_42.yaml \
  --seed-id 42 \
  --run-id spectral_degree_bond_only_seed_42 \
  --generation-seed 42 \
  --generation-id seed_42_n_1024 \
  --num-samples 1024 \
  --device gpu

PYTHONPATH=src python scripts/audit_gdsm_categorical.py \
  --generated-dir outputs/baselines/gdsm_simple/community_small/spectral_degree_bond_only_seed_42/generations/seed_42_n_1024

PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir outputs/baselines/gdsm_simple/community_small/spectral_degree_bond_only_seed_42/generations/seed_42_n_1024 \
  --generated-graphs outputs/baselines/gdsm_simple/community_small/spectral_degree_bond_only_seed_42/generations/seed_42_n_1024/base_graphs.pkl \
  --generated-stage spectral_degree_bond_only \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --num-samples 1024 \
  --output-dir outputs/baselines/gdsm_simple/community_small/spectral_degree_bond_only_seed_42/generations/seed_42_n_1024/evaluation_topology
```

## ego_small — seed 42

```bash
PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage train \
  --dataset ego_small \
  --common-config configs/baselines/common_ego_small.yaml \
  --wrapper-config configs/experiments/gdsm_spectral_degree_explicit/ego_small_seed_42.yaml \
  --seed-id 42 \
  --run-id spectral_degree_bond_only_seed_42 \
  --device gpu

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset ego_small \
  --common-config configs/baselines/common_ego_small.yaml \
  --wrapper-config configs/experiments/gdsm_spectral_degree_explicit/ego_small_seed_42.yaml \
  --seed-id 42 \
  --run-id spectral_degree_bond_only_seed_42 \
  --generation-seed 42 \
  --generation-id seed_42_n_1024 \
  --num-samples 1024 \
  --device gpu

PYTHONPATH=src python scripts/audit_gdsm_categorical.py \
  --generated-dir outputs/baselines/gdsm_simple/ego_small/spectral_degree_bond_only_seed_42/generations/seed_42_n_1024

PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/ego_small_evaluation.yaml \
  --generated-dir outputs/baselines/gdsm_simple/ego_small/spectral_degree_bond_only_seed_42/generations/seed_42_n_1024 \
  --generated-graphs outputs/baselines/gdsm_simple/ego_small/spectral_degree_bond_only_seed_42/generations/seed_42_n_1024/base_graphs.pkl \
  --generated-stage spectral_degree_bond_only \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --num-samples 1024 \
  --output-dir outputs/baselines/gdsm_simple/ego_small/spectral_degree_bond_only_seed_42/generations/seed_42_n_1024/evaluation_topology
```

## qm9 — seed 42

```bash
PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage train \
  --dataset qm9 \
  --common-config configs/baselines/common_qm9.yaml \
  --wrapper-config configs/experiments/gdsm_spectral_degree_explicit/qm9_seed_42.yaml \
  --seed-id 42 \
  --run-id spectral_degree_bond_only_seed_42 \
  --device gpu

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset qm9 \
  --common-config configs/baselines/common_qm9.yaml \
  --wrapper-config configs/experiments/gdsm_spectral_degree_explicit/qm9_seed_42.yaml \
  --seed-id 42 \
  --run-id spectral_degree_bond_only_seed_42 \
  --generation-seed 42 \
  --generation-id seed_42_n_10000 \
  --num-samples 10000 \
  --device gpu

PYTHONPATH=src python scripts/audit_gdsm_categorical.py \
  --generated-dir outputs/baselines/gdsm_simple/qm9/spectral_degree_bond_only_seed_42/generations/seed_42_n_10000

PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-graphs outputs/baselines/gdsm_simple/qm9/spectral_degree_bond_only_seed_42/generations/seed_42_n_10000/molecular_graphs.pkl \
  --dataset qm9_attributed \
  --reference-split test \
  --train-split train \
  --output-dir outputs/baselines/gdsm_simple/qm9/spectral_degree_bond_only_seed_42/generations/seed_42_n_10000/evaluation_molecules \
  --metric-molecule-source raw_valid \
  --require-fcd
```

## zinc — seed 42

```bash
PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage train \
  --dataset zinc \
  --common-config configs/baselines/common_zinc.yaml \
  --wrapper-config configs/experiments/gdsm_spectral_degree_explicit/zinc_seed_42.yaml \
  --seed-id 42 \
  --run-id spectral_degree_bond_only_seed_42 \
  --device gpu

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset zinc \
  --common-config configs/baselines/common_zinc.yaml \
  --wrapper-config configs/experiments/gdsm_spectral_degree_explicit/zinc_seed_42.yaml \
  --seed-id 42 \
  --run-id spectral_degree_bond_only_seed_42 \
  --generation-seed 42 \
  --generation-id seed_42_n_10000 \
  --num-samples 10000 \
  --device gpu

PYTHONPATH=src python scripts/audit_gdsm_categorical.py \
  --generated-dir outputs/baselines/gdsm_simple/zinc/spectral_degree_bond_only_seed_42/generations/seed_42_n_10000

PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-graphs outputs/baselines/gdsm_simple/zinc/spectral_degree_bond_only_seed_42/generations/seed_42_n_10000/molecular_graphs.pkl \
  --dataset zinc_attributed \
  --reference-split test \
  --train-split train \
  --output-dir outputs/baselines/gdsm_simple/zinc/spectral_degree_bond_only_seed_42/generations/seed_42_n_10000/evaluation_molecules \
  --metric-molecule-source raw_valid \
  --require-fcd
```

The synthetic generation count is 1,024 per run; the molecular count is 10,000.
Audit degree preservation before interpreting distributional metrics. Molecular
evaluation above explicitly uses raw valid molecules rather than repaired ones.
