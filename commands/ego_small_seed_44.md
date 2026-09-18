## ego_small — seed 44

Required prior: `outputs/degree_generators/gdsm_final_g345/ego_small/seed_44/checkpoint.pt`.

### Check the supplied config against the prepared data

```bash
PYTHONHASHSEED=44 PYTHONPATH=src python scripts/check_gdsm_categorical_data.py \
  --wrapper-config configs/experiments/gdsm_final_explicit/ego_small_seed_44.yaml \
  --dataset-root outputs/datasets \
  --serialized-dataset ego_small
```

### Train the model

```bash
PYTHONHASHSEED=44 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage train \
  --dataset ego_small \
  --no-common-config \
  --wrapper-config configs/experiments/gdsm_final_explicit/ego_small_seed_44.yaml \
  --dataset-root outputs/datasets \
  --serialized-dataset ego_small \
  --output-root outputs/baselines \
  --seed-id 44 \
  --run-id seed_44_gdsm_final_g345 \
  --device cuda:0
```

### Generate

```bash
PYTHONHASHSEED=44 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset ego_small \
  --no-common-config \
  --wrapper-config configs/experiments/gdsm_final_explicit/ego_small_seed_44.yaml \
  --dataset-root outputs/datasets \
  --serialized-dataset ego_small \
  --output-root outputs/baselines \
  --seed-id 44 \
  --run-id seed_44_gdsm_final_g345 \
  --device cuda:0 \
  --generation-seed 44 \
  --generation-id seed_44_n_1024 \
  --num-samples 1024
```

### Audit

```bash
PYTHONHASHSEED=44 PYTHONPATH=src python scripts/audit_gdsm_categorical.py \
  --generated-dir outputs/baselines/gdsm_simple/ego_small/seed_44_gdsm_final_g345/generations/seed_44_n_1024
```

### Evaluate multiscale structure

```bash
PYTHONHASHSEED=42 PYTHONPATH=src python scripts/evaluate_gdsm_categorical.py \
  --generated-dir outputs/baselines/gdsm_simple/ego_small/seed_44_gdsm_final_g345/generations/seed_44_n_1024 \
  --reference-graphs outputs/datasets/ego_small/test.pkl \
  --seed 42
```

### Evaluate benchmark topology metrics

```bash
PYTHONHASHSEED=42 ORCA_EXEC=/home/quang/orca/orca.out PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/ego_small_evaluation.yaml \
  --generated-dir outputs/baselines/gdsm_simple/ego_small/seed_44_gdsm_final_g345/generations/seed_44_n_1024 \
  --generated-graphs outputs/baselines/gdsm_simple/ego_small/seed_44_gdsm_final_g345/generations/seed_44_n_1024/base_graphs.pkl \
  --base-graphs outputs/baselines/gdsm_simple/ego_small/seed_44_gdsm_final_g345/generations/seed_44_n_1024/initial_graphs.pkl \
  --generated-stage gdsm_final \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --num-samples 1024 \
  --output-dir outputs/baselines/gdsm_simple/ego_small/seed_44_gdsm_final_g345/generations/seed_44_n_1024/evaluation_topology
```
