## community_small — seed 43

Required prior: `outputs/degree_generators/gdsm_final_g345/community_small/seed_43/checkpoint.pt`.

### Check the supplied config against the prepared data

```bash
PYTHONHASHSEED=43 PYTHONPATH=src python scripts/check_gdsm_categorical_data.py \
  --wrapper-config configs/experiments/gdsm_final_explicit/community_small_seed_43.yaml \
  --dataset-root outputs/datasets \
  --serialized-dataset sbm
```

### Train the model

```bash
PYTHONHASHSEED=43 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage train \
  --dataset community_small \
  --no-common-config \
  --wrapper-config configs/experiments/gdsm_final_explicit/community_small_seed_43.yaml \
  --dataset-root outputs/datasets \
  --serialized-dataset sbm \
  --output-root outputs/baselines \
  --seed-id 43 \
  --run-id seed_43_gdsm_final_g345 \
  --device cuda:0
```

### Generate

```bash
PYTHONHASHSEED=43 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset community_small \
  --no-common-config \
  --wrapper-config configs/experiments/gdsm_final_explicit/community_small_seed_43.yaml \
  --dataset-root outputs/datasets \
  --serialized-dataset sbm \
  --output-root outputs/baselines \
  --seed-id 43 \
  --run-id seed_43_gdsm_final_g345 \
  --device cuda:0 \
  --generation-seed 43 \
  --generation-id seed_43_n_1024 \
  --num-samples 1024
```

### Audit

```bash
PYTHONHASHSEED=43 PYTHONPATH=src python scripts/audit_gdsm_categorical.py \
  --generated-dir outputs/baselines/gdsm_simple/community_small/seed_43_gdsm_final_g345/generations/seed_43_n_1024
```

### Evaluate multiscale structure

```bash
PYTHONHASHSEED=42 PYTHONPATH=src python scripts/evaluate_gdsm_categorical.py \
  --generated-dir outputs/baselines/gdsm_simple/community_small/seed_43_gdsm_final_g345/generations/seed_43_n_1024 \
  --reference-graphs outputs/datasets/sbm/test.pkl \
  --seed 42
```

### Evaluate benchmark topology metrics

```bash
PYTHONHASHSEED=42 ORCA_EXEC=/home/quang/orca/orca.out PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir outputs/baselines/gdsm_simple/community_small/seed_43_gdsm_final_g345/generations/seed_43_n_1024 \
  --generated-graphs outputs/baselines/gdsm_simple/community_small/seed_43_gdsm_final_g345/generations/seed_43_n_1024/base_graphs.pkl \
  --base-graphs outputs/baselines/gdsm_simple/community_small/seed_43_gdsm_final_g345/generations/seed_43_n_1024/initial_graphs.pkl \
  --generated-stage gdsm_final \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --num-samples 1024 \
  --output-dir outputs/baselines/gdsm_simple/community_small/seed_43_gdsm_final_g345/generations/seed_43_n_1024/evaluation_topology
```
