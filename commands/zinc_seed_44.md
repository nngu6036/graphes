## zinc — seed 44

Required prior: `outputs/degree_generators/gdsm_final_g345/zinc/seed_44/checkpoint.pt`.

### Check the supplied config against the prepared data

```bash
PYTHONHASHSEED=44 PYTHONPATH=src python scripts/check_gdsm_categorical_data.py \
  --wrapper-config configs/experiments/gdsm_final_explicit/zinc_seed_44.yaml \
  --dataset-root outputs/datasets \
  --serialized-dataset zinc
```

### Train the seed-matched DH-VAE prior

```bash
PYTHONHASHSEED=44 PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/dhvae_final_explicit/zinc_seed_44.yaml
```

### Train the model

```bash
PYTHONHASHSEED=44 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage train \
  --dataset zinc \
  --no-common-config \
  --wrapper-config configs/experiments/gdsm_final_explicit/zinc_seed_44.yaml \
  --dataset-root outputs/datasets \
  --serialized-dataset zinc \
  --output-root outputs/baselines \
  --seed-id 44 \
  --run-id seed_44_gdsm_final_g345 \
  --device cuda:0
```

### Generate

```bash
PYTHONHASHSEED=44 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset zinc \
  --no-common-config \
  --wrapper-config configs/experiments/gdsm_final_explicit/zinc_seed_44.yaml \
  --dataset-root outputs/datasets \
  --serialized-dataset zinc \
  --output-root outputs/baselines \
  --seed-id 44 \
  --run-id seed_44_gdsm_final_g345 \
  --device cuda:0 \
  --generation-seed 44 \
  --generation-id seed_44_n_10000 \
  --num-samples 10000
```

### Audit

```bash
PYTHONHASHSEED=44 PYTHONPATH=src python scripts/audit_gdsm_categorical.py \
  --generated-dir outputs/baselines/gdsm_simple/zinc/seed_44_gdsm_final_g345/generations/seed_44_n_10000
```

### Evaluate multiscale structure

```bash
PYTHONHASHSEED=42 PYTHONPATH=src python scripts/evaluate_gdsm_categorical.py \
  --generated-dir outputs/baselines/gdsm_simple/zinc/seed_44_gdsm_final_g345/generations/seed_44_n_10000 \
  --reference-graphs outputs/datasets/zinc/test.pkl \
  --seed 42
```

### Evaluate molecules (raw-valid cohort)

```bash
PYTHONHASHSEED=42 PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-graphs outputs/baselines/gdsm_simple/zinc/seed_44_gdsm_final_g345/generations/seed_44_n_10000/molecular_graphs.pkl \
  --dataset-root outputs/datasets \
  --dataset zinc \
  --reference-split test \
  --train-split train \
  --metric-molecule-source raw_valid \
  --nspdk-backend eden \
  --nspdk-complexity 4 \
  --nspdk-bond-label-mode hogdiff \
  --require-fcd \
  --fcd-device cuda:0 \
  --output-dir outputs/baselines/gdsm_simple/zinc/seed_44_gdsm_final_g345/generations/seed_44_n_10000/evaluation_molecules
```

### Evaluate QED (same raw-valid cohort)

```bash
PYTHONHASHSEED=42 PYTHONPATH=src python scripts/evaluate_qed_research.py \
  --generated-graphs outputs/baselines/gdsm_simple/zinc/seed_44_gdsm_final_g345/generations/seed_44_n_10000/molecular_graphs.pkl \
  --reference-graphs outputs/datasets/zinc/test.pkl \
  --molecular-report outputs/baselines/gdsm_simple/zinc/seed_44_gdsm_final_g345/generations/seed_44_n_10000/evaluation_molecules/molecular_evaluation_metrics.json \
  --evaluator scripts/evaluate_generated_molecules.py \
  --expected-num-generated 10000 \
  --thresholds 0.5 0.7 \
  --output-dir outputs/baselines/gdsm_simple/zinc/seed_44_gdsm_final_g345/generations/seed_44_n_10000/evaluation_qed
```
