# Minimal GraphER spectral/clustering ablations

Base config:
`configs/experiments/grapher/community_small_topology_spectral_debug.yaml`

The patch adds `--set` overrides to training and adds three generation guidance modes:
- `spectral`
- `clustering`
- `spectral_clustering`

## 1. Spectrum-only

### Train
```bash
PYTHONPATH=src python scripts/train_topology_grapher.py \
  --config configs/experiments/grapher/community_small_topology_spectral_debug.yaml \
  --output-dir outputs/topology_grapher/community_small_ablation_spectral_only/seed_42 \
  --seed 42 \
  --device gpu \
  --set structure_summary_prediction.clustering_coefficient=false \
  --set topology_predictor.loss_weights.spectrum=1.0 \
  --set topology_predictor.loss_weights.moment2=0.0 \
  --set topology_predictor.loss_weights.low_frequency=0.0 \
  --set topology_predictor.loss_weights.clustering_coefficient=0.0
```

### Generate
```bash
PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config configs/experiments/grapher/community_small_topology_spectral_debug.yaml \
  --checkpoint outputs/topology_grapher/community_small_ablation_spectral_only/seed_42/checkpoint.pt \
  --output-dir outputs/topology_generation/community_small_ablation_spectral_only/seed_42 \
  --num-generate 64 \
  --seed 42 \
  --device gpu \
  --set generation.degree_source=train_empirical \
  --set topology_refiner.guidance_mode=spectral \
  --set topology_refiner.spectral_guidance.weight=1.0
```

### Evaluate
```bash
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/grapher/community_small_topology_spectral_debug.yaml \
  --generated-dir outputs/topology_generation/community_small_ablation_spectral_only/seed_42 \
  --output-dir outputs/topology_generation/community_small_ablation_spectral_only/seed_42/evaluation
```

## 2. Clustering-only

The diffusion input is still the noisy spectrum. Only the clean average-clustering target trains the shared spectral Transformer/head, and generation ranks swaps only by predicted clustering.

### Train
```bash
PYTHONPATH=src python scripts/train_topology_grapher.py \
  --config configs/experiments/grapher/community_small_topology_spectral_debug.yaml \
  --output-dir outputs/topology_grapher/community_small_ablation_clustering_only/seed_42 \
  --seed 42 \
  --device gpu \
  --set structure_summary_prediction.clustering_coefficient=true \
  --set topology_predictor.loss_weights.spectrum=0.0 \
  --set topology_predictor.loss_weights.moment2=0.0 \
  --set topology_predictor.loss_weights.low_frequency=0.0 \
  --set topology_predictor.loss_weights.clustering_coefficient=1.0
```

### Generate
```bash
PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config configs/experiments/grapher/community_small_topology_spectral_debug.yaml \
  --checkpoint outputs/topology_grapher/community_small_ablation_clustering_only/seed_42/checkpoint.pt \
  --output-dir outputs/topology_generation/community_small_ablation_clustering_only/seed_42 \
  --num-generate 64 \
  --seed 42 \
  --device gpu \
  --set generation.degree_source=train_empirical \
  --set topology_refiner.guidance_mode=clustering \
  --set topology_refiner.spectral_guidance.weight=0.0 \
  --set topology_refiner.clustering_guidance.weight=1.0
```

### Evaluate
```bash
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/grapher/community_small_topology_spectral_debug.yaml \
  --generated-dir outputs/topology_generation/community_small_ablation_clustering_only/seed_42 \
  --output-dir outputs/topology_generation/community_small_ablation_clustering_only/seed_42/evaluation
```

## 3. Spectrum + clustering

### Train
```bash
PYTHONPATH=src python scripts/train_topology_grapher.py \
  --config configs/experiments/grapher/community_small_topology_spectral_debug.yaml \
  --output-dir outputs/topology_grapher/community_small_ablation_spectral_clustering/seed_42 \
  --seed 42 \
  --device gpu \
  --set structure_summary_prediction.clustering_coefficient=true \
  --set topology_predictor.loss_weights.spectrum=1.0 \
  --set topology_predictor.loss_weights.moment2=0.0 \
  --set topology_predictor.loss_weights.low_frequency=0.0 \
  --set topology_predictor.loss_weights.clustering_coefficient=1.0
```

### Generate
```bash
PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config configs/experiments/grapher/community_small_topology_spectral_debug.yaml \
  --checkpoint outputs/topology_grapher/community_small_ablation_spectral_clustering/seed_42/checkpoint.pt \
  --output-dir outputs/topology_generation/community_small_ablation_spectral_clustering/seed_42 \
  --num-generate 64 \
  --seed 42 \
  --device gpu \
  --set generation.degree_source=train_empirical \
  --set topology_refiner.guidance_mode=spectral_clustering \
  --set topology_refiner.spectral_guidance.weight=1.0 \
  --set topology_refiner.clustering_guidance.weight=1.0
```

### Evaluate
```bash
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/grapher/community_small_topology_spectral_debug.yaml \
  --generated-dir outputs/topology_generation/community_small_ablation_spectral_clustering/seed_42 \
  --output-dir outputs/topology_generation/community_small_ablation_spectral_clustering/seed_42/evaluation
```

## Strict generation-objective ablation

For the cleanest comparison of candidate scoring, train the joint spectrum+clustering checkpoint once and reuse that exact checkpoint for all three generation modes. This keeps the predictor fixed and changes only `topology_refiner.guidance_mode` and weights.
