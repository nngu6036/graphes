# Fixed-k Laplacian diffusion with conditional spectral-tail completion

This implementation keeps the controlled fixed-k information bottleneck while
removing the zero-tail/Gumbel-decoder confound from the earlier ablation.

For an n-node graph, only the first `min(k,n-1)` nontrivial normalized
Laplacian log-gap coordinates are diffused. The remaining high-frequency
coordinates are completed at the endpoint with a training-only conditional
ridge-Gaussian model:

`tail_j | prefix, n ~ Normal(beta_j^T [1, prefix, n/N], sigma_j^2)`.

The model is fit from the training split only. It operates in normalized
log-gap space, so converting the completed state back to Laplacian eigenvalues
preserves non-negativity and ordering. The full donor Laplacian eigenbasis is
then used with the completed spectrum and the original threshold topology
decoder. No degree-budget/Gumbel topology decoder is used in the new configs.

The special `k=full` configuration sets `k=max_nodes-1`. It has no missing tail,
so tail completion is a no-op and does not consume random numbers. This is the
sanity-control configuration and should reproduce the ordinary full-spectrum
path up to the fact that the configuration explicitly records the cutoff.

## Configuration

```yaml
sde:
  eigen_mask: laplacian_nonzero_prefix
  spectral_parameterization: laplacian_log_gap
  spectral_cutoff:
    enabled: true
    k: 8
    selection: low_frequency_nonzero
    tail_policy: conditional_ridge_gaussian
    tail_ridge: 0.01
    tail_min_std: 0.05
    tail_temperature: 1.0
    tail_condition_on_size: true

sample:
  threshold: 0.5
  fixed_k_decoder:
    mode: threshold
```

`tail_temperature=0` uses the conditional mean; `1.0` samples at the fitted
residual scale. Select all fixed-k settings using the validation split and use
the test split only after freezing the model.

## Shipped configs

Configs for Community-small and Ego-small, seeds 42/43/44, and
`k in {2,4,6,8,12,full}` are under:

`configs/experiments/gdsm_laplacian_loggap_topology_degree_g345_fixed_k_tail_explicit/`

The topology auxiliary targets remain degree histogram, topology graphlets
3--5, and ORCA orbit summaries. Rewiring and post-generation repair are off.

## Seed-42 sweep

```bash
SEED=42
N=1024
DEVICE=cuda:0
REFERENCE_SPLIT=val

for DATASET in community_small ego_small; do
  if [ "$DATASET" = community_small ]; then
    SERIALIZED_DATASET=sbm
  else
    SERIALIZED_DATASET=ego_small
  fi

  for K in 2 4 6 8 12 full; do
    CFG="configs/experiments/gdsm_laplacian_loggap_topology_degree_g345_fixed_k_tail_explicit/${DATASET}_k${K}_seed_${SEED}.yaml"
    RUN="seed_${SEED}_${DATASET}_fixed_k${K}_tail_g345_v2"
    GEN_ID="seed_${SEED}_n_${N}"

    PYTHONHASHSEED="$SEED" GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
    python scripts/run_gdsm_simple_baseline.py \
      --stage train \
      --dataset "$DATASET" \
      --common-config "configs/baselines/common_${DATASET}.yaml" \
      --wrapper-config "$CFG" \
      --dataset-root outputs/datasets \
      --serialized-dataset "$SERIALIZED_DATASET" \
      --output-root outputs/baselines \
      --seed-id "$SEED" \
      --run-id "$RUN" \
      --device "$DEVICE"

    PYTHONHASHSEED="$SEED" GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
    python scripts/run_gdsm_simple_baseline.py \
      --stage generate \
      --dataset "$DATASET" \
      --common-config "configs/baselines/common_${DATASET}.yaml" \
      --wrapper-config "$CFG" \
      --dataset-root outputs/datasets \
      --serialized-dataset "$SERIALIZED_DATASET" \
      --output-root outputs/baselines \
      --seed-id "$SEED" \
      --generation-seed "$SEED" \
      --run-id "$RUN" \
      --generation-id "$GEN_ID" \
      --num-samples "$N" \
      --device "$DEVICE"

    GEN="outputs/baselines/gdsm_simple/${DATASET}/${RUN}/generations/${GEN_ID}"
    ORCA_EXEC=/home/quang/orca/orca.out PYTHONPATH=src \
    python scripts/evaluate_graph_generation_report.py \
      --config "configs/experiments/baselines/${DATASET}_evaluation_degree_g345.yaml" \
      --reference-split "$REFERENCE_SPLIT" \
      --generic-mmd-protocol graphrnn \
      --generated-dir "$GEN" \
      --generated-graphs "$GEN/base_graphs.pkl" \
      --output-dir "$GEN/evaluation_${REFERENCE_SPLIT}_fixed_k${K}_tail"
  done
done
```

Use the new exact-isomorphism diagnostics together with Degree/Clustering/Orbit/
Spectral/Graphlet MMD. In particular, verify first that `k=full` recovers the
full-spectrum baseline before interpreting the truncated-k results.
