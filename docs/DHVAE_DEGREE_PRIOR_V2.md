# DH-VAE v4: edge-conditioned degree prior

The generic invariant prior now supports the factorization

\[
p(D,n,m)=p_{train}(n)\,p_\phi(m\mid n)\,p_\phi(D\mid n,m).
\]

This is intended to reduce the upstream degree-distribution error that GraphER
cannot repair because every accepted rewiring preserves the sampled degree
sequence exactly.

## What changed

- **Empirical graph size** remains the default (`sample_num_nodes: empirical`).
- **Explicit edge-count head** learns `p(m | n)`.
- **Edge-conditioned decoder** predicts degree probabilities from `(z,n,m)`.
- **Edge-conditioned learned latent prior** optionally models `p(z | n,m)`.
- **Exact degree-sum sampling** uses dynamic programming to sample categorical
  node degrees conditional on `sum(d_i)=2m`.  Parity and edge count therefore
  hold before graphicality rejection; they are not repaired afterward.
- **Prior-distribution loss** applies a differentiable RBF-MMD surrogate between
  degree histograms decoded from unconditional prior samples and training
  histograms.
- **Aggregate latent moment loss** reduces the aggregate-posterior / learned-prior
  mismatch in addition to the ordinary conditional KL.
- **Degree-only evaluation** now reports a training-empirical resampling reference,
  raw-vs-accepted feasibility, exact-edge-sum diagnostics, and errors in mean
  degree, degree variance, second moment, maximum degree, and wedge counts.

Old architecture-v2/v3 checkpoints remain loadable.  The new edge-conditioned
model writes `architecture_version: 4`.

## Community-small: train

```bash
PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/dhvae/community_small.yaml
```

The direct-training checkpoint is

```text
outputs/degree_generators/sbm/seed_42/checkpoint.pt
```

Alternatively, use the managed baseline wrapper if a wrapper-published checkpoint
is desired:

```bash
PYTHONPATH=src python scripts/run_dhvae_hh_baseline.py \
  --dataset community_small \
  --num-samples 64 \
  --seed-id 42 \
  --run-id seed_42_dhvae_v4 \
  --experiment-config configs/experiments/dhvae/community_small.yaml \
  --device gpu \
  --disable-training-estimates
```

## Cheap degree-only evaluation

Run this before GraphER refinement:

```bash
PYTHONPATH=src python scripts/evaluate_degree_generator.py \
  --config configs/experiments/dhvae/community_small.yaml \
  --num-samples 10000 \
  --skip-constructor-check
```

The comparison table includes:

```text
train_to_test
train_empirical_resample_to_test
posterior_reconstruction_to_test
aggregate_posterior_to_test
standard_normal_prior_to_test
learned_prior_raw_to_test
learned_prior_accepted_to_test
```

The main target is for `learned_prior_accepted_to_test` to approach the
`train_empirical_resample_to_test` degree MMD without simply copying training
sequences.

## Useful edge-count ablation

The default learns edge count:

```yaml
sample_num_edges: model
```

To isolate whether `p(m|n)` is still a bottleneck, temporarily evaluate with:

```yaml
sample_num_edges: empirical
```

This samples `m` from the training conditional distribution while retaining the
learned latent/degree decoder.  It is a training-only empirical diagnostic and
uses no test information.

## GraphER generation with the direct DH-VAE v4 checkpoint

No topology predictor retraining is required merely to change the degree prior.
Override only the invariant-prior checkpoint:

```bash
PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config configs/experiments/grapher/community_small_topology_spectral_graphlet_v2.yaml \
  --output-dir outputs/topology_generation/community_small_dhvae_v4/seed_42 \
  --num-generate 256 \
  --seed 42 \
  --device gpu \
  --set degree_generator.checkpoint_path=outputs/degree_generators/sbm/seed_42/checkpoint.pt
```

Use 128--256 refined graphs during development.  Reserve 1024 x three seeds for
final reporting after the cheap degree-only metrics have improved.

## Legacy configuration

The previous generic DH-VAE settings are retained as:

```text
configs/experiments/dhvae/community_small_legacy_v1.yaml
configs/experiments/dhvae/ego_small_legacy_v1.yaml
```
