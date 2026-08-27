# Spectral + Graphlet-Logit Diffusion Guidance

GraphER separates a **continuous stochastic diffusion trajectory in structural-summary space** from a **discrete degree-preserving rewiring trajectory in graph space**.  They are not assumed to be equivalent.

## Training: diffuse summaries, do not rewire

For each clean graph `G*`, training constructs only the same-degree HH/base endpoint `G_src`.  It extracts

- `Lambda_src`, `Lambda_0`: source and clean Laplacian eigenvalue vectors;
- `z_src,k`, `z_0,k`: source and clean graphlet CLR/logit vectors.

At normalized progress `s in [0,1]` (`0=source`, `1=clean`), the predictor input is sampled directly from an endpoint-conditioned Brownian bridge:

`x_s = (1-alpha_s) x_src + alpha_s x_0 + sigma sqrt(alpha_s(1-alpha_s)) eps`.

No intermediate graph is constructed.  In particular, the continuous state does **not** have to be the spectrum or graphlet distribution of any realizable graph.

For spectra, optional noise projection fixes `lambda_1=0` and keeps the trace `sum(lambda)=2m`; ordering and higher moments are deliberately allowed to leave the graph-realizable manifold.  For graphlets, Gaussian noise is centered separately in each CLR block so every block remains in the zero-sum CLR gauge.

The network is an `x0` predictor.  It receives the continuous current summary, the explicit source endpoint summary, normalized diffusion progress, and a fixed source-graph context.  It predicts the clean spectrum and clean graphlet logits jointly.

`summary_diffusion.storage: streaming` is recommended because it resamples time/noise every epoch instead of reusing a fixed cache.

## Generation: diffusion proposes, rewiring projects

Generation starts from the HH/base graph.  At each reverse-progress step the current **actual graph** is mapped to its spectrum and graphlet logits, the network predicts clean summaries, and the bridge scheduler proposes the next continuous summary target.  GraphER then selects one or more valid double-edge swaps whose realized summaries are closest to that target.

Thus the two state sequences are distinct:

- continuous guide: `Z_src -> ... -> Z_t -> ... -> Z_0`;
- discrete graph: `G_src -> ... -> G_t -> ... -> G_final`.

Generally `Z(G_t) != Z_t`.  Their distance is the projection residual and is reported as a diagnostic.

## Graphlet simplex and logits

For graphlet order `k`, every connected induced graphlet count is divided by `C(n,k)`. A final `disconnected` coordinate contains the remaining probability mass. Each block is therefore a proper probability simplex over induced `k`-node subsets.  A centered log-ratio (CLR) transform maps the simplex to Euclidean coordinates for diffusion; blockwise softmax maps logits back to probabilities.

## Candidate energy

Valid degree-preserving swaps are scored by

`w_spec(t) * D_spec + w_graphlet(t) * D_CLR`.

A global-to-local schedule can emphasize spectrum early and graphlets later. Candidate graphlet states use the exact local-delta update; the full graphlet histogram is not recomputed for every candidate.

## Debugging

Set `topology_refiner.debug.enabled=true`.  `[GraphER/SpectralGraphlet]` lines report clean predictions, current/next targets, global/local weights, candidate counts, per-channel gains, projection residuals, and accepted swaps.

## Community-small v2 commands

Train:

```bash
PYTHONPATH=src python scripts/train_topology_grapher.py \
  --config configs/experiments/grapher/community_small_topology_spectral_graphlet_v2.yaml \
  --output-dir outputs/topology_grapher/community_small_spectral_graphlet_v2/seed_42 \
  --seed 42 \
  --device gpu
```

The startup log should contain:

```text
[GraphER/DiffusionTraining] training path: source summary -> stochastic continuous Brownian bridge -> clean summary; rewiring is generation-only projection ...
```

It should **not** say that spectral-family teacher rewiring trajectories are being prepared.

Generate:

```bash
PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config configs/experiments/grapher/community_small_topology_spectral_graphlet_v2.yaml \
  --output-dir outputs/topology_generation/community_small_spectral_graphlet_v2/seed_42 \
  --num-generate 1024 \
  --seed 42 \
  --device gpu
```

Evaluate:

```bash
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/grapher/community_small_topology_spectral_graphlet_v2.yaml \
  --generated-dir outputs/topology_generation/community_small_spectral_graphlet_v2/seed_42 \
  --output-dir outputs/topology_grapher/community_small_spectral_graphlet_v2/seed_42/evaluation
```
