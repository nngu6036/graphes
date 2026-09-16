# GSDM-Simple reference ladder

`gdsm_simple` is a project-owned spectral reference for controlled GraphER
ablation. It is intentionally smaller than the released GSDM implementation;
results must be labeled **GSDM-Simple**, not GSDM.

## S0: spectral threshold reference

- state: sorted adjacency eigenvalues `lambda`;
- forward process: VP-style discrete Gaussian diffusion on `lambda`;
- denoiser: masked Transformer epsilon predictor;
- generation prior: Gaussian eigenvalues + one empirical training eigenbasis;
- graph size: sampled jointly with the empirical eigenbasis;
- realization: `A_soft = U diag(lambda) U^T`, fixed threshold at 0.5;
- no connectivity repair, degree correction, HH, rewiring, graphlets,
  clustering, or orbit guidance.

The retained high-budget S0 config is:

`configs/baselines/gdsm_simple_community_small_s0.yaml`

## S1: threshold graph + degree-preserving spectral rewiring (implemented)

The default `configs/baselines/gdsm_simple_community_small.yaml` now implements
S1. Training is identical to S0. Only generation changes:

1. sample an empirical training eigenbasis `U` and diffuse a target adjacency
   spectrum `lambda_hat`;
2. reconstruct `A_soft = U diag(lambda_hat) U^T` and threshold at 0.5;
3. call the resulting graph `G0` and extract/freeze its indexed degree sequence;
4. propose ordinary double-edge swaps inside that degree fibre;
5. compute each candidate graph's sorted adjacency eigenvalues, normalized by
   `sqrt(n)` exactly as during training;
6. accept the best candidate only if its spectral RMSE to `lambda_hat` strictly
   improves;
7. repeat up to the configured rewiring budget.

Thus S1 approximately solves

`argmin_{G : deg(G)=deg(G0)} RMSE(lambda(G), lambda_hat)`

without changing the learned GSDM-Simple denoiser. If `G0` is connected, the
default config also requires accepted swaps to keep it connected; a disconnected
threshold reconstruction is not artificially forced to be connected.

Generation writes:

- `threshold_graphs.pkl`: original S0 threshold graphs;
- `base_graphs.pkl`: final S1 refined graphs (used by the common evaluator);
- `target_adjacency_eigenvalues.pkl`: frozen generated spectral targets;
- `rewiring_diagnostics.json`: per-graph and aggregate spectral improvements.

Because rewiring is generation-only, an existing compatible S0 checkpoint can
be reused with the S1 YAML; retraining is not required.

## One-change-at-a-time ladder

| Stage | Change relative to previous stage | Question isolated |
| --- | --- | --- |
| S0 | spectral diffusion + empirical `U` + threshold | How well does the simple spectral generator work? |
| S1 | + post-threshold degree-preserving spectral rewiring | Does discrete realization move the final graph closer to its generated spectrum? |
| S2 | + explicit generated degree sequence and HH source | Is a structured degree-realizable source better than inheriting degrees from thresholding? |
| S3 | + degree-conditioned spectral model / endpoint bridge | Does coupling `D` and `lambda` improve target consistency? |
| S4 | + permutation-invariant eigenspace summary | Does additional global structure improve constrained search? |
| S5 | + graphlet/local structural summary | What is the marginal value of local higher-order guidance? |

Keep the same dataset split, evaluation code, sample count and seed set for
paired comparisons.

## S1 commands

An already-trained run can be reused because S1 is generation-only:

```bash
RUN=seed_42_high_budget
N=1024

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config configs/baselines/gdsm_simple_community_small.yaml \
  --seed-id 42 --run-id "$RUN" --num-samples "$N" --device gpu

GEN_DIR="outputs/baselines/gdsm_simple/community_small/$RUN/generations/seed_42_n_${N}"
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN_DIR" \
  --reference-split test \
  --output-dir "$GEN_DIR/evaluation_test"
```

To reproduce S0 from the same checkpoint, generate with
`gdsm_simple_community_small_s0.yaml` and a different `--generation-id`.
