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

## S1a: lambda-only degree-preserving rewiring (archived ablation)

The previous S1 experiment started from the S0 threshold graph `G0`, froze its
indexed degree sequence, and accepted double-edge swaps only when they reduced
RMSE to the diffusion-predicted adjacency eigenvalues.  It approximately solved

`argmin_{G : deg(G)=deg(G0)} RMSE(lambda(G), lambda_hat)`.

This isolated discrete realization but discarded part of the information used
by GSDM to build `G0`: the sampled empirical eigenbasis `U`.  On
Community-small the lambda-only refiner reduced its own spectral objective but
worsened clustering/orbit MMD, motivating a conservative source-preserving
variant.

The archived config is:

`configs/baselines/gdsm_simple_community_small_s1_lambda_only.yaml`

## S1b: source-preserving spectral refinement (current default)

The default `configs/baselines/gdsm_simple_community_small.yaml` keeps training
identical to S0. Only generation changes.

For every generated sample:

1. sample an empirical training eigenbasis `U` and diffuse a target adjacency
   spectrum `lambda_hat`;
2. reconstruct `A_soft = U diag(lambda_hat) U^T` and threshold at 0.5, giving
   the trusted source graph `G0`;
3. freeze the indexed degree sequence of `G0`;
4. compute the initial spectral residual
   `D_lambda(G0, lambda_hat)` for the whole generated batch;
5. refine only high-residual samples (default: top 25% by residual);
6. for a selected sample, propose ordinary double-edge swaps, but accept a move
   only when the generated eigenvalue target improves;
7. rank candidates with a joint source-preserving energy using:
   - normalized eigenvalue RMSE to `lambda_hat`;
   - a low-rank adjacency-eigenspace projector derived from the same sampled
     `U` used in the GSDM reconstruction;
   - an edge symmetric-difference penalty from `G0`;
8. cap correction at four accepted swaps by default.

The sampled basis is represented by a projector rather than raw eigenvectors so
column sign flips do not change the target.  The dominant mode positions are
selected from `|lambda_hat|`, and the same sorted spectral positions are used
for each candidate graph.

The default source-preserving energy is

`E(G) = w_lambda D_lambda_rel(G) + w_P D_P_rel(G) + w_src D_edge(G, G0)`

with `w_lambda = 1`, `w_P = 1`, and `w_src = 0.1`.  In addition, a hard
projector-worsening tolerance prevents a small lambda gain from destroying the
sampled eigenspace.  Every accepted move must still improve lambda when
`require_lambda_improvement: true`.

Thus the refiner is a **local corrector**, not a second graph generator.
Already-good GSDM samples remain unchanged.

Generation writes:

- `threshold_graphs.pkl`: original paired S0 threshold graphs;
- `base_graphs.pkl`: final refined graphs used by the common evaluator;
- `target_adjacency_eigenvalues.pkl`: frozen generated spectral targets;
- `sampled_basis_indices.pkl`: indices into the checkpoint's empirical basis bank;
- `rewiring_diagnostics.json`: spectral/projector/source/gating diagnostics.

Because S1b is generation-only, an existing compatible S0/S1 checkpoint can be
reused; retraining is not required.

## One-change-at-a-time ladder

| Stage | Change relative to previous stage | Question isolated |
| --- | --- | --- |
| S0 | spectral diffusion + empirical `U` + threshold | How well does the simple spectral generator work? |
| S1a | + lambda-only post-threshold rewiring | Does matching generated eigenvalues alone help? |
| S1b | + source-preserving gated `lambda + P(U)` refinement | Can rewiring improve lambda without destroying information already in GSDM's source? |
| S2 | + explicit generated degree sequence and HH source | Is a structured degree-realizable source better than inheriting degrees from thresholding? |
| S3 | + degree-conditioned spectral model / endpoint bridge | Does coupling `D` and `lambda` improve target consistency? |
| S4 | + permutation-invariant eigenspace summary | Does additional global structure improve constrained search? |
| S5 | + graphlet/local structural summary | What is the marginal value of local higher-order guidance? |

Keep the same dataset split, evaluation code, sample count and seed set for
paired comparisons.

## S1b commands

An already-trained run can be reused because source-preserving refinement is
generation-only:

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
`gdsm_simple_community_small_s0.yaml`. To reproduce the old lambda-only refiner,
use `gdsm_simple_community_small_s1_lambda_only.yaml`.

## Structure3 extension (17 September 2026)

The opt-in configuration `gdsm_simple_community_small_structure3.yaml` now
combines degree/basis conditioning, a matched source-centred spectral process,
size-three clean-summary heads and intermediate structural rewiring. It does
not change the retained S0/S1 configurations or silently replace the main S1
configuration. It requires a new checkpoint. See [the Structure3 guide](GDSM_SIMPLE_STRUCTURE3.md)
for the implemented process, learned degree-prior checks, controls and commands.
