# GSDM-Simple: Structure3 and degree/basis spectral initialization

This opt-in stage extends the project-owned `gdsm_simple` model. It is **not a
reproduction of the published GSDM architecture**, and must be labelled
**GSDM-Simple-Structure3** in comparisons. The original S0, lambda-only S1 and
source-preserving S1 configurations and checkpoint path remain supported.

Primary configuration:

```
configs/baselines/gdsm_simple_community_small_structure3.yaml
```

A new training run is required. Old spectral-only checkpoints do not contain
the conditioning layers or structure heads; generation rejects attempts to use
them for this stage. The main `gdsm_simple_community_small.yaml` remains the
previous S1 configuration so old commands do not silently change experiments.

## 1. Learned outputs and training

The state is the **sorted adjacency spectrum divided by sqrt(n)**, not the
Laplacian spectrum. A masked Transformer predicts the diffusion noise and three
clean-endpoint summaries from the same hidden representation:

| Head | Representation | Default training weight |
|---|---|---:|
| Clustering | 100-bin normalized local-clustering histogram on [0,1] | 1.0 |
| Orbits | `log1p` of four graph-level mean node-orbit counts | 0.5 |
| Graphlets | Normalized counts of connected induced P3 and triangle graphlets | 0.5 |

The four orbit columns are ORCA-compatible columns **0,1,2,3**: edge endpoint,
induced-P3 endpoint, induced-P3 centre, triangle vertex. This is **not** the full
15-orbit size-at-most-four summary. The common evaluator can still report its
usual larger orbit vector; the predictor's scope does not change evaluation.

For each graph, exact triangle and induced-wedge counts are calculated directly.
No ORCA executable is needed for training these four orbit targets. Disconnected
triples are not counted as connected graphlets. A graph with no connected
size-three graphlet has an all-zero graphlet target.

Training minimizes masked epsilon MSE plus clustering KL and CDF MSE, orbit
log-mean MSE, and graphlet KL. The checkpoint is selected using the combined
validation loss. `training_metrics.json` additionally records epsilon MSE,
clustering W1, orbit log RMSE, and graphlet TV.

**The targets come from clean training/validation graphs.** No graphlet summary
is inferred by pretending a noisy eigenvalue vector is a discrete graph. No
rewiring is performed during training.

### Conditioning and source/target pairing

Conditioning contains a padded sorted degree sequence, the spectral anchor,
and sign-invariant basis descriptors: columnwise sum of U^4 and (U^T 1)^2/n.
These descriptors are invariant to eigenvector column sign flips, but are not
claimed invariant to arbitrary rotations within a repeated-eigenvalue subspace.

Training and validation targets are paired with **independently sampled
same-size eigenbases from the training split**. Training uses four cached
pairings per graph by default; validation uses a fixed independent pairing.
The target is not systematically paired with its own eigenbasis. Otherwise the
basis plus its own degree sequence can make the reconstruction task nearly
trivial. An identity pairing can still occur by chance under independent
empirical sampling.

No validation or test eigenbasis enters the sampling bank. A validation size
absent from the training eigenbasis bank is rejected rather than silently
borrowing a held-out basis. This stage retains the empirical bank's size support.

The default 1,250 spectral epochs, four pairings and batch size 16 correspond to
20,000 updates **when the prepared training split has 64 graphs**. This matches
the old S0 update count under that assumption, not total computation: the
auxiliary heads, pairing cache, and separately trained degree prior add work.

## 2. Degree/basis-informed spectral initialization

At generation, the primary configuration samples a degree sequence from the
existing DH-VAE and independently samples a training eigenbasis of the same
size. Basis rows are aligned by their training-graph degree ranks; target degrees
use descending rank order. This is a rank convention, not an isomorphism solver.

For an adjacency matrix A = U diag(lambda) U^T, define

```
B = U diag(U^T 1)
C = U elementwise_squared
```

The anchor fitter solves

```
min_lambda  0.5 ||B lambda - d||^2
          + 0.5 w_diag ||C lambda||^2
          + 0.5 ridge ||lambda||^2
subject to sum(lambda) = 0 and lambda[i] <= lambda[i+1].
```

It then rescales the full vector to satisfy `sum(lambda^2) = sum(d)` and divides
by sqrt(n). These constraints encode row sums, loop-free diagonal preference,
zero trace and an adjacency second-moment identity. The convex constrained fit
uses SciPy SLSQP. A solver failure uses a deterministic sorted trace-constrained
ridge fallback and is recorded as `solver_success=false`.

**An arbitrary sampled U and learned d need not admit an exact simple-graph
spectrum.** Rescaling can worsen the row-sum/diagonal fit. The anchor is a
heuristic continuous source, not an exact eigenpair or degree guarantee.
Diagnostics report solver success, row-sum RMSE, diagonal RMSE, trace residual,
and second-moment residual.

`seed_top_k: null` uses every anchor mode. Setting an integer retains only the
largest-magnitude fitted modes at their sorted positions; unseeded modes have
zero centre. The sparse anchor need not retain the full vector's trace or
second moment; both full-fit and actual-anchor residuals are recorded.

### Matched forward/reverse process

Let s denote the normalized degree/basis anchor. Training uses

```
x_t = s + sqrt(alpha_bar_t) (x_clean - s) + sqrt(1-alpha_bar_t) epsilon.
```

Generation starts from `s + Gaussian noise` and uses the matching centred DDIM
update. Thus it does **not** insert a structured starting vector into a denoiser
trained only for the old zero-centred process. Some random noise remains for
sampling diversity. The terminal Gaussian uses the usual small-alpha_bar
approximation, which is explicitly recorded in the generation manifest.

`initialization.mode: gaussian` sets s=0 during **both training and generation**.
Changing the mode, ridge, diagonal weight or `seed_top_k` requires retraining;
the generation loader checks this contract.

### Learned-degree provenance

The primary degree sampler uses the existing DH-VAE with learned edge-count
sampling, exact degree-sum conditioning, rejection-only feasibility handling,
and `fallback: error`. It never silently falls back to empirical degree
sequences or changes a rejected sequence through repair.

The degree checkpoint's **full training degree-sequence multiset** must match
the spectral checkpoint's training bank. This catches stale/split-mismatched
priors, including duplicate-multiplicity differences. It does not certify the
identity of entire training graphs. A full training-file hash is additionally
checked when the degree checkpoint contains `config.dataset.train_sha256`;
older checkpoints without that field are marked unverified for that hash.

## 3. Intermediate graph realization and rewiring

With 1,000 DDIM evaluations, the default settings select reverse timesteps
**299, 199, 99 and 0**. `every` counts reverse model evaluations, not original
diffusion indices. Reducing the number of sampling steps should be accompanied
by a smaller `every` to retain intermediate events.

At the first event, the predicted clean spectrum is decoded using the sampled
U. A degree-exact feasible graph is constructed using indexed Havel-Hakimi,
connectivity handling, and up to four degree-preserving swaps fitting the soft
U/lambda adjacency scores. This graph is saved as `initial_graphs.pkl`.

**Havel-Hakimi is a discrete feasibility constructor here, not the eigenvalue
prior.** The spectral process started from the degree/basis-centred Gaussian,
not the spectrum of an HH graph. Thresholding U diag(lambda) U^T alone cannot
guarantee the separately sampled degree sequence.

The discrete graph then persists across the selected reverse steps. Each event
predicts new clean spectral/structural targets and freezes them for that event's
local search. Candidate double-edge swaps are rejected if they introduce loops,
duplicate edges, change indexed degrees, disconnect a connected source, or
revisit an already accepted structural state.

The default objective is

```
E(G) = 0.25 D_lambda(G)
     + 1.00 D_clustering(G)
     + 0.25 D_orbit(G)
     + 0.25 D_graphlet3(G)
     + 0.02 D_source_edges(G).
```

Distances are normalized adjacency-spectrum RMSE, histogram W1, log-orbit RMSE,
graphlet TV, and edge disagreement with the first degree-exact realization.
The aggregate structural distance and total objective must both improve by more
than `min_improvement`. Unlike the previous lambda-only refiner, an individual
spectral term is not required to decrease on every accepted move. Neither are
all three structural terms individually required to decrease.

There are at most two accepted structural swaps per event by default (eight
over the four default events), in addition to the separately logged realization
fit. Candidate proposals and exact size-three summary calculations are bounded;
this implementation is intended first for small generic graphs.

### Feedback into the spectral trajectory

After an accepted intermediate structural move, the next DDIM step uses

```
x_clean_guided = (1-rho) x_clean_pred + rho lambda_normalized(G_current),
rho = 0.05 by default.
```

This makes rewiring affect subsequent spectral states, rather than merely
post-processing the last output. `feedback_only_after_accept: true` prevents
feedback when no structural move was accepted. `spectrum_feedback: 0` retains
intermediate graph refinement without coupling it back into DDIM.

The final graph is the persistent degree-exact graph after the last event; it
is not overwritten by a final unconstrained threshold reconstruction.

### Reconciliation of predicted summaries

For degrees d, let W = sum_i choose(d_i,2), P be the induced-P3 count, and T the
triangle count. Necessary identities include

```
P + 3T = W
mean_P3_endpoint = 2P/n
mean_P3_centre = P/n
mean_triangle_vertex = 3T/n.
```

Before refinement, the raw orbit/graphlet predictions are reconciled using
these identities. Mean edge-endpoint count is fixed to mean(d), and the
clustering histogram reserves enough mass at zero for nodes of degree 0 or 1.
Both raw and reconciled predictions are saved.

This projection enforces necessary identities only. It does not prove a graph
exists with all the predicted summaries; fractional expected counts are allowed.
At size three, orbit means and graphlet counts are partly redundant once degrees
are fixed. They are not three independent structural information sources.
Clustering histograms add information about how triangles are distributed.

Targets change across events. Monotonic improvement is guaranteed by acceptance
checks **within a frozen-target event**, not across all events, and does not
imply held-out MMD improvement. Regression heads can also shrink toward
conditional-mean summaries; distributional quality must be measured separately.

## 4. Community-small commands

Use the existing prepared split; do not regenerate it between runs. Run from
the project root.

### Train the degree prior

```bash
PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/dhvae/community_small.yaml
```

That config selects seed 42, device `auto`, and writes
`outputs/degree_generators/sbm/seed_42/checkpoint.pt`. A compatible previously
trained prior may be reused, but generation verifies its degree multiset.
The spectral trainer itself uses true training degrees and does not need the
learned prior file until generation.

### Train Structure3

```bash
CFG=configs/baselines/gdsm_simple_community_small_structure3.yaml
RUN=seed_42_structure3_degree_basis

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage train --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config "$CFG" \
  --seed-id 42 --run-id "$RUN" --device gpu
```

### Generate and evaluate

```bash
N=1024

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage generate --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config "$CFG" \
  --seed-id 42 --run-id "$RUN" --num-samples "$N" --device gpu

GEN="outputs/baselines/gdsm_simple/community_small/$RUN/generations/seed_42_n_${N}"

PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN" \
  --generated-stage gdsm_simple_structure3 \
  --base-graphs "$GEN/initial_graphs.pkl" \
  --reference-split test \
  --output-dir "$GEN/evaluation_test"
```

The source row is the first degree-exact realization, not an S0 sample. Use
validation-only evaluations for tuning; reserve test comparisons for fixed
choices. Existing ORCA requirements for the common evaluator remain unchanged.

A convenience runner is also included:

```bash
bash scripts/run_gdsm_simple_structure3_community_small.sh all
```

It supports stages `degree`, `train`, `generate`, `evaluate`, or `all`, and
`CFG`, `DEGREE_CFG`, `RUN`, `SEED`, `N`, `DEVICE`, `GEN_ID` environment variables.
The separate degree trainer reads its seed/device/checkpoint from its YAML;
changing the runner variables alone does not rewrite that YAML.

## 5. Ablations and saved artifacts

| Configuration suffix after `..._structure3` | Change | New spectral training? |
|---|---|---|
| none | Learned degrees + degree/basis anchor + structure-guided generation | Yes, initially |
| `_empirical_degree` | Sample degrees from the training bank; no learned-prior file needed | No |
| `_gaussian` | Matched zero-centred training and generation; other new components remain | Yes |
| `_no_structure_guidance` | Freeze the first degree-exact realization; no structural swaps or feedback | No |
| `_no_feedback` | Structural graph refinement remains, but DDIM is not modified by it | No |

The Gaussian ablation is **not S0**: it still has degree/basis conditioning,
summary heads and degree-exact realization. Use the existing S0 config/run for
an actual S0 comparison. The no-structure-guidance ablation retains initial
feasibility construction and its spectral-score fitting swaps.

Use distinct generation IDs for generation-only ablations to avoid collisions,
for example `--generation-id structure3_no_feedback_seed_42_n_1024`. Keep seeds,
sample counts and batch sizes fixed for paired comparisons.

Generation writes:

- `base_graphs.pkl`: final persistent degree-exact graphs, consumed by the evaluator.
- `initial_graphs.pkl`: graphs immediately before the first structural event.
- `threshold_graphs.pkl`: final U/lambda threshold decode of **this chain**, diagnostic only.
- `sampled_degree_sequences.pkl`, `sampled_basis_indices.pkl`, `initial_eigenvalue_anchors.pkl`.
- `target_adjacency_eigenvalues.pkl` and `predicted_structure_summaries.pkl` (raw and reconciled).
- `rewiring_diagnostics.json`: per-event predictions, errors, accepted moves, feedback and invariants.
- `manifest.json`: trained checkpoint identity, degree-prior provenance, schema and hashes.
- `intermediate_graphs.pkl` when `save_intermediate_graphs: true`.

`threshold_graphs.pkl` is not an independently generated original-S0 control:
conditioning and potentially feedback have already changed the chain.

## 6. Validation and implementation files

Validation is recorded in `docs/validation/gdsm_structure3/full_suite.log`.
The packaged revision passed **196 tests** on CPU, including the prior tests
and 22 new tests. The new coverage includes all graph-atlas examples up to five
nodes against brute-force size-three counts, moment/sign checks, centred DDIM
algebra, gradients for every head, oracle-target swap acceptance, no-move cases,
managed train/generate round trips, deterministic replay, stale degree-bank
rejection, a small genuinely trained DH-VAE integration, and a feedback-trajectory
check. There is one PyTorch nested-tensor prototype warning.

These are implementation and small synthetic integration checks, not a trained
Community-small quality benchmark. No claim of improved MMD, GPU speed, or
learned-prior quality is made. The archive contains code/configs/tests, not
new benchmark-trained weights or new 1,024-graph evaluation results.

Main additions are `structure3.py`, `degree_initialization.py`,
`structured_model.py`, and `structured_pipeline.py` under
`src/grapher/models/gdsm_simple/`. The original wrapper dispatches to the new
pipeline only when the structured extension is explicitly selected.
