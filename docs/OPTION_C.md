# Option C: categorical nodes and continuous weighted-adjacency diffusion

> **Schema v2 available.** The soft-threshold degree + normalized-Laplacian
> consistency variant is documented in `docs/OPTION_C_SOFT_CONSISTENCY.md` and
> configured under `configs/experiments/option_c_soft_consistency/`. The original
> schema-v1 configs in this document remain supported for matched comparison.


This is a new implementation for the uploaded `graphes(5).zip` codebase. It is
not a switch inside the simple-GDSM wrapper and is not an implementation of
HoG-Diff. Existing source files, runners, dataset configurations and baseline
profiles are left unchanged.

## Entry points and files

- Train: `scripts/train_option_c.py`.
- Generate: `scripts/generate_option_c.py`.
- Implementation: `src/grapher/models/option_c/`.
- Complete configs: `configs/experiments/option_c/{community_small,ego_small,qm9,zinc}.yaml`.
- Explicit commands: `docs/OPTION_C_COMMANDS.sh` and `docs/OPTION_C_EVALUATION.sh`.
- New tests: `tests/test_option_c.py`; validation logs: `docs/option_c_validation/`.

Use the existing GraphES Python environment and run from the repository root
with `PYTHONPATH=src`. No external baseline checkout or degree-VAE checkpoint
is needed. Model/optimizer checkpoints contain state dictionaries, tensors and
primitive metadata. Newer PyTorch uses restricted `weights_only=True` loading;
legacy PyTorch versions that do not expose that argument use the compatibility
path for trusted local checkpoints. Prepared NetworkX
pickles and the local preprocessing cache remain trusted, code-executing pickle
artifacts: do not load untrusted dataset/cache files.

## What is retained, replaced and removed

The empirical node marginal is **the terminal categorical noise distribution**,
not a fixed final atom composition. The existing `MarginalNoise` transition and
posterior helper are reused. The node denoiser still learns clean atom-category
probabilities from the noisy graph. Graph sizes are sampled from the empirical
TRAINING size histogram stored in the checkpoint; no validation sizes or node
counts enter this prior.

The evolving graph state is `(X_t, W_t, mask, t)`: categorical nodes and a
continuous symmetric weighted adjacency. There is no categorical edge state,
edge softmax, independent spectral latent, spectral transformer, eigenbasis
reservoir, spectral anchor, or degree-VAE prior. Clean edge-category indexes are
used only to encode data, count typed graphlets, and decode/refine final graphs.
They are not a second diffusion state.

For generic graphs, physical edge weights are `{0,1}` and `scale=1`. For the
molecular profiles they are `{0,1,2,3}` and `scale=3`. Internally,
`w_0 = physical_weighted_adjacency / scale`. The mapping is keyed by actual
edge attribute VALUE, not by its categorical tensor index. Symmetry, absent
self-loops and padding are enforced in every forward/reverse state.

### Forward corruption

At a uniformly sampled integer timestep `t` in `1,...,T`, independently corrupt
nodes and weights conditional on the same clean graph:

```
Qbar_X(t) = alpha_bar[t] I + (1 - alpha_bar[t]) 1 m_X^T
w_t = sqrt(alpha_bar[t]) w_0 + sqrt(1-alpha_bar[t]) epsilon
```

The cumulative cosine schedule has exact endpoints `alpha_bar[0]=1` and
`alpha_bar[T]=0`. `m_X` is fitted only on training nodes with pseudocount 0.001.
Each unordered pair gets one independent standard-normal noise draw, mirrored
across the diagonal. Averaging two independent matrix noises would incorrectly
halve the variance and is not done. Intermediate weighted matrices can be
negative and are not thresholded into graphs.

### Joint denoiser

A shared node/pair network has four node-pair blocks in the supplied profiles.
Each block uses symmetric pair updates, masked pair-to-node messages, global
pooling, node self-attention without absolute node-index embeddings, and
feed-forward updates. Features include categorical noisy nodes, continuous
noisy weights, weighted row statistics, graph size and time.

The node head produces clean-category logits. A scalar adjacency-regression head
uses pair embeddings and symmetric combinations of soft predicted endpoint node
probabilities. It returns an unbounded real clean matrix, symmetrized and masked.
There is **no edge-category probability head**.

Clean structural-summary heads receive pooled shared node/pair features, soft
node probabilities and differentiable statistics of the predicted adjacency.
They predict separate connected-induced graphlet compositions and masses for
orders 3, 4 and 5, plus the existing untyped clustering histogram and four
log-mean orbit coordinates. Summary losses can therefore train the shared
network and the differentiable node/adjacency outputs as well as their own heads.

### Objective and spectral supervision

The objective is

```
L = 1.0 L_node + 1.0 L_adjacency + 1.0 L_spectral
  + 0.5 L_graphlet + 0.5 L_mass + 0.5 L_clustering + 0.25 L_orbit.
```

Node loss is clean-category cross entropy. Adjacency loss is clean weighted
matrix MSE over each graph's active upper triangle. These losses are normalized
per graph, then averaged. Absent pairs are included; there is no hidden
negative-pair sampling or class reweighting. A one-node graph contributes zero
adjacency loss.

Spectral supervision is computed **from the predicted weighted adjacency**:

```
l_hat = eigvalsh(w_hat_0) / sqrt(n)
l_0   = eigvalsh(w_0)     / sqrt(n)
L_spectral = mean_graph [ mean_rank (l_hat - l_0)^2 ].
```

There is no separate learned `l_hat` head. The gradient passes through
`eigvalsh` to the adjacency head and shared features. Active submatrices are
grouped by node count before decomposition; padded zero eigenvalues must not
be mixed into the sorted active spectrum. The default solver uses CPU float64,
including a differentiable device transfer when neural training is on CUDA.
`spectral.solver_device: model` explicitly runs it on the network's device.
No eigenvectors, numerical diagonal jitter, detach, or straight-through
thresholding enter this loss. Set the spectral loss weight to zero, and its
refinement weight to zero as well, for a retrained spectral-supervision ablation;
then the spectral training eigensolve is skipped.

Sorted eigenvalue losses can be nonsmooth at ties; float64 and using eigenvalues
rather than eigenvector derivatives do not imply universal optimization or
convergence guarantees. Spectral/structural auxiliary losses also change the
pure denoising objective, so this implementation does not claim exact likelihood
or an exact data-distribution reverse process for the combined objective.

### Graphlet targets and preprocessing cache

Targets are computed on CLEAN training/validation graphs, never by thresholding
a noisy matrix. Only training graphs define the typed vocabularies. Each order
has its own normalization, overflow class and connected-subset mass. Orders
larger than a graph are masked; a zero connected-subset mass has no fabricated
histogram target. The reused graphlet engine counts connected induced patterns
exactly, including non-cyclic patterns, and uses exact local deltas in refinement.
The explicit one-million-connected-subset guard raises rather than truncating
or switching to Monte Carlo.

The first seed preprocesses the full training/validation splits; later seeds
reuse `outputs/cache/option_c/<benchmark>/<hash>/`. The cache key includes split
hashes, source dataset metadata, representation/target settings and preprocessing
code hashes. Counts are stored in the existing sparse/indexed representation.
Initial exact graphlet counting can take substantial time on the molecular
splits; it is not repeated every epoch or denoising step.

### Reverse generation

Initialize `X_T ~ Cat(m_X)` and a symmetric standard-Gaussian `w_T`. At each
reverse step the same network predicts clean nodes and clean weights. Nodes
are sampled from the existing exact arbitrary-step categorical posterior mixture.
At the final step they are sampled from the learned clean distribution, **not**
from the empirical marginal and not by argmax.

The default weighted sampler is DDPM with clean-matrix prediction and the
analytic Gaussian posterior mean/variance; skipped steps use the corresponding
cumulative-schedule posterior, not repeated one-step coefficients. Deterministic
DDIM is also available. Both handle the exact zero-signal terminal endpoint
without division by `sqrt(alpha_bar[T])`.

Clean estimates are clipped to the physical palette's range by default for
sampling only. Noisy states are not clipped. After the LAST reverse step, undo
the scaling and quantize by midpoint thresholds. For molecules the thresholds
are `0.5, 1.5, 2.5`; for binary graphs it is `0.5`. A tie selects the larger
physical weight. This directly defines the final edge labels.

### Final-only graphlet refinement

The supplied configs enable at most two accepted same-type swaps AFTER final
thresholding. Targets from the last denoiser call are held fixed. The energy
uses graphlet composition/mass, clustering and orbit discrepancies; optional
spectral discrepancy uses the weighted adjacency; the old edge NLL is replaced
by weighted-adjacency MSE. There are no fabricated edge probabilities.

A swap must lower the configured total and structural energies. Typed degrees,
node labels and simplicity are preserved within this final event; connectedness
is retained only if the event started connected and the configured check is on.
These are not full-trajectory degree or molecular-validity guarantees. Nothing
is projected back into a continuous diffusion state and no intermediate
refinement/feedback is implemented.

Refinement has a separate random generator. At identical checkpoint, neural
seed, step count and batch size, the guided run's `pre_rewire_graphs.pkl` is
exactly the corresponding no-refinement output. This equality is tested. It is
specific to this FINAL-ONLY implementation; it did not hold for the old
interleaved sampler.

## Dataset and experiment settings

The existing dataset YAMLs and prepared splits are reused without changes.
Training fails on missing splits, mismatched known protocol IDs, wrong declared
split sizes, unknown categories, self-loops or oversized graphs. It never
prepares/downloads data or silently filters it. If the prepared dataset's
resolved config is absent, it warns and still records split hashes; in that
case protocol provenance cannot be independently established from the snapshot.

| Config | Prepared directory | Max nodes | Epochs | Train batch | Outputs/seed |
|---|---|---:|---:|---:|---:|
| `community_small.yaml` | `outputs/datasets/sbm` | 20 | 10,000 | 32 | 1,024 |
| `ego_small.yaml` | `outputs/datasets/ego_small` | 18 | 5,000 | 32 | 1,024 |
| `qm9.yaml` | `outputs/datasets/qm9_attributed` | 9 | 200 | 64 | 10,000 |
| `zinc.yaml` | `outputs/datasets/zinc` | 38 | 200 | 32 | 10,000 |

Epochs, train batches and validation intervals are copied from the uploaded
`configs/experiments/grapher_research/<dataset>_g345.yaml` profiles. In particular,
Ego-small is 5,000 epochs in that profile, although the common baseline exposure
file declares 10,000. This new runner does not merge either old wrapper options
or common baseline defaults. The supplied YAML is authoritative and all effective
overrides are saved. To run 10,000 Ego-small epochs instead, explicitly use
`--epochs 10000` or edit its Option-C YAML.

Shared settings: independent seeds 42/43/44; full prepared training split;
500 forward steps and 500 reverse steps; hidden width 128, four attention heads,
feed-forward width 256; AdamW learning rate 2e-4, weight decay 1e-6, gradient clip
1.0; no dropout or EMA by default. Graphlet order weights are equal. Molecular
vocabulary caps per order are 8,192 for QM9 and 16,384 for ZINC. Validation uses
fixed corruption draws (seed 123456), never test metrics. Optional EMA is
implemented, but is disabled in the starting profiles.

These are explicit starting profiles, not tuned performance claims or a claim
of parameter-count matching with the previous spectral-categorical architecture.
Option C requires **new checkpoints**; old GDSM/GraphER checkpoints are rejected.

## Artifacts and resuming

Training output:

```
train/
  checkpoints/best.pt       # minimum fixed-noise validation joint loss
  checkpoints/last.pt       # model + optimizer + RNG for epoch-boundary resume
  manifest.json            # status, split/code hashes, config, versions, steps
  schema.json              # node/size priors and typed graphlet vocabulary
  resolved_config.yaml
  history.jsonl
  training_metrics.json
```

`--resume` restores the last SAVED epoch, including optimizer, corruption RNG,
shuffle RNG and torch RNG. It checks seed, training semantics and data/cache
identity. It permits extending epochs but not silently changing architecture,
representation, graphlet targets or optimization settings. Unsaved epochs after
an interruption are rerun. Exact resume is tested on CPU; bitwise identity across
hardware, different CUDA kernels or dependency versions is not claimed.

Generation output:

```
generation/
  base_graphs.pkl               # FINAL generated graphs, existing evaluator interface
  molecular_graphs.pkl          # same final batch, molecular profiles only
  pre_rewire_graphs.pkl         # raw thresholded batch before optional final swaps
  continuous_adjacencies.pt     # physical matrices before thresholding + unclipped estimates
  diagnostics.json             # per-graph swaps/invariants, connectedness, timing
  resolved_config.yaml
  manifest.json
```

Continuous matrices correspond to PRE-rewiring graphs, not the refined graph.
All requested outputs are retained, including disconnected/chemically invalid
graphs. No valence repair, invalid-output rejection or resample-until-valid policy
is introduced. Use the existing molecular evaluator to assess validity and
quality; the generator does not report uncomputed FCD/NSPDK/validity numbers.

Default generation uses the checkpoint's stored priors/schema and needs no
training pickle or dataset access. Protected overwrite refuses an unrelated
output directory. Explicit `--device gpu` raises when CUDA is unavailable rather
than silently performing a long CPU run.

## Commands, evaluation and validation

All training/generation commands are in `OPTION_C_COMMANDS.sh`. Existing
prepared data is a prerequisite. There is intentionally no multi-stage baseline
runner and no automatic retraining/generation inside either Python entry point.

`OPTION_C_EVALUATION.sh` uses the existing generic report for Community-small and
Ego-small and the existing molecule evaluator for QM9/ZINC. Molecular population
is explicitly `raw_valid`, following `grapher_research/protocol.yaml`; it does not
silently enable HoG-Diff's corrected-molecule metric convention. FCD/EDeN remain
external evaluator dependencies. Exact graphlet TRAINING targets do not change
the shared evaluator's own metrics, kernels or sampling settings.

Run the new tests:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest tests/test_option_c.py -q
```

Validation performed for this delivery: 42 new Option-C tests pass. Together
with the three existing categorical/noise/graphlet regression files, 199 tests
pass and 5 CUDA-dependent tests skip because this environment has no real CUDA.
Checks cover covariance/symmetry, categorical and Gaussian posterior formulas,
threshold mappings, active-node spectral padding, spectral gradient checking,
shared-head gradients, equivariance, train-only vocabularies/cache invalidation,
all-four-profile fixture training/generation, max-node/full-width model passes,
exact CPU resume, EMA/DDIM, real CLI entry points and oracle refinement descent.

The uploaded archive has no prepared dataset splits. Full dataset training,
10,000-molecule generation, benchmark FCD/MMD and GPU throughput were NOT run.
Small-fixture tests demonstrate the implementation path, not generation quality.
