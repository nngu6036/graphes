# GDSM spectral–categorical attributed generation

Implemented against `graphes(20260917-091227).zip`, 17 September 2026.
This is a new opt-in jointly trained variant. Existing S0, S1, Structure3 and
Option-A files/configurations are retained; their checkpoints are not converted.
The new categorical variant actually recomputes eigenvectors from its current
categorical adjacency. This statement does not change the legacy fixed-basis
Option-A implementation still present in the uploaded archive.

## Implemented state and transitions

Use a node category tensor X and symmetric edge category tensor E. E=0 is no
edge; positive categories are actual edge labels. Binary topology is A=(E>0).
Only E determines the graph: there is no second threshold-decoded adjacency,
initial-degree mask, HH graph, edge-count projection, or largest-component repair.

Both node and edge corruption use ordinary marginal categorical noise,
Q(a)=a I+(1-a) 1 m^T. Training-only category counts define m, with an explicitly
recorded positive pseudocount (0.001 per category). Count all unordered pairs,
including absent edges, excluding diagonals and padding. Sample each unordered
pair once and mirror it. The forward process can insert/delete/recolor edges;
node categories can change. No categorical label is assigned an artificial
numeric edge weight in the eigendecomposition.

Index time as 0=clean and T=fully noisy. This variant uses a cosine cumulative
schedule with alpha_bar[0]=1 and alpha_bar[T]=0 exactly. Consequently the finite-T
categorical terminal distribution is exactly the fitted marginal. The last
forward transition is a full marginal redraw. The old linear-beta schedule and
hard threshold settings do not control this variant. This is an intentional new
training/sampling contract, not a drop-in change to an old checkpoint.

The denoiser predicts clean category probabilities. Sampling uses
sum_k p_theta(y0=k | state_t) q(y_s=j | y_t=i,y0=k), for any selected s<t,
including exact skipped-step posteriors. It does not repeatedly take the argmax
of clean predictions. `noise.py` implements this mixture in O(K) per categorical
variable and tests it against enumeration of the full posterior table.

A separate continuous latent z follows source-centred spectral corruption:

    z_t = s + sqrt(alpha_bar_t) (z_clean-s) + sqrt(1-alpha_bar_t) epsilon

where z_clean is the sorted binary-adjacency spectrum divided by sqrt(n).
The new spectral branch directly predicts z_clean (x0 parameterization), not
epsilon, so the exact zero-SNR endpoint causes no x0-from-epsilon division by
zero. A source-centred deterministic DDIM update advances the spectral state
at the matching next noise level. z_t is not assumed equal to eig(A(E_t)).

## Degree prior and initial eigenvectors

For the main configurations, a separately trained ordinary-degree DH-VAE samples
d_prior. A same-size training eigenbasis is sampled independently and used in a
ridge fit for a spectral anchor. The fit penalizes degree row-sum and diagonal
residuals, imposes zero trace, then sorts and normalizes the second moment to
sum(d_prior). These operations can increase residuals; the residuals are saved.
The anchor is not claimed to exactly realize that graph or degree sequence.

The original degree vector is not a separate recurrent neural input and is not
an invariant. The derived anchor remains the spectral coordinate offset and an
input to the spectral branch. Thus this is a soft degree-informed spectral prior,
not deletion of all prior information after the first random draw.

X_T and E_T start independently from their respective marginal distributions;
they are not forced to realize d_prior. The current basis for decoding comes from
E_T, and subsequently from each retained E_s, not from the anchor's sampled basis.
The latter is used only in constructing the initial spectral anchor.

During training, anchors are built from clean training degrees and independently
sampled training eigenbases. Generation uses learned degrees; this substitutes a
learned approximation for the training degree marginal. The prior must be assessed
separately. Gaussian-anchor and empirical-degree alternatives are explicit configs.

The training eigenbasis bank uses a per-size reservoir of at most 32 matrices.
All training degree sequences remain available for provenance checks. An unseen
size uses a recorded random orthogonal basis for the anchor only; validation never
supplies its clean target eigenvectors to the model. No dataset splits are rebuilt.

## Joint network and supervision

The categorical graph encoder and spectral-rank transformer are distinct. Node i
is not eigenvalue token i. A dense symmetric pair encoder supplies node messages,
including no-edge pair features. Pooled graph context conditions the spectral
transformer. Its predicted clean spectrum supplies pair scores through the current
basis; spectral context also conditions the graph encoder. Soft endpoint node
probabilities condition the symmetric edge head, rather than clean teacher labels.

Outputs and default loss weights:

| Output | Target | Weight |
|---|---|---:|
| Clean spectrum | Sorted eig(A)/sqrt(n), MSE | 1.0 |
| Node categories | Masked cross-entropy | 1.0 |
| All-pair edge categories | Upper-triangle cross-entropy, including no edge | 1.0 |
| Typed size-3 graphlets | Conditional histogram KL | 0.5 |
| Connected-triple mass | Squared error | 0.5 |
| Untyped clustering | Histogram KL + CDF squared error | 0.5 |
| Untyped orbit means | log1p mean-count squared error | 0.25 |

Each task is normalized separately. Padding, self-pairs and duplicate lower-triangle
pairs do not contribute to categorical CE. The orbit head has four coordinates:
edge endpoint, induced 3-path endpoint, induced 3-path centre, triangle vertex.
It is not a full ORCA size-at-most-4 predictor. Existing external Orbit MMD is unchanged.

All noisy categorical states and clean targets refer to the same indexed training
graph. Training applies a joint random node permutation to X and E; rank-indexed
spectral vectors are not permuted as node labels. There is no rewiring in training.
Validation noise draws are fixed for reproducible checkpoint selection. The lowest
validation joint loss selects the checkpoint; test graphs are not read during
training (their serialized file is hashed as part of the common split fingerprint).

## Typed graphlets

Count exactly connected induced triples: paths AND triangles, including node
categories and all three edge states in the canonical key. Jointly permute the
three node and edge labels over all six permutations. Edge category zero is
included in the key, distinguishing the missing pair of a path. Arbitrary
integer/string category labels are mapped to stored categorical indices.

Only training graphs define the output vocabulary, plus one overflow coordinate.
Predict the conditional composition histogram and the independent mass
number_of_connected_triples / choose(n,3). Empty-triple cases have zero mass and
masked conditional-histogram loss. No approximation or subset Monte Carlo is used
for k=3. Existing untyped path/triangle reconciliation formulas are NOT applied
to typed class coordinates or to the originally sampled degrees.

A generated unseen typed class contributes to overflow rather than being dropped.
The extra evaluation script instead uses the union of reference/generated typed
classes, with no overflow collapse. It never changes the checkpoint vocabulary.

## Dynamic eigenvectors and feedback

After every categorical posterior draw and any local swaps, torch.linalg.eigh is
applied to the retained binary adjacency. Values and columns stay ordered together.
The next network evaluation receives these eigenpairs; tests check reconstruction
inside each next call, not just in the final saved artifacts.

Equal current eigenvalues admit arbitrary rotations of their eigenvectors. Before
forming U diag(predicted_values) U^T, the implementation averages the predicted
coefficients within numerically equal current-eigenvalue blocks (relative tolerance
1e-6). This makes the score matrix invariant to signs and within-block rotations,
including isolates and the fully edgeless graph. It also restricts spectral
expressivity in those tied subspaces; the categorical graph branch can still
break symmetry stochastically. Raw eigenvector columns are not node features.
Permutation, sign and rotation tests cover this choice.

The retained graph's sorted spectrum can softly blend with the predicted clean
spectrum before the next noise-consistent spectral update. Default strength is
0.05 in the last 20% of time. Eigenbasis recomputation occurs regardless of whether
feedback is enabled or a swap is accepted. Category-only changes are carried in
X/E to the next encoder even when binary eigenvalues stay unchanged.

## Generation-only local guidance

With the full 500-step supplied schedule, posterior sampling and basis refresh
occur 500 times; local guidance events are at destination indices s=100,50,0.
At each event at most 2 same-edge-type double swaps are accepted, selected among
128 proposals / 64 valid candidates by a weighted typed-graphlet/mass/clustering/
orbit/spectral/category-probability objective. The clean prediction is fixed while
that event's candidates are compared. Accepted swaps must improve both the
weighted total and structural part. No permanent rejected-edge table is retained.

Swaps preserve only the current indexed degrees, current node categories and
current per-node typed degrees. The next categorical transition can change all
of these. Swaps preserve connectivity only when that event starts connected;
no global connectivity, chemical-valence, planarity or other ConStruct guarantee
is imposed. The current move set cannot recolor nodes/edges during refinement;
those changes are the responsibility of categorical transitions. Rewiring and
spectral feedback are sampling interventions, not an exact-likelihood guarantee.

## Installation

From the root of the uploaded project:

```bash
unzip -o graphes_gdsm_categorical_changes_20260917.zip -d .
```

For locally edited files, review `graphes_gdsm_categorical_20260917.patch`, then
use `git apply --check` and `git apply`. The patch is against the 09:12:27 upload,
not an earlier project snapshot. No dataset or trained model is shipped. Python
bytecode/cache files are omitted from the full project ZIP.

## QM9: training, generation, audit, evaluation

Prepared splits must already exist at outputs/datasets/qm9_attributed. Categories
match the uploaded project: atomic_num in [6,7,8,9], bond_type in [1,2,3], n<=9.
This version predicts these two channels only; it does not additionally generate
formal charges, hydrogen counts, aromatic flags, or stereochemistry.

```bash
# Preflight does not rebuild data or read test graph contents.
bash scripts/run_gdsm_categorical.sh qm9 check

# New ordinary-degree prior: NOT qm9_typed/checkpoint.pt.
PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/dhvae/qm9_categorical_spectral_prior.yaml

CFG=configs/baselines/gdsm_simple_qm9_categorical.yaml
RUN=seed_42_spectral_categorical
N=1024

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage train --dataset qm9 --no-common-config \
  --wrapper-config "$CFG" --seed-id 42 --run-id "$RUN" --device cuda:0

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage generate --dataset qm9 --no-common-config \
  --wrapper-config "$CFG" --seed-id 42 --generation-seed 42 \
  --run-id "$RUN" --num-samples "$N" --device cuda:0

GEN="outputs/baselines/gdsm_simple/qm9/${RUN}/generations/seed_42_n_${N}"
PYTHONPATH=src python scripts/audit_gdsm_categorical.py --generated-dir "$GEN"

# New, separately named typed/categorical RBF-MMD^2 diagnostics on RAW graphs.
PYTHONPATH=src python scripts/evaluate_gdsm_categorical.py \
  --generated-dir "$GEN" \
  --reference-graphs outputs/datasets/qm9_attributed/test.pkl

# Existing molecular benchmark evaluator; no change to its implementation.
PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-graphs "$GEN/molecular_graphs.pkl" \
  --dataset qm9_attributed --reference-split test --train-split train \
  --metric-molecule-source raw_valid --require-fcd \
  --output-dir "$GEN/evaluation_molecules"
```

The new prior writes
`outputs/degree_generators/qm9_categorical_prior/seed_42/checkpoint.pt`.
Its training device is configured as auto in the prior YAML; the prior CLI has
no --device flag. A compatible existing **ordinary-degree** prior can be reused
by setting its path before spectral-categorical training. Generation verifies the
exact training degree multiset and any recorded training SHA. It fails rather
than silently falling back to empirical degrees or repairing sampled degrees.

`--no-common-config` makes the new experimental budget explicit. Main QM9/ZINC
configs use 200 joint epochs, batch 64/32, hidden size128, 3 layers per branch,
4 spectral heads, 500 noise steps/500 reverse transitions. Degree-prior training
has its own 300-epoch budget. These are starting settings, not tuned results.
Select quality/budget on validation, not the held-out test set.

## ZINC and stage runner

ZINC uses the uploaded prepared split at outputs/datasets/zinc, max38 nodes,
atomic_num [6,7,8,9,15,16,17,35,53], bond_type [1,2,3]. Unsupported categories
raise an error rather than being relabeled. Change the explicit schema and retrain
when a different dataset representation is used.

```bash
# all = preflight, prior-if-absent, joint training, generation, audit, evaluation
DEVICE=cuda:0 N=1024 bash scripts/run_gdsm_categorical.sh qm9 all
DEVICE=cuda:0 N=1024 bash scripts/run_gdsm_categorical.sh zinc all

# Individual stages do exactly what their names state.
DEVICE=cuda:0 bash scripts/run_gdsm_categorical.sh zinc train
DEVICE=cuda:0 N=1024 bash scripts/run_gdsm_categorical.sh zinc generate
N=1024 bash scripts/run_gdsm_categorical.sh zinc audit
N=1024 bash scripts/run_gdsm_categorical.sh zinc evaluate
```

A present prior is reused, but provenance is still verified at generation.
The runner never passes --overwrite. Changing the spectral SEED does not silently
change the seed42 prior configs. FCD and EDeN NSPDK require their existing optional
project dependencies; --require-fcd deliberately fails when FCD is unavailable.
The existing evaluator may also report corrected-valid diagnostics, but the
commands above compute headline validity and distribution metrics from raw-valid
molecules. Generator outputs themselves are not sanitized, repaired or filtered.

## Generic categories and topology ablations

Use `--dataset attributed --no-common-config` for a non-molecular graph dataset.
Place trusted NetworkX lists at outputs/datasets/attributed/{train,val,test}.pkl.
Set node_attribute, edge_attribute and max_nodes in
`configs/baselines/gdsm_simple_attributed_categorical.yaml`; the default names are
node_type and edge_type, max_nodes64. Node/edge labels are scalar categorical
values. This generic template explicitly uses empirical training degrees until a
matching ordinary-degree DH-VAE is supplied. It is not labeled as a learned prior.

```bash
DEVICE=cuda:0 N=1024 bash scripts/run_gdsm_categorical.sh attributed all
```

The CLI also supports --dataset-root and --serialized-dataset without pretending
custom graphs are QM9. The runner accepts DATASET_ROOT, SERIALIZED_DATASET,
OUTPUT_ROOT, CFG, RUN, GID, SEED, GEN_SEED, DEVICE and N environment overrides.
For custom data, train a matching custom prior explicitly rather than allowing
the runner to select an unrelated benchmark's prior.

Community-small and Ego-small categorical configurations use a single dummy node
type and a single positive edge type. They allow a like-for-like topology ablation
of the categorical transition versus the existing eigenvalue-only models. They
reuse the existing matching ordinary-degree priors. Their 1000-epoch starting
budgets are not claimed to be update-matched to all older runs. Their runner's
evaluate stage invokes both the additional typed metrics and the unchanged common
graph evaluator (ORCA must be available for the latter).

## Controlled ablations

QM9 and ZINC include these complete YAMLs:

- `_categorical_no_guidance.yaml`: same trained architecture/noise/prior, no swaps;
  generation-only ablation under the same RUN and different generation ID.
- `_categorical_empirical_prior.yaml`: new training run, explicitly empirical
  initialization instead of a learned prior.
- `_categorical_categorical_only.yaml`: new training run, Gaussian anchor, no
  spectral-to-category/summary conditioning, no spectral loss, feedback or swaps.
  The current implementation still allocates/computes the spectral branch; it is
  a predictive-dependence ablation, NOT an optimized runtime categorical baseline.

```bash
VARIANT=no_guidance RUN=seed_42_spectral_categorical DEVICE=cuda:0 N=1024 \
  bash scripts/run_gdsm_categorical.sh qm9 generate

VARIANT=no_guidance RUN=seed_42_spectral_categorical N=1024 \
  bash scripts/run_gdsm_categorical.sh qm9 evaluate

# These defaults use separate run IDs automatically.
VARIANT=empirical_prior DEVICE=cuda:0 N=1024 \
  bash scripts/run_gdsm_categorical.sh qm9 all
VARIANT=categorical_only DEVICE=cuda:0 N=1024 \
  bash scripts/run_gdsm_categorical.sh qm9 all
```

Disable only `spectrum_feedback` for an additional runtime ablation. Disabling it
still updates the basis and passes the categorical graph to the next model call.
The checkpoint rejects runtime changes to the trained noise, vocabulary, anchor
construction, loss contract and spectral-conditioning mode. Faster sampling may
reduce sample.steps below diffusion.steps; categorical posteriors use the exact
s-to-t cumulative kernel, not the wrong adjacent-step transition.

## Artifacts and evaluation interpretation

`initial_graphs.pkl` is the terminal marginal categorical graph, not an HH graph.
`base_graphs.pkl` is the final attributed graph; `molecular_graphs.pkl` is the same
batch under the molecular evaluator alias. `final_pre_rewire_graphs.pkl` is the
last posterior sample before the last swaps. The latter and final graph have equal
current degrees/typed degrees by design, but initial and final need not match.
These are trajectory-stage comparisons, not independent no-guidance baselines.

Other saved artifacts: sampled/final degrees, anchors, final normalized eigenvalues
and eigenvectors, final model predictions, category mappings/marginals, graphlet
vocabulary, per-event rewiring energies, degree/category-change counters, and
checkpoint/data/output hashes. Set save_trajectory=true for full retained X/E
states at every selected reverse time; this can produce large files.

`audit_gdsm_categorical.py` verifies actual file hashes, simple graph/category
representation, node count, final event-local invariants, eigenpair reconstruction
and orthogonality. It reports, rather than requires, initial/prior degree changes.

`evaluate_gdsm_categorical.py` adds biased RBF MMD squared (Euclidean features,
sigma1) for node histograms, edge-category histograms including no edge, connected
triple mass, and typed connected triple class mass augmented with disconnected
mass. It evaluates raw graphs and uses fully resolved external typed classes.
These are NOT replacements for GraphRNN/ORCA, FCD or NSPDK. By default every saved
generated and reference graph is used; --max-reference explicitly enables seeded
reference subsampling. Pairwise kernel calculation is blocked for memory, not
subsampled silently. No training/validation vocabulary leakage is introduced.

## Validation and limitations

Run `PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q`.
See docs/validation/gdsm_categorical_validation.log for actual execution results.
Tests cover closed-form noise/posteriors versus enumeration, exact endpoint,
variable-size masks, singleton/edgeless cases, typed graphlets versus brute force,
permutation/sign/degenerate-basis symmetry, gradients through both directions of
coupling, oracle swap improvements, basis propagation between model calls,
trained DH-VAE integration/provenance rejection, audit/metric outputs and legacy
regressions. The real-CUDA tests are skipped in this CPU-only environment.

The generic CLI smoke run trained on five small synthetic categorical graphs,
generated12, audited actual degree changes/eigenpairs, and ran the extra attributed
evaluator. A separate tiny molecule-fixture smoke run exercised the QM9 CLI and
existing molecular evaluator using a proxy NSPDK backend with FCD skipped because
those dependencies were absent. It is a compatibility check, not a QM9 benchmark.
No full QM9/ZINC training, 1024-graph evaluation, GPU throughput or chemical validity
claim is made. New settings and guidance weights need validation experiments.

This is a dense O(n^2) categorical encoder with eigendecomposition and candidate
search. It does not retain the eigenvalue-only model's low pairwise compute cost.
Exact typed counting/refiner rescoring can dominate larger graphs. More efficient
incremental typed deltas are a future optimization, not claimed in this patch.
Checkpoint resume is unsupported in the existing managed simple-GDSM interface.

## Relationship to the referenced models

ConStruct (Madeira et al., NeurIPS2024) uses marginal node noise but absorbing edge
noise and a structural projector. This implementation uses marginal noise for BOTH
nodes and edges and deliberately has neither that projector nor its constraints.
The endpoint-classification posterior-mixture idea follows discrete categorical
diffusion as in DiGress/ConStruct. The spectral coupling, typed summary supervision
and late swap intervention are the experimental GraphES extension.

Reference formulations: ConStruct Section3.3 and AppendixA.1–A.2;
Primary implementation references (no copied source code added):
https://github.com/cvignac/DiGress/blob/main/src/diffusion/noise_schedule.py
https://github.com/cvignac/DiGress/blob/main/src/diffusion/diffusion_utils.py
https://github.com/manuelmlmadeira/ConStruct
