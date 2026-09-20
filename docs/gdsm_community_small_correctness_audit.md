# Community Small GDSM correctness audit

The active graphlet summaries, spectral diffusion equations, and local rewiring
passed the checks below. A separate bug was found in the **evaluation spectral
histogram**. Correcting that metric requires reevaluation of existing graphs;
it does not change their topology or establish that training quality improves.

## Commands and active implementation

[grapher.txt](../data/experiments/grapher.txt) launches seeds 41 through 45 using
the corresponding `configs/experiments/gdsm_final_explicit/community_small_seed_*.yaml`.
The [seed 42 configuration](../configs/experiments/gdsm_final_explicit/community_small_seed_42.yaml)
uses 10,000 epochs, batch size 32, 500 training and sampling diffusion steps,
the full prepared training split, and 1,024 generated graphs. `--no-common-config`
means these explicit settings control the run. `GDSM_EIGH_BACKEND=cpu` selects
CPU eigendecomposition while the neural network runs on `cuda:0`.
All five Community Small configurations were compared: their model settings
match, with the expected seed-specific DHVAE checkpoint paths.

`extensions.attributed_categorical.enabled: true` selects
[`categorical/pipeline.py`](../src/grapher/models/gdsm_simple/categorical/pipeline.py).
The outer `structural_summary: none` and `degree_preserving_rewiring: false`
belong to a different path; they do not disable the active categorical summary
heads or `attributed_categorical.guidance`.

## Structural summaries

The active [multiscale counter](../src/grapher/models/gdsm_simple/categorical/multiscale.py)
counts connected, induced graphlets of orders 3, 4, and 5 exactly. An independent
oracle checked all 208 nonempty graph-atlas graphs with at most six nodes and
recovered the expected 2, 6, and 21 connected unlabelled graphlet types. Each
order has its own normalized histogram and connected-subset mass; the mass
denominator is `choose(n, k)`. Vocabulary construction uses training data only,
with an overflow category for unseen types.

The [topology summaries](../src/grapher/models/gdsm_simple/categorical/data.py)
also passed checks: the clustering histogram has 100 bins, and the four orbit
features are `log1p` of the mean degree, induced-path endpoint count,
induced-path centre count, and triangle count. These four features are a
deliberate smaller target than the report's 15 ORCA orbit features.

The report's graphlet MMD is not the training loss. Its
[evaluation configuration](../configs/experiments/baselines/community_small_evaluation.yaml)
allows 8,192 sampled subsets. Order-five counting becomes sampled at `n >= 18`,
where `choose(n, 5) > 8192`; training and local refinement remain exact. This
introduces evaluation sampling variance, not a confirmed counting bug.

## Spectral diffusion and the evaluation correction

The model learns sorted **binary-adjacency eigenvalues divided by `sqrt(n)`**.
The [forward and reverse equations](../src/grapher/models/gdsm_simple/categorical/noise.py)
use a source-centred Gaussian process and an x0-predicting DDIM update with
matching cosine endpoints. The prediction is masked, zero-trace, sorted, and
bounded. Current adjacency eigenvectors are recomputed after every categorical
update and every refinement event. Repeated eigenspaces are handled without
depending on arbitrary eigenvector signs or rotations.

An independent 500-step oracle check on path, cycle, and community-like SBM
graphs produced final spectrum error `1.19e-7`, maximum intermediate trajectory
error `1.10e-5`, and adjacency reconstruction error `1.19e-7`. Padding stayed
exactly zero.

The report instead compares **normalized-Laplacian spectral histograms**.
These are different descriptors, so minimizing the training spectral loss does
not directly minimize the reported spectral MMD. The shared
[`spectral_histogram`](../src/grapher/rewiring_mlp/properties/summary.py)
implementation had three evaluation defects:

- Isolated vertices contributed eigenvalue one instead of zero.
- Roundoff just outside `[0, 2]` could discard endpoint eigenvalues.
- Roundoff around internal bin boundaries changed histograms under node
  relabelling. Reproductions gave histogram L1 differences of about 1.2667 for
  a six-cycle and 0.178947 for a 20-node community graph.

The correction uses the proper normalized Laplacian and stabilizes numerical
roundoff at histogram boundaries. Report format v7 records the revised
descriptor. Reevaluate every compared baseline with the same implementation;
do not mix old and corrected spectral MMD values. This correction does not
require retraining or generation.

## What rewiring guarantees

The active [refiner](../src/grapher/models/gdsm_simple/categorical/refiner.py)
uses same-type double-edge swaps. Oracle checks covered 875 valid swaps and
145 incremental graphlet-count updates. Accepted swaps preserve each current
node's degree and typed degree, and preserve connectivity when the input to
that event is connected. Acceptance requires both lower weighted total energy
and lower structural energy against the network's predicted clean summaries.
Those predictions are not an oracle for held-out distribution quality.

With the current 500-step schedule, guidance runs at remaining steps 100, 50,
and 0, with at most two accepted swaps per event: **at most six per graph**.
Later categorical transitions can undo earlier structural changes and change
degrees. Only the final event's at most two swaps have no subsequent diffusion
transition. Connectivity preservation does not itself repair disconnected
graphs.

The DHVAE degree sample builds a **soft spectral anchor**. Initial categorical
graphs are independently sampled marginal noise, and final degrees need not
match the prior. The spectral prediction similarly informs the decoder without
forcing the final adjacency to realize that spectrum.

The commands label `initial_graphs.pkl` as the report's base graphs. Therefore
`gdsm_simple_base_to_test` measures terminal noise, not generation without
rewiring. Comparing `final_pre_rewire_graphs.pkl` with `base_graphs.pkl` isolates
only the final event. Measuring all guidance effects requires a separate
generation with guidance disabled, holding checkpoint, generation seed, sample
count, sampling schedule, and spectral feedback fixed.

## Recorded results and next diagnostics

The following values are copied from `grapher.txt`, not recomputed. Spectral
values use the earlier evaluator and need replacement.

| Comparison | Degree MMD | Clustering MMD | Orbit MMD | Spectral MMD (old) | Graphlet MMD |
|---|---:|---:|---:|---:|---:|
| Train to test | 0.004733 | 0.009889 | 0.000399 | 0.020486 | 0.010610 |
| Final seed 41 | 0.012453 | 0.074927 | 0.010830 | 0.052038 | 0.101208 |
| Final seed 42 | 0.009559 | 0.080085 | 0.020576 | 0.057560 | 0.117862 |
| Final seed 43 | 0.015680 | 0.059785 | 0.014582 | 0.051978 | 0.073168 |

The final graphs improve substantially over the saved noise baseline, but
clustering, orbit, and graphlet discrepancies remain above train-to-test
values. The spectral evaluator correction alone cannot explain those other
gaps.

1. Reevaluate existing saved graphs for all baselines with report v7 and the
   same reference split and metric settings.
2. Read each run's `train/training_metrics.json`, `train/manifest.json`, and
   `train/resolved_config.yaml`. Check selected epoch, train/validation gaps,
   and the separate spectral, edge, graphlet-order, mass, clustering, and orbit
   losses before changing training budgets or weights.
3. Inspect each generation's `rewiring_diagnostics.json`: accepted steps and
   tested candidates per event, energy changes, connectedness, prior-degree
   preservation, and eigenpair reconstruction error. Zero accepted swaps need
   interpretation alongside candidate counts and prediction accuracy.
4. Compare the saved pre-final and final graphs, then run the controlled
   no-guidance generation if a complete guidance ablation is needed. Evaluate
   distributions as well as the refiner's own energy.

One remaining hypothesis is conditioning generalization: preparation fixes
one sampled basis/anchor per training graph, while generation uses new sampled
degree/basis combinations. This may matter on a small dataset, but is not a
demonstrated bug or justification for changing hyperparameters without results.

The categorical, multiscale, eigensolver, spectral-summary, and report test
suites passed **162 tests, with five skipped**. Separate focused agent runs
overlap with those suites and are not added to this total.
The independent oracle checks above supplement those tests. Server checkpoints,
training histories, and generated artifacts were unavailable locally, and no
full experiment was rerun. This audit establishes the tested invariants and
the evaluation correction; it does not establish the remaining cause of poor
generation quality.
