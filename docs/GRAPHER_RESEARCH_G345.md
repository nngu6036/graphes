# GraphER research profiles: spectral--categorical backbone and graphlets 3/4/5

## Status and naming

These are runnable, prespecified **starting profiles for new experiments**, not
validation-tuned hyperparameters or new benchmark results. The body architecture,
noise schedule, and guidance budget of the successful QM9 size-3 run are retained;
only the graphlet output representation is enlarged in the main QM9 profile.
Keep that existing checkpoint and its result as the k=3 reference.

It is reasonable to name the combined method **GraphER**. Describe its backbone
as a **project-owned spectral--categorical adaptation inspired by GSDM and
DiGress**, not as the official GSDM implementation. The source wrapper explicitly
states that its minimal spectral implementation is not a reproduction claim.
Use an independently rerun official GSDM as a separate baseline when available.
Do not transfer spectral-only speed/likelihood claims or ConStruct's projector
constraint guarantees to this hybrid sampler.

Suggested method description:

> GraphER combines a GSDM-inspired spectral--categorical generative backbone
> with learned multi-scale structural targets and generation-time rewiring.
> The joint denoiser predicts clean node and edge categories, adjacency spectra,
> and connected induced graphlet summaries. At selected reverse steps, local
> edge swaps reduce mismatch to predicted structural targets before sampling
> continues. Each local swap preserves the current graph's typed degrees;
> subsequent categorical transitions remain free to change topology and types.

The initially sampled degree sequence is **not** a global invariant. The derived
anchor remains a soft input and coordinate reference. Eigenvectors are updated
from the retained categorical adjacency after every reverse transition.
There is no whole-trajectory connectivity or chemical-validity guarantee.

## Interpretation of the user-reported QM9 result

The reported batch has 1,024 requested graphs, 1,001 raw-valid molecules,
999 unique valid molecules, and 455 novel unique valid molecules. Native validity
is 97.75390625%, uniqueness is 999/1001 = 99.8002%, and novelty is
455/999 = 45.5455%. FCD 0.554195 and NSPDK MMD 0.00158317 use the raw-valid
subset, not the 1,024 corrected molecules. The 100% corrected validity is a
separate repair diagnostic. Do not label it native GraphER validity.

The result is promising, but a single report does not identify the benefit of
rewiring or establish a protocol-matched ranking. It has no graphlet MMD values
and no accepted-swap diagnostics. It also cannot establish memorization from
training overlap alone. Compare fixed graph representation, dataset fingerprints,
formal-charge policy, sample counts, valid-molecule selection, and exact metric
implementations before claiming comparable performance. Report independent
training-seed variability, not only repeated sampling from one checkpoint.

## Why a support patch is included

The preceding categorical branch explicitly rejected graphlet orders other than
3. Adding `sizes: [3,4,5]` to an unpatched config would not activate them. This
release adds exact connected induced typed counting for orders 3,4,5, independent
per-order output normalization and mass prediction, exact local swap deltas,
sparse training target storage, and an external multi-order evaluator.
The legacy size-3 representation and existing checkpoints remain readable.
**A new multiscale model must be retrained.** Do not load a size-3 checkpoint with
a size-3/4/5 configuration or reuse its prediction-head weights as a full model.
The previous robust float64/CPU eigensolver fix is retained unchanged.

## Installation

From the corrected categorical project root:

```bash
unzip -o grapher_g345_research_bundle.zip -d .
```

The bundle includes all changed/new source files, the numerical solver file,
28 model/ablation YAML files, a protocol YAML, scripts, tests, and this guide.
It does not contain or overwrite prepared data or trained checkpoints. A full
project archive and a source patch against the preceding eigensolver-fixed
archive are also supplied. Review the diff before replacing locally edited code.

## Configuration paths and training budgets

| Dataset | Main config under `configs/experiments/grapher_research/` | Max nodes | Epochs | Batch | Validate every |
|---|---|---:|---:|---:|---:|
| Community-small | `community_small_g345.yaml` | 20 | 10000 | 32 | 100 |
| Ego-small | `ego_small_g345.yaml` | 18 | 5000 | 32 | 50 |
| QM9 | `qm9_g345.yaml` | 9 | 200 | 64 | 5 |
| ZINC | `zinc_g345.yaml` | 38 | 200 | 32 | 5 |

For the repository's configured 64-graph Community-small training split and
128-graph Ego-small training split, those horizons give 20,000 optimizer updates
each. They are not claims about the user's currently mounted split; the check
stage reports actual graph counts. Actual update counts are recorded in training
history and the manifest. Molecular models retain the existing 200-epoch horizon.
Do not describe different raw epoch counts as equal compute across datasets.

All profiles use hidden width 128, three layers per branch, four attention heads,
feed-forward width 256, zero dropout, AdamW learning rate 0.0002, weight decay
0.000001, gradient clipping 1.0, 500 training noise steps, and 500 reverse steps.
No unimplemented EMA or learning-rate-scheduler keys are placed in the YAMLs.
Every training graph is used (`max_train_graphs: null`). The checkpoint is selected
by minimum joint validation denoising loss using fixed validation corruption RNG,
not by test FCD/MMD. There is no automatic metric-tuned early stopping or exact
resume from an interrupted optimizer in this trainer.

The seed-42 ordinary DH-VAE paths match the prior release. The new runner resolves
both model and degree-prior configs for SEED=43/44 and uses separate prior
checkpoints for those seeds. It does **not** change the prepared data split seed.
A matching existing ordinary-degree prior can be reused. A typed-signature prior
is not a substitute. Prior training and generation provenance are checked.

## Graphlet semantics and controls

The operative YAML block is:

```yaml
graphlets:
  sizes: [3, 4, 5]
  size_weights: [1.0, 1.0, 1.0]
  connected_only: true
  counting: exact_connected
  clustering_bins: 100
  max_vocab_per_size: 8192
  min_train_count: 1
  max_connected_subsets: 1000000
```

The cap is `null` for Community-small/Ego-small, 8192 per order for QM9,
and 16384 per order for ZINC. Each order has one **additional** overflow bin.
These caps are resource settings, not observed vocabulary sizes or tuned optima.
Training occurrence counts determine retained patterns; ties are broken by the
canonical key. All omitted patterns contribute to the overflow count and mass.
The checkpoint schema records observed/retained class counts and retained
occurrence fractions. Inspect per-order overflow before drawing graphlet-fidelity
conclusions; change caps before retraining when needed, not during generation.

All connected induced topologies are counted: paths, stars, trees, triangles,
cycles, chorded patterns, and cliques when present. This is not a cycle-only
basis. Node and edge categories are permuted together during canonicalization.
For the untyped datasets, the complete connected topological class counts are
2, 6 and 21 at orders 3, 4 and 5, as verified against the graph atlas in tests.

For each order k, the model separately predicts a conditional composition H_k
and selected mass m_k = (# connected induced k-subsets) / binomial(n,k).
Histograms are NOT passed through a single softmax across all sizes. Per-order
loss and energy weights are normalized, so adding orders does not automatically
triple the total graphlet weight. Histogram loss is masked for graphs with no
connected k-subsets. Mass loss is masked when n<k and targets zero for available
orders with no connected subsets. Independent graphlet and mass validation losses
are logged per order. Untyped orbit supervision stays at the four coordinates
through size 3; standard external ORCA evaluation remains through size 4.

Counting is exact. The limit on connected subsets is a **fail-loud resource guard**,
not a Monte Carlo sample count or a silent truncation. Candidate scoring updates
only connected subsets in the union graph that contain a changed pair, and tests
compare those deltas to complete recounts. Training stores sparse indices/counts
rather than a dense dataset-by-vocabulary matrix. The evaluator also uses sparse
features. Full ZINC preprocessing and exact late-stage counting can still be
expensive; this release has no GPU throughput estimate.

## Joint losses and generation settings

The aggregate loss weights remain spectral=1, node=1, edge=1, graphlet=0.5,
mass=0.5, clustering=0.5 and orbit=0.25. Equal weights inside the graphlet and mass
blocks average their per-order contributions. The graphlet output heads grow;
the body architecture is unchanged, but total parameter count is not identical.

The main profile keeps structural events at destination timesteps 100,50,0,
with at most two accepted swaps per event, proposal budget 128 and valid-candidate
budget 64. This is deliberately conservative relative to increasing both graphlet
size and refinement budget at once. Spectral feedback is 0.05 in the final 20%
of the schedule. Positive total-energy and structural-energy improvement are both
required. Connectivity is preserved by swaps only when the event input is already
connected. The next categorical transition may change it again.

Changing loss weights, vocabulary/counting settings, graphlet orders or the
spectral-conditioning architecture requires retraining. Guidance timing and
enabling/disabling guidance can be changed for checkpoint-matched sampling
ablations. There is no claim that exact order-5 guidance must improve FCD/MMD.

## Commands for all four datasets

Run from the project root. The default eigensolver policy keeps the neural network
on the requested device but evaluates graph eigenpairs in CPU float64:

```bash
export GDSM_EIGH_BACKEND=cpu
export DEVICE=cuda:0
```

Check and profile before a long run (profile reads only a small training prefix):

```bash
bash scripts/run_grapher_research.sh qm9 check
bash scripts/run_grapher_research.sh qm9 profile
```

Train the prior as needed, train the joint model, generate, audit and evaluate:

```bash
SEED=42 bash scripts/run_grapher_research.sh community_small all
SEED=42 bash scripts/run_grapher_research.sh ego_small all
SEED=42 bash scripts/run_grapher_research.sh qm9 all
SEED=42 bash scripts/run_grapher_research.sh zinc all
```

Stages can be run separately: `degree`, `train`, `generate`, `audit`, `evaluate`.
Default requested generation counts are 1024 for topology and 10000 for molecules.
They are read from `configs/experiments/grapher_research/protocol.yaml`; use the
same requested counts for every method in a comparison. For a small engineering
smoke run only, override e.g. `N=32`; that does not change the training budget.

Three independent training/prior seeds, on the same prepared split:

```bash
for seed in 42 43 44; do
  SEED="$seed" bash scripts/run_grapher_research.sh qm9 all
done
```

The runner writes seed-resolved YAMLs to
`outputs/research_configs/<dataset>/<run>/<variant>/`. A differing config under
an existing resolved path raises an error instead of silently reusing old results.
Use a new RUN for a changed experiment. It does not rebuild data or pass overwrite.
The run ID is `seed_<seed>_grapher_g345`; a typical output directory is:

```text
outputs/baselines/gdsm_simple/qm9/seed_42_grapher_g345/generations/seed_42_n_10000
```

The internal CLI model ID remains `gdsm_simple` to avoid breaking prior artifacts;
that string is not a paper naming or reproduction claim.

### Direct commands for the seed-42 QM9 profile

```bash
CFG=configs/experiments/grapher_research/qm9_g345.yaml
RUN=seed_42_grapher_g345
N=10000

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage train --dataset qm9 --no-common-config \
  --wrapper-config "$CFG" --seed-id 42 --run-id "$RUN" --device cuda:0

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage generate --dataset qm9 --no-common-config \
  --wrapper-config "$CFG" --seed-id 42 --generation-seed 42 \
  --run-id "$RUN" --num-samples "$N" --device cuda:0
```

Use the runner for seeds other than 42, so the prior paths are also resolved.

## Required ablations and attribution of gains

Each dataset has seven complete YAMLs; there is no custom YAML-inheritance syntax:

| Variant | Training | Question |
|---|---|---|
| main | new g345 joint training | Combined framework |
| no_guidance | reuse main checkpoint | Does interleaved rewiring help with the same trained predictor? |
| final_only | reuse main checkpoint | Is final-only refinement competitive? |
| backbone_only | new training | Do summary supervision and rewiring help beyond the spectral--categorical backbone? |
| categorical_only | new training | Does the spectral mechanism help beyond node/edge diffusion? |
| k3 | new training | Single-order controlled baseline |
| k34 | new training | Added benefit of order 5 after orders 3 and 4 |

`no_guidance` retains spectral feedback; it isolates the explicit swap intervention,
not all possible sampling guidance. `final_only` retains the same two-step per-event
cap, so its total budget is smaller; it is **not a compute-matched proof of the value
of interleaving**. Match budgets in a separate comparison and report actual counts.
The categorical-only control retains unused spectral calculations and summary
head parameters for implementation compatibility, so it is not a speed-optimized
DiGress implementation or a fair wall-clock baseline for an official model.

Example checkpoint-matched ablation:

```bash
VARIANT=no_guidance SEED=42 bash scripts/run_grapher_research.sh qm9 generate
VARIANT=no_guidance SEED=42 bash scripts/run_grapher_research.sh qm9 audit
VARIANT=no_guidance SEED=42 bash scripts/run_grapher_research.sh qm9 evaluate
```

A trained ablation:

```bash
VARIANT=backbone_only SEED=42 bash scripts/run_grapher_research.sh qm9 all
```

The runner automatically shares the main RUN for no_guidance/final_only and uses
separate training runs for backbone_only/categorical_only/k3/k34. Generation IDs
also have ablation suffixes. Random generators for neural sampling and refinement
are separate. Keep N, batch size, training seed and generation seed fixed for paired
sampling. Paths diverge after intervention; initial graphs alone are not the backbone
baseline, and `final_pre_rewire_graphs.pkl` isolates only the last event.

### Immediately test the already successful legacy k=3 checkpoint

This uses the previous script/config, not the new multiscale profile:

```bash
VARIANT=no_guidance RUN=seed_42_spectral_categorical \
DEVICE=cuda:0 N=1024 GDSM_EIGH_BACKEND=cpu \
  bash scripts/run_gdsm_categorical.sh qm9 generate

VARIANT=no_guidance RUN=seed_42_spectral_categorical \
DEVICE=cuda:0 N=1024 GDSM_EIGH_BACKEND=cpu \
  bash scripts/run_gdsm_categorical.sh qm9 evaluate
```

Use the actual existing RUN name. This does not overwrite the guided 1024 batch.
It is the highest-priority controlled comparison for attributing the reported FCD.

## Evaluation protocol

Molecular main evaluation uses raw-valid molecules, requires FCD and does not
allow the optional NSPDK proxy flag. It does not resample until a valid target
count is reached. Report raw validity over all requested samples and FCD/NSPDK
over the realized raw-valid subset, with actual generated/reference counts.
Posthoc correction and projected-charge validity are supplementary only. Do not
compare the main table to corrected-molecule FCD or substitute the proxy NSPDK
values. Canonical SMILES/deduplication/novelty conventions must match baselines.
The no-formal-charge and implicit-hydrogen representation is unchanged.

The separate `categorical_metrics.json` now reports graphlets at all requested
orders on **all raw graphs**, including invalid molecular graphs. For each order:

* joint-mass RBF MMD^2, retaining disconnected-subset mass;
* selected connected-subset mass RBF MMD^2;
* conditional histogram RBF MMD^2 on graphs with positive selected count, with
  that restricted denominator reported explicitly;
* external vocabulary size and overflow relative to the frozen training vocabulary.

External vocabulary is the union of generated/reference patterns and has no
training-overflow collapse. It never changes the checkpoint vocabulary. This
metric is intentionally separate from standard GraphRNN/ORCA, NSPDK and FCD.
The older molecular report's optional graphlet fields can still be null; use the
separate per-order file rather than relabeling an uncomputed field.

The topology evaluator remains unchanged: GraphRNN Degree/Clustering/Orbit MMD,
ORCA through size 4, no silent change to orbit definition. Set `ORCA_EXEC` to the
existing executable if needed. Training/test distances are finite-sample diagnostic
references, **not mathematical lower bounds**. Use matched-size reference draws
for a sampling-floor diagnostic and keep their randomness separate from training.

The protocol calls for three independent training/prior seeds and mean plus sample
standard deviation. It does not automatically aggregate metric files: report each
seed, valid count and dataset fingerprint before aggregation. Also retain generation
wall-clock time, graphlet preprocessing/profile time, accepted-swap counts, model
parameter count, hardware/software versions and externally measured peak memory.
No claim of bit-identical CUDA results across machines is made.

## Validation in this environment

The full suite reports **395 passed, 8 skipped**, with one existing nested-tensor
warning. The eight skipped tests require CUDA. There are 57 added tests for exact
counting/canonicalization through order 5, local deltas including recolors/deletions,
train-only vocabulary/caps, sparse targets, independent heads/masks/gradients,
managed training/generation/auditing/evaluation and all 28 model profiles.
Old size-3 tests still pass. Seed-43 runner resolution was checked for all four
datasets, including matching model/prior paths without changing split seeds.
A 38-node synthetic labelled ladder verifies exact local deltas against full recounts.

No full Community-small, Ego-small, QM9 or ZINC run was performed here. There are
no new FCD/MMD results, real-data vocabulary sizes or GPU throughput measurements.
The user's reported QM9 metrics were not independently rerun.

## Sources to credit in the paper

* Luo, Mo and Pan. Fast Graph Generation via Spectral Diffusion. IEEE TPAMI,
  2024. DOI 10.1109/TPAMI.2023.3344758; arXiv 2211.08892. The paper's acronym is GSDM.
* Vignac et al. DiGress: Discrete Denoising Diffusion for Graph Generation.
  ICLR 2023; arXiv 2209.14734. Marginal categorical node/edge diffusion.
* Madeira et al. Generative Modelling of Structurally Constrained Graphs.
  NeurIPS 2024. ConStruct's edge-absorbing constraint projector is related but
  not the sampler implemented here.
* Preuer et al. Frechet ChemNet Distance: A Metric for Generative Models for
  Molecules in Drug Discovery. JCIM 2018. DOI 10.1021/acs.jcim.8b00234.

The attached ConStruct paper's Appendix G uses 10000 QM9 samples per sampling
run and includes formal charges as extra node labels. Its reported metrics must
not be treated as a directly matched benchmark for the 1024-sample, atom-identity
configuration without aligning these protocol differences.
