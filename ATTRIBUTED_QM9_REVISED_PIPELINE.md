# Revised attributed GraphER pipeline (QM9 spectral + attributed graphlets)

This note traces the implementation used by the revised config
`configs/experiments/grapher/qm9_attributed_spectral_graphlet_v2_enriched_source.yaml`.
It also explains how the user's original generation/evaluation commands map to the code.

## Important command distinction

`run_attributed_grapher.py` is **generation only**. The predictor it loads must first be trained with
`train_attributed_grapher.py`.

Recommended revised commands:

```bash
PYTHONPATH=src python scripts/train_attributed_grapher.py \
  --config configs/experiments/grapher/qm9_attributed_spectral_graphlet_v2_enriched_source.yaml \
  --output-dir outputs/attributed_grapher/qm9_spectral_graphlet_v2_enriched_source/seed_42 \
  --seed 42 \
  --device gpu

PYTHONPATH=src python scripts/run_attributed_grapher.py \
  --config configs/experiments/grapher/qm9_attributed_spectral_graphlet_v2_enriched_source.yaml \
  --output-dir outputs/attributed_generation/qm9_spectral_graphlet_v2_enriched_source/seed_42 \
  --num-generate 1024 \
  --seed 42 \
  --device gpu

PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-graphs \
    outputs/attributed_generation/qm9_spectral_graphlet_v2_enriched_source/seed_42/molecular_graphs.pkl \
  --dataset-root outputs/datasets \
  --dataset qm9_attributed \
  --reference-split test \
  --train-split train \
  --output-dir \
    outputs/attributed_grapher/qm9_spectral_graphlet_v2_enriched_source/seed_42/evaluation \
  --require-fcd \
  --hogdiff-compatible-metrics
```

The canonical project QM9 split in `configs/datasets/qm9.yaml` is 104,665 train / 13,083 validation /
13,083 test. This experiment uses all 104,665 training graphs and caps validation at 2,000.

---

# Training code path

## `scripts/train_attributed_grapher.py`

- `main()` begins around line 368.
- Lines ~430-460 build the categorical vocabulary from training only. For this config it is C/N/O/F
  (`[6,7,8,9]`) and single/double/triple bonds (`[1,2,3]`), with edge index 0 reserved for no-edge.
- The graphlet basis is fit from at most 20,000 training molecules and contains attributed, connected,
  induced graphlet classes for `k=3,4`, plus overflow/disconnected simplex coordinates defined by the
  attributed graphlet code. Validation/test never define this vocabulary.
- Lines ~522-545 instantiate `AttributedSpectralDiffusionIterableDataset` for train and validation.
  The new config enables a persistent SQLite endpoint cache.
- Lines ~570-606 instantiate `AttributedSpectralGraphletTransformerPredictor`. The original dynamic
  predictor and the new hard-invariant summary head are trained jointly in one checkpoint.
- Batch size is 128, epochs 30, AdamW LR `1.5e-4`, weight decay `5e-5`.
- One stochastic bridge state is sampled per training molecule per epoch (`samples_per_graph=1`,
  `paths_per_graph=1`), so one epoch has 104,665 examples and `ceil(104665/128)=818` training batches.
  Across 30 epochs the model sees 3,139,950 training bridge states and performs 24,540 optimizer steps.
- Validation is capped at 2,000 examples, i.e. 16 batches/epoch.

## `src/grapher/rewiring_mlp/attributed/spectral_data.py`

`resolve_attributed_diffusion_endpoints` constructs the fixed source endpoint using the target's indexed
atom type and per-bond-type degree. This is stricter than the generation-time hard invariant and is used to
construct a chemically compatible paired source.

`build_attributed_spectral_diffusion_examples` (around line 318):

1. Construct source `G_src` and clean target `G*`.
2. Compute two Laplacian spectra for each endpoint:
   - topology Laplacian,
   - bond-order-weighted Laplacian.
3. Compute exact attributed graphlet simplex statistics for `k=3,4`.
4. Transform graphlet probabilities to CLR coordinates.
5. Sample Brownian endpoint-conditioned continuous summaries:

   `lambda_t = (1-alpha) lambda_src + alpha lambda_clean + noise`,

   with `spectral_sigma=0.18`, cosine schedule, lambda1 fixed at zero, and trace-preserving spectral noise.

   For graphlet CLR coordinates the same endpoint-conditioned idea uses `graphlet_sigma=0.30`.
6. There is **no rewiring trajectory during training**. The conditioning topology remains the typed source.

New persistent endpoint cache (`AttributedSpectralDiffusionIterableDataset`, around line 588):

- first access to a graph constructs the typed source, dual endpoint spectra and exact attributed graphlets;
- a compressed endpoint record is stored in SQLite;
- later epochs reload the fixed endpoints and only resample bridge progress/noise and shared node relabeling.

This amortizes the expensive typed constructor + exact graphlet endpoint work over 30 epochs.

## `src/grapher/rewiring_mlp/attributed/spectral_model.py`

### Dynamic source-conditioned branch

`_graph_context` (around line 245):

- node input contains atom one-hot, normalized ordinary degree, per-bond-type degrees, graph size, time and mask;
- edge input contains no-edge/bond category one-hot, adjacency/mask, time, endpoint degree features,
  typed-degree pair features and graph size;
- six dense edge-aware MPNN layers use hidden dim 256 and edge dim 128;
- masked node/present-pair/absent-pair pooling yields a 256-d graph context.

`_spectral_outputs` (around line 376):

- each spectral channel has `n` tokens; the two channels are flattened to `2n` Transformer tokens;
- each token receives current lambda, source lambda, relative spectral rank, progress, graph size and mask;
- channel embeddings tell the Transformer whether the token is topology or bond-weighted;
- Transformer: dim 256, 6 layers, 8 heads, FF dim 1024;
- the decoder predicts positive eigenvalue gaps, not arbitrary eigenvalues;
- lambda1 is exactly zero, ordering is enforced by cumulative positive gaps, and each channel is rescaled to
  the exact source trace. Topology trace is `2m`; bond-weighted trace is twice total bond order.

Graphlet heads predict a residual in CLR space for every fixed graphlet block and convert it back to a
probability simplex with softmax.

### New hard-invariant summary branch

`_invariant_summary_outputs` (around line 452) predicts a fixed structural target `S(I)` without using source
adjacency. Its inputs are exactly the information preserved by the current cross-type rewiring kernel:

- atom categories + ordinary degrees as permutation-invariant node tokens,
- graph size,
- global bond-type counts/proportions.

It deliberately does **not** condition on per-node typed degree, because cross-type bond reassignment is
allowed to change typed degree/weighted valence subject to valence/RDKit constraints.

It predicts:

- clean topology spectrum,
- clean bond-weighted spectrum,
- clean attributed graphlet CLR/probability summaries.

### Joint loss

The existing dynamic predictor loss uses config weights:

- full dual spectrum: overall 1.0,
- topology channel 1.0,
- bond-weighted channel 1.25,
- second-moment regularizer 0.20,
- extra low-frequency loss 0.75 over lambda2..lambda5,
- graphlet CLR Smooth-L1 2.0,
- graphlet probability KL 0.75.

The new invariant-summary branch receives the same clean endpoint targets and the same spectral/graphlet
component losses. Its complete auxiliary loss is multiplied by `invariant_summary: 0.35`.

Thus:

`L_total = L_dynamic + 0.35 L_invariant_summary`.

---

# Generation code path

## `scripts/run_attributed_grapher.py`

`main()` starts around line 264.

- Lines 290-312 load the config and prepared train/test splits.
- Lines 314-329 load the attributed checkpoint and its training-only categorical/graphlet vocabularies.
- `--num-generate 1024` overrides any YAML sample count and requests 1,024 returned molecules.
- The config uses `generation.invariant_source: learned`, therefore a typed-degree VAE is loaded.
- `max_attempts_per_graph=64` allows resampling/reconstruction if a source attempt is invalid.
- Source validity is a hard generation gate: `require_rdkit_source_validity=true`.

For each requested molecule (loop around line 450):

1. Sample a typed invariant from the learned typed-degree VAE.
2. Construct a connected attributed source with the typed backtracking constructor.
3. Extract the actual hard rewiring invariant from the realized source:
   atom types, indexed ordinary degrees, global bond-type counts.
4. Require raw RDKit validity of the source.

### New source enrichment

Before the dynamic refiner, `enrich_attributed_graph_with_invariant_summary` is called.

- Predict one fixed target `S(I)` from the source hard invariant.
- Run at most 12 valid cross-type double-edge swaps against this fixed target.
- Candidate budget is adaptive: 128 proposals / 32 valid candidates initially, increasing to 256 / 64.
- Both dual spectra and attributed graphlet CLR distance are used with fixed weights.
- Because the target and energy are fixed, energies are comparable across these 12 steps and the best visited
  graph is a meaningful enriched base.
- The output remains connected and preserves atom types, indexed ordinary degrees and global bond-type counts.

### Main state-conditioned refiner

The existing dynamic predictor then refines the enriched base for at most 64 accepted swaps, but its graph
encoder still receives the original typed source as `conditioning_graph`. This matches training: the model
was trained with a fixed typed source topology while only spectrum/graphlet summaries move continuously.

The current graph contributes its actual current topology spectrum, bond-weighted spectrum and graphlet CLR.
The model predicts a clean summary, then the code constructs the next reverse-summary target using:

- spectral clean mix: cosine, 0.10 -> 1.0,
- graphlet clean mix: cosine, 0.05 -> 1.0.

Global-to-local weights move from:

- spectrum 1.00 -> 0.30,
- graphlets 0.25 -> 2.25.

Prediction horizon anneals from 4 accepted swaps/prediction to 1.

The revised main config uses `return_best_state: false`. This is important: the predictor and bridge target
are refreshed along the trajectory, so energies from different reverse steps are different functions and
cannot be compared as one global objective. Returning the last accepted state is mathematically consistent.
`return_best_state: true` remains valid for source enrichment because enrichment uses one fixed target.

---

# Candidate search and runtime revisions

## `src/grapher/rewiring_mlp/attributed/spectral_graphlet_refiner.py`

`_propose_attributed_candidates` (around line 625):

- sample two current edges with four distinct endpoints;
- construct two topological reconnections;
- if removed bond types differ, enumerate both assignments of those two types to each reconnection, yielding
  up to four attributed successors;
- perform topology/connectivity validation with a virtual-swap BFS before constructing a NetworkX candidate;
- compute weighted-valence changes only for the four affected atoms before materialization;
- materialize only survivors;
- retain the full valence checker as a safety assertion;
- reject visited attributed states.

Hard kernel invariants are:

- atom types,
- indexed ordinary degrees,
- global bond-type counts,
- connectivity.

Per-node typed degree and weighted valence are **not** invariants under cross-type reassignment; valence is a
validity constraint instead.

`_prepare_candidate_states` (around line 931):

- exact attributed graphlet changes are updated locally from the four changed bonds;
- all candidate topology/bond Laplacians are assembled into a batch;
- `batched_attributed_laplacian_spectra` uses `torch.linalg.eigvalsh` on CUDA when available, falling back to
  NumPy; this replaces two Python-dispatched eigensolvers per candidate.

Main candidate budgets are now adaptive:

- beginning: 256 proposals / 64 valid,
- end: 512 / 128,
- cosine interpolation.

The hard maximum remains the old 512 / 128. The enrichment stage uses smaller 128/32 -> 256/64 budgets.

RDKit is deliberately not run on every candidate. Cheap structural/spectral/graphlet filters rank candidates
first; RDKit checks only the top 16 eligible candidates.

---

# Complexity with this QM9 config

Let `n` be heavy atoms, `m` bonds, `C` valid candidates and `P` raw proposals. QM9 molecules are small, so
multiplicity across 104,665 training examples and 1,024 generated samples matters more than asymptotic `n`.

Training endpoint preparation (first cached pass):

- typed backtracking constructor: combinatorial worst case, explicitly bounded by configured restarts/backtracks;
- each dual spectrum: two symmetric eigendecompositions, `O(n^3)`;
- exact attributed graphlets through k=4: worst-order `O(n^4)` enumeration/canonicalization;
- later epochs avoid repeating these fixed endpoint calculations.

Neural training per batch:

- dense MPNN: all node pairs, approximately quadratic in `n` times hidden projections;
- spectral Transformer: sequence length `2n`, attention `O((2n)^2 d)`, FF roughly `O(2n d d_ff)`;
- invariant summary head: pooled node-token MLP plus lightweight spectrum/graphlet heads.

Generation per decision:

- proposal validity: roughly `O(P(n+m))`, without a graph copy for raw invalid proposals;
- batched dual spectra: `O(C n^3)` numerical work, but batched/GPU-dispatched;
- local k=4 attributed graphlet delta: only subsets touching changed edges, approximately `O(C n^2)` in subset
  count rather than full `O(C n^4)` recounting;
- RDKit: at most 16 shortlisted eligible candidates per decision.

Source enrichment adds up to 12 search steps, so it is not guaranteed to reduce raw wall time by itself. Its
purpose is to start the 64-step dynamic refiner from a structurally better base. The batching/fast-validity and
adaptive-budget changes are intended to offset search cost; end-to-end runtime should still be benchmarked.

---

# Evaluation command

`evaluate_generated_molecules.py` does not generate new molecules. It loads the 1,024 serialized outputs.

At lines ~657-675 it loads:

- generated `molecular_graphs.pkl`,
- full declared test reference split (13,083 unless capped by CLI),
- full train split (104,665 unless capped) for novelty.

At lines ~687-713:

- compute strict raw RDKit validity first;
- independently attempt deterministic evaluation-only valence correction;
- `--hogdiff-compatible-metrics` selects the corrected-valid set for distributional metrics.

The headline `validity` remains raw, full-RDKit validity even in HOG-compatible mode.

NSPDK:

- HOG-compatible mode forces EDeN neighborhood-pair NSPDK,
- complexity 4,
- HOG-Diff bond-label convention,
- computed between the test reference molecules and corrected-valid generated molecules.

FCD:

- HOG-compatible mode uses corrected-valid generated SMILES;
- reference FCD statistics are cacheable under the dataset evaluation-cache directory;
- `--require-fcd` raises an error if a compatible FCD backend cannot produce the metric.

The evaluator writes `molecular_evaluation_metrics.json`, raw-valid SMILES, and corrected-valid SMILES.

To measure whether source enrichment itself helps, evaluate the additional output separately:

```bash
PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-graphs \
    outputs/attributed_generation/qm9_spectral_graphlet_v2_enriched_source/seed_42/enriched_base_graphs.pkl \
  --dataset-root outputs/datasets \
  --dataset qm9_attributed \
  --reference-split test \
  --train-split train \
  --output-dir \
    outputs/attributed_grapher/qm9_spectral_graphlet_v2_enriched_source/seed_42/evaluation_enriched_base \
  --require-fcd \
  --hogdiff-compatible-metrics
```

This gives a clean three-stage molecular diagnostic:

1. `typed_source_graphs.pkl`,
2. `enriched_base_graphs.pkl`,
3. `molecular_graphs.pkl`.

All three share the hard rewiring invariant; changes in molecular distributional metrics therefore isolate the
value of enrichment and final refinement rather than changes in ordinary degree/atom/global bond counts.
