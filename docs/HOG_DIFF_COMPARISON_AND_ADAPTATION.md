# HOG-Diff vs GraphER: implementation comparison and adaptation

This note is based on the uploaded repositories `HOG-Diff.zip` and
`graphes(20260828-151852).zip`.  It focuses on two code paths requested for
comparison: molecular-generation evaluation and spectral diffusion.

## 1. Executive conclusion

The two models use spectral information in fundamentally different ways.
HOG-Diff is a continuous score/SDE model over atom features and Laplacian
eigenvalues.  During spectral denoising it holds an eigenvector basis fixed and
reconstructs a dense pseudo-adjacency from the current eigenvalues.  GraphER is
an x0-style summary predictor whose continuous spectral state is only a
training-time target/conditioning variable; generation remains a walk over
simple, connected, degree-constrained discrete graphs using valid edge
rewiring.

Therefore the parts of HOG-Diff that can be transferred safely are the
*endpoint-conditioned OU marginal*, metric implementations, caching strategy,
and diagnostic ideas.  Copying HOG-Diff's eigenbasis reconstruction or reverse
Euler/Langevin updates into GraphER would bypass the constrained rewiring state
space and remove GraphER's exact structural guarantees.

Implemented adaptations in this repository:

1. HOG-Diff/EDeN NSPDK for molecular evaluation, while retaining the old
   GraphER proxy only as a diagnostic/backend option.
2. Strict raw RDKit validity is now the headline GraphER validity; corrected
   validity remains a separately reported diagnostic.
3. Explicit HOG-Diff-compatible distribution-metric mode that evaluates
   uniqueness/novelty/FCD/NSPDK from corrected molecules without weakening the
   raw-validity headline.
4. Cached reference FCD statistics when the installed `fcd_torch` API exposes
   `precalc`.
5. A HOG-Diff/GOUB OU-bridge marginal as an optional GraphER spectral-summary
   training path, with exact endpoint behavior and GraphER's trace/lambda1
   projections retained.
6. Streaming OU coefficients are cached once per schedule instead of rebuilt
   per training state.
7. HOG-inspired OU ablation configs for Community-small, Ego-small, and QM9.

## 2. Molecular evaluation comparison

### 2.1 HOG-Diff evaluation path

Primary files:

- `HOG-Diff/sampler.py`: `Sampler._mol_eval_fn`
- `HOG-Diff/utils/mol_utils.py`: `tensor2mol`, `correct_mol`,
  `valid_mol_can_with_seg`, `mols_to_nx`
- `HOG-Diff/evaluation/mol_evaluator.py`: `get_all_metrics`, `FCDMetric`
- `HOG-Diff/evaluation/mol_nspdk_evaluator.py`: `nspdk_eval_fn`
- `HOG-Diff/evaluation/eden.py`: EDeN neighborhood-pair vectorizer

The actual sequence is:

1. `tensor2mol(..., correct_validity=True)` constructs every molecule.
2. `correct_mol` repeatedly lowers the highest-order bond incident to the atom
   reported by RDKit's valence-property sanitization until the valence check
   passes.
3. `valid_mol_can_with_seg` canonicalizes the corrected molecule and can keep
   only the largest connected component.
4. `all_valid_wd` records whether the molecule passed the initial valence check
   *without correction*. `Sampler._mol_eval_fn` reports the mean of this flag as
   `validity` and logs it as `validity w/o corr.`.
5. The SMILES passed to `get_all_metrics` are nevertheless generated from the
   *corrected* molecule list. Thus HOG-Diff's FCD, uniqueness and novelty are
   based on corrected outputs, not the same raw subset used by the reported
   no-correction validity.
6. NSPDK is also evaluated on the corrected RDKit molecules.

A subtle point is that HOG-Diff's no-correction flag is based on
`SANITIZE_PROPERTIES`/valence checking in `check_valency`, whereas GraphER's raw
validity uses full `Chem.SanitizeMol`. They should therefore not be assumed to
be bit-for-bit equivalent validity tests.

### 2.2 HOG-Diff NSPDK

HOG-Diff converts RDKit molecules to labeled NetworkX graphs:

- node label = atom symbol;
- edge label = `int(bond.GetBondTypeAsDouble())`;
- EDeN `vectorize(..., complexity=4, discrete=True)`;
- biased MMD with a linear kernel:
  `mean(K_XX) + mean(K_YY) - 2 mean(K_XY)`.

For a linear kernel this is exactly the squared Euclidean distance between the
mean EDeN feature vectors. The GraphER port computes that equivalent sparse
form, avoiding dense O(n^2) Gram matrices.

HOG-Diff caches the reference vector/statistic by `PYTHONHASHSEED`. The GraphER
port is more conservative: persistent EDeN caching is enabled only when
`PYTHONHASHSEED` is explicitly fixed, and the cache key also hashes the actual
reference labeled graphs, complexity, and bond-label mode. This avoids a stale
cache being reused for a different dataset/split.

### 2.3 HOG-Diff FCD

`get_all_metrics` caches `compute_intermediate_statistics(test, ...)`, and FCD
uses a precomputed reference representation. GraphER previously recomputed FCD
through whichever direct API was available. The updated evaluator now uses
`FCD.precalc(reference_smiles)` when supported and caches the reference object
under a content hash of the reference SMILES. It falls back to the old APIs for
compatibility.

### 2.4 GraphER behavior before this adaptation

`GraphER/scripts/evaluate_generated_molecules.py` already had two useful
features that HOG-Diff does not provide as clearly:

- full raw RDKit sanitization of every serialized generated graph;
- bounded deterministic correction with a configurable maximum number of
  correction steps and explicit correction diagnostics.

However, two protocol problems were present:

1. the headline `validity` was the *corrected* validity even though
   `validity_without_correction` was also available;
2. `nspdk_mmd` was a custom hashed rooted-neighborhood-pair proxy, not the
   EDeN NSPDK implementation used by HOG-Diff/graph-generation benchmarks.

Both are corrected in the updated code.

### 2.5 Updated GraphER molecular protocol

Default strict GraphER mode:

- `validity` = raw full RDKit validity, no correction;
- `validity_with_correction` = evaluation-only diagnostic;
- uniqueness/novelty/FCD/NSPDK use raw-valid molecules by default;
- NSPDK backend defaults to EDeN/HOG-Diff-compatible features;
- the legacy hashed proxy remains in the report and can be selected explicitly.

HOG-Diff distribution-metric reproduction mode:

```bash
PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-graphs <molecular_graphs.pkl> \
  --dataset-root outputs/datasets \
  --dataset qm9_attributed \
  --reference-split test \
  --train-split train \
  --hogdiff-compatible-metrics \
  --output-dir <evaluation_dir>
```

This switches uniqueness/novelty/FCD/NSPDK to corrected molecules and uses the
HOG-Diff EDeN bond-label convention. The headline GraphER validity remains raw
full-RDKit validity so the no-repair protocol is not weakened.

For ZINC, an additional diagnostic is available:

```bash
--nspdk-bond-label-mode categorical
```

The default `hogdiff` mode reproduces HOG-Diff's
`int(bond.GetBondTypeAsDouble())`, which collapses aromatic 1.5 to integer 1.
`categorical` instead preserves GraphER's categorical aromatic-bond id.

## 3. Spectral diffusion comparison

### 3.1 HOG-Diff spectral representation

Primary files:

- `HOG-Diff/utils/dataloader.py`: `transform_adjs`
- `HOG-Diff/utils/sde.py`: `VPSDE`, `OUBridge`
- `HOG-Diff/utils/losses.py`: `DenoisingScoreMatching`
- `HOG-Diff/utils/solver.py`: reverse predictor, Langevin corrector,
  `cal_adj_from_eig`
- `HOG-Diff/models/ScoreNet.py`: `ScoreNet`, `ConScoreNet`
- `HOG-Diff/sampler.py`: phase-1/phase-2 sampling and eigenbasis handling

`transform_adjs` constructs the combinatorial Laplacian and computes
`torch.linalg.eigh`. For Laplacian mode, HOG-Diff stores the eigenvalues and
eigenvectors in descending eigenvalue order. Molecular adjacency is a scalar
bond-order latent with single/double/triple represented as 1/3, 2/3, 1 before
forming the Laplacian.

The important modeling choice is that HOG-Diff diffuses **eigenvalues**, not the
eigenvectors. Given perturbed eigenvalues `la_t`, it reconstructs a dense
spectral state with a fixed basis:

`L_t = U diag(la_t) U^T`, followed by off-diagonal negation in Laplacian mode.

The resulting pseudo-adjacency is supplied to the score network. The network
also derives thresholded adjacency, random-walk landing features, and
shortest-path-distance one-hot features from that reconstructed spectral state.

### 3.2 HOG-Diff two-stage process

HOG-Diff has two distinct spectral stages:

1. a first model (normally VPSDE) generates the higher-order/lifted state;
2. a conditional OU/GOUB bridge model transforms that state toward the
   original graph distribution.

For the OU bridge, the endpoint marginal has mean

`m_t * x_clean + n_t * mu_source`

and a time-dependent standard deviation that is zero at both endpoints.
HOG-Diff trains a denoising score through

`(score * std + noise)^2`

for atom features and eigenvalues. Generation then uses a reverse SDE
predictor (Euler-Maruyama) and optionally a Langevin corrector.

### 3.3 HOG-Diff empirical eigenbasis conditioning

The repository sampler contains a material implementation detail: phase-2
sampling obtains `u0` from a randomly selected training dataset item and passes
that original-graph eigenbasis into the OU sampler. The evolving adjacency is
then reconstructed from generated eigenvalues and this sampled `u0`.
Consequently, the released implementation generates/denoises eigenvalues but
does not independently generate the phase-2 eigenvector basis.

This is very different from GraphER and is intentionally **not** copied.
GraphER's discrete graph at every rewiring step has its own Laplacian and hence
its own eigenvectors; no training-graph eigenbasis is required during GraphER
generation.

### 3.4 GraphER spectral path before this adaptation

Primary files:

- `src/grapher/rewiring_mlp/generic/spectral.py`
- `src/grapher/rewiring_mlp/generic/summary_diffusion.py`
- `src/grapher/rewiring_mlp/generic/spectral_data.py`
- `src/grapher/rewiring_mlp/generic/spectral_model.py`
- attributed equivalents under `rewiring_mlp/attributed/`

GraphER uses ascending combinatorial-Laplacian eigenvalues. It does not model or
hold eigenvectors fixed. For generic graphs it uses one topology spectrum; for
attributed graphs it uses two channels: topology and bond-weighted Laplacian
spectra.

Training previously sampled a Brownian endpoint-conditioned summary state:

`x_p = (1-alpha_p) x_source + alpha_p x_clean + sigma sqrt(alpha_p(1-alpha_p)) eps`.

The predictor is an x0 predictor: it estimates the complete clean spectrum,
with positive gaps and trace normalization enforcing zero first eigenvalue,
nondecreasing eigenvalues, and the degree-fixed trace. Generation does not run
a continuous reverse SDE. Instead the predicted clean summary ranks valid
rewiring candidates, so the actual graph trajectory remains in the constrained
discrete graph space.

### 3.5 Adapted OU marginal in GraphER

`SummaryDiffusionConfig` now supports:

```yaml
summary_diffusion:
  bridge: ou_bridge
  ou_num_scales: 500
  ou_schedule: linear
  ou_eps: 0.005
  spectral_sigma: 0.20
```

The implementation ports the coefficient construction from HOG-Diff's
`OUBridge.marginal_prob` but maps time to GraphER's convention:

- GraphER progress 0 = HH/base source;
- GraphER progress 1 = clean data graph.

The sampled state is still only a continuous summary. GraphER-specific
projections are retained:

- lambda_1 noise can be fixed at zero;
- zero-mean noise on lambda_2:n can keep the spectral trace exactly fixed;
- attributed topology and bond-weighted channels are handled independently.

The HOG/GOUB coefficient arrays are cached with `lru_cache`, which is important
for GraphER's streaming dataset because otherwise an N=200--800 schedule would
be rebuilt for every sampled training state.

The OU change is deliberately optional. Existing
`bridge: brownian_endpoint_conditioned` configs remain backward compatible.
The HOG-inspired configs isolate the spectral change: graphlet CLR diffusion
remains Brownian so the ablation does not simultaneously change both guidance
channels.

## 4. What was deliberately not ported

### Fixed-eigenvector adjacency reconstruction

Not ported. A matrix reconstructed as `U diag(lambda_t) U^T` is generally dense
and does not obey GraphER's exact degree sequence, simplicity, or connectivity
constraints. It is useful inside HOG-Diff's continuous score model but should
not become GraphER's actual generated state.

### Euler-Maruyama/Langevin reverse SDE

Not ported into the graph state. Direct continuous eigenvalue updates followed
by adjacency reconstruction would bypass the rewiring kernel. A future
constrained analogue could use stochastic candidate selection/correction inside
the valid swap set, but the state itself should remain a valid graph.

### Training-set eigenbasis sampling

Not ported. GraphER should retain the stronger property that generation does
not require an eigenbasis taken from a training graph.

### HOG higher-order lifted graph as GraphER source

Not ported as a source graph because a HOG lifted/filtered graph generally does
not share the degree invariant that GraphER must preserve. Higher-order
information can instead enter as predicted graphlet/spectral guidance.

## 5. Recommended next experiments

The adaptations above are low-risk because they do not change GraphER's
constrained state space. The next HOG-derived ideas should be evaluated as
separate ablations rather than merged immediately:

1. **OU vs Brownian marginal, same model and budgets.** This is now directly
   supported by the supplied configs.
2. **Noise/SNR-aware denoising objective.** HOG-Diff's DSM weights prediction
   through the marginal standard deviation. GraphER currently uses clean-x0
   Smooth-L1 at all times. A principled score/noise auxiliary head could improve
   high-noise training, but it should be added only together with a defined
   constrained reverse-projection rule.
3. **Spectral-state structural features.** HOG-Diff feeds random-walk landing and
   shortest-path features from its noised spectral pseudo-adjacency. A safe
   GraphER variant could derive analogous features from a *conditioning-only*
   spectral reconstruction or from the current valid rewired graph, without
   replacing the graph state itself.
4. **Stochastic constrained corrector.** Translate Langevin correction to a
   temperature-controlled distribution over valid swaps (GraphER already has
   softmax/sample selection machinery) rather than adding continuous noise to
   adjacency entries.

## 6. New ablation configurations

- `configs/experiments/grapher/community_small_topology_spectral_graphlet_v2_hogdiff_ou.yaml`
- `configs/experiments/grapher/ego_small_topology_spectral_graphlet_v2_hogdiff_ou.yaml`
- `configs/experiments/grapher/qm9_attributed_spectral_graphlet_hogdiff_ou.yaml`

The OU schedule parameters mirror the corresponding HOG-Diff configs where
meaningful:

- Community-small: N=500, linear theta schedule;
- Ego-small: N=200, cosine theta schedule;
- QM9: N=800, linear theta schedule.

The parent GraphER `spectral_sigma` is retained for controlled comparison. HOG's
`max_sigma` is not copied directly because its molecular spectral latent is
constructed from bond weights divided by three, whereas GraphER uses its own
mean-degree-normalized topology/bond-weighted spectral channels.

### Community-small

```bash
PYTHONPATH=src python scripts/train_topology_grapher.py \
  --config configs/experiments/grapher/community_small_topology_spectral_graphlet_v2_hogdiff_ou.yaml \
  --output-dir outputs/topology_grapher/community_small_spectral_graphlet_v2_hogdiff_ou/seed_42 \
  --seed 42 \
  --device gpu

PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config configs/experiments/grapher/community_small_topology_spectral_graphlet_v2_hogdiff_ou.yaml \
  --output-dir outputs/topology_generation/community_small_spectral_graphlet_v2_hogdiff_ou/seed_42 \
  --num-generate 1024 \
  --seed 42 \
  --device gpu
```

### Ego-small

```bash
PYTHONPATH=src python scripts/train_topology_grapher.py \
  --config configs/experiments/grapher/ego_small_topology_spectral_graphlet_v2_hogdiff_ou.yaml \
  --output-dir outputs/topology_grapher/ego_small_spectral_graphlet_v2_hogdiff_ou/seed_42 \
  --seed 42 \
  --device gpu

PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config configs/experiments/grapher/ego_small_topology_spectral_graphlet_v2_hogdiff_ou.yaml \
  --output-dir outputs/topology_generation/ego_small_spectral_graphlet_v2_hogdiff_ou/seed_42 \
  --num-generate 1024 \
  --seed 42 \
  --device gpu
```

### QM9 attributed

```bash
PYTHONPATH=src python scripts/train_attributed_grapher.py \
  --config configs/experiments/grapher/qm9_attributed_spectral_graphlet_hogdiff_ou.yaml \
  --output-dir outputs/attributed_grapher/qm9_spectral_graphlet_hogdiff_ou/seed_42 \
  --seed 42 \
  --device gpu

PYTHONPATH=src python scripts/run_attributed_grapher.py \
  --config configs/experiments/grapher/qm9_attributed_spectral_graphlet_hogdiff_ou.yaml \
  --output-dir outputs/attributed_generation/qm9_spectral_graphlet_hogdiff_ou/seed_42 \
  --num-generate 1024 \
  --seed 42 \
  --device gpu
```

Use the same evaluation command as the Brownian run. For cross-codebase HOG-Diff
metric compatibility add `--hogdiff-compatible-metrics`; for the strict GraphER
paper protocol, omit it.

## 7. Files changed

Core changes:

- `scripts/evaluate_generated_molecules.py`
- `src/grapher/rewiring_mlp/evaluation/eden.py` (ported evaluator dependency)
- `src/grapher/rewiring_mlp/evaluation/molecular_nspdk.py`
- `src/grapher/rewiring_mlp/generic/summary_diffusion.py`
- `src/grapher/rewiring_mlp/generic/spectral_data.py`
- `src/grapher/rewiring_mlp/attributed/spectral_data.py`

Tests:

- `tests/test_evaluate_generated_molecules.py`
- `tests/test_summary_diffusion_training.py`
- `tests/test_attributed_spectral_graphlet_diffusion.py`

Ablation configs are listed in Section 6.


## External baseline wrapper status (2026-09-04)

The external HOG-Diff baseline is now implemented under
`src/grapher/models/hog_diff/`. It uses isolated two-stage training/sampling,
GraphER split projection, raw pre-correction molecular export, immutable
checkpoint/source validation, and the common baseline artifact contract. See
`docs/HOG_DIFF_WRAPPER.md` for setup and commands.
