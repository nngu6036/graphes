# Degree-exact spectral topology with bond-only categorical labels

## What changes

This is an **opt-in model variant**, selected by
`extensions.attributed_categorical.topology.mode: spectral_degree`.
The old `categorical` mode and its old checkpoint shapes remain supported.
Train a new denoiser under a new run ID. Removing the no-edge output class and
changing the training topology process are not compatible with simply loading
an old joint edge/no-edge checkpoint.

The uploaded implementation uses eigenvalues of the **binary adjacency matrix**,
normalized by `sqrt(n)`. This change deliberately retains that convention; it
does not silently switch the model to a graph Laplacian.

The split is now:

1. An explicit binary topology determines which edges exist.
2. Spectral scores and optional structural guidance change that topology only
   through valid degree-preserving swaps.
3. On generic graphs, the only edge label is deterministically one: there is no
   edge classifier and no categorical edge sampling.
4. On attributed graphs, the edge classifier predicts only real bond categories
   on current or newly proposed edges. It cannot create or delete an edge.

Zero is still the storage/input sentinel for absent pairs, padding and the
matrix diagonal. It is **not** an output category of the bond classifier.
Serialized graph vocabulary indices therefore remain compatible with the
existing graph decoders and molecular evaluators.

## Why a hard topology constraint is necessary

Eigenvalues alone do not identify the adjacency or its indexed degrees. For
example, the 5-node star and a 4-cycle plus an isolated node both have adjacency
eigenvalues `[-2, 0, 0, 0, 2]`, but their sorted degrees are respectively
`[1,1,1,1,4]` and `[0,2,2,2,2]`. This counterexample is also a regression test.

With full eigenvectors and eigenvalues an exact symmetric adjacency is
`A = U diag(lambda) U^T`. A predicted spectrum combined with the current basis
only yields a real-valued proposal, not necessarily a binary graph. Independent
thresholding does not preserve its row sums.

The new decoder instead stays in

`Omega(d) = {A: A=A^T, diag(A)=0, A_ij in {0,1}, A 1=d}`.

It starts with a Havel-Hakimi realization of the **indexed** degree sequence.
Optional random switches diversify this realization without changing degrees.
For connected profiles, components are first merged using degree-preserving
switches; an impossible connected sequence raises a clear error. No degree
rounding, edge dropping, or largest-component filter is used.

Given the existing degeneracy-aware spectral proposal `S`, the decoder makes
bounded local improvements to `sum_{i<j} A_ij S_ij` inside `Omega(d)`. Because
`sum_ij A_ij` is fixed, this is equivalent to improving `||A-S||_F^2` subject to
those constraints. **The optimizer is local, not an exact global projection or
an exact inverse of the predicted spectrum.** A state with no accepted swap is
still a valid degree-exact state.

## One generation step

The implemented order is:

1. Encode the noisy graph and spectral state once. Predict the clean spectrum,
   structural summaries, node categories and real-bond probabilities on current
   edges, sharing the existing graph/spectral backbone.
2. At the configured structural-guidance events, optionally apply same-type
   structural swaps to the current noisy graph.
3. At the separately configured topology events, improve spectral pair scores
   through binary degree-preserving swaps. Bond types cannot constrain edge
   existence in this decoder.
4. Query the bond head on the selected topology, reusing the encoded graph and
   spectral features rather than rerunning the transformer. Structural targets
   remain unchanged by this output query.
5. Sample node categories as before. For edges that existed in the previous
   topology, use the real-bond reverse posterior. A newly introduced edge has
   no previous bond observation, so use a fresh time-s marginal mixture:

   `p_birth(b_s) = alpha_bar[s] * p_theta(b_0) + (1-alpha_bar[s]) * m_bond`.

   At `s=0`, this is the predicted clean bond distribution. Never pass a
   no-edge sentinel to the bond chain as a genuine observed category.
6. Sample each existing unordered edge once, mirror the result, and force every
   other entry to zero. Check the support and indexed degree invariants.
7. Recompute eigenpairs from the realized binary topology and perform the
   existing source-centred spectral update/feedback.

The structural-guidance stage precedes the final bond draw in this variant.
Consequently, `final_pre_rewire_graphs.pkl` and the final graphs coincide here;
the manifest explicitly explains this compatibility artifact. In legacy mode
that artifact keeps its original meaning.

## Training changes and limitations

* Bond marginals are fitted using **real training edges only**. The legacy
  all-pair edge marginal remains recorded for provenance but is not the bond
  noise distribution.
* Topology is corrupted by a finite number of random degree-preserving switch
  attempts proportional to time and edge count. The spectral encoder never
  receives the clean graph as its input topology merely to mask bond labels.
* Existing corrupted edges receive real-bond noise. Newly introduced topology
  edges receive marginal bond noise, without invented clean bond targets.
* Bond cross-entropy is supervised only on real clean edges, including clean
  edges missing from the corrupted topology. Such pairs are queried **after**
  graph/spectral encoding. The clean query mask does not enter that encoder,
  the spectral prediction, or structural-summary pooling. Tests check this
  non-leakage property and permutation equivariance.
* The topology corruption is finite-swap augmentation. Its terminal topology
  is not claimed to be uniformly mixed or exactly equal in law to the
  randomized Havel-Hakimi initializer. The local topology decoder is not an
  analytically exact reverse kernel for that augmentation. This is a
  degree-constrained guided sampler, not a proof of exact joint diffusion
  sampling or improved graph-distribution quality.

## Exactly what is, and is not, preserved

Guaranteed by construction and checked during generation:

- Node count; simple, symmetric, loop-free binary topology.
- Every node's sampled **ordinary degree**, at initialization and every step.
- Bond sampling leaves the selected binary topology unchanged.
- Connectedness when `topology.require_connected=true` and its enforced
  connectivity-preserving transition settings are used.

Not guaranteed by ordinary degree preservation:

- Atom identities, which remain categorical variables in this implementation.
- Per-node typed degrees (counts of single/double/triple bonds).
- Bond-order sums, chemical valence, or RDKit validity.
- Exact realization of predicted spectra or graphlet summaries.
- Uniform exploration of all graphs with the requested degrees.
- Improved degree MMD: the output degree distribution is the degree prior's
  distribution, so an inaccurate prior cannot be corrected by this sampler.

For example, changing one single bond to a double bond preserves the number of
neighbors but changes its endpoints' bond-order sums. Stronger molecular
constraints require a joint atom/typed-degree or valence-constrained assignment;
independent bond recoloring cannot promise those invariants.

## Configuration and compatibility

Twenty standalone YAMLs are supplied in:

`configs/experiments/gdsm_spectral_degree_explicit/`

They cover `community_small`, `ego_small`, `qm9`, `zinc`, each with seeds
`41,42,43,44,45`, matching the existing uploaded final experiment configurations.
Training budgets, model sizes, graphlet orders 3/4/5, dataset settings and
seed-specific ordinary-degree-prior checkpoint paths are retained. The ordinary
degree prior can be reused when trained on the same frozen training split; the
spectral/categorical denoiser must be retrained.

The new topology parameters are explicitly written in each YAML:

```yaml
topology:
  mode: spectral_degree
  require_connected: true
  train_swap_attempts_per_edge: 10.0
  initial_swap_attempts_per_edge: 10.0
  every: 10
  start_fraction: 1.0
  max_steps_per_event: 2
  proposal_budget: 128
  valid_candidate_budget: 64
  preserve_connectivity_if_connected: true
  min_improvement: 1.0e-08
```

These are explicit initial experiment settings, not benchmark-tuned values.
The terminal step is always considered for topology decoding; other events use
`every` and `start_fraction`. Degrees stay exact between topology events too.
Set `topology.every: 1` to attempt decoding at every reverse step, at additional
computational cost. Setting `max_steps_per_event: 0` disables spectral swaps;
setting `guidance.enabled: false` disables only the separate structural swaps.
Neither ablation enables categorical insertion/deletion.

`guidance.weights.edge` is explicitly zero: a conditional bond probability is
not an edge-existence score. Configuration validation rejects a nonzero value
in the new mode, rather than silently making no-edge energy comparisons.

The legacy extension switches `degree_conditioning`, `hh_initialization` and
`degree_preserving_rewiring` stay false. They control an older, different code
path. The new exact topology is controlled only by the new `topology` section.

## Files

- `categorical/topology.py`: exact initializer, finite-swap topology corruption,
  connectivity handling and constrained spectral-score topology updates.
- `categorical/degree_step.py`: topology-first reverse step and edge-birth logic.
- `categorical/noise.py`: masked real-bond sampling and bond-only noise functions.
- `categorical/model.py`: K-real-bond output head, sparse unordered-edge queries,
  generic-graph bypass, masked bond loss and cached-feature re-query.
- `categorical/pipeline.py`: training, marginals, initialization, sampling,
  checkpoint validation, manifests and per-step invariant assertions.
- `categorical/config.py`, `refiner.py`, `evaluation.py`: explicit validation,
  bond/existence energy separation and stronger output/trajectory audits.
- `tests/test_gdsm_spectral_degree.py`: 58 tests for the new mode/configuration
  combinations, including generic and attributed train/generate integration.

All source paths above are under `src/grapher/models/gdsm_simple/`.

## Running and auditing

See `SPECTRAL_DEGREE_BOND_ONLY_COMMANDS.md` for explicit commands using the existing
`scripts/run_gdsm_simple_baseline.py` entry point. No new launcher rewrites or
silently selects model parameters.

The generated `rewiring_diagnostics.json` should report:

```json
{
  "prior_degree_preservation_rate": 1.0,
  "initial_degree_preservation_rate": 1.0,
  "indexed_prior_degree_preservation_rate": 1.0,
  "categorical_degree_change_steps_mean": 0.0,
  "no_edge_output_category": false,
  "typed_degree_preservation_guaranteed": false
}
```

`no_edge_output_category: false` means there is no such classifier output. It
does not mean the serialized matrix has no zeros. `indexed_degree_checks_per_graph`
is `sample.steps + 1`, counting initialization. Save trajectories to additionally
verify all recorded intermediate states with `audit_gdsm_categorical.py`.

## Validation performed

CPU regression command:

```bash
PYTHONPATH=src pytest -q \
  tests/test_gdsm_spectral_degree.py \
  tests/test_gdsm_categorical.py \
  tests/test_gdsm_categorical_multiscale.py \
  tests/test_gdsm_categorical_eigh.py \
  tests/test_gdsm_final_addon.py \
  tests/test_gdsm_simple.py \
  tests/test_gdsm_structure3.py \
  tests/test_gdsm_option_a.py -rs
```

Result: **303 passed, 5 skipped**. The five skips require a real CUDA device.
The test summary is retained in `SPECTRAL_DEGREE_TEST_RESULTS.txt`.
Small CPU end-to-end generic and attributed fixtures were trained and sampled;
all initial, intermediate and final indexed-degree checks passed. No full
Community-small, Ego-small, QM9 or ZINC benchmark training was performed here,
and no GPU speed, molecular-validity or MMD/FCD improvement is claimed.
