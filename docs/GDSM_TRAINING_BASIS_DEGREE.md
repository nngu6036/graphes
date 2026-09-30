# Fixed training-eigenvector proposals with exact ordinary-degree constraints

## Scope of this implementation

This change implements the **generation-only experiment** discussed for the
uploaded `graphes(20260930-065815).zip` source. It reuses the existing spectral
and bond-only denoiser and the existing degree-preserving topology decoder.
It does not restore a learned generic edge/no-edge head.

The decisive change is the source of the spectral proposal matrix:

```text
previous sampler: U = eigenvectors(current discrete graph), updated each step
new sampler:      U = one same-size training-bank eigenbasis, fixed throughout
```

The new sampler uses `S_t = U diag(sqrt(n) * z_hat_0(t)) U^T`, where the existing
model predicts ascending **binary-adjacency eigenvalues divided by sqrt(n)**.
It then searches for a simple graph with the sampled indexed ordinary degrees
whose edges score well under `S_t`. This is an adjacency representation, not a
Laplacian representation.

The training forward corruption, loss functions, model parameter shapes and
checkpoint format remain unchanged. In particular, training still uses the
current corrupted graph's basis when conditioning the pair/summary layers.
**The fixed-training-basis sampler is therefore an explicit conditioning shift
at generation, not a newly trained exact GSDM reverse SDE.** Existing bond-only
v2 checkpoints can run this ablation without retraining. Its quality must be
measured; the correctness tests do not establish improved MMD.

## Algorithm

1. Sample an ordinary degree sequence `d` from the configured existing prior.
   Preserve its indexed assignment throughout the generated trajectory.
2. Sample one `n x n` eigenvector matrix from the checkpoint's same-size training
   reservoir. Keep its stored row ordering and ascending-eigenvalue column
   ordering. Do not take eigenvectors from validation/test graphs.
3. Use that same donor basis in the existing `degree_anchor` routine to obtain
   the soft spectral anchor. Initialize the spectral latent as anchor plus
   Gaussian noise. This anchor is a ridge/trace/moment heuristic, **not an exact
   binary-degree realization**; the noisy latent need not have row sums `d`.
4. Construct a feasible indexed Havel–Hakimi graph and apply the existing random
   switch warm-up. These operations, rather than the eigenvalue anchor, establish
   the exact ordinary degrees. The supplied profiles also require connectedness.
5. At each reverse step, predict a clean normalized spectrum and form the score
   matrix with the **same donor U**. Mask its diagonal and padding. Do not use
   current-graph eigenvalue degeneracy blocks to average donor coefficients.
6. Apply the existing bounded score-improving 2-switch decoder. Each allowed
   switch preserves degrees and simplicity; disconnecting switches are rejected
   when connectivity preservation is configured.
7. For generic graphs, present edges receive their single possible label. For
   attributed graphs, sample real bond categories on the decoded support only.
   Absent pairs are not a category in the bond head. Existing surviving-edge and
   newly-created-edge bond posterior rules are retained.
8. Run the existing optional structural-guidance events, which preserve the same
   ordinary degrees. Advance the spectral latent with the existing reverse step.
   Never replace the donor U with the new graph's eigenvectors.
9. Compute the **actual final graph** eigenpairs once for saved output/audit.
   They are stored separately from the sampled decoder basis and generally
   differ from it.

For frozen scores, the decoder approximately maximizes
`sum_{i<j} A_ij S_ij` subject to `A 1 = d`, symmetry, zero diagonal, binary entries
and optional connectedness. It is a finite local search, **not** an exact global
maximum-weight factor solver. Feasibility is exact even if no improving switch
is found. The final graph need not have the predicted eigenvalues or donor basis.
Optional structural guidance may subsequently change this frozen-score objective.

## Configuration and compatibility

The opt-in fields under `gdsm_simple.extensions.attributed_categorical` are:

```yaml
spectrum_feedback: 0.0
save_trajectory: false
save_degree_trajectory: true
topology:
  mode: spectral
  decoder: degree_preserving
  basis_source: training_bank
  threshold: 0.5
  max_swaps_per_step: 2
  proposal_budget: 128
  initial_random_swaps_per_edge: 4
  preserve_connectivity: true
```

`threshold` is retained for compatibility and is unused by the constrained
switch decoder. Other settings, including the existing degree-generator path,
remain explicit in the complete YAML profiles.

`basis_source: current_graph` remains the default, so replacing the source code
alone does not silently change existing experiments. `basis_source: training_bank`
requires spectral/bond-only mode, `decoder: degree_preserving`, degree-basis
initialization and `spectrum_feedback: 0.0`.

The feedback requirement is deliberate: the sorted eigenvalues of a projected
current graph are not its coefficients in the fixed donor basis. This version
**does not** feed those values back into donor coordinates. A separately designed
basis-aware feedback rule would be a different experiment.

Thirteen complete profiles are supplied in
`configs/experiments/gdsm_training_basis_degree_explicit/`:
Community-small, Ego-small, QM9 and ZINC for seeds 42, 43 and 44, plus a
Community-small seed-42 current-basis control. Full training splits, graphlet
orders 3/4/5, original training budgets, degree-prior checkpoints and projection
budgets are unchanged. The new profiles disable feedback and enable compact
indexed-degree traces. The control differs from its fixed-basis counterpart in
**only `topology.basis_source`**; feedback is zero in both arms.

Existing `gdsm_spectral_topology_bond_only_checkpoint_v2` checkpoints are accepted.
Old checkpoints lacking the newly added provenance/default fields are also
supported: their saved bank is used directly and missing source-graph indices
are recorded as null, not invented. Original edge/no-edge categorical v1
checkpoints remain incompatible with this bond-only architecture.

A fresh training run is also supported, but it trains the same denoiser objective.
Use a new run ID; do not overwrite the completed checkpoint merely to test the
new generation rule. See `GDSM_TRAINING_BASIS_COMMANDS.md` for both workflows.

## Training-bank selection and provenance

The bank already existed in the supplied checkpoint preparation code. It is a
per-size reservoir built only while reading training graphs. The supplied
profiles retain at most 32 bases per node count. The new sampler is **uniform
over the retained same-size reservoir**, not over every graph in the entire
training dataset at inference.

A missing size is an explicit error. There is no Haar fallback, resizing,
truncation, validation/test lookup, or switch back to the current-graph basis.
The requested size prior must be supported by the checkpoint bank.

New training checkpoints additionally record each retained donor's training-row
index, indexed degrees, normalized eigenvalues and basis digest. Reservoir
replacement updates the basis and its metadata together. Existing checkpoints
without this metadata retain their original bank, with null training-row IDs.
Node rows are left in stored donor order; degree-nearest donor selection and
node-assignment optimization are **not** added by this first ablation.

## Saved artifacts and audits

The new mode adds:

- `sampled_training_eigenvectors.pkl`: the fixed decoder basis for each returned
  graph; do not confuse it with `final_eigenvectors.pkl`.
- `sampled_training_basis_records.pkl`: donor size/index, content hash, checkpoint
  and training-split provenance, row-order convention and fallback status.
- `degree_trajectories.pkl`: timesteps and indexed-degree vectors at initialization
  and after every reverse transition, when `save_degree_trajectory: true`.

The existing `predicted_summaries.pkl` additionally stores each final fixed-basis
spectral proposal. Existing graph outputs, final graph eigenpairs and molecular
output aliases remain compatible with the evaluators.

`audit_gdsm_categorical.py` now verifies the sampled basis's orthogonality/digest,
its exact membership in the referenced checkpoint bank when the checkpoint is
available, the final proposal's reconstruction from that donor and the predicted
spectrum, all saved degree vectors, and the existing bond-support/final-graph
checks. Full graph trajectories are cross-checked against the degree vectors
when both are saved. These are distinct audit scopes: degree vectors are not
independent full-adjacency snapshots.

A portable generation folder may no longer have access to its original
checkpoint. In that case the audit reports `sampled_basis_matches_checkpoint:
null` and `sampled_basis_checkpoint_check: checkpoint_unavailable`; it does not
claim to have verified checkpoint membership. Available artifact and proposal
checks still run.

Key new fields for a successful fixed-basis run are:

```text
fixed_training_eigenbasis: true
sampled_basis_matches_checkpoint: true
final_spectral_proposal_matches_saved_training_basis: true
recorded_basis_updates_per_graph: 1
recorded_decoder_basis_updates_after_initialization: 0
prior_indexed_degree_preservation_rate: 1.0
recorded_categorical_degree_change_steps_mean: 0.0
saved_intermediate_degree_vectors_verified: true
```

There is one recorded decoder-basis initialization, not 501 changing bases.
The main sampler computes actual graph eigenpairs only at the final step in
this mode. This does not count eigenvalue calculations performed inside optional
structural-guidance candidate scoring, and is **not** a total runtime estimate.

## Guarantees and limitations

The guarantee is **indexed ordinary degree preservation**, simplicity and the
configured connectivity constraint throughout the discrete generated graph
trajectory. The categorical bond sampler cannot add or remove support edges.

It is not a guarantee of fixed atom types throughout sampling, typed degrees,
bond-order sums, chemical valence/validity, an exact spectrum, a globally optimal
score projection, or improved generation quality. The final same-type local
rewiring step preserves its own input atom/typed-degree configuration; that
local fact does not make the whole bond-recolouring trajectory typed-degree
preserving.

The current joint denoiser still uses graph-state and structural heads. This
patch implements a GSDM-style **proposal basis** inside that existing model; it
does not replace the entire model with the paper's architecture. Training with
matched fixed-basis conditioning would require a separate training design and
controlled evaluation, not merely a claim of checkpoint compatibility.

## Files changed

- `categorical/training_basis.py`: strict same-size bank sampling, basis hashes
  and direct fixed-basis spectral reconstruction.
- `categorical/model.py`: optional `proposal_basis` input; no new parameters.
- `categorical/pipeline.py`: opt-in fixed-basis sampling, bank provenance, separate
  donor/final bases, degree traces and explicit generation/training contracts.
- `categorical/config.py`: validated generation controls with legacy defaults.
- `categorical/evaluation.py`: donor/proposal/degree-trace audit extensions.
- `tests/test_gdsm_training_basis.py`: 48 new test cases, including generic and
  attributed generation, old-v2 compatibility, source provenance, replay
  determinism, malformed inputs and semantic corruption detection.

All categorical paths above are under `src/grapher/models/gdsm_simple/`.

## Validation

The focused GDSM regression suite completed with **347 passed and 5 CUDA-only
skips**, including all 48 new cases. The complete project suite completed with
**637 passed, 8 CUDA-only skips and 2 failures**. Both failures are existing
SPECTRE epoch-budget assertions (Community-small 500 versus expected 10,000;
Ego-small 270 versus expected 10,000), reproduced against the unmodified uploaded
archive. They were not altered by this patch.

Separate CLI smoke runs used tiny synthetic generic/attributed datasets, two
training epochs, four generation transitions and three returned graphs each.
Both training/generation/audit workflows passed. These are not benchmark QM9 or
Community-small quality measurements. No full dataset training, 1,024/10,000-sample
quality experiment, GPU run or FCD/NSPDK benchmark was performed here.

See `GDSM_TRAINING_BASIS_TEST_REPORT.json` for the machine-readable test record.
