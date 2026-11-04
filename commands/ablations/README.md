# Separate ablation studies by dataset

Each command file runs one controlled comparison from the latest spectral
GraphES ablation. The shared implementation is `commands/ablation.sh`; model and
sampling parameters remain in the explicit YAML files under
`configs/experiments/grapher_paper_ablation/`.

## Command files

| Dataset | Study | Command file | Compared outputs |
|---|---|---|---|
| Community-small | Primary decoder × rewiring | `commands/ablations/community_small/primary.sh` | T, D, TR, DR |
| Community-small | Decoder temperature | `commands/ablations/community_small/decoder_temperature.sh` | D, D0 |
| Community-small | Rewiring budget | `commands/ablations/community_small/rewiring_budget.sh` | D, R1, R12, DR |
| Community-small | Summary supervision | `commands/ablations/community_small/summary_supervision.sh` | D, S0 |
| Ego-small | Primary decoder × rewiring | `commands/ablations/ego_small/primary.sh` | T, D, TR, DR |
| Ego-small | Decoder temperature | `commands/ablations/ego_small/decoder_temperature.sh` | D, D0 |
| Ego-small | Rewiring budget | `commands/ablations/ego_small/rewiring_budget.sh` | D, R1, R12, DR |
| Ego-small | Summary supervision | `commands/ablations/ego_small/summary_supervision.sh` | D, S0 |
| QM9 | Attributed rewiring | `commands/ablations/qm9/rewiring.sh` | T, TR |
| ZINC | Attributed rewiring | `commands/ablations/zinc/rewiring.sh` | T, TR |

T is threshold decoding. D is stochastic exact-degree decoding with temperature
0.4. D0 uses temperature 0 with the same saved degree targets. TR and DR apply
rewiring to the corresponding saved attributed or generic endpoint. The budget
study includes 0 (D), 1 (R1), uniform 1--2 (R12), and the saved per-graph DR
budget. Current generic matched YAMLs allow DR budgets of 1--8 steps despite
`B1to4` appearing in the training run names. Exported diagnostics retain the
actual saved requested budgets; TR uses those same per-graph budgets when
rewiring T. Run names do not override the recorded sampling settings.

S0 is a separately trained checkpoint with the total topology-summary loss
multiplier set to zero. D and S0 both use the stochastic exact-degree decoder
with no rewiring; the comparison does not use DR as its reference.

The current molecular generator supports T/TR. Molecular D, D0 and DR scripts
are intentionally absent because the generic exact-degree decoder has not been
integrated into that generator. A feasibility analysis is given below.

## Examples

From the repository root, for illustration using Community-small:

```bash
bash commands/ablations/community_small/primary.sh
bash commands/ablations/community_small/decoder_temperature.sh
bash commands/ablations/community_small/rewiring_budget.sh
bash commands/ablations/community_small/summary_supervision.sh
```

The corresponding Ego-small scripts have the same filenames under `ego_small/`.
For the attributed comparison:

```bash
bash commands/ablations/qm9/rewiring.sh
bash commands/ablations/zinc/rewiring.sh
```

Each file exposes its dataset, study, seeds, sample count, device and stage
switches. Defaults are seeds 41–45, 1,024 graphs for each generic dataset and
10,000 molecules for each molecular dataset. The reference checkpoint is reused
unless training is requested. Model settings are unchanged from the preceding
implementation.

```bash
# Select the three seeds used in an earlier experiment protocol.
SEEDS='42 43 44' bash commands/ablations/community_small/primary.sh

# Request training when the reference checkpoints have not been created.
DO_TRAIN=1 bash commands/ablations/ego_small/primary.sh

# Inspect the planned operations without executing jobs or writing outputs.
DRY_RUN=1 SEEDS='42' bash commands/ablations/qm9/rewiring.sh

# Evaluate saved populations without launching generation or S0 training.
DO_GENERATE=0 bash commands/ablations/community_small/summary_supervision.sh
```

Existing `DATASET_ROOT`, `OUTPUT_ROOT`, `REPORT_ROOT`, `ORCA_EXEC`, `N`, `DEVICE`,
`REFERENCE_SPLIT` and `RUN_TEMPLATE` overrides remain available. File paths
containing spaces are supported. Verified reuse uses the Linux `flock` utility
to serialize preparation of shared generations and checkpoints.

## Reference checkpoints for the 200-epoch protocol

The final and ablation YAMLs require **200 training epochs**, including the
separately trained S0 control. The default run IDs match the current main
commands for seeds 41--45:

| Dataset | Reference training run ID (`{seed}` is the seed) |
|---|---|
| `community_small` | `seed_{seed}_community_small_grapher_final_tau0p4_B1to4_g345_v2` |
| `ego_small` | `seed_{seed}_ego_small_grapher_final_tau0p4_B1to4_g345_v2` |
| `qm9` | `seed_{seed}_qm9_grapher_final_B1to4_g345_v2` |
| `zinc` | `seed_{seed}_zinc_grapher_final_B1to4_g345_v2` |

The earlier Community-small `*_grapher_final_g345_v1` runs recorded 10,000
training epochs. Selecting those runs produces
`training.train.epochs: expected 200, recorded 10000`. A final checkpoint trained
for 10,000 epochs cannot be reused as the required 200-epoch checkpoint. The
verifier continues to reject that mismatch and leaves the existing artifacts
untouched. `TRAIN_EPOCHS` is unsupported; a nonempty value is rejected. Clear a
previously exported value before rerunning:

```bash
unset TRAIN_EPOCHS
```

When the current 200-epoch runs are absent, create them under the v2 names with:

```bash
DO_TRAIN=1 bash commands/ablations/community_small/primary.sh
```

This creates missing reference checkpoints in separate runs from the old v1
artifacts and reuses compatible completed v2 runs. Training runs for the full
200 epochs; the recorded final checkpoint epoch must also equal 200. If an
existing v2 run is incompatible, select an unused `RUN_TEMPLATE` to create a
separate 200-epoch run:

```bash
RUN_TEMPLATE='seed_{seed}_{dataset}_grapher_final_{profile}_g345_v3_200epochs' \
  DO_TRAIN=1 bash commands/ablations/community_small/primary.sh
```

The verifier also requires a matching training configuration, dataset identity,
train/validation/test split fingerprints, and artifact hashes. `DO_TRAIN=1` and
`OVERWRITE=1` do not bypass those checks for incompatible saved runs under
verified reuse. The YAML budgets, saved manifests and existing checkpoints are
not rewritten to make an incompatible run pass.

The new sampled population uses the independent generation ID
`paper_ablation_v1_seed_<seed>_n_<N>` beneath the selected existing run.
`RUN_TEMPLATE` remains available for an explicitly different training run.

```bash
DRY_RUN=1 SEEDS=42 bash commands/ablations/community_small/primary.sh
```

## Shared populations, independent study reports

The first generic study prepares the shared seven-branch generation (T, D, TR,
DR, D0, R1 and R12). Each study evaluates and reports only its selected branches.
The subsequent study files reuse that saved population after validating its
provenance and requested IDs. They do not create independent reverse trajectories
for each comparison. A summary-supervision run separately prepares S0 as needed.

### Resume a completed generic population after sealing fails

An earlier generic producer completed reverse sampling but omitted the matched
branch export and embedded graph sample IDs. A subsequent seal could therefore
reject `base_graphs.pkl` at sample `41:0` after all 1,024 graphs had been saved.
The missing matched export protocol also prevented the remaining branch checks
from succeeding.

Rerun the same study command with the same settings:

```bash
DO_TRAIN=1 bash commands/ablations/community_small/primary.sh
```

With verified reuse, `DO_TRAIN=1` reuses the compatible completed 200-epoch
checkpoint; it trains only when the checkpoint is absent.

The preparation step uses the saved threshold graphs (T), exact-degree graphs
(D), final graphs (DR), continuous endpoints, projected degrees and summaries.
It exports the missing controls from those endpoints and resumes verification
without repeating reverse sampling. Existing parent pickles and the generation
manifest retain their original bytes; matched provenance and branch records are
written separately to `matched_ablation_manifest.json` and
`ablations/ablation_manifest.json`. No overwrite, new generation tag or epoch
change is needed for this recovery; the training budget remains 200 epochs.

When an older generation manifest lacks effective sampler settings, recovery
requires the original `.ablation_generation_intent.json` written by `begin` for
that generation. Its configuration digest, file hash and training context must
match the requested operation. Missing or conflicting evidence rejects recovery
instead of inferring settings from the current YAML.

Unlabelled T, D and DR graphs are associated with requested sample IDs only by
the saved list order anchored to the verified parent file hashes and matched
provenance. An embedded ID that conflicts with this order still fails. Derived
TR, D0, R1 and R12 outputs carry explicit IDs and failure records; count, order,
checkpoint, dataset and artifact checks remain mandatory.

### Resume a completed molecular population after sealing fails

Earlier QM9 and ZINC producers also saved unlabelled graph lists. A completed
10,000-molecule population could then fail sealing at sample `44:0`. With the
original generation intent retained, rerun the study under its original settings
and tag. For the completed QM9 seed 44 population, export, seal and evaluate the
saved outputs with:

```bash
SEEDS=44 DO_GENERATE=0 bash commands/ablations/qm9/rewiring.sh
```

The ordinary command also resumes this recovery when generation is enabled.
Preparation exports T from `pre_refinement_graphs.pkl` and TR from
`base_graphs.pkl` as byte-identical copies. It records their parent hashes and
ordered sample identities in `molecular_ablation_manifest.json` and the identical
`ablations/ablation_manifest.json`, then seals the population for verified reuse.
The existing graph pickles, generation manifest and trained checkpoint retain
their original bytes. This recovery performs no neural training, reverse
sampling, molecule correction or replacement draws, and preserves the requested
10,000 outputs, including invalid molecules.

Recovery requires the original `.ablation_generation_intent.json` to prove the
configuration file hash, effective sampling settings and training/data context.
The recorded 200-epoch checkpoint, decode modes and temperatures, attribute
diffusion steps, RNG policy and rewiring settings must match. Unlabelled graphs
receive positional identities only through verified parent hashes; conflicting
embedded IDs, altered order, counts or hashes reject reuse. Keep the same seed,
sample count, run template, dataset root and generation tag when retrying.

### Exact-degree connectivity solver

A generation that stopped with `remained disconnected after 16 connectivity-cut
rounds` reached the cut-iteration cap while solving the connected exact-degree
problem. At that cap, the solver now uses a single-commodity flow formulation
to enforce connectivity with the same indexed degrees, edge scores and sampled
Gumbel perturbation. The requested graph and its optimization objective remain
fixed. Diagnostics record the connectivity method and the number of cut rounds.

After updating, retry the same study command with the same settings. Completed
checkpoints and saved populations remain available for verified reuse. Truly
infeasible degree targets and solver timeouts remain errors; failed outputs
retain their existing failure accounting.

All generic studies use the same generation ID by default:

```text
paper_ablation_v1_seed_<seed>_n_<N>
```

Reports have distinct study directories:

```text
outputs/ablations/grapher_paper/<dataset>/<study>/n_<N>/evaluation_<split>/
```

Each selected branch's evaluator report is also study-specific, under
`evaluation_<split>/<study>/`, so a later study cannot overwrite another study's
evaluation status or measurements. The new entry files default to `REUSE=1`.

The legacy all-studies command remains available as `commands/ablation.sh` and
keeps its previous aggregate-report location.

Reuse checks compare the selected saved outputs, IDs, requested counts, hashes,
checkpoint and data identity, and the available effective sampling/refinement
settings. Invalid or incompatible existing outputs are reported; they are not
silently replaced. Recorded partial D0 outputs remain partial. A failed optional
D0 does not block a primary study that does not select it.

With `REUSE=1`, `DO_TRAIN=1` creates a missing checkpoint and reuses a matching
completed one. Incompatible artifacts are rejected even if `OVERWRITE=1` is set.
To request a separate population, use `GENERATION_TAG`; the corresponding S0
override is `S0_GENERATION_TAG`. `REUSE=0` explicitly selects the legacy runner's
creation/overwrite behavior instead of verified reuse.

Existing generations are verified against their saved effective configuration
and parent/branch manifests. An original pending generation intent supplies the
missing provenance needed to complete export and sealing after the failures
described above. A molecular population with neither a valid reuse receipt nor
its matching original intent remains unverified and is retained for inspection.
Verified recovery uses the existing population and generation tag.

The summarizer records per-seed paired differences, explicit missing or failed
controls, and separate report locations for each study.

## Verification

Regression tests use isolated synthetic checkpoints, split fingerprints, graph
populations and matched sidecars to verify strict 200-epoch reuse, parent hash
and ID association, changed provenance, corrupt or reordered artifacts, and
recovery intent checks. Tiny saved-endpoint export tests exercise branch
creation without reverse diffusion. A tiny evaluator run also checks the
dataset-root override, recorded generic validity bounds, and their effect on
the reported validity rate. Molecular fixtures prepare and seal unlabelled T/TR
parents, preserve their bytes, and reject conflicting identities, settings and
provenance. A wrapper regression confirms recovery proceeds with generation
disabled and never calls the neural runner. These fixtures and shell syntax
checks do not constitute a full training run or benchmark validation.

## Can exact-degree decoding be applied to attributed graphs?

Yes, as a decoder of the **binary support before attribute generation**. This is
a proposed extension; the command-file update does not enable it.

The reusable interface is `degree_constrained_realization.realize_graph`.
`hybrid_attributed_loggap.generate` currently thresholds the continuous endpoint
and then passes that support to `_attribute_diffusion_decode`. Replacing that
threshold step with exact-degree realization is compatible with the conditional
attribute model. The attribute decoder recomputes degree features and a binary
Laplacian eigensystem from the supplied support.

The pipeline would first project the continuous Laplacian's diagonal to a
feasible ordinary-degree vector, then realize a binary D topology with that
vector. It would generate attributes on that fixed support and finally apply
the optional attributed rewiring.

Categorical generation would preserve the selected binary degree vector because
it assigns real bond categories only to existing edges. Existing strict
attributed rewiring could then preserve the resulting ordinary degrees, atom
labels, bond-type degrees, neighbor-type/bond-type signatures and bond valence.

### Ordinary degrees and typed degrees are different constraints

An ordinary degree specifies how many neighbors a node has. A typed degree also
specifies how many incident edges have each bond category. Weighted valence sums
the corresponding bond orders. The generic solver constrains the ordinary degree
only; it cannot prescribe typed degrees or valence before categorical generation.
Prescribing those quantities beforehand requires a typed realization optimizer
with bond-category variables and additional constraints.

### A molecular projection needs chemical bounds

The current generic projector permits degrees up to `n-1`. For example, the
six-node complete graph is simple, connected and graphical, but every node has
degree five. None of the current QM9 endpoint decoder's allowed neutral atom
types can support that degree: its configured capacities are C=4, N=3, O=2 and
F=1, as defined in `src/grapher/models/grapher/attribute_decoding.py`.

A molecular adaptation should bound the projected degrees using the supported
atom/charge vocabulary and make the trace-derived edge-count target consistent
with those bounds. Atom/bond masks remain necessary after projection. Degree
bounds alone do not establish validity under the desired sampled atom types,
bond assignments, aromaticity rules or dataset-specific sanitization policy.

QM9's final configuration enables `constraint_mode: atom_bond_valence`. ZINC's
final configuration uses `constraint_mode: none` and includes additional atom
types. The current neutral C/N/O/F endpoint mask cannot simply be treated as a
general ZINC valence constraint. ZINC therefore needs a vocabulary-appropriate
policy for a chemically constrained realization claim.

### Retraining and comparison design

A first decoder-only experiment can use the existing checkpoint: its spectral
model already produces the required continuous endpoint, and its attribute
model already conditions on binary support. Better validity or distributional
quality must still be measured. The attribute model is trained on clean training
topologies; changing the decoder changes its inference-time conditioning
distribution. Any later retraining should be a separate experiment.

A molecular four-way study should share the spectral endpoint between T and D,
then generate attributes separately conditional on each topology. Their atom and
bond assignments should not be claimed identical. TR must start from its exact
saved attributed T sample, and DR from its exact saved attributed D sample,
without attribute resampling within those pairs. Attribute randomness should be
isolated per requested ID/branch, with matched rewiring budgets assigned per ID.

The attached older “Hybrid Target Summary” manuscript describes a degree/typed
signature generator and a Havel–Hakimi or typed source constructor. The current
spectral implementation instead obtains the ordinary target from the generated
Laplacian diagonal. These are distinct source-generation designs.
