# Attributed log-gap GSDM: masking fix and constrained decoding

Updated 4 October 2026, based on `graphes(20261004-003802).zip`.

## What changed

The Laplacian log-gap / PPGN / random-walk and shortest-path architecture is
unchanged. Generic generators, other attributed branches, the existing experiment
YAML files, and frozen dataset settings are retained. This update applies the
code-audit recommendations to `gdsm_simple/attributed_loggap.py`; it does not
replace this branch with the earlier HH/rewiring manuscript design.

1. Training samples one categorical mask for each undirected edge and mirrors
   the Boolean mask before one-hot encoding. A masked bond no longer contains
   half of its ground-truth one-hot label. Missing clean edges remain legitimate
   unknown-category targets, false-positive current edges receive MASK, and
   nonedges/padding receive zero categorical inputs.
2. Node and bond loss terms return differentiable zero when no unknown labels
   are selected. They do not switch to copying visible labels. Logs include
   supervised-target counts so zero-support accuracy values are identifiable.
3. Three generation-only decoder modes are available: `none`, `atom_degree`,
   and `atom_bond_valence`. Each keeps the same two neural passes, no no-edge
   category, and no topology editing.
4. Independent topology and attribute random-number streams are the new
   attributed-generation default. Per-sample hashes of all decoder inputs and
   serialized topology checks enable a verified paired comparison.
5. `--strict-raw-metrics` rejects correction flags that would change the primary
   distribution-metric population. Explicit raw-valid plus HOG-Diff-compatible
   flags now raise an error instead of silently choosing corrected molecules.
6. Evaluations include overlapping failure flags, an explicitly prioritized
   primary classification, connectedness, and atom/bond marginals over **all**
   generated graphs, including invalid draws.

See [the explicit commands](ATTRIBUTED_MASKFIX_VALENCE_COMMANDS.md) and
[the validation report](ATTRIBUTED_MASKFIX_VALIDATION.md).

## Compatibility and training provenance

**Train a fresh checkpoint with a new run ID for final results.** Do not resume
or reuse a legacy training run as evidence of leakage-free training. The new
state and training manifest contain:

```text
categorical_training_contract: undirected_mask_no_visible_target_fallback_v2
```

Checkpoint tensor shapes and the existing checkpoint format are unchanged.
Legacy checkpoints load for exploratory decoder comparisons but emit a warning
and are labeled `legacy_unverified` in the generation manifest. The training
reuse guard does not silently return a pre-fix checkpoint. Existing exact
structure-target caches remain usable: those clean targets were not affected
by the corruption bug.

The existing epoch budget and final-configured-epoch checkpoint selection are
unchanged. Decoder temperatures, constraint mode, and ordering should be chosen
using validation results; the update makes no claim that they are optimal.

## Decoding controls

```yaml
sample:
  separate_attribute_rng: true
attributed:
  decode:
    node_mode: sample
    edge_mode: sample
    node_temperature: 1.0
    edge_temperature: 0.6
    two_pass: true
    constraint_mode: atom_bond_valence
    edge_order: random
    infeasible_policy: retain
```

`none` is the unconstrained reference (A). `atom_degree` adds degree-compatible
atom selection only (B). `atom_bond_valence` additionally reserves and enforces
bond-order capacity (C). Argmax remains available. Temperature scaling is
applied after masking, and each undirected bond is sampled once then mirrored.
`edge_order` supports seeded `random` and `lexicographic`; random is recommended
for the pilot and is recorded per graph. Lexicographic order is an ordering
sensitivity control, not a claim of permutation invariance.

### Supported constraint representation

The optional hard masks are deliberately restricted to atomic numbers 6, 7,
8, 9 (neutral C/N/O/F), integer bond orders 1, 2, 3, and implicit hydrogens. A
single-bond category must be present. Constrained modes reject other configured
vocabularies instead of guessing valences for them. The capacity table is C=4,
N=3, O=2, F=1. This follows the neutral-element valence convention documented in
[the RDKit Book](https://www.rdkit.org/docs/RDKit_Book.html#valence-calculation-and-allowed-valences).

This model branch does not generate formal charges. It is not a charged-atom,
aromatic-bond, or general ZINC valence solver. Charged or unsupported graphs are
marked outside the neutral diagnostic scope by the evaluator.

### Atom and bond feasibility

An atom category is allowed only when its capacity is at least the binary
degree already supplied by the generated topology. Once atoms are selected,
reserve one capacity unit for **every** incident edge before assigning any bond
order. The extra budget is `capacity(atom_i) - degree_i`. A bond of order `b`
consumes `b - 1` units of extra budget at both endpoints.

For feasible atom assignments, this leaves a single bond available for every
remaining edge and guarantees that the sum of incident bond orders does not
exceed the declared capacity. Constraints are applied during sampling, not by
lowering bond orders after sampling an invalid molecule.

A topology with degree above every supported atom capacity has no feasible
assignment. With the default `infeasible_policy: retain`, the atom draw at that
node is left unconstrained, incident bonds that have no extra capacity receive
single bonds, and the graph is returned with an explicit failure flag. No edge
is removed and the sample is not dropped or replaced. `infeasible_policy: error`
is an optional fail-fast diagnostic, not the benchmark default.

This is a sequential constrained decoder, not exact sampling from the original
independent categorical model conditioned on validity. Edge ordering can affect
the output distribution. Better valence feasibility does not imply better FCD,
NSPDK, typed graphlets, or complete chemical validity.

## Scope of guarantees

Attribute decoding preserves the sampled binary topology and its unweighted
node degrees; generation asserts this on the serialized graph. On feasible
neutral inputs, mode C also satisfies the declared weighted-valence bounds.
It does not guarantee a prescribed degree sequence throughout spectral
diffusion, connectivity, all RDKit sanitization conditions, or chemical utility.
All modes preserve the complete requested-sample denominator.

## Paired comparison and new artifacts

Keep the same checkpoint, topology sampling settings, generation seed, batch
size, sample count, precision, hardware, and software environment for A/B/C.
The topology Torch generator uses the generation seed; the attribute generator
uses `(generation_seed + 104729) % (2**63 - 1)`. The NumPy donor-basis stream is
separate. A different decoder therefore cannot consume the next topology
batch's random draws.

`sample.separate_attribute_rng: false` restores the legacy shared RNG arrangement
for historical investigations. Such a run must not be assumed paired with other
decoders merely because seeds match. New RNG separation changes exact samples
relative to the original archive; compare only explicitly matched runs.

New `topology_pairing.json` records each sample's digest over `sample_x`, soft
adjacency, flags, eigenbasis, spectral state, eigenvalues, and binary topology.
`verify_gdsm_decoder_pairing.py` checks these digests, sample counts, checkpoint
identity, and actual indexed topology loaded from the graph pickle. It exits
nonzero on a mismatch. These files are verification metadata, **not replayable
state caches**; the supplied commands rerun diffusion for each variant. Identical
hashes are checked, rather than assuming cross-device determinism.

`attribute_prediction_diagnostics.json` records per-graph constraint activation,
bond-allocation order, capacity failure flags, connectedness, and complete
sample accounting. Confidence values retain their existing untempered,
unmasked head-probability interpretation; they are not constrained-posterior
confidence estimates. Graphlet summaries remain auxiliary predictions, not
post-decoding exact graphlet counts.

## Raw evaluation and migration

Use `base_graphs.pkl`, not a valid-only SMILES file, for a complete validity
denominator. Primary commands set `--strict-raw-metrics`,
`--metric-molecule-source raw_valid`, and explicitly select all connected
induced graphlets of sizes 3, 4, 5. The historical evaluator defaults
(`simple_cycle`, sizes 3–6) are intentionally retained for other workflows;
old ring-only rows must be labeled separately from all-connected-graphlet rows.

The evaluator still computes charge-projection and correction diagnostics, but
under the strict flag they are not the primary FCD/NSPDK/uniqueness/novelty/
graphlet population. Corrected-molecule comparisons remain opt-in secondary
reports. The EDeN backend can be selected independently from correction flags.
Explicit `--fcd-use-corrected` without strict mode remains possible and warns
when it creates a mixed-population report.

Failure flags can overlap. For a mutually exclusive count, the evaluator uses
this priority: raw valid; empty topology; outside neutral scope; topology
infeasible; atom-degree incompatibility; bond-valence excess; other conversion
or sanitization failure. Disconnectedness is separate because a disconnected
molecular representation can pass toolkit sanitization. Marginals cover all
generated graph records, not just the valid subset.

## Experiment sequence

Use one corrected seed-42 checkpoint and 1,024 validation-pilot samples per
variant. Verify pairing before comparing raw validity, fidelity metrics,
constraint activations, marginals, and invalidity classes. Report the raw-valid
population size for FCD and NSPDK; these can differ despite identical topology
batches. Select settings using validation data. Then freeze the settings and
run independent training seeds 42, 43, 44, generating 10,000 molecules per final
molecular run against test references. Full prepared training splits and typed
graphlet orders 3–5 are retained. The 20,000-graph vocabulary-discovery limit is
not a cap on the number of training graphs.

No measured improvement on QM9 or ZINC is asserted by this source update.
