# GraphER

GraphER is the final spectral graph generator in this repository. The implementation combines:

- full Laplacian log-gap spectral diffusion;
- learned structural summaries (degree, connected induced graphlets of orders 3--5, and orbit summaries);
- random-length constraint-preserving rewiring;
- fixed-topology categorical atom/bond diffusion for molecular datasets; and
- strict same-bond-type stochastic rewiring for attributed graphs, additionally preserving each node's neighbor-atom-type-by-bond-type signature and per-node weighted bond valence.

The final workflow uses five independent seeds: **41, 42, 43, 44, and 45**.

## Repository layout

```text
commands/                         executable end-to-end commands
configs/datasets/                 frozen dataset protocols
configs/experiments/grapher_final/ final model configs (4 datasets x 5 seeds)
configs/evaluation/               generic-graph evaluation configs
scripts/run_grapher_baseline.py   training/generation entry point
scripts/evaluate_graph_generation_report.py
scripts/evaluate_generated_molecules.py
scripts/evaluate_qed.py
src/grapher/models/grapher/       GraphER implementation
```

## Environment

Run commands from the repository root and expose the package through `PYTHONPATH=src`. GPU commands default to `cuda:0`; override with `DEVICE=...`. ORCA defaults to `/home/quang/orca/orca.out`; override with `ORCA_EXEC=...`.

## Prepare datasets

Prepare all four datasets with:

```bash
export QM9_SDF=/path/to/gdb9.sdf
export QM9_UNCHARACTERIZED=/path/to/uncharacterized.txt
export ZINC_SMILES=/path/to/250k_rndm_zinc_drugs_clean_3.csv
export ZINC_TEST_INDICES=/path/to/valid_idx_zinc250k.json
commands/prepare_datasets.sh
```

Equivalent individual commands are:

```bash
PYTHONPATH=src python scripts/prepare_generic_dataset.py --dataset community_small --root outputs/datasets
PYTHONPATH=src python scripts/prepare_generic_dataset.py --dataset ego_small --root outputs/datasets

PYTHONPATH=src python scripts/prepare_qm9_dataset.py \
  --config configs/datasets/qm9.yaml --source sdf \
  --sdf-file "$QM9_SDF" --uncharacterized-file "$QM9_UNCHARACTERIZED" \
  --root outputs/datasets

PYTHONPATH=src python scripts/prepare_zinc_dataset.py \
  --config configs/datasets/zinc.yaml \
  --smiles-file "$ZINC_SMILES" --test-indices-file "$ZINC_TEST_INDICES" \
  --root outputs/datasets
```

## Train, generate, and evaluate

Each command trains, generates, and evaluates seeds 41--45.

```bash
commands/community_small.sh
commands/ego_small.sh
commands/qm9.sh
commands/zinc.sh
```

Run everything sequentially with:

```bash
commands/run_all.sh
```

Useful overrides:

```bash
DEVICE=cuda:1 REFERENCE_SPLIT=val commands/community_small.sh
N=10000 REFERENCE_SPLIT=test commands/qm9.sh
```

Use the validation split while selecting settings, then freeze the design and evaluate once on test.

### Generic graphs

The generic commands evaluate both the spectral graph before rewiring and the final diversified graph in one report. In addition to degree, clustering, orbit, spectral, and graphlet MMD, the evaluator now reports four complementary diversity quantities: (1) exact-isomorphism novelty against the complete training split, (2) exact isomorphism-class uniqueness, (3) mean/median nearest-training graded structural distance, and (4) an epsilon-novelty / held-out-coverage curve. The graded distance is a permutation-invariant WL/component structural pseudodistance in [0,1], intended to distinguish near-copies from genuinely larger departures; it is not exact graph-edit distance. Legacy WL-hash novelty and exact class-frequency TV diagnostics are retained for backward comparison.

### Molecular graphs

The molecular diversification stage is stricter than ordinary same-bond rewiring.  A candidate swap is accepted only when it preserves, for every node, the multiset of `(neighbor atom type, bond type)` incidences.  Because the swap also uses the same bond type on both removed and inserted edges, ordinary degree, typed degree, bond-type counts, and the per-node weighted bond-valence vector are exact invariants.  With `rdkit_candidate_filter: true`, every changed candidate must also pass RDKit sanitization; a raw-valid source therefore cannot become invalid through rewiring.

The QM9 and ZINC commands evaluate both `pre_refinement_graphs.pkl` and final `base_graphs.pkl`. Each population receives:

- validity, uniqueness, novelty, NSPDK, FCD, and attributed graphlet MMD; and
- RDKit QED statistics, QED yield, threshold yields, and Wasserstein distance to the selected reference split.

QED outputs are saved under each molecular evaluation directory in `qed/qed_metrics.json`.

## Output paths

Training and generation artifacts are stored under:

```text
outputs/baselines/grapher/<dataset>/<run-id>/
```

The generated population uses:

```text
pre_refinement_graphs.pkl   graph before random rewiring
base_graphs.pkl             final diversified graph
```

## Final configs

Only final-design experiment configs are retained:

```text
configs/experiments/grapher_final/
  community_small_seed_{41..45}.yaml
  ego_small_seed_{41..45}.yaml
  qm9_seed_{41..45}.yaml
  zinc_seed_{41..45}.yaml
```

## Generation-only eigenbasis perturbation ablation

GraphES can now perturb the sampled training Laplacian eigenbasis without
retraining.  The perturbation is a product of small Givens rotations acting
only on non-constant Laplacian modes; the constant zero mode is fixed and
orthogonality is preserved up to floating-point error.

Two variants are available under `sample.eigenbasis_perturbation.stage`:

- `reverse`: perturb the basis before reverse diffusion and retain that basis
  throughout the complete reverse trajectory;
- `decode`: run the complete reverse process in the original sampled basis,
  then perturb the basis only for final continuous-Laplacian reconstruction
  and thresholding.

Example generation-only override:

```yaml
grapher:
  sample:
    eigenbasis_perturbation:
      enabled: true
      stage: decode       # reverse or decode
      method: givens
      pairing: adjacent
      sigma: 0.025
      num_rotations: 4
      max_abs_angle: 0.25
      seed_offset: 730001
      preserve_zero_mode: true
  graphlet_refinement:
    enabled: false
```

The perturbation uses a dedicated RNG stream, so the `decode` variant with the
same generation seed retains the same donor-basis indices, diffusion noise and
generated eigenvalues as the unperturbed run.  This makes it a paired test of
spectral realization alone.  The `reverse` variant uses the same donor choices
and base diffusion-noise stream but changes the matrix-valued reverse states, so
its denoised spectrum may also change.

Each perturbed generation writes
`eigenbasis_perturbation_diagnostics.json`, including the Givens pairs and
angles for every graph, Frobenius basis change, orthogonality error and
constant-zero-mode error.  The continuous eigenvalue multiset is unchanged by
`decode` perturbation before thresholding; the spectrum of the final binary
graph can still change because thresholding is nonlinear.

A validation sweep for generic graphs is provided by:

```bash
DATASET=community_small SEED=42 RUN=<existing-trained-run> \
  bash commands/eigenbasis_perturbation_generic.sh
```

The script evaluates both stages over `sigma` in
`{0.005, 0.01, 0.025, 0.05, 0.1}` using the existing MMD and graded diversity
report.  For QM9/ZINC, use `commands/eigenbasis_perturbation_molecular.sh`;
molecular rewiring is explicitly disabled so the experiment isolates basis
perturbation.

## Stochastic degree-constrained spectral realization

The generic Laplacian model now supports a generation-time alternative to hard
thresholding under `sample.degree_constrained_realization`. The decoder
projects the generated Laplacian diagonal to an indexed graphical degree
sequence, then solves a Gumbel-perturbed maximum-weight binary realization
problem under exact degree constraints. Optional connectivity is enforced by
iterative cut generation. This stage is part of topology decoding and does not
use post-generation rewiring.

The paired outputs are:

```text
pre_realization_graphs.pkl                 deterministic threshold control
base_graphs.pkl                            stochastic exact-degree realization
projected_realization_degree_sequences.pkl indexed degree targets
degree_constrained_realization_diagnostics.json
```

Community-small configs and the end-to-end command are:

```text
configs/experiments/grapher_degree_constrained_realization/
commands/community_small_degree_constrained_realization.sh
```

Run validation first:

```bash
REFERENCE_SPLIT=val commands/community_small_degree_constrained_realization.sh
```

See `STOCHASTIC_DEGREE_CONSTRAINED_SPECTRAL_REALIZATION.md` for the objective,
constraints, limitations, and diagnostics.
