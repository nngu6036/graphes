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

The generic commands evaluate both the spectral graph before rewiring and the final diversified graph in one report. Metrics include degree, clustering, orbit, spectral and graphlet MMD, WL novelty, exact-isomorphism novelty, isomorphism uniqueness, and support-frequency TV distances.

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
