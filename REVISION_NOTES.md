# Revision notes

## High-level changes

- Repaired package imports from the old `empirical_comparison` namespace to the local `grapher` package.
- Added missing package initializers and a working lazy registry for datasets and models.
- Centralized degree-sequence, connected Havel-Hakimi, candidate construction, graph-to-data, and double-edge-swap helpers in `src/grapher/generation/rewiring.py`.
- Replaced script-local duplicate utilities with lightweight compatibility wrappers.
- Added checkpoint loaders in `src/grapher/models/checkpoint.py` so training, sampling, and evaluation scripts share the same model-loading path.
- Filled empty model config files and renamed `grapher_generic_yaml` to `grapher_generic.yaml`.
- Updated `requirements.txt` with missing runtime libraries and removed unused heavyweight dependencies.

## Script-level changes

- `scripts/prepare_dataset.py`: keeps the existing dataset split workflow and now uses the repaired local registry.
- `scripts/train_dhvae_model.py`: trains the size-conditioned DH-VAE degree prior on degree histograms and graph-size embeddings, saving a structured checkpoint.
- `scripts/generate_dhvae_samples.py`: new/repaired degree-sequence sampler with graphicality and connected-realizability filtering.
- `scripts/train_grapher_model.py`: trains the existing GraphER scorer from target-aware teacher labels but target-free candidate sets.
- `scripts/generate_grapher_samples.py`: loads the DH-VAE checkpoint and GraphER checkpoint, constructs connected canonical sources, and samples graph rewiring trajectories.
- `scripts/evaluate_grapher_metrics.py`: evaluates graph samples and degree-sequence samples with degree, clustering, spectral, motif-proxy, structural summary, validity, connectedness, uniqueness, and novelty metrics.
- `scripts/evaluate_degree_sequence.py`, `scripts/evaluate_generic_metrics.py`, `scripts/evaluate_molecular_metrics.py`, and `scripts/eval.py`: compatibility aliases that dispatch to the unified evaluator.

## Minimal model adaptations

- `GraphER` now imports shared rewiring utilities from the package instead of from `scripts/utils.py`.
- `GraphER` has a small fallback GIN layer when `torch_geometric` is unavailable, allowing generic smoke tests to run in minimal environments.
- Legacy independent-count degree-prior checkpoints should be retrained with DH-VAE.


## Size-conditioned DH-VAE degree prior update

- Replaced the legacy independent-count degree-prior internals with the paper-aligned size-conditioned Degree-Histogram VAE.
- `src/grapher/models/model_dhvae.py` now provides `DHVAE`, explicit degree-zero histogram encoding, size embeddings, masked degree logits for `k < n`, multinomial histogram sampling, and a multinomial NLL reconstruction loss.
- `scripts/train_dhvae_model.py` now trains on `(histogram, graph_size)` pairs, stores the empirical graph-size distribution in checkpoints, and records DH-VAE-specific metadata.
- `scripts/generate_dhvae_samples.py` now samples graph sizes from the checkpoint's empirical size distribution and exposes `--temperature` for the DH-VAE categorical degree probabilities before multinomial sampling.
- `src/grapher/models/checkpoint.py` now requires DH-VAE checkpoints and gives an explicit retraining message for legacy degree-prior checkpoints.
- Updated `configs/models/dhvae.yaml`, `README.md`, and the model registry to use DH-VAE naming.

## Smoke tests performed

The revised codebase was syntax-checked and package-import checked:

```bash
PYTHONPATH=src python -m compileall -q scripts src
PYTHONPATH=src python - <<'PY'
import importlib, pkgutil, grapher
for m in pkgutil.walk_packages(grapher.__path__, grapher.__name__ + '.'):
    importlib.import_module(m.name)
print('package import ok')
PY
```

A small DH-VAE degree-prior smoke run was executed on a tiny SBM dataset:

```bash
PYTHONPATH=src python scripts/prepare_dataset.py --dataset sbm --config /tmp/graphes_dhvae_smoke/sbm_small.yaml --output-root outputs_dhvae_smoke/datasets --force
PYTHONPATH=src python scripts/train_dhvae_model.py --dataset sbm --model-config /tmp/graphes_dhvae_smoke/dhvae_small.yaml --dataset-config /tmp/graphes_dhvae_smoke/sbm_small.yaml --dataset-root outputs_dhvae_smoke/datasets --device cpu --run-id 0
PYTHONPATH=src python scripts/generate_dhvae_samples.py --dataset sbm --model-config /tmp/graphes_dhvae_smoke/dhvae_small.yaml --num-samples 4 --device cpu --run-id 0 --force --max-attempts 200
```

The smoke checkpoint stores `model_name: dhvae` and `architecture: size_conditioned_dhvae`, and the generated sequences had the sampled graph size exactly by construction.

## Ego-CiteSeer dataset update

- Added `configs/datasets/ego_citeseer.yaml` with GraphRNN-style defaults: 200 graphs, radius-3 ego extraction, and 4--18 node filtering.
- Added `src/grapher/datasets/ego_citeseer.py`, which can build from a local `ind.citeseer.graph` pickle or from `torch_geometric.datasets.Planetoid(root, name="CiteSeer")`.
- Registered `ego_citeseer` in `src/grapher/registry.py`, so all existing scripts accept `--dataset ego_citeseer`.
- Extended `scripts/prepare_dataset.py` with `--raw-graph-path` for local CiteSeer pickle preparation and updated dataset-preparation documentation.
- Updated the Ego-CiteSeer builder to use `min(num_graphs, available_candidates)` after filtering. This avoids crashes when the config requests more ego graphs than CiteSeer can provide under the selected radius/node-count filters; `strict_num_graphs: true` restores failure-on-shortfall behavior.

## Dataset statistics script

- Added `src/grapher/datasets/statistics.py` with reusable NetworkX summary functions.
- Added `scripts/print_dataset_statistics.py` to print aggregate and split-level statistics for registered datasets and sample pickles.
- The script loads prepared `train.pkl` / `val.pkl` / `test.pkl` files, or builds them from the dataset config when missing.
- It reports graph count, node range, edge range, max degree, node/edge means, average degree, density, connectedness, clustering, transitivity, triangle counts, planarity, component diagnostics, isolate counts, degree histograms, graphicality and connected-feasibility checks, WL-hash uniqueness, and optional exact-isomorphism uniqueness.
- It also supports DH-VAE degree-sequence pickle statistics through `--input-pkl`.
- Useful flags include `--json-out`, `--csv-out`, `--raw-graph-path`, `--download-root`, `--max-graphs`, `--max-graphs-per-split`, `--strict-num-graphs`, `--rebuild`, `--save-built`, `--full`, `--include-wl-hashes`, `--include-planarity`, `--include-path-stats`, and `--include-exact-isomorphism`.

## ZINC SMILES preparation update

- ZINC preparation is now routed through `scripts/prepare_zinc_from_smiles.py` only. The generic `scripts/prepare_dataset.py --dataset zinc` path now exits with a clear error message.
- Disabled the PyG-backed `ZINCDatasetBuilder.build()` path because PyG ZINC stores categorical atom-type ids, not explicit atomic numbers. The attributed molecular pipeline needs `atomic_number`/`z` node attributes for validity checks and molecule reconstruction.
- Added `src/grapher/datasets/zinc_utils.py` with a shared preparation hint and atomic-number coverage validation.
- `save_dataset_splits(dataset="zinc", ...)` now verifies that every node has `atomic_number` or `z` before persisting split files and records `zinc_atomic_number_stats` in metadata.
- Updated `configs/datasets/zinc.yaml` to document the SMILES/RDKit source and removed stale PyG atom-type mapping comments.
- Updated `README.md`, `requirements.txt`, and statistics-script error messages to point users to `prepare_zinc_from_smiles.py` for ZINC.

## Generic GraphER complete-action revision

- Updated `GraphER` from the old anchor-edge/second-edge scorer to a complete action scorer over `RewireAction(e1, e2, orientation)`.
- Added graph, degree-sequence, normalized-time, orientation, and local structural conditioning to the generic action scorer.
- Added target-free complete-action candidate enumeration utilities in `grapher.generation.rewiring`.
- Updated `scripts/train_grapher_model.py` so target-aware candidates are used only for offline teacher selection, while neural training receives `{teacher action} + target-free negatives`.
- Updated generation so GraphER samples target-free complete rewiring actions with `action_temperature`.
- Old second-edge GraphER checkpoints now require retraining.
