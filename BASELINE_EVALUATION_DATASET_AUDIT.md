# GraphER baseline evaluation and dataset-protocol audit

This audit compares the attached HOG-Diff, GraphRNN, DeFoG, DiGress and GraphER source trees. It distinguishes **native baseline behavior** from the **common GraphER rerun protocol**. The attached sources are the basis for every source-specific statement below.

## Executive findings

1. **The old GraphER generic evaluator is not the same as the historical GraphRNN/SPECTRE evaluator.** It used 20 clustering bins, a sample-adaptive EMD bandwidth, and converted the 15-dimensional orbit vector into a normalized histogram before applying an EMD kernel. GraphRNN, HOG-Diff Community/Ego, and DiGress/DeFoG `comm20` use 100 clustering bins, fixed EMD bandwidths, and a Gaussian L2 kernel on the mean per-node orbit vector.
2. **Community-small is not the same dataset in every repository.** DiGress and DeFoG explicitly download SPECTRE `community_12_21_100.pt`; native GraphRNN instead generates 3000 community graphs with different size/inter-community parameters. The old GraphER config regenerated another approximation rather than pinning the SPECTRE file.
3. **Ego-small also has multiple identities.** Native GraphRNN builds radius-1 Citeseer ego graphs with 4--20 nodes; DeFoG downloads EDGE `Ego.pkl`; HOG-Diff expects a preprocessed `ego_small` pickle (not included in the archive) with max node count 18. The old GraphER config rebuilt Citeseer egos with 4--18 nodes and deterministic first-selection.
4. **QM9 has no single shared split across the attached repositories.** DiGress/DeFoG split the raw 133,885 indices with seed 42 into 100,000 train, 10% test and the remainder validation *before* excluding uncharacterized records. HOG-Diff uses `valid_idx_qm9.json` as test and all remaining records as train, with no validation split. GraphER uses the 130,831 characterized pool and a clean 80/10/10 split. Therefore published molecular values are not automatically protocol-compatible with GraphER reruns.
5. **GraphER's old ZINC protocol is a fixed 12k subset, while HOG-Diff and DeFoG target full ZINC250k.** In addition, old GraphER retained aromatic bond class 4, whereas both attached HOG-Diff and DeFoG code kekulize ZINC into single/double/triple bonds.
6. **Training budget is a major confounder.** The attached DiGress and DeFoG Comm20 experiment configs specify 1,000,000 epochs. The current GraphER DiGress wrapper config uses 200,000 epochs, and prior DeFoG Community runs also used 200,000. This can independently explain weaker DiGress/DeFoG results and should be fixed before drawing ranking conclusions.

## 1. Generic graph evaluation logic

| Component | GraphRNN | DiGress | DeFoG | HOG-Diff | Old GraphER | Aligned GraphER |
|---|---|---|---|---|---|---|
| Degree descriptor | `nx.degree_histogram` | same lineage | same lineage | same | normalized degree histogram | same distribution |
| Degree kernel on Community | Gaussian EMD, sigma=1 | `compute_emd=True` -> Gaussian EMD | `Comm20SamplingMetrics(... compute_emd=True)` | Gaussian EMD for `community_small` | Gaussian EMD with **median/adaptive sigma** | Gaussian EMD, **sigma=1** |
| Clustering bins | **100** | **100** | **100** | **100** | **20** | **100** |
| Clustering kernel | Gaussian EMD, sigma=0.1, distance scaling=100 | same when `compute_emd=True` | same for Comm20 | same for Community/Ego | adaptive Gaussian EMD, no `/100` coordinate scaling | same as GraphRNN |
| Orbit descriptor | ORCA 4-node orbit totals / n | same | same | same | per-node vector is computed, then **renormalized by vector sum** | mean per-node 15-orbit vector, no histogram normalization |
| Orbit kernel | Gaussian L2, sigma=30 | same for Comm20 | same for Comm20 | Gaussian L2, sigma=30 | Gaussian **EMD**, sigma=1 | Gaussian L2, **sigma=30** |
| MMD estimator | biased `E[kxx]+E[kyy]-2E[kxy]` | biased | biased (`abs` wrapper) | biased | biased | biased |

Relevant attached paths:

- GraphRNN: `eval/stats.py`, `eval/mmd.py`
- DiGress: `src/analysis/spectre_utils.py`, `src/analysis/dist_helper.py`
- DeFoG: `src/analysis/spectre_utils.py`, `src/analysis/dist_helper.py`
- HOG-Diff: `evaluation/stats.py`, `evaluation/mmd.py`
- GraphER: `scripts/evaluate_graph_generation_report.py`, `src/grapher/rewiring_mlp/evaluation/metrics.py`

### Dataset-specific evaluator exception

DeFoG's `EgoSamplingMetrics` sets `compute_emd=False`, so its native Ego metrics use its Gaussian-TV mode rather than the GraphRNN EMD mode. HOG-Diff Ego and native GraphRNN use the EMD family. Therefore **native DeFoG Ego numbers must not be mixed directly with the common GraphER EMD table**. The common paper evaluator should use one declared protocol for every rerun; the aligned GraphER implementation uses the GraphRNN/SPECTRE EMD convention because the manuscript explicitly describes Gaussian EMD metrics.

## 2. Molecular evaluation logic

| Item | HOG-Diff | DiGress | DeFoG | GraphER |
|---|---|---|---|---|
| Raw validity | available internally | sanitization-based | sanitization-based | **headline metric: strict raw sanitization** |
| Correction/relaxed validity | `tensor2mol` defaults to MoFlow-style correction | relaxed validity path | relaxed validity path | correction is diagnostic only |
| Disconnected molecule handling | can keep largest connected component | converts to largest fragment for reported SMILES | largest fragment logic | strict raw graph is not silently replaced for headline validity |
| Uniqueness/novelty source | corrected/valid molecules in normal HOG pipeline | based on relaxed-valid set | based on relaxed-valid set | strict-valid by default; HOG-compatible mode available |
| NSPDK | EDeN NSPDK, complexity=4 | not present in attached core evaluator | not present in attached core evaluator | EDeN-compatible implementation; `--hogdiff-compatible-metrics` available |
| FCD | `fcd_torch` | not part of attached core RDKit metrics | code exists but `molecular_metrics.py` forcibly sets `compute_fcd=False` | `fcd_torch`; can use strict-valid or corrected set |

Conclusion: the attached molecular evaluators do **not** share one validity convention. GraphER should retain strict raw validity because that is the manuscript's stated metric, while optionally reporting HOG-compatible corrected-distribution metrics as a secondary compatibility row/diagnostic.

## 3. Community-small dataset/config comparison

### Native GraphRNN

`create_graphs.py` handles `community*` by:

- choosing each community size once from 12--17;
- generating **3000** graphs;
- using `p_inter=0.01`;
- then `main.py` shuffles with seed 123 and uses 80% train / 20% test;
- validation is the first 20% of the complete shuffled list, hence it overlaps training.

This is **not** the SPECTRE Community-small artifact used by DiGress/DeFoG.

### DiGress

- dataset name `comm20`;
- downloads SPECTRE `community_12_21_100.pt`;
- experiment config: 1,000,000 epochs, batch size 256, 8 layers;
- intended split formula is 64/16/20 for a 100-graph dataset.

The attached loader has two suspicious legacy issues: `self.num_graphs=200` despite the `..._100.pt` filename, and `process()` appends each graph to `data_list` twice. These should not be copied into the common GraphER protocol.

### DeFoG

- dataset name `comm20`;
- downloads the same SPECTRE `community_12_21_100.pt`;
- attached Comm20 experiment: 1,000,000 epochs, batch size 256, 8 layers;
- splitter also declares `self.num_graphs=200` for Comm20 and uses seed 1234.

### HOG-Diff

- `configs/cs.yaml`: `name: community_small`, `max_node: 20`, `test_split: 0.2`;
- the raw generic graph pickle is not included in the attached archive, so its exact graph identities cannot be proven from the attachment alone;
- loader treats the first 20% of the stored list as test and the remainder as training.

### Old GraphER

- regenerated 100 stochastic-block graphs locally;
- total size 12--20 but equal two-block construction yields even sizes;
- `p_in=0.70`, cross-edge expectation controlled by `p_inter=0.05`;
- split 70/10/20, seed 0.

This is a different graph population from the explicit DiGress/DeFoG SPECTRE artifact.

### Updated GraphER recommendation

- pin SPECTRE `community_12_21_100.pt` directly;
- verify exactly 100 graphs;
- use a clean, disjoint 64/16/20 split with seed 0;
- do not reproduce the attached DiGress duplicate-append or hardcoded-200 bugs;
- evaluate all rerun methods with the common GraphRNN/SPECTRE metric protocol.

## 4. Ego-small dataset/config comparison

### Native GraphRNN

- Citeseer largest connected component;
- radius-1 ego graph around every node;
- retain 4--20 nodes;
- shuffle and retain 200;
- 80/20 train/test; validation overlaps training.

### DeFoG

- downloads EDGE `Ego.pkl`;
- number of graphs is read from the artifact;
- seed 1234;
- train 80%, test 20%; validation is the first 20% and therefore overlaps training;
- native evaluator uses Gaussian-TV rather than Comm20's EMD setting.

### HOG-Diff

- `configs/ego.yaml`: `max_node: 18`, `test_split: 0.2`;
- raw preprocessed pickle is not included, so exact identity cannot be proven from the archive;
- first 20% stored graphs are test.

### DiGress

No Ego-small dataset config is shipped in the attached repository.

### Old GraphER

- rebuilds from Citeseer;
- radius 1, 4--18 nodes;
- deterministic `selection: first`, 200 graphs;
- 70/10/20 split, seed 0.

### Updated GraphER recommendation

- pin EDGE `Ego.pkl` rather than independently rebuilding Citeseer egos;
- use a clean disjoint 64/16/20 split, seed 1234;
- keep the common paper evaluator independent of DeFoG's native TV exception.

## 5. QM9 dataset/config comparison

### DiGress / DeFoG

Both attached QM9 dataset implementations:

- start from the raw 133,885 QM9 records;
- set `n_train=100000`;
- set `n_test=int(0.1 * n_samples)`;
- assign the remainder to validation;
- use `dataset.sample(frac=1, random_state=42)`;
- **then** skip the official uncharacterized indices during processing;
- remove hydrogens in the no-H experiment;
- native no-H training config: 1000 epochs, batch size 1024, 9 layers.

Therefore final processed split counts are not the same as a direct 80/10/10 split of the 130,831 characterized molecules.

### HOG-Diff

- `max_node=9`, four atom channels;
- `load_smiles()` reads `valid_idx_qm9.json` as test and uses the complement for train;
- source comment states 133,885 total, 120,803 train, 13,082 test;
- no validation split in that loader;
- molecule preprocessing kekulizes;
- evaluation requests 10,000 samples and metrics validity/uniqueness/novelty/FCD/NSPDK.

### GraphER

- explicitly excludes 3,054 official uncharacterized records, yielding 130,831 characterized graphs;
- uses a deterministic 104,665/13,083/13,083 80/10/10 split, seed 42;
- heavy atoms only, kekulized;
- C/N/O/F node vocabulary;
- raw-validity is the primary metric.

### Recommendation

Do **not** silently switch the common QM9 split to one baseline's native split: the attached baselines disagree. Keep the clean GraphER common split for all reruns, but mark it `rerun_only` and do not copy published molecular numbers into the same table unless their source split/evaluator is proven compatible. Use a separate native-reproduction run when auditing a baseline implementation.

## 6. ZINC dataset/config comparison

### HOG-Diff

- full `zinc250k`;
- max 38 nodes, 9 atom channels;
- test identities from `valid_idx_zinc250k.json`, remainder train;
- no validation split in the HOG loader;
- `Chem.Kekulize` is applied in preprocessing;
- evaluation requests 10,000 molecules.

### DeFoG

- full ZINC250k source;
- `remove_h=True`, `aromatic=False`, max-node behavior consistent with the 38-node benchmark;
- explicit `Chem.Kekulize(..., clearAromaticFlags=True)`;
- attached experiment: 300 epochs, batch size 256, 12 layers;
- uses a ZINC validation-index file; parts of the attached split-writing code deserve independent verification before treating it as a gold reference.

### DiGress / GraphRNN

No native ZINC configuration is included in the attached DiGress or GraphRNN source.

### Old GraphER

- fixed 12,000-sample ZINC250k subset;
- 10,000/1,000/1,000 train/val/test;
- preserves aromatic bonds as category 4 (`kekulize=false`).

### Updated GraphER recommendation

Keep the 12k subset if it is the declared paper benchmark, but make the incompatibility with full-ZINC250k published values explicit. Align the graph representation with HOG-Diff/DeFoG by setting:

```yaml
kekulize: true
retain_aromatic_bonds: false
edge_categories: [1, 2, 3]
```

This change requires regenerating the ZINC dataset and retraining molecular models.

## 7. Changes implemented in the updated GraphER tree

### Generic evaluation

`evaluate_graph_generation_report.py` now supports:

```bash
--generic-mmd-protocol graphrnn
--generic-mmd-protocol graphes_adaptive
```

`graphrnn` is the default and uses fixed historical benchmark kernels. `graphes_adaptive` preserves the old GraphER behavior only for backward diagnostics. The selected protocol is written into the JSON report.

### Community-small

`configs/datasets/community_small.yaml` now pins SPECTRE `community_12_21_100.pt` and uses disjoint 64/16/20 splitting.

### Ego-small

`configs/datasets/ego_small.yaml` now pins EDGE `Ego.pkl` and uses disjoint 64/16/20 splitting with seed 1234.

### Generic source loading

`src/grapher/data/builders.py` now supports pinned `spectre_pt` and `networkx_pickle` sources and allows `split.seed` to be specified explicitly.

### QM9

The actual GraphER common split is retained, but the config now states a protocol ID, `rerun_only` comparison scope, and the attached native split differences.

### ZINC

The fixed 12k scope is retained, but preprocessing is changed to kekulized single/double/triple bonds to align with attached HOG-Diff and DeFoG representations. `prepare_zinc_dataset.py` now supports this representation.

## 8. Recommended verification order

### A. Re-score existing outputs first — no retraining

Use the updated evaluator on your current generated graph files:

```bash
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/grapher/community_small_topology_graphlet.yaml \
  --generated-dir "$GEN_DIR" \
  --generic-mmd-protocol graphrnn \
  --output-dir "$GEN_DIR/evaluation_graphrnn_protocol"
```

Run this on the existing GraphRNN, DiGress, and DeFoG batches. This isolates the evaluator effect.

For a backward comparison only:

```bash
--generic-mmd-protocol graphes_adaptive
```

### B. Fix training budget before model-quality conclusions

For Community-small, native attached DiGress and DeFoG configs specify 1,000,000 epochs. Re-run at that horizon (or explicitly justify a smaller common optimization budget) before concluding that GraphRNN is superior.

### C. Rebuild canonical Community/Ego datasets — retraining required

```bash
PYTHONPATH=src python scripts/prepare_generic_dataset.py \
  --dataset community_small \
  --root outputs/datasets

PYTHONPATH=src python scripts/prepare_generic_dataset.py \
  --dataset ego_small \
  --root outputs/datasets
```

Because these commands change graph identities/splits relative to your old prepared data, archive the old `outputs/datasets/sbm` and `outputs/datasets/ego_small` first if you need to reproduce previous results.

### D. Rebuild ZINC only when ready to rerun molecular models

The updated ZINC config changes aromatic representation. Existing ZINC checkpoints are not compatible with the new prepared categorical state.

## 9. Test status

Focused metric/dataset/wrapper tests pass. The complete runnable suite passes with the two pre-existing missing-script collection tests excluded:

- 256 tests passed
- 585 subtests passed
- 1 warning

The two existing collection blockers are unrelated to this audit:

- missing `scripts/run_defog_grapher.py`
- missing `scripts/run_research_protocol.py`
