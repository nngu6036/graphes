# CatFlow-Inspired Endpoint and Graphlet GraphER

This is the main revised generic-graph implementation. It does not replace the
complete graph prediction with a target-summary generator.

At every intermediate graph \(G_t\), one shared model predicts

\[
q_\theta(X_1\mid G_t,t,D),\qquad
q_\theta(E_1\mid G_t,t,D),\qquad
q_\theta(H_1\mid G_t,t,D).
\]

Here \(X_1\) and \(E_1\) are clean endpoint node and dense edge/no-edge
categorical states. \(H_1\) is a distribution over graphlet histograms plus
the connected-subgraph mass. The graphlet head is complementary: it supplies
higher-order topology that factorized edge marginals do not explicitly model.

CatFlow converts endpoint categorical marginals into a continuous marginal
velocity. This implementation instead:

1. samples an endpoint graph from the node/edge categorical probabilities;
2. samples graphlet histograms from Dirichlet heads and connected masses from
   Beta heads;
3. enumerates or samples valid double-edge swaps from the current graph;
4. scores each swap by categorical, endpoint-log-probability, and graphlet
   improvement;
5. applies one selected swap and predicts again at the next state.

The sampled endpoint is guidance only. It may have the wrong degree sequence
or be disconnected and is never installed directly. Every realized update
comes from the valid rewiring action set, which preserves degree and, when
configured, connectivity.

## Main files

| File | Purpose |
| --- | --- |
| `src/grapher/hybrid/data.py` | Aligned Havel-Hakimi teacher paths, shared relabel augmentation, categorical batches, and fixed graphlet basis |
| `src/grapher/hybrid/model.py` | Joint clean-endpoint node/edge predictor and graphlet-distribution heads |
| `src/grapher/hybrid/refiner.py` | Hybrid candidate scoring and constraint-preserving rollout |
| `scripts/train_hybrid_endpoint_grapher.py` | Training and validation |
| `scripts/run_hybrid_endpoint_grapher.py` | Generation and evaluation |
| `configs/experiments/sbm_hybrid_endpoint_graphlet.yaml` | Runnable SBM configuration |

The old `summary_generator` and GraphER-Opt route remains available as a
legacy baseline in [`LEGACY_PIPELINES.md`](LEGACY_PIPELINES.md). The hybrid
training script rejects a summary-generator block unless the configuration
explicitly marks it as a legacy baseline.

## Installation

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a PyTorch build appropriate for the machine, then install the remaining
dependencies:

```bash
python -m pip install torch
python -m pip install -r requirements.txt
export PYTHONPATH=src
```

For a CUDA system, replace the generic PyTorch command with the command for the
installed CUDA driver. PyTorch is intentionally not pinned in
`requirements.txt`.

All commands below assume that they are run from the repository root. Config,
dataset, checkpoint, and output paths are relative to that directory.

## Training procedure

The hybrid route trains one shared endpoint-and-graphlet predictor. It does
not train a rewiring-action selector and does not backpropagate through a
generation-time rewiring rollout.

### 1. Prepare the dataset

```bash
PYTHONPATH=src python scripts/prepare_generic_dataset.py \
  --dataset sbm \
  --root outputs/datasets
```

The default dataset configuration creates 500 connected SBM graphs and splits
them into 400 training, 50 validation, and 50 test graphs. Because
`dataset.build_if_missing: true`, training can build missing splits
automatically. Explicit preparation is recommended because it prints and saves
the dataset verification report.

Existing split files are reused. Re-run the preparation command after changing
`configs/datasets/sbm.yaml`.

### 2. Construct endpoint-training examples

For every training graph \(G_1\), the code constructs an aligned source
\(G_0\) whose node \(i\) has the same degree as node \(i\) in \(G_1\).
The preprocessing stage then:

1. records the labelled-node degree sequence \(D\);
2. builds a connected Havel-Hakimi source satisfying
   \(\deg_{G_0}(i)=\deg_{G_1}(i)\);
3. constructs an offline path using valid degree- and
   connectivity-preserving double-edge swaps that reduce edge disagreement
   with \(G_1\);
4. selects at most eight approximately evenly spaced intermediate states;
5. applies the same random node permutation to \(G_t\) and \(G_1\).

Each supervised example is

\[
(G_t,t,D)\longrightarrow
\left(
X(G_1),E(G_1),
H_3(G_1),H_4(G_1),
M_3(G_1),M_4(G_1)
\right),
\]

where \(X\) and \(E\) are clean endpoint node and dense edge/no-edge
categories, \(H_k\) is the normalized connected graphlet histogram, and
\(M_k\) is the connected-subgraph mass. The shared relabeling preserves node
correspondence while preventing the model from exploiting a fixed label order.

By default, the teacher searches for at most 32 swaps, examines up to 64
candidates per step, retains at most eight states per graph, and estimates each
\(k=3,4\) target using 2,048 sampled node subsets. Trajectories are built
before epoch 1, so this preprocessing can be the slowest part of a full run.

If the offline path stalls before reaching \(G_1\), the code appends \(G_1\)
only as the final \(t=1\) denoising example. It is not counted as an accepted
teacher transition.

### 3. Train the predictor

Run the full configured experiment:

```bash
PYTHONPATH=src python scripts/train_hybrid_endpoint_grapher.py \
  --config configs/experiments/sbm_hybrid_endpoint_graphlet.yaml
```

The training command reads these YAML blocks:

| Configuration block | Training role |
| --- | --- |
| `dataset` | Dataset name, split location, build behavior, and optional graph limits |
| `categorical_state` | Node categories and present-edge categories; index 0 is reserved for no-edge |
| `graphlet_prediction` | Graphlet sizes, connected-only rule, backend, and sample count |
| `endpoint_trajectory` | Aligned source construction and offline teacher-path settings |
| `endpoint_predictor` | Architecture, optimizer, loss weights, device, epochs, and checkpoint path |

`generation`, `degree_generator`, `constructor`, `hybrid_refiner`, and
`evaluation` are generation-time blocks and do not change predictor training.
The root `seed` defaults to 42 in the supplied experiment; a `--seed` argument
overrides it.

The default optimization settings are:

| Setting | Default |
| --- | ---: |
| Epochs | 100 |
| Batch size | 4 |
| Optimizer | AdamW |
| Learning rate | \(3\times10^{-4}\) |
| Weight decay | \(10^{-5}\) |
| Gradient-norm limit | 5 |
| Hidden / edge / graph dimensions | 128 / 64 / 128 |
| Message-passing layers | 4 |
| Graphlet sizes | \(k=3,4\) |
| Sampled subsets per graphlet size | 2,048 |

The loss is

\[
\mathcal L=
\mathcal L_{\mathrm{node}}+
\mathcal L_{\mathrm{edge}}+
\mathcal L_{\mathrm{graphlet\ mean}}+
0.1\mathcal L_{\mathrm{Dirichlet}}+
0.25\mathcal L_{\mathrm{connected\ mass}}.
\]

For the featureless SBM configuration, there is one node category, so
\(\mathcal L_{\mathrm{node}}=0\). The edge cross-entropy uses class weights
`0.25` for no-edge and `1.0` for a present edge. The graphlet heads learn
Dirichlet distributions over the normalized histograms, while Beta heads learn
their connected-subgraph masses.

Validation runs after every epoch. The checkpoint with the smallest validation
loss is saved. There is no early stopping, learning-rate scheduler, or resume
option in the current training CLI.

Default outputs:

```text
outputs/hybrid_endpoint/sbm/checkpoint.pt
outputs/hybrid_endpoint/sbm/training_report.json
```

The checkpoint stores the model, categorical vocabulary, graphlet basis,
summary configuration, experiment configuration, and best-epoch report.
`training_report.json` stores the full epoch history and teacher-trajectory
diagnostics.

Monitor these validation metrics:

- `loss`: weighted total validation loss used for checkpoint selection;
- `present_edge_recall`: recall on target edges;
- `edge_accuracy`: accuracy over all node pairs;
- `graphlet_mae`: error of the predicted graphlet means;
- `graphlet_mass_mae`: error of the connected-subgraph mass.

Most node pairs are no-edges, so inspect `present_edge_recall` together with
`edge_accuracy`.

### 4. Run a smoke training check

Use a small run before launching the full experiment:

```bash
PYTHONPATH=src python scripts/train_hybrid_endpoint_grapher.py \
  --config configs/experiments/sbm_hybrid_endpoint_graphlet.yaml \
  --max-train-graphs 4 \
  --max-val-graphs 2 \
  --epochs 1 \
  --batch-size 2 \
  --device cpu \
  --output-dir outputs/hybrid_endpoint/sbm/smoke
```

This writes:

```text
outputs/hybrid_endpoint/sbm/smoke/checkpoint.pt
outputs/hybrid_endpoint/sbm/smoke/training_report.json
```

If the dataset splits do not exist, the smoke command first builds the full
500-graph dataset and only then applies the four/two-graph limits.

Run the focused regression tests:

```bash
PYTHONPATH=src pytest -q \
  tests/test_hybrid_endpoint_model.py \
  tests/test_hybrid_rewiring_guidance.py
```

### 5. Training command-line overrides

| Option | Effect |
| --- | --- |
| `--output-dir DIR` | Writes `checkpoint.pt` and `training_report.json` to `DIR` |
| `--epochs N` | Overrides `endpoint_predictor.epochs` |
| `--batch-size N` | Overrides `endpoint_predictor.batch_size` |
| `--max-train-graphs N` | Limits training graphs for debugging |
| `--max-val-graphs N` | Limits validation graphs for debugging |
| `--seed N` | Overrides the experiment seed |
| `--device auto\|cpu\|cuda\|cuda:0` | Selects the PyTorch device |

Without `--output-dir`, the checkpoint is written to
`endpoint_predictor.checkpoint_path` from the YAML file and
`training_report.json` is written beside it. If a custom output directory is
used, pass its checkpoint explicitly to the generation command.

The predictor is independent of the degree generator during training. The
default generation configuration samples empirical training degree sequences;
a trained `DegreeHistogramVAE` is needed only after setting
`generation.degree_source: learned`.

## Generation-time candidate score

For a valid swap \(a\), the fixed generation-time refiner uses

\[
s(a)=
\lambda_{\mathrm{cat}}\Delta_{\mathrm{cat}}(a)+
\lambda_{\mathrm{prob}}\Delta_{\log q}(a)+
\lambda_{\mathrm{gl}}\Delta_{\mathrm{graphlet}}(a).
\]

The graphlet term includes both per-size normalized graphlet histograms and
connected-subgraph mass. Expensive graphlet evaluation can be restricted to
the categorical/probability top-\(K\). If those pre-graphlet weights are zero,
all candidates are evaluated so a tied shortlist cannot discard the best
graphlet action.

## Generate and evaluate graphs

Generate with the default checkpoint:

```bash
PYTHONPATH=src python scripts/run_hybrid_endpoint_grapher.py \
  --config configs/experiments/sbm_hybrid_endpoint_graphlet.yaml \
  --output-dir outputs/hybrid_endpoint/sbm/generated
```

Generate from the smoke-test checkpoint:

```bash
PYTHONPATH=src python scripts/run_hybrid_endpoint_grapher.py \
  --config configs/experiments/sbm_hybrid_endpoint_graphlet.yaml \
  --checkpoint outputs/hybrid_endpoint/sbm/smoke/checkpoint.pt \
  --num-generate 2 \
  --output-dir outputs/hybrid_endpoint/sbm/smoke_generated
```

Generation writes:

```text
outputs/hybrid_endpoint/sbm/generated/coarse_graphs.pkl
outputs/hybrid_endpoint/sbm/generated/hybrid_refined_graphs.pkl
outputs/hybrid_endpoint/sbm/generated/report.json
```

In `report.json`, require:

```text
degree_preservation_rate                 = 1.0
constructor_target_degree_match_rate     = 1.0
final_target_degree_match_rate           = 1.0
connectedness_rate                       = 1.0
```

The target-degree rates compare sorted degree multisets, so they are invariant
to node ordering and constructor relabeling. The separate
`predictor_sampled_endpoint_degree_match_rate` is only a guidance diagnostic:
the endpoint predictor may sample an infeasible graph, but that graph is never
installed directly. The realized graph changes only through valid double-edge
swaps.

## Scope

The supplied configuration implements generic topology graphs. Attributed
molecular generation additionally needs typed rewiring actions, bond-attribute
transfer, valence rejection, and node-relabel actions when atom types may
change. Edge-only swaps cannot act on the predicted node categories.

Dense endpoint edge tensors use \(O(N^2)\) memory. Large sparse graphs need a
candidate-pair or sparse endpoint decoder.
