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
legacy baseline. The hybrid training script rejects a summary-generator block
unless the configuration explicitly marks it as a legacy baseline.

## Training examples

For every training graph \(G_1\), the code constructs an aligned source
\(G_0\) whose node \(i\) has the same degree as node \(i\) in \(G_1\).
Target-aware valid swaps produce intermediate states. Each supervised example
is

\[
(G_t,t,D)\longrightarrow(X(G_1),E(G_1),H(G_1)).
\]

Current and target graphs receive the same random relabeling. This preserves
node correspondence while preventing the model from exploiting a fixed label
order. Training and validation use their own dataset splits.

The loss is

\[
\mathcal L=
\lambda_X\mathcal L_X+
\lambda_E\mathcal L_E+
\lambda_H\mathcal L_{\text{graphlet mean}}+
\lambda_D\mathcal L_{\text{Dirichlet}}+
\lambda_M\mathcal L_{\text{connected mass}}.
\]

For featureless generic graphs, there is one node category, so the node loss
is exactly zero. The node head becomes non-trivial only for attributed data.

## Candidate score

For a valid swap \(a\), the selector uses

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

## Run

From the repository root:

```bash
export PYTHONPATH=src

python scripts/prepare_generic_dataset.py \
  --dataset sbm \
  --root outputs/datasets

python scripts/train_hybrid_endpoint_grapher.py \
  --config configs/experiments/sbm_hybrid_endpoint_graphlet.yaml

python scripts/run_hybrid_endpoint_grapher.py \
  --config configs/experiments/sbm_hybrid_endpoint_graphlet.yaml \
  --output-dir outputs/hybrid_endpoint/sbm/generated
```

Quick training check:

```bash
python scripts/train_hybrid_endpoint_grapher.py \
  --config configs/experiments/sbm_hybrid_endpoint_graphlet.yaml \
  --max-train-graphs 4 \
  --max-val-graphs 2 \
  --epochs 1 \
  --batch-size 2 \
  --output-dir outputs/hybrid_endpoint/sbm/smoke

python scripts/run_hybrid_endpoint_grapher.py \
  --config configs/experiments/sbm_hybrid_endpoint_graphlet.yaml \
  --checkpoint outputs/hybrid_endpoint/sbm/smoke/checkpoint.pt \
  --num-generate 2 \
  --output-dir outputs/hybrid_endpoint/sbm/smoke_generated
```

Check `degree_preservation_rate = 1.0` and
`connectedness_rate = 1.0`. `sampled_target_degree_match_rate` is a
diagnostic, not a validity requirement.

## Scope

The supplied configuration implements generic topology graphs. Attributed
molecular generation additionally needs typed rewiring actions, bond-attribute
transfer, valence rejection, and node-relabel actions when atom types may
change. Edge-only swaps cannot act on the predicted node categories.

Dense endpoint edge tensors use \(O(N^2)\) memory. Large sparse graphs need a
candidate-pair or sparse endpoint decoder.
