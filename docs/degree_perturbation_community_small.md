# Degree-perturbation priors for Community-small

## Scope

This generation-only addition implements options 1–4 discussed for an empirical
training-degree prior. It reuses a trained GraphER checkpoint. No retraining,
DH-VAE prior sampling, new training targets, or dataset rebuilding is required.
The joint checkpoint's learned encoder/decoder features still condition the
predictor on the **actual perturbed degree histogram**.

The pipeline is:

`training degree D -> optional valid D' perturbation -> fresh connected HH(D') -> existing GraphER`

All kernels preserve the parent node count and edge count. The refiner preserves
D', not the unperturbed D. Kernels see training degrees only; no held-out degree
sequence, graphlet target, or original training adjacency is used to perturb D.

## Implemented options

| Option | Config method | Operation | Extra invariant |
|---|---|---|---|
| 1 | `unit_transfer` | Decrement one degree and increment another; test bounds, graphicality, connected feasibility, and multiset change. Both balancing and concentrating directions are allowed. | None beyond n,m |
| 2 | `moment_preserving` | Replace a 4-entry degree block by a different block with identical sum and squared sum; validate the full resulting sequence. | Exact `sum(d*d)` |
| 3 | `edge_relocation` | Construct a temporary connected HH witness from D; relocate one edge endpoint while keeping the witness simple and connected; extract D'. | None beyond n,m |
| 4 | `interpolation` | Select a nearby, distinct training degree multiset with the same n,m; interpolate sorted degrees and use dependent randomized rounding with exact degree sum; validate D'. | None beyond n,m |

For option 3 the temporary adjacency is **discarded**. The generation script
constructs a fresh source using only the accepted degree summary. No original
training edges are copied into the prior.

Option 4 uses alpha=0.5 and the 8-nearest-partner distance boundary, including
ties. Partners must be within half-sorted-L1 distance 4. The rounded result must
differ from both current and partner multisets; merely returning the other
training parent is rejected. Rounding preserves the exact sum. Graphicality is
checked after rounding, not assumed from interpolation.

A multiset may differ from its selected parent but coincide with a different
training example. This is allowed by default and separately reported as
`novel_vs_training: false`. `require_novel: true` rejects all training multisets,
but changes the kernel and can lower acceptance substantially.

## Conservative starting settings

The four provided configs request a perturbation independently with probability
0.25 and request one accepted operation. The other samples take an explicit
unperturbed empirical branch. Per-step candidate validation budget is 256. The
maximum half-sorted-L1 distance from the original parent is 4. Degrees remain in
[1,n-1] for n>1 (the singleton [0] is supported). The joint checkpoint's maximum
degree is enforced as an additional bound: degrees are never clipped.

All five configs, including the new empirical control, fix:

- checkpoint: `best_joint`, training seed 42 (the user's epoch-17 checkpoint);
- refiner: 32 steps, 1024 proposals, no valid-candidate cap;
- clustering/orbit/spectral weights: 0.25 / 1.0 / 0.0;
- source enrichment: disabled;
- dataset: the existing prepared Community-small split, without rebuilding;
- sample count: 64 per generation seed; selection/evaluation reference: validation.

These are starting settings, not optimized hyperparameters or measured quality
improvements. This patch does not change predictor-time scheduling, candidate
scoring, orbit counting, or training.

## Failed requests are visible, not silently resampled

Some parents cannot be changed by a specified kernel. In particular, a degree
multiset with minimum variance for its n,m cannot have a different multiset with
the same second moment. Interpolation can also lack distinct same-(n,m) partners,
or its rounded proposals may only reproduce the two parents.

The experiment configs explicitly use:

```yaml
generation:
  degree_source: train_empirical_perturbed
  degree_rng_mode: independent
  degree_perturbation:
    method: unit_transfer  # or moment_preserving / edge_relocation / interpolation
    probability: 0.25
    steps: 1
    max_attempts: 256
    failure_policy: keep_original
    max_distance: 4.0
    require_novel: false
```

`keep_original` means a **logged identity transition on the same parent**. It does
not substitute another method, redraw another parent, or repair degrees. Failed
multi-step requests are rolled back, and rolled-back operations are recorded.
Consequently the realized changed fraction may be lower than 0.25. Compare it
alongside MMD; an unchanged output is not counted as a successful perturbation.

The API's default failure policy is `error`. To use strict generation, override:

```bash
--set generation.degree_perturbation.failure_policy=error
```

A strict failure aborts generation and writes `degree_prior_report.json` before
raising. This avoids silently conditioning the sampled parent distribution on
whether a perturbation was possible. The existing downstream HH constructor
retains its existing retry behavior; if constructor failures change the returned
parent batch, the comparison script detects the fingerprint mismatch.

## Reproducible comparisons

The new configs use a parent-sampling RNG independent of source construction,
refinement, the mixture coin, and perturbation rejection counts. Each perturbation
sample has separate mixture and kernel streams. This provides matching parent
samples and mixture flags across methods, barring downstream constructor retries.

**Run the new `empirical` control.** Historical empirical experiments used a
shared degree/construction RNG. The new control is not expected to reproduce
those historical source batches. With perturbation probability zero, the new
perturbation pipeline exactly matches the new independent empirical control;
this is covered by an end-to-end test of source and final graph fingerprints.

`parent_degree_fingerprint` must match within a sampling seed. Source graph
fingerprints generally should not match after a successful perturbation.

## Installation

Extract the full archive into a new working directory, or extract the patch-only
archive over a copy of `graphes(20260912-001739).zip` / the corresponding checkout.
Archive paths are relative to the repository root (`src/`, `scripts/`, etc.).
Keep existing `outputs/` data and checkpoints. No dependency changes are needed.

## Prior-only preflight (no checkpoint needed)

Run from the repository root:

```bash
PYTHONPATH=src python scripts/diagnose_degree_perturbations.py \
  --config configs/experiments/grapher/community_small_degree_perturb_unit_transfer.yaml \
  --output-dir outputs/degree_perturbation_diagnostics/community_small/seed_42 \
  --num-samples 256 --seed 42 --probability 1.0
```

This examines all four methods on matched training-parent samples. It prints
requested, changed, novel, and fallback rates and writes a report per method.
Probability 1.0 here probes feasibility; it **does not change the generation
configs' probability of 0.25**. The diagnostic never loads a neural checkpoint,
so an optional `--max-degree` can reproduce a known checkpoint support ceiling.
Actual generation obtains that ceiling directly from the loaded joint model.
This preflight does not report graph-generation quality or MMD.

## One command per option (generation AND evaluation)

The runner defaults to generation seeds 42,43,44, 64 samples each, GPU generation,
and validation evaluation. It refuses to overwrite an existing completed run.

```bash
# Required fresh empirical control
bash scripts/run_community_small_degree_perturbation_ablation.sh empirical

# Option 1
bash scripts/run_community_small_degree_perturbation_ablation.sh unit_transfer

# Option 2
bash scripts/run_community_small_degree_perturbation_ablation.sh moment_preserving

# Option 3
bash scripts/run_community_small_degree_perturbation_ablation.sh edge_relocation

# Option 4
bash scripts/run_community_small_degree_perturbation_ablation.sh interpolation
```

Or run the full matrix once:

```bash
bash scripts/run_community_small_degree_perturbation_ablation.sh all
```

Do not run `all` after the individual commands into the same output root; the
runner intentionally refuses to overwrite the completed outputs.

To start with only generation seed 42:

```bash
SEEDS="42" bash scripts/run_community_small_degree_perturbation_ablation.sh all
```

To explicitly request perturbation on every sample, use a **new output root**:

```bash
PERTURB_PROBABILITY=1.0 \
OUT_ROOT=outputs/topology_generation/community_small_degree_perturbation_rho100/ckpt_seed_42 \
  bash scripts/run_community_small_degree_perturbation_ablation.sh all
```

This still logs impossible perturbations as identity transitions under the
provided failure policy. It does not promise 100% changed degree multisets.

Runner environment overrides: `PYTHON`, `CKPT`, `SEEDS`, `NUM_GENERATE`, `DEVICE`,
`OUT_ROOT`, `REFERENCE_SPLIT`, and `PERTURB_PROBABILITY`. Use validation while
selecting settings. Reserve test evaluation for a locked final configuration.

## Explicit Python commands for each option

Shared setup (run in the same shell):

```bash
set -euo pipefail
CKPT=outputs/topology_grapher/community_small_joint_degree_multicheckpoint/seed_42/checkpoints/best_joint/checkpoint.pt
CFGROOT=configs/experiments/grapher
GENROOT=outputs/topology_generation/community_small_degree_perturbation/ckpt_seed_42
SEED=42
```

### Option 1 — unit transfer

```bash
CFG="$CFGROOT/community_small_degree_perturb_unit_transfer.yaml"
OUT="$GENROOT/generation_seed_${SEED}/unit_transfer"
PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config "$CFG" --checkpoint "$CKPT" --output-dir "$OUT" \
  --num-generate 64 --seed "$SEED" --device gpu
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config "$CFG" --generated-dir "$OUT" --reference-split val \
  --output-dir "$OUT/evaluation_val"
```

### Option 2 — second-moment-preserving blocks

```bash
CFG="$CFGROOT/community_small_degree_perturb_moment_preserving.yaml"
OUT="$GENROOT/generation_seed_${SEED}/moment_preserving"
PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config "$CFG" --checkpoint "$CKPT" --output-dir "$OUT" \
  --num-generate 64 --seed "$SEED" --device gpu
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config "$CFG" --generated-dir "$OUT" --reference-split val \
  --output-dir "$OUT/evaluation_val"
```

### Option 3 — edge-endpoint relocation on a temporary witness

```bash
CFG="$CFGROOT/community_small_degree_perturb_edge_relocation.yaml"
OUT="$GENROOT/generation_seed_${SEED}/edge_relocation"
PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config "$CFG" --checkpoint "$CKPT" --output-dir "$OUT" \
  --num-generate 64 --seed "$SEED" --device gpu
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config "$CFG" --generated-dir "$OUT" --reference-split val \
  --output-dir "$OUT/evaluation_val"
```

### Option 4 — training-neighbor interpolation

```bash
CFG="$CFGROOT/community_small_degree_perturb_interpolation.yaml"
OUT="$GENROOT/generation_seed_${SEED}/interpolation"
PYTHONPATH=src python scripts/run_topology_grapher.py \
  --config "$CFG" --checkpoint "$CKPT" --output-dir "$OUT" \
  --num-generate 64 --seed "$SEED" --device gpu
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config "$CFG" --generated-dir "$OUT" --reference-split val \
  --output-dir "$OUT/evaluation_val"
```

For the explicit new control, use the same commands with
`community_small_degree_perturb_empirical.yaml` and output suffix `/empirical`.

## Metrics and audit outputs

Each run saves the existing `coarse_graphs.pkl`, `topology_refined_graphs.pkl`,
and `report.json`, plus:

- `degree_prior_report.json`: kernel config, parent and output fingerprints,
  actual changed/novel fractions, exact failure reasons, attempted operations,
  candidate rejection counts, and per-sample parent/output degrees;
- `sampled_degree_sequences.json`: actual invariants passed into HH and GraphER.

The prior report includes all prior attempts, plus `returned_records` for only
successfully returned graphs. Summary metrics use the returned records. Graph
metrics remain in `evaluation_val/graph_mmd_metrics.csv` and
`evaluation_val/graph_evaluation_report.json`.

After all three sampling seeds are evaluated:

```bash
PYTHONPATH=src python scripts/summarize_degree_perturbation_evaluations.py \
  --generation-root outputs/topology_generation/community_small_degree_perturbation/ckpt_seed_42 \
  --reference-split val --seeds 42 43 44
```

For a seed-42 pilot, use `--seeds 42`. The summary prints metric means and writes
`summary_val/rows.csv`, `summary_val/aggregates.csv`, and `summary_val/comparison.json`.
It rejects incompatible checkpoints, references, evaluator settings, refiner
settings, and unmatched parent-degree batches. It checks that source and final
Degree MMD agree. Means and sample SDs are across generation seeds for a single
checkpoint, not independent training runs or pooled-graph MMD.

## Validation scope

The implementation is tested using small explicit graphs and randomly initialized
joint-model checkpoints for CLI integration. The uploaded source archive does
not contain the user's trained checkpoint or prepared Community-small split.
No Community-small quality result, improved MMD, or real-data acceptance rate is
claimed by these tests. See `docs/degree_perturbation_validation.json` and the
included test logs for the exact validation outcome, including the inherited
floating-point warm-start test failure reproduced on the unmodified archive.
