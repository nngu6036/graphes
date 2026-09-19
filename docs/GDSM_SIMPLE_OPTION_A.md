# GSDM-Simple-Structure3: Option A (free-degree spectral decoding)

This update adds `extensions.generation_mode: spectral_decode`. The previous
`degree_constrained` mode remains the default for old configurations, and the
CUDA-device-alias fix is retained. The implementation diff is relative to
`graphes_gdsm_structure3_20260917_device_fixed.zip`.

## What changed

The sampled degree sequence `d_prior` defines a source-centred adjacency-spectrum
anchor `s(d_prior, U)`. It does **not** define a degree-constrained graph state.

For every reverse model evaluation:

1. Predict diffusion noise and clean clustering/orbit/graphlet summaries.
2. Compute the predicted clean normalized adjacency spectrum `x0_hat`.
3. Sort its valid coefficients and decode `U diag(sqrt(n) * x0_hat) U.T`.
4. Threshold at `sample.threshold`, symmetrize, clear the diagonal, and retain
   all `n` nodes. These operations produce the actual current discrete graph.
5. At selected guidance events, refine this freshly decoded graph with valid
   double-edge swaps. Reconcile size-3 count targets with **its current degrees**,
   not the originally sampled degree vector. The source-distance reference and
   visited-state set are local to this event.
6. Feed the representable component of an accepted swap back into the next DDIM
   step. The next spectral decode can again change the degrees and edge count.
7. Return the last decoded graph **after** its final local rewiring event. Do not
   decode once more after this correction, which would discard that final event.

There is no HH realization, degree projection, original edge-count projection,
connectivity repair, largest-component extraction, or resampling based on the
final graph. The existing learned degree prior still performs its own declared
feasibility/rejection checks **before** creating the spectral anchor.

This remains the project-owned **GSDM-Simple** model, not a claim of reproducing
the published GSDM architecture. The state is an **adjacency** spectrum divided
by `sqrt(n)`, not a Laplacian spectrum. Graph size and the sampled eigenbasis stay
fixed within a sample; node degrees and edge count do not.

## Strict initialization-only versus checkpoint compatibility

The old Structure3 denoiser receives four conditioning blocks: degree vector,
anchor, and two sampled-basis descriptors. Merely removing the global degree
constraint would still leave the old degree vector as a soft neural condition.

The new primary configurations set:

```yaml
extensions:
  degree_conditioning: true
  generation_mode: spectral_decode
  initialization:
    mode: degree_basis
    conditioning: initialization_only
```

The architecture flag `degree_conditioning: true` is retained for the existing
structured network interface. Under `initialization_only`, the **degree and
anchor conditioning blocks are zero** during both training and generation;
only the two basis-descriptor blocks remain. Degrees enter via the spectral
initialization, not as a separately supplied vector at each denoiser call.

The matching source-centred VP/DDIM scheduler remains:

```
x_t = s + sqrt(alpha_bar_t) * (x_clean - s) + sqrt(1-alpha_bar_t) * noise
x_T approximately s + noise
```

The offset `s` remains known to the scheduler throughout denoising. This is a
coordinate offset, not a requirement that the decoded graph realize `d_prior`.
This is not a one-time warm start substituted into an unmatched zero-centred
sampler. Training still uses clean-graph summaries and no graph rewiring.

**Strict mode requires a new spectral-model training run**, because its neural
conditioning differs. The trained DH-VAE can be reused when it belongs to the
same prepared training split. The checkpoint contract rejects an attempt to
silently change conditioning at generation.

Four configurations are supplied:

| Dataset | Configuration | Spectral checkpoint |
|---|---|---|
| Community-small | `configs/baselines/gdsm_simple_community_small_structure3_option_a.yaml` | Retrain strict mode |
| Ego-small | `configs/baselines/gdsm_simple_ego_small_structure3_option_a.yaml` | Retrain strict mode |
| Community-small | `configs/baselines/gdsm_simple_community_small_structure3_option_a_legacy_conditioning.yaml` | Reuse previous Structure3 checkpoint |
| Ego-small | `configs/baselines/gdsm_simple_ego_small_structure3_option_a_legacy_conditioning.yaml` | Reuse previous Structure3 checkpoint |

The compatibility configurations implement free-degree Option-A decoding but
retain the old **soft** degree/anchor conditioning. They are not the strict
initialization-only experiment. Older S0/S1 checkpoints lack the structure heads
and cannot be used for either variant.

The primary settings retain the previous starting training budgets (Community:
1,250 epochs; Ego: 625; batch size 16; four training-basis pairings), not newly
tuned budgets. The actual number of updates depends on the prepared split size.
Do not rebuild the dataset between degree training, spectral training, and
held-out evaluation.

## Basis-aware feedback

The sampled basis `U` is fixed, while a rewired graph generally has different
eigenvectors. Passing its sorted eigenvalues back with the old `U` would not
re-encode that graph. Option A instead uses the local adjacency delta:

```
delta_A = A_after_swap - A_before_swap
z_new = Iso(z + rho * diag(U.T @ delta_A @ U) / sqrt(n))
```

`Iso` is the equal-weight least-squares projection onto nondecreasing
coefficients, implemented with pool-adjacent violators. It avoids permuting the
association between coefficients and the fixed basis columns. The updated
coefficients are then restored to the model-coordinate order before the next
DDIM update. The noise prediction from the current step is retained.

This is a **sampling heuristic**, not an exact graph re-encoding or a proof of
sampling from the unguided model distribution. A fixed basis cannot represent
all swap directions. The event log reports projection residuals and the effective
coefficient change; some accepted swaps can have zero representable feedback.
Intermediate corrections can also be partly undone by later denoising or
thresholding. Only the final correction is guaranteed to be the returned graph.

```yaml
extensions:
  spectral_decode:
    connectivity: unconstrained
    feedback_mode: fixed_basis_delta
    save_degree_trajectory: false
  structure_guidance:
    start_fraction: 0.35
    every: 100
    max_steps_per_event: 2
    spectrum_feedback: 0.05
    feedback_only_after_accept: true
```

With 1,000 model evaluations, decoding occurs 1,000 times and local structural
rewiring runs at timesteps 299, 199, 99 and 0. `every` counts reverse model
evaluations. The last evaluation is always a guidance event. Different sampling
step counts can change these event times.

Option A applies feedback only after an accepted swap and never at the terminal
event, regardless of the legacy `feedback_only_after_accept` flag. No swap means
no delta to feed back. Setting `spectrum_feedback: 0` is a generation-only
ablation: earlier swaps cannot affect later spectral states and are overwritten
by later decoding. Final-event swaps still affect the returned graph.

`initialization.ensure_connected` and `realization_fit_steps` are legacy
constructor settings and are ignored in this branch. `degree_preserving_rewiring:
true` enables local swaps; it is **not** a global degree-invariance switch.

## Run strict Option A

Extract the changes-only ZIP from the project root, or use the complete project
archive. Do not overwrite locally edited source files without first comparing
the supplied diff.

The runner supports both datasets and does not overwrite outputs or rebuild data:

```bash
cd /home/quang/graphes

# Community-small: check, reuse/train DH-VAE, train the strict model,
# generate, audit, and evaluate.
export ORCA_EXEC=/home/quang/orca/orca.out
DEVICE=cuda:0 N=1024 \
  bash scripts/run_gdsm_simple_option_a.sh community_small all

# Ego-small uses its own degree checkpoint, data, and spectral checkpoint.
DEVICE=cuda:0 N=1024 \
  bash scripts/run_gdsm_simple_option_a.sh ego_small all
```

Stages can be run separately: `check`, `degree`, `train`, `generate`, `audit`,
`evaluate`, and `evaluate-local`. The `all` stage reuses an existing configured
DH-VAE checkpoint, subject to generation's provenance validation. The degree
trainer reads its device from its degree YAML (`auto`), not the runner's `DEVICE`
variable. `DEVICE` controls spectral training and generation.

### Explicit Community-small commands

```bash
CFG=configs/baselines/gdsm_simple_community_small_structure3_option_a.yaml
COMMON=configs/baselines/common_community_small.yaml
RUN=seed_42_structure3_option_a
N=1024
GID=seed_42_n_${N}_option_a
GEN=outputs/baselines/gdsm_simple/community_small/${RUN}/generations/${GID}

# Only needed when the existing degree checkpoint does not match this split.
PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/dhvae/community_small.yaml

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage train --dataset community_small \
  --common-config "$COMMON" --wrapper-config "$CFG" \
  --seed-id 42 --run-id "$RUN" --device cuda:0

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage generate --dataset community_small \
  --common-config "$COMMON" --wrapper-config "$CFG" \
  --seed-id 42 --generation-seed 42 --run-id "$RUN" \
  --generation-id "$GID" --num-samples "$N" --device cuda:0

PYTHONPATH=src python scripts/audit_gdsm_option_a.py --generated-dir "$GEN"

export ORCA_EXEC=/home/quang/orca/orca.out
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN" --generated-graphs "$GEN/base_graphs.pkl" \
  --generated-stage gdsm_simple_option_a \
  --base-graphs "$GEN/initial_graphs.pkl" --reference-split test \
  --generic-mmd-protocol graphrnn \
  --output-dir "$GEN/evaluation_test"
```

For Ego-small, replace the dataset names in these commands with `ego_small`.
The Ego prior path is `outputs/degree_generators/ego_small/seed_42/checkpoint.pt`;
the Community prior path remains `outputs/degree_generators/sbm/seed_42/checkpoint.pt`.
The runner selects these automatically.

The evaluator writes metrics only. No `--max-graphs` cap is set here, so the
complete generated batch is evaluated. ORCA uses the existing external orbit protocol,
not only the four predicted size-at-most-three orbit coordinates.

### Reuse the current Structure3 checkpoint without retraining

This tests the new graph-decoding mechanism while keeping the trained soft
conditioning unchanged. It does not remove the original degree input from the
network. Set `RUN` to the exact existing trained run, not a new run name.

```bash
VARIANT=legacy_conditioning \
RUN=seed_42_structure3_degree_basis DEVICE=cuda:0 N=1024 \
  bash scripts/run_gdsm_simple_option_a.sh community_small generate

VARIANT=legacy_conditioning \
RUN=seed_42_structure3_degree_basis N=1024 \
  bash scripts/run_gdsm_simple_option_a.sh community_small evaluate
```

Use `ego_small` for the analogous existing Ego checkpoint. Both compatibility
and strict generation use a new generation ID ending in `_option_a` to avoid
collisions with old samples. Supply `GEN_ID` to keep additional ablations apart.

## Outputs and interpretation

| Artifact | Meaning |
|---|---|
| `base_graphs.pkl` | Final graph after final-event local rewiring |
| `initial_graphs.pkl` | Decoded graph before the first guidance event, not HH |
| `first_decoded_graphs.pkl` | Decoded graph at the first reverse model evaluation |
| `final_pre_rewire_graphs.pkl` | Input to final-event local rewiring |
| `threshold_graphs.pkl` | Same final pre-rewire decode; not an independent unguided baseline |
| `sampled_degree_sequences.pkl` | Degree vectors used to build the initial anchors |
| `final_degree_sequences.pkl` | Actual final indexed degrees |
| `target_adjacency_eigenvalues.pkl` | Final clean spectral prediction before the final swap |
| `final_graph_adjacency_eigenvalues.pkl` | Actual normalized adjacency spectra of final graphs |
| `rewiring_diagnostics.json` | Per-event and aggregate structural and degree audits |
| `degree_trajectories.npz` | Optional `[graphs, reverse_steps, max_nodes]` degree array |
| `intermediate_graphs.pkl` | Optional decoded/corrected graph pair at every guidance event |

`degree_trajectories.npz` is enabled by
`extensions.spectral_decode.save_degree_trajectory: true`. It stores timesteps
and true node counts; trailing degree slots are padding. Optional graph snapshots
are enabled by `structure_guidance.save_intermediate_graphs: true`.

Important diagnostic fields:

```
local_rewiring_degree_preservation_rate                 # must be 1
initial_to_final_degree_multiset_preservation_rate      # allowed to be below 1
prior_to_final_degree_multiset_preservation_rate        # allowed to be below 1
mean_spectral_degree_change_steps                      # measured across all decodes
mean_feedback_events
final_connected_rate
mean_final_isolates
```

Initial-versus-final Degree MMD is no longer constrained to equality. It can
still coincide because of threshold stability, unchanged degree histograms, or
rounding; equality alone does not prove an implementation error. The audit reads
individual graphs and event records rather than inferring changes from MMD.

**Final-pre-rewire versus final Degree MMD must still match for the same batch**,
because those graphs differ only by the final local swaps. Run `evaluate-local`
to isolate that event, versus `evaluate` for first-guidance-decode to final.
Neither comparison is a controlled independent no-guidance baseline: disable
rewiring and regenerate with a separate generation ID for that experiment.

Simple, undirected and loop-free output is guaranteed by decoding and valid
swaps. Global connectedness is deliberately **not** guaranteed. A local swap
preserves connectivity when its input graph is connected; later decoding can
still disconnect it. No quality improvement in held-out MMD is guaranteed.

## Validation

The full suite passes **246 tests**, including **29 new Option-A tests**, with
three existing real-CUDA tests skipped because no GPU is available. Coverage
includes strict conditioning, checkpoint compatibility, every-step decoding,
changing degrees with a fixed basis, padding, local degree invariance,
basis-aware feedback, no-feedback ablation, learned DH-VAE integration, CLI
configuration parsing, artifact audits, and deterministic generation replay.

A small real CPU round-trip uses four six-node training graphs and two training
epochs. In its six generated graphs, three change degree multisets between the
first guidance decode and final output, while all local rewiring events preserve
degrees. This is an implementation test, not a Community-small benchmark or a
claim about model quality. The test uses eight reverse steps and a larger
feedback weight than the production starting configuration.

No full Community-small or Ego-small retraining, 1,024-graph benchmark, GPU
execution, or performance measurement was performed for this update.

The shared training/generation CLI was also run end-to-end on the six-node CPU
smoke dataset, and the separate audit CLI confirmed its saved graph degrees.
Changed Python files were syntax-checked with the Python 3.9 grammar; runtime
tests used the installed Python/PyTorch versions recorded in the validation log.
