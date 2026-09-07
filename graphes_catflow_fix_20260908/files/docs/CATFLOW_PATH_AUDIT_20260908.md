# CatFlow wrapper audit and correction — 8 September 2026

## Executive finding

The supplied CatFlow source and the previous GraphER wrapper contain a **training/sampling probability-path mismatch**. The wrapper inherited a fixed additional Gaussian perturbation from `conditional_velocity("normal", ...)`, but used the straight-line CatFlow velocity during generation. More epochs do not make these two definitions consistent.

The user's shell commands correctly route training, generation, and evaluation. In particular, omitting `--n-epochs 200000` from the generation command does **not** reset the training budget to the YAML value.

This audit inspects the uploaded archives and the previously supplied wrapper. It does not include the user's actual checkpoint, run logs, processed graph splits, or generated graphs. The local `/home/quang/CatFlow` may differ from the uploaded source. Consequently, this is a confirmed implementation inconsistency in the inspected code, not proof that it alone accounts for the user's reported MMD values.

The previous wrapper's smoke tests established execution and data compatibility, not mathematical or benchmark equivalence. That distinction was not adequately checked before delivery.

## 1. Primary source evidence

The line-numbered excerpts are in `validation/catflow_v2/source_excerpts.md`. The preserved pre-fix worker excerpt is explicitly labelled; line numbers should not be compared with the revised worker.

| Evidence | File and location | Finding |
|---|---|---|
| S1 | Uploaded `flow_matching.py`, lines 27–39 | `x_t=(1-t)*x_0+t*x_1`, followed by a fixed `0.5` Gaussian perturbation; edge noise is symmetrized. |
| S1 | Uploaded `flow_matching.py`, lines 128–145 and 581–590 | Generation starts from a Gaussian with only a `1e-6` extra perturbation and uses `(softmax(pred)-state)/(1-t)`. |
| S2 | Previous wrapper worker, lines 91–103 | Calls the uploaded noisy training path directly for node and edge tensors. |
| S2 | Previous worker, lines 118–131 | Saves final epoch only; generation normally copies EMA weights. |
| S2 | Previous worker, lines 79–117 | Validation uses raw model parameters and fresh random time/noise draws. |
| S3 | `external_cli.py`, training/generation routing | Training overrides are recorded; generation takes trained options and only applies sampling/runtime overrides. |
| S4 | Uploaded `utils.py`, model factory | Uses the supplied categorical transformer inputs, not an automatically added spectral/structural-feature pipeline. |

Inspected uploaded `flow_matching.py` SHA256:

```text
c1d3ab645a62599f3ccd8b25cb90bb91fee24269842d05795e62e12f6da4593b
```

No claim is made that this uploaded archive is identical to any particular official Git commit. The patch leaves every `external/catflow/` source file unchanged and also does not modify an external checkout selected with `CATFLOW`.

## 2. Probability-path mismatch

For the straight-line path,

```text
x_t = (1-t) z + t x_1
v_theta(x_t,t) = (mu_theta(x_t,t)-x_t)/(1-t)
```

where `mu_theta` is the endpoint-category posterior mean. This pairing is supported by Eq. (3) and the categorical posterior-mean construction in Eijkelboom et al., *Variational Flow Matching for Graph Generation*, arXiv:2406.04843v2 (external theoretical reference: https://arxiv.org/html/2406.04843v2).

The supplied normal training branch instead samples

```text
x_t = (1-t) z + t x_1 + 0.5 epsilon
```

with an independent, time-independent noise draw. For edges, both Gaussian draws are symmetrized consistently. That symmetrization does not eliminate the mismatch. At `t=1`, the training example is still noisy rather than equal to the target; at `t=0.95`, the extra-noise coefficient is ten times the intended source-noise coefficient. A classifier fitted to the altered path estimates a different posterior, so plugging it into the unaltered straight-line velocity is not the matching flow.

Removing the extra noise restores the stated linear path. Deriving a different flow for a deliberately noisy endpoint would be a different alternative; the correction here does not silently make that modelling change.

### Numerical diagnostic independent of neural-network training

A one-binary-edge experiment uses the **exact Bayes posterior**, avoiding undertraining and optimizer error. Both variants start from the same Gaussian samples and integrate the same straight-line velocity to `t=0.95` using NumPy RK4, 400 steps, 200,000 samples, seed 1729.

| Target edge probability | Exact posterior under linear training | Exact posterior under fixed-extra-noise training |
|---|---:|---:|
| 0.20 | 0.199325 | 0.061970 |
| 0.35 | 0.349615 | 0.243475 |
| 0.50 | 0.498685 | 0.498685 |

These are diagnostic edge frequencies, **not Community-small experiments, MMD scores, or expected recovered benchmark values**. They show that the mismatch can strongly bias sampling even with perfect posterior estimation. In a learned graph model, the direction and size of the effect must be measured rather than inferred from this toy example.

The source probe also measures residual noise at `t=1`: node standard deviation approximately 0.5021, off-diagonal edge standard deviation approximately 0.3524. Edge symmetrization explains the smaller per-coordinate value.

## 3. Secondary issues and ruled-out explanations

### Checkpoint selection and validation

The previous worker saved only the final epoch. Its validation loss measured raw parameters with changing random perturbations, while sampling used EMA parameters. This made the validation trace less directly relevant to the sampled model and provided no best-checkpoint fallback. It does not prove that the user's run overfit.

The revision validates EMA parameters on fixed random time/noise draws from the held-out validation split, restores raw training parameters and RNG state afterwards, and retains both a best-validation snapshot and final weights. The selected checkpoint never uses test MMD. Denoising validation loss remains an imperfect proxy for graph-generation quality.

### Epoch override and update accounting

`--n-epochs` is an alias for the training epoch argument. The saved training configuration is reused for generation; the generation command must not repeat training overrides. An end-to-end CPU test trained with a YAML value of 1,000 and CLI override of 3, then generated without repeating the override. The saved checkpoint and manifest correctly reported 3 epochs and 6 optimizer updates.

The supplied Community-small protocol requests 64 training graphs, 16 validation graphs and 20 test graphs. With a batch size of 128, that would be one optimizer update per epoch. The diagnostic reads the actual exported split rather than assuming those counts. Thus 200,000 epochs can mean 200,000 updates; the number alone does not establish either undertraining or overtraining.

### Data and sampler checks

The inspected dense encoding distinguishes valid absent edges `[1,0]`, present edges `[0,1]`, and zeroed padding/diagonal `[0,0]`. Regression tests cover this distinction. No edge-label inversion was found.

For an unattributed graph there is one node category, so node cross-entropy is zero; edge cross-entropy still trains topology. Generated node counts are sampled from the training split. Disconnected graphs and isolated nodes remain in exported samples; this patch does not filter or repair them to improve metrics.

The default `dopri5` sampler is adaptive. `sample.steps` does not set its number of integration steps in the inspected implementation; it controls Euler, and in the patch explicitly controls the other supported fixed-grid solvers. Increasing this field while retaining `dopri5` is not a remedy for the probability-path mismatch.

The endpoint `t_end=0.95` and final argmax were not identified as wrapper-only bugs. They remain unchanged to avoid combining several experimental changes.

### Scope of reproduction

The uploaded runnable pipeline and the generic-graph wrapper are not a verified reproduction of all paper experiments. Appendix D.1 describes additional structural/spectral features; the supplied active model factory does not automatically use them. The correction fixes the demonstrated path mismatch and checkpoint policy, not all possible differences from the paper. No paper-level performance is claimed.

## 4. Files changed

The changes include an explicit `catflow_path.py`, the CatFlow worker, wrapper defaults, all five existing CatFlow dataset configurations, a new `catflow_community_small_linear_v2.yaml`, regression tests, the diagnostic script, and audit documentation. The shared external wrapper has only a small manifest change to report the worker's real checkpoint-selection policy rather than hard-code final-epoch selection.

New default training settings:

```yaml
train:
  path: linear
  validation_every: 100
  validation_repeats: 4
  checkpoint_every: 1000
sample:
  checkpoint_selection: auto
```

`auto` selects best-validation weights for new checkpoint bundles and final weights for legacy checkpoints. Legacy generation emits a warning and retains its previous semantics; changing a training YAML does not transform old weights into a corrected-path model. Explicit `train.path: upstream` is retained only for labelled legacy diagnostics.

The new Community-small v2 configuration keeps the previous architecture, learning rate, EMA, batch size and default sampler. Its 10,000-epoch budget is a **diagnostic starting point, not a tuned or paper-converged setting**. Extending the budget requires a new experiment; this patch does not implement training resumption.

## 5. Apply the patch safely

The small patch installer checks file hashes against the previously supplied GraphER archive, supports a dry run, and backs up replaced files. It refuses to overwrite files that were locally modified since that archive; those conflicts should be reviewed and merged rather than forced. Files already matching the patch are skipped. No datasets, checkpoints, generations, or external model source files are changed.

From the existing GraphER repository root, after extracting the patch to `/tmp/graphes_catflow_fix_20260908`:

```bash
python /tmp/graphes_catflow_fix_20260908/apply_catflow_fix.py --repo "$PWD" --dry-run
python /tmp/graphes_catflow_fix_20260908/apply_catflow_fix.py --repo "$PWD"
```

The full project archive is an alternative for a separate checkout. Preserve the existing `seed_42` run for diagnosis.

## 6. Diagnose the existing run

```bash
export CATFLOW="/home/quang/CatFlow"
export CATFLOW_PYTHON="$(command -v python)"

PYTHONPATH=src "$CATFLOW_PYTHON" scripts/diagnose_catflow_baseline.py \
  --run-dir outputs/baselines/catflow/community_small/seed_42 \
  --generated-dir outputs/baselines/catflow/community_small/seed_42/generations/seed_42_n_1024 \
  --json-out outputs/catflow_seed42_audit.json
```

Check `source_matches_training`, `upstream_path_probe`, the actual saved epoch, EMA update count, checkpoint and generated-graph hashes, and train/generated mean degree, clustering, triangles, connectivity and isolate fraction. A source probe alone cannot establish the trained probability path when the external file differs from the recorded training hash. Only load trusted local checkpoint and graph pickle files.

Do not edit the external source in place to attempt to repair an existing run: source-fingerprint checks are intentional, and the old model learned a different training law.

## 7. Corrected training, generation, evaluation

Use a new run ID. These commands do not overwrite the original 200,000-epoch run.

```bash
export CATFLOW="/home/quang/CatFlow"
export CATFLOW_PYTHON="$(command -v python)"
CFG="configs/baselines/catflow_community_small_linear_v2.yaml"
RUN="seed_42_linear_v2"

PYTHONPATH=src python scripts/run_catflow_baseline.py \
  --stage train \
  --dataset community_small \
  --wrapper-config "$CFG" \
  --seed-id 42 \
  --run-id "$RUN" \
  --device gpu \
  --n-epochs 10000

PYTHONPATH=src python scripts/run_catflow_baseline.py \
  --stage generate \
  --dataset community_small \
  --wrapper-config "$CFG" \
  --seed-id 42 \
  --run-id "$RUN" \
  --num-samples 1024 \
  --device gpu

GEN_DIR="outputs/baselines/catflow/community_small/$RUN/generations/seed_42_n_1024"

PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN_DIR" \
  --generated-stage catflow_linear_v2 \
  --output-dir "$GEN_DIR/evaluation_report"
```

The sampler log identifies the selected EMA epoch. Reuse the exact same prepared train/validation/test files and evaluation protocol for comparison with the old result. Do not select configurations using repeated test-set metric sweeps.

## 8. Validation and limits

The selected regression suite covers path identities, symmetric edge noise, correct masking and category encoding, real upstream-transformer gradients, deterministic EMA validation and parameter/RNG restoration, independent best-checkpoint snapshots, native CPU training and checkpoint reload, exact sample counts, and shared wrapper/evaluator regressions. The machine-readable report records the final test command and result.

Additional checks performed:

* Old and revised workers generated exactly equal adjacency, node-count and node-type arrays from the same legacy checkpoint and seed using Euler. Legacy behavior is not silently reinterpreted.
* A managed CLI training/generation run verified that the command-line epoch override survives generation without being specified again.
* The exact-posterior diagnostic above verifies a distributional consequence of the path mismatch without a learned network.

**Not validated:** the user's actual 200,000-epoch checkpoint; any real Community-small MMD recovery; converged full-architecture runs; GPU execution; the default adaptive `dopri5` solver; or molecular benchmark quality. Local native numerical smoke tests used the small supplied transformer on synthetic graphs and Euler sampling. This environment has CPU-only PyTorch and lacks `torchdiffeq`; installing the optional solver dependencies failed because the package index was unreachable.

Do not interpret passing execution tests as benchmark equivalence. Retraining on the corrected path and measuring the unchanged evaluation protocol is still necessary to establish the effect on the user's metrics.
