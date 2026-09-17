# Categorical GDSM: eigensolver convergence fix

This update applies to `graphes_gdsm_categorical_20260917.zip`, the attributed
spectral–categorical project supplied immediately before the reported QM9 error.
It changes **one existing source file**:

```
src/grapher/models/gdsm_simple/categorical/spectral.py
```

It also adds numerical regression tests and a dataset-free preflight script.
No configurations, parameters, checkpoint formats, datasets, learned degree
priors, categorical transitions, graphlet conventions or refinement rules change.

## Why the exception occurs at this location

The old `eigenpairs` groups the current categorical graphs by active node count,
forms `(edges > 0).float()`, and calls batched `torch.linalg.eigh` directly on the
training device. The reported call failed on CUDA. This computation is already
under `torch.no_grad()`: the reported exception is in the forward feature
calculation, not an eigenvector-gradient calculation.

The traceback establishes a solver convergence failure, but does not establish
the precise CUDA-library cause or the spectrum of the particular failing input.
Singular/repeated-eigenvalue adjacencies are legitimate inputs. For instance, an
edgeless graph has an identically zero spectrum. They must not be dropped or
modified merely to make a solver succeed.

Official PyTorch references:

- https://docs.pytorch.org/docs/stable/generated/torch.linalg.eigh.html
- https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html

PyTorch documents possible numerical/backend differences and recommends higher
precision as one mitigation, without guaranteeing that float64 fixes all cases.
The patch therefore includes an independent CPU solve and checks the results.
It does not claim that a particular CUDA version caused this user's failure.

## New solver policy

`GDSM_EIGH_BACKEND=auto` is the default:

1. Validate shapes, finite/integer/nonnegative categorical labels, boolean prefix
   masks, symmetric active edges, and zero active diagonals. Nonfinite labels
   are rejected BEFORE binarization, rather than silently becoming edges/nonedges.
2. Extract active graph blocks (not padding) and construct the unchanged binary
   adjacency in float64. Edgeless graphs receive their exact zero spectrum and
   identity eigenbasis without calling an eigensolver.
3. Solve nonempty-edge blocks with float64 `torch.linalg.eigh` on their current
   device. Validate finite outputs, ascending eigenvalues, eigen-equation
   residuals and orthonormality. Float64 quality tolerance is `1e-8`.
4. On a numerical convergence/quality failure, solve the same matrices using
   CPU float64 NumPy/LAPACK. If a CPU batch fails, retry individual graphs;
   a remaining failure uses SciPy's symmetric QR eigensolver (`driver="ev"`).
   Those results must pass the same checks. There is no random perturbation,
   random-basis substitution, graph repair, or skipped graph.
5. After a successful fallback, warn and remember that device/node-count group
   for the lifetime of the process. Later groups of that size use CPU directly,
   avoiding repeated failing solver attempts. The cache is a numerical routing
   policy, NOT a graph/eigenpair cache: every changed graph is recomputed.
6. Return float32 eigenpairs to the original device, retaining the existing
   sorted eigenvalue / square-root-of-node-count normalization and padding
   contract.

`GDSM_EIGH_BACKEND=cpu` bypasses the torch eigensolver and uses CPU float64 LAPACK
from the beginning. The neural network, losses, and optimizer still use the
requested GPU. No call to `torch.backends.cuda.preferred_linalg_library` or other
global CUDA backend configuration is made.

Out-of-memory errors, illegal memory access, and unrelated programming errors
are re-raised rather than disguised as numerical convergence failures. If all
CPU solvers fail, the operation raises with matrix size/index information; it
never supplies fabricated eigenpairs.

## Install

From the project root, after placing the changes ZIP there:

```bash
cd /home/quang/graphes
unzip -o graphes_gdsm_categorical_eigh_fix.zip -d .
```

For locally modified source, inspect the `.patch` file and use `git apply --check`
before applying it. Neither archive contains checkpoints or dataset replacements.

## Preflight on the user's GPU

For the reported failure, begin with CPU eigensolving and GPU model computation:

```bash
export GDSM_EIGH_BACKEND=cpu

PYTHONPATH=src python scripts/check_gdsm_categorical_eigh.py \
  --device cuda:0 \
  --max-nodes 9 \
  --graphs 256 \
  --output-json outputs/diagnostics/gdsm_categorical_eigh_preflight.json
```

This creates synthetic graphs (including empty graphs, cliques, paths, cycles,
and random graphs), verifies eigenpair reconstruction, and exercises the real
categorical corruption and joint loss with forward/backward optimizer steps.
It does not read, rebuild, or change QM9. The JSON reports actual Python,
PyTorch, CUDA build, GPU device, and solver-routing counters.

## Restart the reported training command

The environment variable below is read only by the patched eigensolver:

```bash
export GDSM_EIGH_BACKEND=cpu

CFG=configs/baselines/gdsm_simple_qm9_categorical.yaml
RUN=seed_42_spectral_categorical

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage train \
  --dataset qm9 \
  --no-common-config \
  --wrapper-config "$CFG" \
  --seed-id 42 \
  --run-id "$RUN" \
  --device cuda:0
```

The failed training attempt normally removes its temporary staging directory.
The same run ID can be reused if no completed training output already occupies
it. Do not remove an existing successful checkpoint or pass `--overwrite`
without checking what it would replace. A new run ID is always an alternative.

The numerical fix does not invalidate categorical model checkpoints. Existing
ordinary-degree DH-VAE checkpoints remain usable under the same provenance
checks. The failed job needs to be restarted: the current categorical trainer
does not restore optimizer progress from a failed partial run.

For generation, use the same `GDSM_EIGH_BACKEND=cpu` setting in that process and
the existing generation command. The common `eigenpairs` function covers model
forward calls, generation initialization, and every updated categorical graph.
Dynamic eigenvectors and freely changing degrees are not disabled by this fix.

To try the float64 device solver with automatic recovery instead:

```bash
export GDSM_EIGH_BACKEND=auto
```

Routing is a runtime numerical option and is not part of the checkpoint's learned
configuration. Keep it recorded with experiment commands. Different floating-point
backends need not produce bitwise-identical stochastic runs; CPU transfers may
also affect speed. No GPU throughput measurements are claimed here.

## Validation performed here

Environment: Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, SciPy 1.17.0.
CUDA was unavailable.

- Complete repository suite: **338 passed, 8 skipped**, one existing nested-tensor
  warning. The eight skips require real CUDA. See
  `docs/validation/gdsm_categorical_eigh_fix.log`.
- New eigensolver tests: **42 passed, 4 skipped**, including all nonempty graph-atlas
  entries, mixed-size padding, repeated eigenvalues, finite quality checks,
  forced torch and CPU solver failures, per-graph QR fallback, invalid-input
  rejection, non-numerical error propagation, and full model gradients.
- An injected-convergence-failure test performs joint training, generation,
  saved-eigenpair auditing, and checks that degrees can still change.
- Existing categorical suite with `GDSM_EIGH_BACKEND=cpu`: **50 passed, 1 skipped**.
- CPU-only, forced-CPU synthetic preflight: 256 graphs, maximum 9 nodes, 8 optimizer
  steps; finite losses/gradients; maximum returned-eigenpair reconstruction error
  `1.1896619600548775e-07`.
- CPU-only, automatic-policy synthetic preflight: 128 graphs, maximum 38 nodes,
  4 optimizer steps; finite losses/gradients; maximum reconstruction error
  `1.2537639324683328e-07`.
- Changed Python files were syntax-parsed with the Python 3.9 grammar. This is
  NOT execution testing under the user's Python 3.9/PyTorch/CUDA installation.

The user's actual CUDA convergence error was not reproduced on hardware here.
Failure recovery was exercised through injected matching exceptions. No full
QM9/ZINC training, 1,024-graph benchmark, or GPU runtime test was run.
