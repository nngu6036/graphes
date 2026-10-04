# Validation report — 4 October 2026

## Focused tests

**61 passed** for the updated attributed model, categorical masking/constraints,
raw evaluation controls, and molecular input accounting.

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_gdsm_simple_attributed_loggap.py \
  tests/test_gdsm_attributed_masking_and_constraints.py \
  tests/test_molecular_raw_protocol_and_failures.py \
  tests/test_molecular_generated_inputs.py
```

Coverage includes partial/full/no masking, missing clean bonds, false-positive
current edges, padding, zero-loss gradients, stochastic decoding, atom-degree
feasibility, reserved bond-order capacity, reordered vocabularies, infeasible
and singleton topologies, unsupported chemistry, RNG separation, fresh toy
training/checkpoint/generation, three-way topology-state pairing across multiple
batches, serialized-topology mutation detection, all nine new YAML configs,
legacy-checkpoint provenance, conflicting evaluation flags, failure accounting,
and a raw-valid evaluator smoke test using the explicitly named proxy backend.

The training smoke tests use tiny synthetic molecular graphs and reduced
models. They are not QM9 model-quality measurements.

## Full repository comparison

The same complete test command was run on the untouched uploaded archive and
on the updated repository in the same CPU environment:

| Repository | Passed | Failed | Setup errors | Skipped |
|---|---:|---:|---:|---:|
| Original uploaded archive | 652 | 18 | 18 | 8 |
| Updated repository | 690 | 18 | 18 | 8 |

**All 36 failing/error test IDs are identical.** The 38 added tests pass; no new
failing/error test ID was introduced in this environment. The full repository
suite is not wholly passing. Inherited issues are deliberately not hidden by
marking tests xfail or changing expectations.

The inherited failures are in:

- `test_baseline_exposure_budgets.py`: two SPECTRE epoch-budget expectation
  mismatches (500/270 versus 10,000).
- `test_gdsm_training_basis.py`: mismatches between existing fixed-training-basis
  tests/configs and the older spectral-categorical implementation, including
  unsupported `proposal_basis`, missing `basis_bank_provenance`, and unknown
  `basis_source` / `save_degree_trajectory` settings.

Those branches were not rewritten as part of this attributed log-gap update.
Use the raw logs and machine-readable ID comparison for exact details.

## Other checks

`python -m compileall -q src scripts tests` completed successfully. Every Bash
block in the command guide passed `bash -n`. The molecular evaluator and pairing
verification CLI help commands both loaded successfully. The release packaging
step checks ZIP integrity and confirms that every original non-cache file is
retained, with original configurations unmodified.

## Environment and limits

Tests ran on Python 3.13.5 with Torch 2.10.0+cpu, NetworkX 3.6.1, NumPy 2.3.5,
SciPy 1.17.0, PyYAML 6.0.3, pytest 9.0.2, and RDKit 2025.9.4.

GPU behavior, full training runs, production EDeN NSPDK/FCD, and benchmark
quality were **not** validated. EDeN, `fcd_torch`, and `torch_geometric` were not
installed in this test environment. Optional-backend tests retain their normal
skip behavior; no production metric is replaced silently by a proxy.

No dataset files, trained benchmark checkpoints, or generated benchmark samples
were supplied or added. The delivered archive preserves all original source,
configuration, test, and other non-cache files. It omits Python bytecode and
pytest caches, which are reproducible runtime artifacts, including stale cache
files present in the upload.

## Raw evidence

[Focused test log](validation/pytest_focused.txt) ·
[Updated full-suite log](validation/pytest_full_updated.txt) ·
[Original full-suite log](validation/pytest_full_original.txt) ·
[Machine-readable comparison](validation/comparison.json) ·
[Test environment](validation/environment.json)
