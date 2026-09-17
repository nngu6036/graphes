# GSDM-Simple Structure3: device-alias hotfix

## Failure and cause

Generation with `--device gpu` resolves the runtime device to `torch.device("cuda")`.
The loaded DH-VAE parameters report a concrete device such as `cuda:0`.
The previous embedded-sampler guard compared these objects directly and raised
`ValueError: Embedded degree model must already be on the sampler device.`
The problem is runtime placement validation, not trained weights or graph summaries.

## Compatibility

No retraining is required. Reuse the existing Structure3 checkpoint, DH-VAE
checkpoint, wrapper configuration, and run ID. There are no architecture,
checkpoint-format, training-objective, graphlet, or rewiring changes. The fix
retains the existing learned-degree provenance and no-fallback checks.

## Immediate workaround (without installing this patch)

Change `--device gpu` to an explicit logical CUDA device, normally
`--device cuda:0`, in the same generation command. Keep CFG and RUN unchanged.
Use the appropriate explicit index when intentionally targeting another visible GPU.

```bash
N=1024
PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config "$CFG" \
  --seed-id 42 \
  --run-id "$RUN" \
  --num-samples "$N" \
  --device cuda:0
```

## Permanent changes

1. `DegreeVAESampler` resolves an unindexed CUDA device to the current concrete
   GPU before validation or loading. Explicit indices are preserved. CPU aliases
   are normalized to the device reported by CPU tensors. Subsequent sampling
   remains on the pinned device even if the process's current CUDA device changes.
2. Structure3 passes the loaded DH-VAE model's actual parameter device to the
   embedded sampler, rather than forwarding the original runtime alias.
3. Genuine mismatches still raise an error. The message now reports the model,
   resolved sampler, and originally requested devices. The guard is not removed,
   and the patch does not silently move an embedded model or fall back to CPU.

The source-level fix changes only these two files:

```
src/grapher/models/dhvae_hh/degree_sampler.py
src/grapher/models/gdsm_simple/structured_pipeline.py
```

The ZIP also adds device regression tests, this guide, and a validation log.
No configuration or checkpoint files are overwritten.

## Install

Place `graphes_gdsm_structure3_device_fix.zip` in the existing project root, then:

```bash
cd /home/quang/graphes
unzip -o graphes_gdsm_structure3_device_fix.zip -d .
```

The archive overwrites the two source files listed above. For a locally modified
copy of those files, review and apply the provided unified diff instead of
replacing whole files:

```bash
git apply --check graphes_gdsm_structure3_device_fix.patch
git apply graphes_gdsm_structure3_device_fix.patch
```

The unified diff contains only the two source changes; the ZIP additionally
includes the new tests and documentation. After installing the fix, the original
`--device gpu` command works with the existing checkpoints; explicit `cuda:N`
also remains supported. No new training run ID is needed.

## Validation

CPU environment: Python 3.13, PyTorch 2.10.0+cpu; no CUDA GPU available.

- Original code against the new regressions: 17 failed, 4 passed, 3 skipped.
  These failures cover the alias bug, unpinned sampling, and missing diagnostic
  details; they are not 17 distinct generation failures.
- Patched code, new regressions: 21 passed, 3 skipped.
- Patched code, complete repository test suite: 217 passed, 3 skipped, 1 warning.

CUDA metadata and current-device selection were mocked in CPU-runnable tests.
The three real CUDA allocation tests are included but were skipped here. The
complete suite also ran the existing small CPU Structure3 train/generate and
trained DH-VAE integration tests. No real-GPU generation or 1,024-graph benchmark
was performed. The single full-suite warning concerns PyTorch nested tensors.

Run locally, from the project root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=src python -m pytest -q tests/test_degree_sampler_device.py tests/test_gdsm_structure3.py
```

The `torch.load(weights_only=False)` FutureWarning in the reported traceback is
separate from the fatal device check. This patch does not modify checkpoint
loading policy or suppress warnings.

## API reference

PyTorch's `torch.device` documentation describes unindexed CUDA devices as the
current GPU and tensor devices as concrete placement metadata:
https://docs.pytorch.org/docs/main/tensor_attributes.html#torch.device
