# GraphES molecular learning-ablation reference compatibility

## Diagnosed result (from the user's manifest inspection)

- QM9 seeds 41--45: the parent manifest records
  `sampling.separate_attribute_rng=true` but lacks
  `effective_sample_config`.
- ZINC seed 44: the same legacy-manifest format.
- ZINC seeds 41, 42, 43, 45: both records exist and agree.

The original A0/A1/A2 preflight rejected the six legacy manifests before
training. It did not show that their topology/attribute random streams were
shared.

## Fix

The updated `src/grapher/experiments/learning_ablation.py`:

1. Uses `effective_sample_config` from a modern generation manifest if present.
2. For a legacy manifest, obtains the sample configuration only from the
   *sealed molecular ablation export*, never from a guessed default, current
   YAML contents, or an unverified sidecar.
3. Verifies the original manifest SHA-256, molecular export SHA-256, reuse
   receipt SHA-256 binding, generation context, seed, dataset, checkpoint
   SHA-256, and training manifest SHA-256.
4. Requires both the recorded generation RNG policy and the recovered
   sample configuration to specify `separate_attribute_rng=true`.
5. Requires explicit `use_ema=false`, as in the existing A-controls contract.
6. Records the recovery source and evidence hashes in the frozen-topology
   reference; rechecks them at attribute-only training initialization.

No original checkpoint, generation manifest, graph population, or ablation
receipt is modified.

## Validate locally

From the GraphES repository root:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_learning_attribute_reference_compat.py -v
```

## Run the previously blocked experiments

Pilot one reference:

```bash
bash commands/paper_missing.sh learning --datasets qm9 --seeds 41 --variants A0 A1 A2 --execute
```

If it succeeds, process the remaining seeds without rerunning seed 41:

```bash
bash commands/paper_missing.sh learning --datasets qm9 --seeds 42 43 44 45 --variants A0 A1 A2 --execute
bash commands/paper_missing.sh learning --datasets zinc --variants A0 A1 A2 --execute
```

Recompute molecular learning summaries:

```bash
bash commands/paper_missing.sh learning_summary --datasets qm9 zinc --variants A0 A1 A2 --execute
```

If preflight now rejects a missing or mismatching sealed receipt, stop and
inspect that exact reference. Do not hand-edit a manifest or regenerate
under the same generation ID. A genuinely unverifiable reference requires
a fresh, explicitly named generation.
