# GraphER HH / Spectral / Graphlet Guidance Factorial

This experiment isolates **generation-time candidate-ranking guidance** while
holding the trained checkpoint, degree prior, HH constructor, source RNG stream,
proposal budget, and evaluation split fixed.

The four reported cases are:

1. **HH**: the common Havel--Hakimi source batch (no refinement).
2. **HH + spectra**: only Laplacian-eigenvalue discrepancy ranks valid swaps.
3. **HH + graphlet**: only the predicted connected induced graphlet histogram ranks valid swaps.
4. **HH + spectra + graphlet**: equal-weight normalized spectral and graphlet discrepancies rank swaps.

Edge, clustering-histogram, and orbit-summary energies are set to zero in all
refined cases. The common checkpoint is still the jointly trained checkpoint;
therefore this is a **guidance ablation**, not a retraining/auxiliary-loss
ablation.

## Primary unconditional test

```bash
PYTHONPATH=src python scripts/run_guidance_factorial_ablation.py \
  --profile configs/experiments/grapher/ablations/community_small_guidance_factorial.yaml \
  --degree-source learned
```

Outputs are written below:

```text
outputs/topology_generation/community_small_joint_edge_laplacian_graphlets345/seed_42/guidance_ablation/learned/
```

The script verifies that the ordered `coarse_graphs.pkl` batch has the same
fingerprint for all refined modes. Since every swap is degree-preserving, Degree
MMD must also remain unchanged across HH and all refined cases.

## Degree-controlled structural test

To reduce degree-prior mismatch as a confound, repeat the same factorial using
the existing edge-relocation degree source:

```bash
PYTHONPATH=src python scripts/run_guidance_factorial_ablation.py \
  --profile configs/experiments/grapher/ablations/community_small_guidance_factorial.yaml \
  --degree-source edge_relocation
```

This is a diagnostic structural comparison rather than the primary unconditional
model result.

## Important interpretation

The current production GraphER config also uses edge, clustering, and orbit
energies. Therefore **HH + spectra + graphlet** in this factorial is deliberately
not the same as the current full GraphER row. It answers the narrower question:
what do spectral and induced-graphlet guidance contribute when other ranking
terms are removed?

Because all cases reuse the same jointly trained checkpoint, the spectral-only
row can still benefit indirectly from representations learned under auxiliary
heads during training. To test the causal effect of auxiliary *training losses*,
a second experiment must retrain separate checkpoints with the excluded losses
set to zero.
