# Molecular final-connectivity ablation

## Question

Does rejecting disconnected **final** molecular graphs and resampling replacements improve molecular distribution quality without changing GraphER training, the learned checkpoint, local rewiring, or the requested returned sample count?

This is a sampling-only intervention. It does **not** take the largest connected component, delete fragments, add bonds, or sanitize/repair a returned graph. A disconnected final draw is rejected as a whole trajectory and a new trajectory is sampled.

## Variants

- **Baseline (`VARIANT=main`)**: the existing GraphER sampler. It returns every final categorical graph, including disconnected outputs.
- **Connectivity filter (`VARIANT=connectivity_filter`)**: reuses the exact main checkpoint and generation seed, but requires the final returned graph to be connected. Disconnected final draws are rejected and replaced until `N` connected outputs are returned, subject to `max_attempt_multiplier=10.0`.

The fixed configuration is:

```yaml
final_acceptance:
  require_connected: true
  max_attempt_multiplier: 10.0
```

The baseline resolves the default `require_connected: false`.

## Primary experiment

Run three paired seeds on QM9 and ZINC. Train the main checkpoint exactly once per seed. Then run both sampling variants from that checkpoint.

```bash
# Example: ZINC, seeds 42/43/44, 10,000 returned molecules per variant.
SEEDS="42 43 44" N=10000 DEVICE=cuda:0 \
  scripts/run_molecular_connectivity_ablation.sh zinc

# QM9
SEEDS="42 43 44" N=10000 DEVICE=cuda:0 \
  scripts/run_molecular_connectivity_ablation.sh qm9
```

The driver assumes these checkpoints already exist:

```text
outputs/baselines/gdsm_simple/<dataset>/seed_<seed>_grapher_g345/
```

If a checkpoint is missing, train it first with the normal research command, e.g.

```bash
SEED=42 VARIANT=main scripts/run_grapher_research.sh zinc train
```

## Required reporting

For each paired seed report:

- native validity, uniqueness, novelty, NSPDK, and FCD;
- returned-output connectedness;
- **raw final connectedness rate before rejection**;
- number of attempted trajectories;
- number rejected for disconnectedness;
- generation yield = returned / attempted;
- runtime if comparing cost.

The connectivity-filter run must not be reported simply as “100% connected” without the raw rejection/yield statistics, because the returned distribution is conditioned on final connectedness.

Generation writes `final_acceptance_diagnostics.json` and also records the acceptance policy and counts in `manifest.json` and `rewiring_diagnostics.json`. When rejections occur, the rejected raw graphs are preserved in `rejected_disconnected_graphs.pkl` for failure analysis; they are never included in `molecular_graphs.pkl`.

## Paired interpretation

The generated sets are paired by checkpoint and initial RNG seed, but after the first rejection the random streams necessarily diverge because the filtered sampler draws replacement trajectories. Therefore, treat FCD/NSPDK as paired **run-level** measurements by seed, not molecule-by-molecule paired outcomes.

The summarizer checks checkpoint SHA256, generation seed, requested count, and returned count before computing per-seed deltas and mean/sample-standard-deviation summaries:

```bash
python scripts/summarize_molecular_connectivity_ablation.py \
  --baseline-dir <seed42-baseline> --fixed-dir <seed42-filtered> \
  --baseline-dir <seed43-baseline> --fixed-dir <seed43-filtered> \
  --baseline-dir <seed44-baseline> --fixed-dir <seed44-filtered> \
  --output-dir outputs/ablations/zinc_final_connectivity
```

A negative `FCD` or `NSPDK` delta means the connectivity-conditioned sampler improved that metric; a positive delta means it worsened it. Always interpret these changes together with generation yield.

## Why filtering is final-only

GraphER's categorical reverse transitions can add/delete/recolor edges after earlier guidance events. Enforcing connectedness only at terminal initialization would therefore not guarantee a connected final molecule. Final-sample acceptance directly enforces the property on the object that is actually returned while leaving the learned reverse process and event-local same-type rewiring unchanged.
