# Patch notes: Degree Sequence Generator integration

This patch adds a dedicated degree-sequence generator to the coarse-to-fine graph generation pipeline.

## Added

- `src/grapher/generators/degree_vectorizer.py`
  - Converts graph degrees to permutation-invariant degree histograms.
  - Repairs sampled degree sequences into graphical, connected-feasible sequences.
  - Stores empirical node counts and empirical degree sequences for robust fallback.

- `src/grapher/generators/degree_vae.py`
  - Implements `DegreeHistogramVAE`, a VAE over degree histograms.
  - Uses multinomial-style degree histogram reconstruction and KL regularization.

- `src/grapher/generators/degree_sampler.py`
  - Implements `DegreeVAESampler` and `EmpiricalDegreeSampler`.

- `scripts/train_degree_generator.py`
  - Trains the degree-histogram VAE.

- `scripts/verify_degree_generator.py`
  - Reports graphicality, connected-feasibility, constructor validity, and degree MMD.

- `tests/test_degree_generator.py`
  - Smoke test for sampled degree summaries.

- `configs/experiments/sbm_report_degreevae.yaml`
  - Report-grade config with `degree_generator.enabled: true`.

## Updated

- `src/grapher/properties/sampler.py`
  - Adds `HybridSummarySampler`, which merges structural summaries from SummaryVAE with degree summaries from DegreeVAE.

- `src/grapher/pipeline/coarse_to_fine.py`
  - Wraps the configured summary sampler with the degree generator when `degree_generator.enabled: true`.

- `scripts/build_rewiring_teacher.py`
  - Uses the same hybrid sampler for teacher-target summaries.

- `scripts/verify_stage.py`
  - Adds `--stage degree_generator` and makes `summary_generator` verification honor the hybrid degree sampler.

## Typical workflow

```bash
PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/sbm_report_degreevae.yaml \
  --output-dir outputs/degree_generators/sbm_report \
  --epochs 300 \
  --batch-size 32 \
  --beta 0.005 \
  --degree-weight 5.0 \
  --edge-moment-weight 0.1 \
  --seed 42

PYTHONPATH=src python scripts/verify_degree_generator.py \
  --config configs/experiments/sbm_report_degreevae.yaml \
  --num-samples 1000

PYTHONPATH=src python scripts/verify_stage.py \
  --stage degree_generator \
  --config configs/experiments/sbm_report_degreevae.yaml
```

After the degree generator passes, rebuild the teacher cache and retrain the rewiring selector using `configs/experiments/sbm_report_degreevae.yaml`.
