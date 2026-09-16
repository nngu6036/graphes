# GSDM-Simple reference ladder

`gdsm_simple` is a project-owned spectral reference for controlled GraphER
ablation. It is intentionally smaller than the released GSDM implementation;
results must be labeled **GSDM-Simple**, not GSDM.

## S0: spectral reference (implemented)

- state: sorted adjacency eigenvalues `lambda`;
- forward process: VP-style discrete Gaussian diffusion on `lambda`;
- denoiser: masked Transformer epsilon predictor;
- generation prior: Gaussian eigenvalues + one empirical training eigenbasis;
- graph size: sampled jointly with the empirical eigenbasis;
- realization: `A_soft = U diag(lambda) U^T`, fixed threshold at 0.5;
- no connectivity repair, largest-component filter, degree correction, HH,
  rewiring, graphlets, clustering, or orbit guidance.

This is the reference against which every GraphER addition should be measured.

## Planned one-change-at-a-time ladder

| Stage | Change relative to previous stage | Question isolated |
| --- | --- | --- |
| S0 | GSDM-Simple reference | How well does spectral eigenvalue diffusion + empirical `U` work? |
| S1 | + degree histogram conditioning | Does knowing the invariant improve spectral prediction? |
| S2 | + HH source / endpoint bridge | Does a degree-realizable source improve the reverse path? |
| S3 | + degree-preserving rewiring realization | What is gained/lost by replacing threshold reconstruction with constrained realization? |
| S4 | + permutation-invariant eigenspace summary | Does additional global structure improve the constrained search? |
| S5 | + graphlet summary | What is the marginal value of local higher-order guidance? |

Each stage should keep the same dataset split, evaluation code, sample count and
seed set. Use a new config and run ID for every stage. Do not modify S0 in place.

## S0 commands

```bash
RUN=seed_42
N=1024

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage train \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config configs/baselines/gdsm_simple_community_small.yaml \
  --seed-id 42 --run-id "$RUN" --device gpu

PYTHONPATH=src python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset community_small \
  --common-config configs/baselines/common_community_small.yaml \
  --wrapper-config configs/baselines/gdsm_simple_community_small.yaml \
  --seed-id 42 --run-id "$RUN" --num-samples "$N" --device gpu

GEN_DIR="outputs/baselines/gdsm_simple/community_small/$RUN/generations/seed_42_n_${N}"
PYTHONPATH=src python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN_DIR" \
  --output-dir "$GEN_DIR/evaluation_report"
```
