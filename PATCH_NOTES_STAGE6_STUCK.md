# Stage 6 runtime patch

The previous tuning config appeared to hang because candidate scoring called
`distance_to_summary` for every sampled rewiring candidate. That function first
computed a full graph summary, including spectral histograms and Python graphlet
orbit counts. Orbit counts are O(n^4) without ORCA, and spectral eigendecomposition
is also expensive for hundreds of thousands of candidates.

Changes:
- `distance_to_summary` now computes only descriptors with non-zero energy weights.
- Evaluation can skip orbit MMD with `evaluation.compute_orbit: false`.
- `scripts/run_coarse_to_fine.py` handles skipped metrics safely.
- Added `configs/experiments/sbm_v0_stage6_fast.yaml`.
- Made `sbm_v0_stage6_tune.yaml` safer by disabling orbit/spectral in refinement
  and reducing candidate budget/steps.

Recommended command:

```bash
PYTHONPATH=src python scripts/run_coarse_to_fine.py \
  --config configs/experiments/sbm_v0_stage6_fast.yaml \
  --num-generate 20 \
  --output-dir outputs/coarse_to_fine/sbm_v0_stage6_fast \
  --debug
```

After the fast run works, increase `steps` and `candidate_budget` gradually.
Do not enable `orbit_weight` in refinement unless ORCA is installed and the graph
sizes are small. Keep `evaluation.compute_orbit: false` during tuning.
