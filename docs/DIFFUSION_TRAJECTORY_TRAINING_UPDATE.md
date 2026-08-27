# Diffusion-Trajectory Training Update

This revision removes the spectral-family dependency on oracle rewiring teacher trajectories.

## Previous behavior

Training first generated discrete valid graphs by spectrum-guided rewiring and then used those graph states as predictor inputs. This implicitly treated the discrete rewiring path as the diffusion path.

## Current behavior

For every clean target graph, training constructs/loads only the fixed same-degree source graph and extracts the source/clean structural endpoints. Intermediate training states are sampled directly in continuous summary space from an endpoint-conditioned Brownian bridge.

- Spectrum: continuous eigenvalue coordinates with optional lambda_1 and trace-preserving noise projection.
- Graphlets: continuous CLR/logit coordinates with blockwise zero-mean Gaussian noise.
- No intermediate graph is created.
- No edge swap is performed when preparing spectral-family training data.
- `summary_diffusion.storage: streaming` resamples bridge time/noise each epoch.

The predictor is explicitly conditioned on the fixed source endpoint (source graph context, source spectrum, and source graphlet logits) and on the continuous current summary state.

At generation, the initial source endpoint remains fixed as neural conditioning, while the actual rewired graph supplies the current realized summary. Rewiring is therefore a discrete projection of continuous denoising guidance rather than the definition of the diffusion trajectory.

## Expected training log

A corrected spectral+graphlet training run begins with lines similar to:

```text
Preparing streaming continuous summary-diffusion samples (...)
[GraphER/DiffusionTraining] training path: source summary -> stochastic continuous Brownian bridge -> clean summary; rewiring is generation-only projection and is not used to create training states.
```

The old message `Preparing eager topology trajectories (guidance=spectral_...)` should not appear for spectral-family training.
