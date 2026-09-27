# Option C v2: soft degree + normalized-Laplacian consistency

This variant keeps the original Option-C stochastic state and standalone entry
points:

- categorical node diffusion with the training empirical marginal as terminal noise;
- continuous symmetric weighted-adjacency diffusion;
- no categorical edge head;
- no independently diffused spectrum;
- no degree VAE or sampled eigenbasis.

It changes the structural losses attached to the predicted clean weighted
adjacency. The original schema-v1 configs remain supported for controlled
comparison; the new configs live under
`configs/experiments/option_c_soft_consistency/` and use `schema_version: 2`.
Existing schema-v1 checkpoints remain loadable, but a schema-v1 checkpoint
cannot be used with a schema-v2 config.

## 1. Soft edge presence

The denoiser still predicts a clean scaled weighted matrix `W_hat`. For structural
consistency we do not hard threshold it during training. Instead

```
A_soft = sigmoid((W_hat - tau) / temperature)
```

where `tau` and `temperature` are converted from physical edge units to the
scaled diffusion representation. The supplied configs use
`soft_threshold_physical: 0.5`, exactly the midpoint used by the final decoder
between no edge (0) and the first positive edge weight (1). This is enforced by
config validation. `temperature_physical: 0.1` is an explicit starting value,
not a tuned optimum.

For Community-small/Ego-small (`scale=1`), the soft threshold is at 0.5. For
QM9/ZINC (`scale=3`), the network state stores weights divided by 3, so the same
physical threshold is internally 1/6 and the temperature is internally 1/30.
The diagonal and padding are masked to zero.

## 2. Degree consistency

Let

```
d_hat_i = sum_j A_soft[i,j]
d_i     = sum_j 1[W_0[i,j] > 0]
```

The new loss compares ordinary degrees after normalizing both by `n-1`:

```
L_degree = mean_active_nodes [ ((d_hat_i - d_i)/(n-1))^2 ].
```

This directly trains the matrix that will later be thresholded, instead of asking
an auxiliary summary head to predict the degree distribution. It does not impose
a hard degree constraint during sampling.

## 3. Normalized-Laplacian spectral consistency

The old Option-C spectral loss used eigenvalues of the predicted weighted
adjacency divided by `sqrt(n)`. Schema v2 instead constructs a normalized
Laplacian from `A_soft`:

```
D_ii  = sum_j A_soft[i,j]
Lnorm = diag(1[D_ii > eps]) - D^{-1/2} A_soft D^{-1/2}
```

and minimizes the mean squared error between its sorted eigenvalues and the
normalized-Laplacian eigenvalues of the clean discrete topology.

The clean target follows the same isolate convention as the repository generic
graph evaluator: isolated nodes contribute zero normalized-Laplacian eigenvalues.
This aligns the training representation with the `Spectral MMD` descriptor used
by `evaluate_graph_generation_report.py` (the evaluator subsequently bins these
eigenvalues into a 20-bin histogram for MMD).

No hard threshold enters this loss. Gradients flow through the sigmoid,
normalized-degree factors, `eigvalsh`, the adjacency head, and shared denoiser.
As before, active graph blocks are extracted before eigendecomposition so padded
zeros do not become fake eigenvalues. The supplied profiles use CPU float64 for
the small eigensolves; neural training may remain on CUDA.

## 4. Objective

The supplied starting objective is

```
L = 1.0 L_node
  + 1.0 L_adjacency
  + 1.0 L_degree
  + 1.0 L_normlap_spectral
  + 0.5 L_graphlet
  + 0.5 L_mass
  + 0.5 L_clustering
  + 0.25 L_orbit.
```

`L_adjacency` remains weighted clean-matrix MSE, so molecular bond order is still
learned. `L_degree` and `L_normlap_spectral` operate on soft *edge presence* and
therefore target unweighted molecular/generic topology. The graphlet, mass,
clustering, and orbit heads are unchanged.

These weights and temperature are starting settings. They have not been tuned on
held-out benchmark metrics and no performance improvement is claimed before
running the experiment.

## 5. Refinement consistency

The final same-type rewiring module remains optional and final-only. Under schema
v2 its spectral energy is also changed to normalized-Laplacian consistency:

- the fixed continuous target matrix is soft-thresholded with the same threshold
  and temperature;
- candidate discrete graphs use their actual binary topology;
- both are compared in normalized-Laplacian eigenvalue space.

The weighted-adjacency MSE term remains unchanged. A separate degree energy is
not added because a same-type double-edge swap already preserves ordinary and
typed degrees within the final refinement event.

## 6. Backward compatibility

- `configs/experiments/option_c/*.yaml` remain schema v1 and preserve the old
  weighted-adjacency spectral objective.
- `configs/experiments/option_c_soft_consistency/*.yaml` are schema v2.
- The shared training/generation scripts read the schema version and select the
  corresponding loss semantics.
- Generation contracts include the consistency configuration, so mixing a v1
  checkpoint with a v2 config is rejected.
- The checkpoint loader includes the legacy-PyTorch compatibility fix: on old
  versions that do not expose `weights_only`, trusted local checkpoints are
  loaded without passing the unsupported keyword.

## 7. New tests

`tests/test_option_c_soft_consistency.py` covers:

- all four schema-v2 profiles;
- physical/scaled soft-threshold equivalence;
- degree consistency behavior;
- clean normalized-Laplacian targets (`P3 -> [0,1,2]`);
- padding and permutation invariance;
- gradient flow through soft threshold + normalized-Laplacian eigensolve;
- joint-loss backpropagation into the adjacency head;
- small end-to-end training/generation; and
- checkpoint contract separation between schema v1 and v2.

The focused regression suite used for this patch reports 290 passed and 5 skipped
(CUDA-only skips).
