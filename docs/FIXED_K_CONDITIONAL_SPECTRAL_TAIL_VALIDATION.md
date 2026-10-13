# Validation: fixed-k conditional spectral tail

Validation was run on the updated repository after implementing conditional
high-frequency spectral-tail completion.

Focused fixed-k tests:

```text
tests/test_gdsm_fixed_k_spectral_tail.py
4 passed
```

The tests cover:

- fitting the training-only conditional ridge-Gaussian tail model;
- non-zero sampled tail coordinates for truncated k;
- nonnegative, nondecreasing completed Laplacian spectra;
- threshold decoder use rather than the stochastic degree-budget decoder;
- all shipped Community-small/Ego-small k/seed configs;
- the k=full no-op contract, including no extra RNG consumption.

Relevant generic spectral/structure regression suite:

```text
60 passed
```

Relevant attributed regression tests that import the common sampler:

```text
43 passed
```

A broad `tests/test_gdsm*.py` run on the updated repository produced:

```text
420 passed, 5 skipped, 16 failed, 18 errors
```

The same run on the untouched uploaded repository produced:

```text
416 passed, 5 skipped, 16 failed, 18 errors
```

The failure/error set is inherited from the uploaded repository and is centered
on `tests/test_gdsm_training_basis.py` (for example the existing
`save_degree_trajectory`, `proposal_basis`, and basis-provenance contract
mismatches). The conditional-tail implementation introduces four new passing
tests and no new failure category.

No full 10,000-epoch GPU Community-small/Ego-small sweep was run in this
environment. The shipped configs should first be selected on validation, with
`k=full` used as the baseline-recovery sanity check.
