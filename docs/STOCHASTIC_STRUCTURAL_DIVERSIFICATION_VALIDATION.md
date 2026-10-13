# Validation: stochastic structural diversification

The implementation was validated against the uploaded repository on 2026-10-13.

Focused tests:

```
PYTHONPATH=src pytest -q \
  tests/test_gdsm_soft_rewiring.py \
  tests/test_gdsm_simple_structure_summary.py \
  tests/test_gdsm_simple_laplacian_loggap_graphlet.py \
  tests/test_gdsm_simple_laplacian_loggap_options.py \
  tests/test_gdsm_topology_degree_histogram_g346.py
```

Result: `22 passed`.

Broader relevant suite:

```
PYTHONPATH=src pytest -q \
  tests/test_gdsm_simple*.py \
  tests/test_gdsm_soft_rewiring.py \
  tests/test_gdsm_topology_degree_histogram_g346.py \
  tests/test_gdsm_topology_graphlet_k6.py
```

Result: `65 passed`.

The new tests cover source-anchored trust-region filtering, reproducible random
step budgets, config validation, and an end-to-end CPU generation smoke test in
which degree preservation and trust-region preservation are asserted.

The complete suite was also run.  The updated tree produced `722 passed, 18
failed, 18 errors, 8 skipped`; the unmodified uploaded tree produced `718
passed, 18 failed, 18 errors, 8 skipped`.  The failure/error set is inherited
from the uploaded repository (SPECTRE exposure-budget assertions and the
separate GDSM training-basis/categorical branch).  No new full-suite failure or
error IDs were introduced.
