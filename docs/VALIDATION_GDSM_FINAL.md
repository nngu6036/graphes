# Validation scope

Executed on the configuration/orchestration add-on, not the full GraphES source:

```
python -m compileall -q scripts tests
bash -n scripts/run_gdsm_final.sh
python -m pytest tests/test_gdsm_final_addon.py -q
```

Result: **30 passed**; Python and Bash syntax checks passed. Local RDKit version:
2025.09.4. The QED tests include real RDKit descriptor calculations on small SMILES
fixtures; repository integration is exercised with an explicit fixture adapter.
No synthetic fixture is a benchmark result.

The latest repository archive could not be materialized. The existing model,
CUDA kernels, live dataset files, full-training behavior, full graphlet vocabulary,
FCD, NSPDK and wall-clock performance have not been independently run here.
The launcher has runtime preflight checks for installed multiscale support,
CUDA when requested, frozen split files/counts/node limits, data aliases, ordinary
prior type, required molecular backends, and ORCA.

The add-on never asserts that the old project's separate 395-test result was run
for this package. Its reported test count is only the 30 local tests included here.
