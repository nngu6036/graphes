# Refactor notes

This refactor narrows the archive to the paper-facing ordinary-degree hybrid
route and its molecular baseline. The review used repository-wide reference
search, import-closure inspection, Ruff/Vulture static analysis, byte
compilation, and the test suite.

## Removed slices

- summary-only target generation and its coarse-to-fine pipeline;
- the disconnected learned action selector and GraphER-Opt implementation;
- alternative molecular attribute-flow and mixture-CatFlow models;
- forwarding aliases, verification scripts, and exploratory plotting scripts;
- corresponding experiment configurations and tests;
- generated `__pycache__`, `*.pyc`, Ruff, and pytest caches.

The maintained command surface is now:

- dataset preparation: `prepare_generic_dataset.py`,
  `prepare_qm9_topology_dataset.py`;
- degree prior: `train_degree_generator.py`,
  `evaluate_degree_generator.py`;
- hybrid model: `train_hybrid_endpoint_grapher.py`,
  `run_hybrid_endpoint_grapher.py`;
- reports: `evaluate_graph_generation_report.py`,
  `evaluate_generated_molecules.py`.

## Dependency-boundary changes

- Moved `EdgeAwareMPNNLayer` out of a deleted molecular baseline into
  `hybrid/layers.py`.
- Moved the degree sampler factory beside its sampler implementations in
  `generators/degree_sampler.py`.
- Kept degree-sequence repair local to `degree_vae.py`, its only maintained
  consumer.
- Removed the empty legacy `pipeline` package.

## Compactness changes

- Inlined one-use incidental wrappers in dataset loading, hybrid prediction,
  endpoint label sampling, candidate filtering/selection, and summary distance
  code.
- Removed unused graphlet wrappers, JSON adapters, action permutation helpers,
  molecular indexing constants, report constants, imports, duplicate function
  definitions, and unused prediction fields.
- Removed the redundant second YAML load in dataset split creation.
- Kept action validation, graphlet canonicalization/counting, teacher-path
  construction, and molecular validity checks as named functions because they
  are reusable correctness boundaries.
- Retained one attributed-nauty aggregation primitive as a future integration
  boundary for the paper's currently missing attributed graphlet vocabulary.

## Robustness and disclosure

- Strict typed rewiring now requires same-type candidate sampling and
  removed-type preservation to be enabled together.
- RDKit candidate checking works independently of the local valence switch.
- Supplied refiners now score direct predicted pair probabilities and graphlet
  means by default; sampled-endpoint Hamming guidance remains an ablation.
- Molecular train/run entry points and the QM9 configuration explicitly label
  the route as a non-conformant ordinary-degree-plus-attributes baseline.
- Added `pyproject.toml`, optional dependency groups, pytest/Ruff settings, and
  `.gitignore`; replaced documentation that referenced missing files.

## Scope decision

This was a review and consolidation, not a research reimplementation. The
refactor does not silently claim to add the missing typed-signature prior,
typed constructor/teacher, attributed graphlet model, or policy/`STOP` head.
Those gaps and their recommended implementation order are documented in
`IMPLEMENTATION_AUDIT.md`.
