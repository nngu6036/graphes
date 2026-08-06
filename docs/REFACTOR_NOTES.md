# Refactor and roadmap completion notes

The original archive retained only an ordinary-degree energy refiner and a
topology-first molecular baseline. The current tree extends that compact route
to the complete typed and learned-selector pipeline while preserving the
existing module boundaries.

## Added implementation slices

- typed-signature representation, feasibility checks, graph-size-conditioned
  VAE, checkpointing, and evaluation;
- exact typed residual-demand construction with postcondition diagnostics;
- attributed connected graphlet canonicalization and train-only vocabularies;
- explicit invariant/size predictor inputs and hierarchical conditioning;
- randomized invariant-matched hard/soft teachers and streaming trajectories;
- a shared learned selector with `STOP`, policy-only, energy-only, and hybrid
  modes;
- typed QM9/ZINC generation and strict molecular preparation/validation;
- standardized distance metrics, stage studies, ablations, cost frontiers, and
  safe fixed-seed protocol execution.

## Maintained command surface

- preparation: `prepare_generic_dataset.py`, `prepare_qm9_dataset.py`,
  `prepare_zinc_dataset.py`;
- invariant priors: `train_degree_generator.py`,
  `evaluate_degree_generator.py`;
- predictor/refiner: `train_hybrid_endpoint_grapher.py`,
  `run_hybrid_endpoint_grapher.py`;
- reports: `evaluate_graph_generation_report.py`,
  `evaluate_generated_molecules.py`;
- studies: `run_research_protocol.py`.

## Compatibility and disclosure

- The serialized name `sbm` remains an alias for the Community-small
  benchmark; its canonical configuration is
  `configs/datasets/community_small.yaml`.
- Legacy `candidate_budget` remains readable, but current configurations use
  separate proposal and valid-candidate budgets.
- Connected graphlet mass remains available as a named ablation/diagnostic and
  is disabled in main losses.
- Unspecified research choices and unavailable external adapters are explicit
  `NotImplementedError` placeholders, listed in the README and implementation
  audit.
