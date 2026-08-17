# Refactor and decoupled topology notes

The generic route now lives in `grapher.rewiring_mlp.generic` and is statistically coupled
through `p(A,Y)=p_top(A)p_attr(Y|A)` while being computationally staged. This
revision implements the topology factor only. The previous endpoint-coupled
implementation remains in `grapher.rewiring_mlp.attributed` for attributed experiments.

## Added implementation slices

- topology-only batches with no terminal adjacency labels;
- a graph-level graphlet/clustering/orbit predictor with no node/pair heads or no-edge category;
- completed-base structural teachers whose candidate proposals cannot access target edges;
- checksum-verified completed base pools with deterministic one-to-one matching;
- graphlet/clustering/orbit energy refinement with prediction refresh after each accepted swap,
  visited-state rejection, and explicit `STOP`;
- distinct topology checkpoint, configuration, report, and diagnostic schemas;
- topology-aware research diagnostic aggregation without fake pair metrics;

Retained attributed implementation slices include:

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
- generic topology: `train_topology_grapher.py`,
  `run_topology_grapher.py`;
- retained attributed endpoint path: `train_hybrid_endpoint_grapher.py`,
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
- Old generic endpoint checkpoints and selector checkpoints are incompatible
  with the topology path and must be retrained. Their graphlet encoder and
  teacher both depended on terminal pair information.
- The initial topology release uses exact energy selection. A learned shortlist
  is deferred because the old selector was trained with graphlet-gain features
  that were unavailable at shortlist inference time.
- Connected graphlet mass remains available as a named ablation/diagnostic and
  is disabled in main losses; clustering and orbit are active maintained targets.
- Unspecified research choices and unavailable external adapters are explicit
  `NotImplementedError` placeholders, listed in the README and implementation
  audit.
