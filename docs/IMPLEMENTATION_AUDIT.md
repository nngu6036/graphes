# Graph-ER implementation audit

This audit records the implementation state after completing the roadmap in
`README.md`. The normative method reference is `docs/DESIGN_CONTRACT.md`.

## Verdict

The generic and attributed molecular pipelines are implemented end to end:
invariant sampling, invariant-matched construction, hierarchical summary
prediction, teacher supervision, learned selection with `STOP`, constrained
rewiring, and evaluation. The implementation fails explicitly at boundaries
that require an unspecified research choice or unavailable external system.

## Requirement status

| Requirement | Status | Implementation |
| --- | --- | --- |
| Size-conditioned ordinary invariant prior | Implemented | Degree-histogram VAE, feasibility filtering, strict fallback diagnostics |
| Atom-typed-degree invariant prior | Implemented | Training-only signature vocabulary, graph-size conditioning, incidence/valence checks |
| Connected ordinary constructor | Implemented | Havel-Hakimi plus repair and postcondition validation |
| Exact typed constructor | Implemented | Simultaneous residual-demand search, compatibility/connectivity pruning, backtrack/restart diagnostics |
| Explicit invariant and size conditioning | Implemented | Predictor batches carry ordinary or typed invariants, graph size, and normalized `t/T` |
| Hierarchical prediction | Implemented | Pair head consumes fixed/predicted node categories; graphlet head consumes soft pair predictions |
| Attributed graphlets | Implemented | Joint topology/node/edge canonicalization, train-only vocabulary, overflow coordinate |
| Invariant-matched teacher | Implemented | Randomized sources, shared validators, pair-plus-graphlet energy, hard/soft targets, cached `STOP` |
| Memory-bounded trajectories | Implemented | Iterable streaming dataset for full molecular splits |
| Selector modes | Implemented | Energy-only, policy-only, hybrid shortlist, positive-energy gate, learned `STOP` |
| Candidate accounting | Implemented | Separate proposal and valid-candidate budgets; domain checks precede graphlet scoring |
| Typed generation | Implemented | Learned typed prior and typed constructor are wired into QM9/ZINC generation |
| Diagnostics and studies | Implemented | Stage decomposition, constructor bias, consistency, ablations, Pareto, three-seed aggregation |

## Conformance points

- Generic swaps preserve indexed ordinary degree; molecular swaps preserve
  indexed per-bond-type degrees and weighted valence.
- Candidate graphs are simple and undirected, and optionally connected.
- Missing configured node/edge categorical attributes are errors.
- The main graphlet objective excludes connected-mass prediction; that quantity
  is retained only as an external diagnostic/explicit ablation.
- Energy-only mode scores every valid candidate. Policy shortlisting occurs
  only in hybrid mode.
- Teacher time is the actual action step divided by the configured horizon.
  An unreached clean target is never appended as a fake trajectory state.
- Topology-only and attributed evaluation paths remain distinct.

## Explicitly unsupported boundaries

The following APIs raise `NotImplementedError`:

- exact projection of infeasible target summaries;
- hierarchical/coarsened summaries;
- exact constrained state-space coverage for graphs above eight nodes;
- chemical stability, synthesizability, and 3D-quality oracles;
- external baseline execution without a pinned argv adapter; and
- automatic ZINC source download.

These are research definitions or external integrations, not silently
approximated features.

## Verification

Run the following from the repository root:

```bash
python -m pytest -q
ruff check src scripts tests
ruff format --check src scripts tests
python -m compileall -q src scripts tests
```

The suite includes constructor property tests, typed-invariant trajectory
checks, attributed canonicalization, singleton handling, equivariance,
selector/STOP behavior, evaluation metrics, and safe research-protocol
execution.
