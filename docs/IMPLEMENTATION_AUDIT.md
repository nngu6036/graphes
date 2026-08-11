# Graph-ER implementation audit

This audit records the implementation state after completing the roadmap in
`README.md`. The normative method reference is `docs/DESIGN_CONTRACT.md`.

## Verdict

The generic pipeline is now decoupled end to end: ordinary DH-VAE sampling,
connected Havel-Hakimi construction, topology-only graphlet prediction, a
target-adjacency-free graphlet teacher, and graphlet-energy rewiring with
`STOP`. The existing attributed molecular endpoint pipeline remains isolated
and operational pending the attribute-only CatFlow/DeFoG migration.

Topology graphlets use a portable fixed canonicalizer, exact connected-subset
enumeration, and exact local count deltas for each proposed switch. This avoids
empty sparse-Grid targets and environment-dependent checkpoint coordinates.

## Requirement status

| Requirement | Status | Implementation |
| --- | --- | --- |
| Size-conditioned ordinary invariant prior | Implemented | Degree-histogram VAE, feasibility filtering, strict fallback diagnostics |
| Atom-typed-degree invariant prior | Implemented | Training-only signature vocabulary, graph-size conditioning, incidence/valence checks |
| Connected ordinary constructor | Implemented | Havel-Hakimi plus repair and postcondition validation |
| Exact typed constructor | Implemented | Simultaneous residual-demand search, compatibility/connectivity pruning, backtrack/restart diagnostics |
| Generic invariant and size conditioning | Implemented | Topology batches carry current adjacency, indexed ordinary degrees, graph size, and `t/T` |
| Decoupled generic prediction | Implemented | Graphlet heads consume topology encoder state directly; no node/pair head or no-edge class exists |
| Generic graphlet teacher | Implemented | Random valid proposals and cached graphlet targets; no terminal adjacency access |
| Attributed hierarchical prediction | Retained | Legacy pair head consumes fixed/predicted node categories; attributed graphlet head consumes soft pair predictions |
| Attributed graphlets | Implemented | Joint topology/node/edge canonicalization, train-only vocabulary, overflow coordinate |
| Attributed invariant-matched teacher | Retained | Randomized sources, shared validators, pair-plus-graphlet energy, hard/soft targets, cached `STOP` |
| Generic memory-bounded trajectories | Implemented | Iterable streaming dataset processes one target graph at a time |
| Attributed memory-bounded trajectories | Retained | Iterable streaming dataset for full molecular splits |
| Generic selection | Implemented | Exact graphlet-energy scoring, positive-improvement gate, and explicit `STOP` |
| Attributed selector modes | Retained | Energy-only, policy-only, hybrid shortlist, positive-energy gate, learned `STOP` |
| Candidate accounting | Implemented | Separate proposal and valid-candidate budgets; domain checks precede graphlet scoring |
| Typed generation | Retained | Learned typed prior and typed constructor remain wired into the legacy QM9/ZINC route |
| Diagnostics and studies | Implemented | Stage decomposition, constructor bias, consistency, ablations, Pareto, three-seed aggregation |

## Conformance points

- Generic topology checkpoints cannot load endpoint checkpoints; retraining is
  required because the old graphlet encoder consumed soft pair predictions.
- Generic swaps preserve indexed ordinary degree; molecular swaps preserve
  indexed per-bond-type degrees and weighted valence.
- Candidate graphs are simple and undirected, and optionally connected.
- Missing configured node/edge categorical attributes are errors.
- The main generic topology graphlet objective excludes connected-mass
  prediction; that quantity is retained only as an external diagnostic or
  explicit ablation.
- Generic energy mode scores every retained valid candidate against one frozen
  graphlet prediction. A learned generic shortlist is intentionally deferred.
- Teacher and generation time both use the actual action step divided by the
  configured horizon.
- One cached exact graphlet target is reused across all paths/states from one
  target, including teacher energy computation.
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
checks, attributed canonicalization, singleton handling, generic topology
equivariance and `STOP`, retained endpoint selector behavior, evaluation
metrics, and safe research-protocol execution.
