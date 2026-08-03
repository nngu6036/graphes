# Graph-ER implementation audit

Scope: the hybrid predictor/refiner, molecular constraints, graphlet summaries,
the hybrid train/run entry points, and their supplied SBM/QM9 configurations.
The normative reference is `docs/DESIGN_CONTRACT.md`.

## Verdict

The repository implements a usable **generic, energy-only, ordinary-degree
constrained refiner**.  Its valid double-edge swaps preserve indexed ordinary
degrees and, when enabled, connectivity.  The supplied QM9 inference
configuration also performs same-bond-type swaps and applies valence and RDKit
filters.

It does **not** yet implement the paper's attributed model end to end.  In
particular, the typed invariant prior/constructor and attributed graphlet model
are absent, and the QM9 teacher trajectory is generated in the ordinary-degree
state space while inference is restricted to a typed state space.  Results from
the current QM9 path therefore should not be presented as the full molecular
Graph-ER model described in the paper.

## Requirement-by-requirement status

| Requirement | Status | Evidence and gap |
| --- | --- | --- |
| Simple, undirected, degree-preserving double-edge swaps | **Implemented** | `refinement/rewiring.py::is_valid_action` enforces four distinct endpoints, existing removed edges, absent inserted edges, no self-loops/duplicates, and optional connectivity. `hybrid/refiner.py::refine_graph_with_hybrid_predictions` rechecks indexed ordinary degrees and connectivity after selection. |
| Indexed generic source/target alignment and joint relabeling | **Implemented** | `hybrid/data.py::aligned_havel_hakimi_source` constructs an HH source with the target node's indexed ordinary degree; endpoint example construction jointly permutes source and terminal graph; collation checks per-node degree equality. |
| Randomized, invariant-constrained generic source construction | **Partial** | Connectivity repair is present, but `nx.havel_hakimi_graph` is deterministic and the supplied configs disable generation-time relabeling (`configs/experiments/*hybrid_endpoint_graphlet.yaml`, `constructor.random_relabel: false`). There is no randomized tie-breaking among feasible HH choices or uniform constructor baseline under the same mask. |
| Generic size-conditioned DH-VAE and feasibility filtering | **Partial** | `generators/degree_vae.py` conditions degree logits on size, masks degrees `>= n`, and uses multinomial proposals plus graphical/connected checks. The default model uses a conditional GMM prior rather than the paper's standard normal, and failed draws may be repaired or replaced by an empirical fallback instead of being reported strictly as rejected samples. |
| Explicit `I0`, graph size, and typed/ordinary invariant encoding in the predictor | **Partial** | `HybridEndpointBatch` carries normalized ordinary degrees and time. It has no explicit invariant object, typed-degree tensor, or graph-size feature, so `hybrid/model.py` encodes ordinary degree only. |
| Typed-signature prior `(node category, edge-type degree vector)` | **Missing** | No typed-signature vectorizer, histogram VAE, parity/compatibility feasibility checker, or typed sampler exists. The only learned/empirical samplers return ordinary degree sequences (`src/grapher/generators/degree_sampler.py`). |
| Constraint-aware typed source constructor | **Missing** | `construction/coarse.py` realizes only a sorted ordinary degree sequence. QM9 generation then samples atom labels and bond labels after topology construction in `run_hybrid_endpoint_grapher.py` and `molecular/constraints.py`. This is the independent attribute attachment explicitly excluded by the paper. There is no residual typed-demand search, endpoint-compatibility mask, backtracking, or exact completion fallback. |
| Dense indexed node/pair prediction with symmetric no-edge-aware pair output | **Implemented** | `hybrid/model.py` emits node logits and dense symmetric pair logits, including no-edge, and trains pair CE on all upper-triangular real pairs with configurable class weights. |
| Hierarchical endpoint and graphlet conditioning | **Partial** | Current node categories influence message passing, but the pair head does not consume the predicted node posterior and the graphlet heads do not consume soft node/pair predictions. `hybrid/model.py` computes edge logits and graphlet heads as parallel projections of hidden states. In molecular mode, fixed atom one-hots should replace the unnecessary copy-prediction head. |
| Connected induced graphlet target and score for generic graphs | **Partial** | Normalized connected topology graphlets and fixed Monte Carlo sampling are available in `properties/summary.py`. `GraphletBasis` uses the complete topology atlas rather than a training-only vocabulary plus overflow class, and supplied configs model sizes 3-4 rather than the paper default 3-5. Connected mass is trained and remains available as an ablation, but its default refiner weight is now zero. |
| Attributed graphlet canonicalization, prediction, scoring, and evaluation | **Missing** | Hybrid extraction always calls `default_topology_canonicalizer` and `GraphletBasis` always uses topology keys. One attributed nauty aggregation primitive remains in `src/grapher/utils/motifs.py`, but no hybrid/property/evaluation path calls it. Consequently atom and bond categories do not affect the QM9 graphlet target or energy; exact ORCA evaluation is also topology-only. |
| Strict same-edge-type swaps and molecular validity during inference | **Implemented** | `hybrid/refiner.py` samples same-type candidates, transfers the removed type to both inserted edges, and filters with valence and/or RDKit checks. The QM9 config enables both typed-preservation switches, and configuration validation prevents enabling only one. |
| Typed invariant preservation during molecular training | **Missing** | The teacher source copies node attributes but creates HH edges without target bond attributes. Teacher candidates/actions are topology-only and collation checks only ordinary degrees. Thus QM9 intermediate states need not share the target per-node typed degrees, bond counts, or weighted valence, while inference permits only same-type moves. |
| State-conditioned pair/graphlet energy with one fixed prediction per step | **Implemented for energy-only mode** | A prediction is held fixed while candidates are scored and is refreshed after every accepted step in the supplied configs. The pair gain is the changed-pair difference in log probability, which is exactly the full all-pairs energy difference because unchanged terms cancel; graphlet distance uses current-minus-candidate improvement. Supplied configs use soft pair probabilities and graphlet means rather than sampled Hamming guidance. |
| Energy-only, policy-only, hybrid selector, and explicit `STOP` | **Partial** | Energy-only greedy/softmax selection is implemented. There is no policy input, policy shortlist, learned stop logit, or selector loss in `scripts/train_hybrid_endpoint_grapher.py`; the disconnected legacy selector was removed during consolidation. |
| Terminal-target teacher trajectories | **Partial** | Every stored predictor example shares the terminal node/pair/graphlet labels. The teacher itself greedily minimizes topology edge disagreement only: it does not score the true categorical-plus-graphlet energy, form a temperature-controlled distribution, cache candidate actions, or cache `STOP`. A stalled path appends the terminal graph as a non-trajectory `t=1` example. |
| Generation loop and hard stopping | **Partial** | The runner constructs a source, predicts, scores valid swaps, refreshes by default, and stops on empty/no-improving candidates or maximum steps. It has no explicit learned `STOP`; `refresh_prediction_every > 1` remains available only as the frozen-prediction ablation. |
| QM9 invariant generation | **Missing** | Even `degree_source: oracle` extracts only the held-out graph's sorted ordinary degrees. Atom/bond categories are newly sampled from training priors, so this is not the paper's oracle typed-invariant diagnostic and cannot isolate rewiring error. |
| ZINC molecular mode | **Missing** | There is no hybrid ZINC config or typed prior. `DEFAULT_MAX_VALENCE` is QM9-specific and the runner does not pass a configurable table; the initializer filters allowed generated bonds to single/double/triple, so aromatic ZINC cannot be initialized as described in the paper. |
| Required diagnostics and controlled comparisons | **Partial** | The hybrid report records ordinary-degree preservation, connectivity, accepted steps, diagnostic endpoint degree match, and final RDKit validity. It does not report per-node typed-invariant preservation, prior feasibility versus constructor success, restarts/trials/yield, proposal/pass rates, proposals per accepted swap, STOP rate, pair NLL/macro-F1, consistency residual, runtime breakdown, or matched-source uniform-swap controls. |
| Conformance tests | **Partial** | Existing tests cover generic equivariance, ordinary degree/connectivity, endpoint target sharing, and a single same-type valence example. There is no end-to-end test that a molecular teacher source shares the target typed invariant, no attributed-graphlet test, no typed-prior/constructor test, no hybrid policy/STOP test, and no trajectory-wide typed-degree/bond-count assertion. |

## Additional scoring, teacher, and robustness findings

- The node prediction does not guide any action: every score/trace hard-codes
  `node_guidance_gain = 0.0`. Node logits are used only to construct a
  diagnostic endpoint sample; unused prediction fields were removed.
- Strict typed rewiring now validates that same-type candidate sampling and
  removed-type preservation are enabled together.
- Missing node/edge attributes silently map to the first configured category
  in `hybrid/data.py`. This hides absent bond labels in molecular teacher
  sources instead of failing invariant validation.
- RDKit candidate checking is independent of the local valence switch; either
  check can be enabled alone or together.
- With `accept_only_improving: true`, the configured `patience` cannot accumulate
  stalled accepted moves: a non-improving best action stops before `stalled` is
  updated.
- Training time is normalized by the number of stored states, including the
  optionally appended clean endpoint, while inference uses `step/(steps-1)`.
  Stalled/short trajectories therefore do not use the same `t/T` convention as
  generation.

## Recommended implementation order

1. Introduce a typed-signature representation, typed feasibility checks, typed
   DH-VAE/empirical sampler, and one typed residual-demand constructor used by
   both training and generation.
2. Make molecular teacher trajectories use that typed source and the exact same
   same-type/domain candidate validator as inference; assert indexed typed
   invariants after every state.
3. Wire attributed canonicalization into `SummaryConfig`, `GraphletBasis`, target
   extraction, candidate scoring, and evaluation. Build model vocabularies from
   train only with an overflow coordinate.
4. Feed explicit `I0`, graph size, and typed degrees to the predictor; condition
   graphlet heads on soft pair predictions. For fixed molecular node categories,
   remove the copy head and supply one-hot categories directly.
5. Make the soft pair discrepancy the default energy. Add a hybrid selector and
   `STOP` only if policy/hybrid results will be claimed; otherwise describe the
   implementation and experiments explicitly as energy-only.
6. Add rejection/yield/search diagnostics and the missing typed, attributed
   graphlet, teacher, selector, and matched-control tests before reporting the
   molecular experiments.

## Verification performed

- Ruff lint and format checks pass for every retained source, script, and test.
- Static Python byte-compilation succeeds for every retained Python file.
- The full retained suite passes: **31 tests passed** on CPU with NetworkX,
  PyTorch, and RDKit installed in an isolated test environment.
- All eight maintained command-line entry points import successfully and
  return `--help`; both supplied hybrid refiner configurations parse with the
  stricter typed-mode validation.
- A wheel builds and installs successfully from `pyproject.toml` without
  runtime dependency resolution.
