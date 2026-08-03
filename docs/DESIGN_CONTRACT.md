# Graph-ER implementation contract

This document translates *Graph-ER: Structure-Guided Degree-Constrained Graph Generation* into an implementation-facing contract. Page references use the numbered PDF pages.

Labels:

- **REQUIRED**: semantic behavior needed for paper conformance.
- **OPTIONAL**: a supported feature or approximation that may be disabled.
- **REPORTED**: behavior that experiments must measure or audit.
- **UNSPECIFIED**: the paper leaves the exact implementation choice open.

## 1. Graph representation and invariants

### 1.1 Common graph state

- **REQUIRED**: represent a simple, undirected graph; valid states have no self-loops or parallel edges (§3, pp. 3-4).
- **REQUIRED**: the source, every accepted intermediate state, and the returned graph are connected (§5.1, p. 6; §E.1, p. 19).
- **REQUIRED**: distinguish a graph-level invariant sample `I0` from its indexed assignment to nodes.
- **REQUIRED**: once node records are assigned, every accepted swap preserves each indexed node's invariant, not merely the sorted graph-level histogram (§5.1, p. 6).
- **REQUIRED**: pass `I0`, graph size, and normalized time `t/T` to the state-conditioned model (§5.2, p. 6; §D.1, pp. 18-19).

### 1.2 Generic invariant

- **REQUIRED**: the generic invariant is the ordinary-degree multiset, represented by its permutation-invariant histogram `h_D` (§4.1, p. 5).
- **REQUIRED**: every accepted action leaves every indexed degree unchanged (§E.1, Proposition E.1, p. 19).
- **REQUIRED**: a source/final batch sharing invariant samples has identical degree distributions; unequal source/final degree MMD is an implementation or accounting error (§I.2, p. 22).

### 1.3 Attributed invariant

- **REQUIRED**: use complete node signatures `tau_v = (x_v, (d_v,r)_r)`, containing node category and incident count for every edge category (§4.4, p. 5; §A.1, Eqs. 9-12, p. 13).
- **REQUIRED**: model a multiset/histogram of complete typed signatures. Do not sample ordinary degrees and independently attach node/edge labels (§A, p. 13).
- **REQUIRED**: preserve node categories and every per-node typed degree; this also preserves ordinary degrees, global edge-category counts, and molecular weighted valence (§A.1, Eqs. 13-14, pp. 13-14; §E.1, p. 19).
- **REQUIRED**: QM9 uses heavy atoms `{C,N,O,F}`, bonds `{single,double,triple}`, and implicit hydrogens (§A.4, p. 15).
- **REQUIRED**: ZINC uses the declared preprocessing vocabulary, including aromatic bonds only when retained by preprocessing (§A.4, p. 15).

## 2. Invariant prior and feasibility

### 2.1 Generic DH-VAE

- **REQUIRED**: condition the encoder/decoder on graph size and consume a permutation-invariant degree histogram (§4.1-4.2, p. 5).
- **REQUIRED**: decode a categorical distribution over admissible degrees and sample exactly `n` entries through a multinomial (§4.2, Eq. 3, p. 5).
- **REQUIRED**: for connected graphs with `n > 1`, exclude degree zero; retain the single-node degree-zero case (§4.2, p. 5).
- **REQUIRED**: train with multinomial negative log-likelihood plus KL regularization (§4.3, Eq. 4, p. 5).
- **REQUIRED**: reject odd degree sums, failed Erdos-Gallai graphicality, and sequences with no connected simple realization (§4.3, p. 5).
- **REQUIRED**: sample graph size from the declared size prior; the paper uses the empirical training distribution (§4.3, p. 5).

### 2.2 Typed DH-VAE

- **REQUIRED**: decode complete typed signatures through a size/domain-masked softmax and sample exactly `n` signatures with a multinomial (§A.2, Eqs. 15-17, p. 14).
- **REQUIRED**: enforce local signature admissibility for maximum degree, valence, bond order, charge, and other declared rules (§A.1, pp. 13-14).
- **REQUIRED**: check even incidence totals per edge type and graphical/connected realizability of aggregate degrees (§A.3, p. 14).
- **REQUIRED**: simultaneous realization allows at most one typed edge per unordered pair and respects endpoint compatibility (§A.3, Eqs. 18-20, p. 14).
- **OPTIONAL**: category-wise graphicality tests are inexpensive necessary filters but are not sufficient (§A.3, p. 14).
- **OPTIONAL**: exact backtracking or integer-feasibility checks may follow cheap filters, especially for small molecular graphs (§A.3-A.4, p. 14).
- **REPORTED**: prior feasibility/rejection must be separated from constructor success (§A.2-A.4, pp. 14-15).

## 3. Invariant-constrained source constructor

- **REQUIRED**: expand `I0` into randomized indexed node records and residual ordinary or typed demands (Algorithm 1, p. 17).
- **REQUIRED**: produce a simple, connected `G0` that exactly realizes `I0` and all enabled domain constraints (§5.1, p. 6; Algorithm 1, p. 17).
- **REQUIRED**: a typed edge `(i,j,r)` is feasible only when both endpoints have residual demand for `r`, the pair is unused, endpoint categories are compatible, and the residual instance remains completable (§A.4, p. 14).
- **REQUIRED**: maintain component/connectivity state and use it in feasibility decisions (Algorithm 1, p. 17).
- **REQUIRED**: backtrack or restart on an infeasible branch and fail only after the declared search budget is exhausted (§A.4, p. 15; Algorithm 1, p. 17).
- **REQUIRED**: verify invariant equality, simplicity, connectedness, and domain validity before returning the source (Algorithm 1, p. 17).
- **REQUIRED**: randomize tie-breaking so output is not determined by storage order (§E.3, p. 20).
- **REQUIRED**: for training, preserve the target's indexed degree/signature assignment and apply any permutation jointly to `G0` and `G*` (§5.1, p. 6; Algorithm 4, p. 18).
- **REQUIRED**: typed construction ranks feasible choices with training-only `p_init(r | tau_i, tau_j)` plus feasibility/connectivity information (§A.4, Eq. 21, pp. 14-15).
- **OPTIONAL**: exact completion checking may be disabled when a declared search/restart method is used.
- **REQUIRED BASELINE**: support uniform selection under the identical feasibility mask (§C.1, p. 16).
- **REPORTED**: record restarts, trials per returned source, constructor success, and failure reasons (§A.4, p. 15).

## 4. State-conditioned predictor and losses

### 4.1 Encoder and heads

- **REQUIRED**: use a permutation-equivariant GNN/graph transformer over `G_t`; graph pooling may add context but must preserve node embeddings for pair decoding (§D.1, pp. 18-19).
- **REQUIRED**: node inputs contain current category and ordinary/typed signature; edge inputs contain current edge category (§D.1, p. 18).
- **REQUIRED**: predict a categorical terminal label for every unordered pair, including currently absent pairs (§5.2, p. 6; §B.1, p. 15).
- **REQUIRED**: the pair alphabet contains `no-edge` plus all supported edge categories (§B.1, p. 15).
- **REQUIRED**: use an endpoint-symmetric pair decoder (§D.2, p. 19).
- **REQUIRED**: for mutable attributed node categories, predict per-node terminal categories first and condition pair prediction on both endpoint posteriors (§5.2, p. 6; §B.1, Eqs. 22-23, p. 15).
- **REQUIRED FOR PAPER'S MOLECULAR MODE**: atom categories are invariant, so supply fixed one-hot values rather than training a copying head (§B.1, p. 15; Table 5, p. 21).
- **REQUIRED**: graphlet heads consume current-state information and soft node/all-pairs predictions (§5.2, p. 6; §B.3, p. 16).
- **REQUIRED**: predictions are indexed node/pair marginals, not only graph-level category histograms.

### 4.2 Losses

- **REQUIRED**: train node/pair heads with categorical cross-entropy against the indexed terminal graph (§B.3, Eq. 29, p. 16).
- **REQUIRED**: use class weighting or negative-pair sampling for no-edge imbalance (§5.2, p. 6; §B.3, p. 16).
- **REQUIRED**: train each available graphlet head against the normalized terminal histogram with a declared divergence (§B.3, Eq. 29, p. 16).
- **OPTIONAL**: add soft ordinary/typed-degree consistency residuals from incident pair probabilities (§B.1, Eq. 24, p. 15).
- **REQUIRED**: expose active node, pair, graphlet, consistency, and selector losses and weights (§5.4, Eq. 8, p. 7; §B.3, Eq. 29, p. 16).
- **UNSPECIFIED**: exact encoder, dimensions, divergence family, optimizer, and numeric weights.

## 5. Connected induced graphlets

- **REQUIRED**: use connected, induced graphlets; default modeled sizes are `{3,4,5}` (§3.3, p. 4; §B.2, pp. 15-16; Tables 4-5, pp. 20-21).
- **REQUIRED**: generic canonicalization preserves topology; attributed canonicalization also preserves node and edge categories (§3.3, p. 4; §B.2, p. 15).
- **REQUIRED**: build each model vocabulary from training data only and add one unseen/overflow class (§B.2, Eq. 25, pp. 15-16).
- **REQUIRED**: normalize by the number of connected induced graphlets of that size and mask head/loss when `|V| < k` (§B.2, Eqs. 26-28, p. 16).
- **OPTIONAL**: report connected-subgraph mass externally; it is not in the normalized predicted histogram (§B.2, Eq. 28, p. 16).
- **OPTIONAL**: use exact counts or fixed Monte Carlo subsets. Candidate comparisons must reuse identical sampled subsets at one step (§B.2, p. 16).
- **OPTIONAL**: incrementally update affected graphlet counts instead of fully recounting (§B.4, p. 16).
- **REQUIRED**: external graphlet MMD uses the union of generated/reference classes, not the model overflow bin (§F.3, p. 20).

## 6. Rewiring candidates, energy, and selector

### 6.1 Candidate validity

- **REQUIRED**: choose two edges with four distinct endpoints and enumerate both cross-reconnection orientations (§3.2, p. 4; Algorithm 2, p. 17).
- **REQUIRED**: reject self-loops, duplicate edges, disconnecting moves, and invariant violations before scoring (§5.3, p. 6; Algorithm 2, p. 17).
- **REQUIRED**: deduplicate actions and maintain separate proposal and valid-candidate budgets (Algorithm 2, p. 17).
- **REQUIRED IN STRICT TYPED MODE**: both removed edges have the same category and both inserted edges retain it (§3.2, p. 4; §E.1, p. 19).
- **REQUIRED**: enforce enabled compatibility, locality, and candidate-level chemical checks before acceptance (Algorithm 2, p. 17).
- **REQUIRED**: always expose `STOP` (§5.3, p. 6; §D.3, p. 19).
- **OPTIONAL**: locality masks and toolkit chemical filters may be enabled; configuration/effects must be reported (§E.4, p. 20).

### 6.2 Energy and selection

- **REQUIRED**: hold `S_hat_t` fixed while comparing all candidates at step `t` (§5.3, pp. 6-7; §B.4, p. 16).
- **REQUIRED**: energy combines weighted all-pairs categorical discrepancy with weighted divergences for every enabled graphlet head (§B.4, Eq. 30, p. 16).
- **REQUIRED**: improvement is current energy minus successor energy (§B.4, Eq. 31, p. 16).
- **REQUIRED**: support energy-only, policy-only, and hybrid modes (§5.3, p. 7).
- **REQUIRED**: energy-guided modes accept only positive current-step improvement (§5.3, p. 7; §C.5, p. 18).
- **REQUIRED FOR HYBRID**: policy shortlists and current-step energy rescores (§D.3, p. 19).
- **REQUIRED FOR POLICY**: a shared scorer is candidate-order independent and learns the teacher distribution including `STOP` (§D.3, Eq. 32, p. 19).
- **OPTIONAL**: visited-state/tabu memory may prevent cycles; a maximum step count remains required (§E.2, p. 20).
- **UNSPECIFIED**: proposal distribution, energy divergences, and numeric score weights.

## 7. Teacher trajectories and training

- **REQUIRED**: construct a randomized connected source with the same indexed invariant as terminal `G*` (§5.4, p. 7; Algorithms 3-4, p. 18).
- **REQUIRED**: extract fixed terminal node labels, all-pairs labels, and graphlet histograms once per trajectory (Algorithm 3, p. 18).
- **REQUIRED**: cache every intermediate state, `t/T`, `I0`, and those same terminal targets (Algorithm 3, p. 18).
- **REQUIRED**: enumerate valid candidates and score teacher actions against true terminal targets, not current predictor estimates (Algorithm 3, p. 18).
- **REQUIRED**: form a declared distribution over improving actions; cache `STOP` and terminate when none improves (Algorithm 3, p. 18).
- **REQUIRED**: train the predictor on cached intermediate states paired with their terminal graph (§5.4, p. 7; Algorithm 4, p. 18).
- **OPTIONAL**: train a selector to imitate teacher distributions including `STOP`; energy-only generation does not require it (Algorithm 4, p. 18).
- **UNSPECIFIED**: §5.4 describes a combined objective while Algorithm 4 describes staged cached training. Either is conforming if all active supervision is retained.

## 8. Generation and stopping

- **REQUIRED**: sample graph size and `I0`, reject infeasible draws, and construct valid `G0` before rewiring (Algorithm 5, p. 19).
- **REQUIRED**: at each step predict `S_hat_t`, build valid candidates, rank in the configured mode, and apply at most one action (§5.5, p. 7; Algorithm 5, p. 19).
- **REQUIRED**: recompute predictions after every accepted swap, but not separately for same-step candidates (§5.5, p. 7; §B.4, p. 16).
- **REQUIRED**: stop on `STOP`, empty candidates, no acceptable energy improvement, exhausted search, or maximum accepted steps (§5.5, p. 7).
- **REQUIRED**: return the last accepted state without post-hoc degree, connectivity, or molecular correction (§5.5, p. 7; Table 5, p. 21).
- **REQUIRED**: do not claim global monotonic descent because prediction recomputation changes the energy between steps (§5.6, p. 7; §E.2, p. 20).

## 9. Diagnostics and controlled comparisons

- **REPORTED**: invariant fidelity, feasibility, constructor success, trials/output, final preservation, and end-to-end yield (§F.4, Eq. 33, p. 21; §I.3, pp. 22-23).
- **REPORTED**: candidate pass/rejection rate, proposals per accepted swap, accepted swaps, `STOP` rate, connectivity, and runtime (§I.4-I.6, pp. 23-24).
- **REPORTED**: pair NLL/macro-F1, graphlet error, and soft consistency residual on held-out indexed states (§F.3, p. 20; §I.6, p. 24).
- **REPORTED**: generic degree, clustering, four-node orbit, connected-induced graphlet MMD, and connectedness (§6.1, p. 8).
- **REPORTED**: molecular pre-repair RDKit validity, uniqueness, novelty, NSPDK, FCD, graphlet fidelity, invariant preservation, and yield (§6.1, p. 8; §F.4, p. 21).
- **REQUIRED CONTROL**: compare identical invariant samples/sources for full refinement, `G0`, and uniform valid swaps with matched accepted-swap counts (§6.3, p. 9).
- **REQUIRED ABLATIONS**: graphlet-only, pair-only, prediction frozen at `G0`, energy-only, and policy-only (§6.3, p. 9).
- **REQUIRED EXTENDED ABLATIONS**: oracle invariant, random feasible constructor, greedy teacher, uniform proposals, small candidate budget, step-budget sweep, and molecular chemical-filter removal (§I.3-I.6, pp. 23-24).
- **REPORTED**: break runtime/memory down by prior, construction, predictor, validation, selector, graphlet updates, and molecular checks (§I.7, p. 25).

## 10. Minimum conformance tests

- Joint graph/invariant permutation permutes node/pair outputs and leaves graph-level graphlet output unchanged (§E.3, p. 20).
- Every candidate has four distinct endpoints and is simple, connected, invariant preserving, and domain valid.
- Every accepted typed swap preserves each node's typed-degree vector exactly.
- Source and final states have identical indexed invariants after every trajectory.
- Jointly permuting `G0` and `G*` retains training-label alignment.
- Pair output is symmetric and covers all `n(n-1)/2` unordered pairs.
- Pair-only has no graphlet loss/score; graphlet-only has no pair head/loss/score (§I.5, pp. 23-24).
- Frozen mode predicts once at `G0`; full mode predicts after each accepted swap (§I.5, p. 24).
- Same-step energy comparison uses one fixed prediction and one fixed sampled graphlet subset.
- Evaluation graphlets do not alter training vocabulary/model dimensions.
- Final molecular validity is measured before repair or correction.

## 11. Explicit non-conformance patterns

- Independently attaching atom/bond labels after sampling ordinary degrees.
- Preserving ordinary degree while changing a node's typed degrees.
- Swapping different bond types in strict molecular mode.
- Predicting only global category histograms rather than indexed pairs including `no-edge`.
- Training source/terminal graphs without consistent node correspondence.
- Guiding every graph toward one empirical graphlet mean instead of state-conditioned per-graph predictions.
- Recomputing predictions separately for same-step candidates.
- Comparing sampled candidate graphlets with different Monte Carlo subsets.
- Repairing invalid final graphs instead of rejecting invalid actions.

The paper leaves several hyperparameters blank and contains unresolved placeholders. Judge conformance by required semantics, not by an unstated architecture, optimizer, threshold, or numeric budget.
