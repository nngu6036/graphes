# Decoupled generic topology generator

## Scope

This stage models an unlabelled simple graph (A) independently of the future
attribute mechanism while preserving the conditional factorization

\[
p(A,X,R)=p_{\mathrm{top}}(A)\,p_{\mathrm{attr}}(X,R\mid A).
\]

Only (p_{\mathrm{top}}) is implemented here. QM9 and ZINC still use the
retained attributed endpoint path until the attribute-only CatFlow/DeFoG stage
is added.

## Training flow

For every terminal training graph (G^\star):

1. Extract its ordinary degree sequence and one cached connected-induced
   graphlet target (H(G^\star)).
2. Construct a connected Havel-Hakimi realization with the same degree
   sequence and optionally randomize it with valid switches.
3. At state (G_t), propose double-edge swaps without consulting the terminal
   adjacency.
4. Score each candidate with the frozen target:

   \[
   \Delta_t(a)=D(H(G_t),H(G^\star))
   -D(H(T_a(G_t)),H(G^\star)).
   \]

5. Follow a hard or soft distribution over positive-improvement actions, or
   cache `STOP` when none improves.
6. Train `TopologyGraphletPredictor` on each visited state using only graphlet
   mean/distribution loss and optional connected-mass loss.

Connected topology graphlets are enumerated exactly. Candidate counts are
updated by exact local deltas over connected subsets containing one of the
removed or added edges, so sparse Grid graphs do not collapse to empty
histogram blocks and candidates do not require full-graph recounts. The same
cached terminal histogram is reused by teacher scoring and supervision.

## Generation flow

1. Sample a feasible degree sequence with DH-VAE. Maintained generic configs
   use rejection-only postprocessing, so an invalid raw decoder draw is never
   silently projected into a different degree sequence.
2. Construct and validate a connected Havel-Hakimi source (G_0).
3. Predict a graphlet target
   \(widehat H_t=f_\theta(G_t,d,t/T)\).
4. Propose valid switches and score

   \[
   \Delta_t(a)=D(H(G_t),\widehat H_t)
   -D(H(T_a(G_t)),\widehat H_t).
   \]

5. Apply the best positive-improvement action, recompute the prediction from
   the accepted state, and repeat. Stop on no candidate, no positive gain, a
   revisited-state mask, or the step budget.

Every accepted action preserves indexed node degrees, simplicity, and
connectivity. The local decrease guarantee holds for the frozen
current-step prediction; recomputing the prediction means no global monotonic
energy claim is made.

Degree-prior diagnostics distinguish raw graphicality and connected
feasibility from repair, fallback, and accepted output validity. The maintained
topology path disables repair and fallback; the fields remain explicit so
ablation runs cannot be mistaken for native DH-VAE yield.

Generation writes `topology_refined_graphs.pkl`. The old
`hybrid_refined_graphs.pkl` duplicate is emitted only when the explicit legacy
alias option is enabled.

## Deliberate incompatibilities

- `topology_graphlet_predictor_v1` does not load an endpoint checkpoint.
- Topology checkpoints pin the portable lexicographic graph6 canonicalizer;
  their coordinates do not depend on whether nauty is installed.
- Generic topology configs contain no categorical vocabulary, edge/no-edge
  class weights, edge loss, consistency loss, pair guidance, or endpoint
  sampling.
- The old learned selector is not used. Its training features contained exact
  graphlet gains although shortlist inference did not, creating a distribution
  shift. Exact energy selection is the clean reference implementation.
- Graphlet equality does not identify a unique adjacency. Teacher or generation
  `STOP` denotes a constrained local graphlet solution, not exact graph
  reconstruction.
