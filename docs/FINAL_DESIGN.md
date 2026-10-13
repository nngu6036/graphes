# GraphER final design

GraphER uses full Laplacian log-gap spectral diffusion, learned structural-summary heads, and a random-length constraint-preserving rewiring process.

- Generic graphs: exact degree-preserving double-edge swaps guided by topology graphlets and orbit summaries.
- Attributed graphs: same-bond-type double-edge swaps preserving node types, ordinary degree, typed degree, bond-type counts, and weighted valence.
- Graphlet orders: 3, 4, and 5.
- No rejection sampling, degree-sequence spectral anchor, or post-hoc repair is used.
