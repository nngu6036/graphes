# GraphER enriched source + runtime optimizations

This revision adds a degree-conditioned source-enrichment stage and removes the
largest avoidable costs in the spectral+graphlet generation path.

## Generation factorization

The new optional factorization is

```
D ~ degree prior
  -> HH(D)
  -> S_D = DegreeSummaryNet(D)
  -> short fixed-target degree-preserving rewiring toward S_D
  -> G_base (best visited enriched realization)
  -> existing state-conditioned spectral+graphlet GraphER refiner
  -> G_final
```

`DegreeSummaryNet` is trained jointly in the existing spectral+graphlet
checkpoint.  Its input is the sorted degree multiset only.  It predicts a
trace-constrained clean Laplacian spectrum and the k=3,4,5 graphlet CLR
simplexes.  The auxiliary loss uses the same clean targets as the main x0
predictor and is multiplied by `topology_predictor.loss_weights.degree_summary`.

The main refiner starts from `G_base` but keeps the original HH realization as
its neural conditioning graph/source summary.  This avoids changing the
training distribution of the existing state-conditioned predictor.

## Runtime changes

1. Candidate validity uses a virtual-swap BFS instead of materializing a
   NetworkX candidate just to run `nx.is_connected`.
2. Candidate Laplacian eigenspectra are stacked and solved in batches.  `auto`
   uses `torch.linalg.eigvalsh` on CUDA and NumPy batched `eigvalsh` on CPU.
3. Candidate budgets can start small, anneal with progress, and expand only on
   a plateau up to the existing hard maximum.
4. `collate_spectral_examples` no longer recomputes the HH eigenspectrum when
   the example already carries current/source spectra.
5. Streaming diffusion caches fixed HH/clean endpoint spectra and graphlet
   summaries across epochs when `source_randomization_steps=0`; only bridge
   time/noise is resampled each epoch.
6. Existing exact local graphlet-delta updates are retained.

## Recommended config

`configs/experiments/grapher/community_small_topology_spectral_graphlet_v3_enriched_source.yaml`

It keeps the original 256-step final-refinement hard maximum but uses an
adaptive candidate search of 512/128 -> 1024/256, with one plateau expansion up
to the original 2048/512 maximum.  The source enrichment uses at most 32
accepted swaps and a smaller 256/64 -> 512/128 search.

Generation now also saves `enriched_base_graphs.pkl`.  The generic evaluation
script automatically adds an `enriched_base_to_test` row when this file is
present, which exposes whether the learned degree-conditioned enrichment
actually improves HH before the main refiner.
