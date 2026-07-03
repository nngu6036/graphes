# Cleanup report

This fresh-start branch removes the previous GraphER research code paths that are not needed for the new proposal.

## Removed from the old codebase

- DH-VAE degree-prior training and sampling scripts.
- Learned GraphER training scripts.
- Molecular GraphER scripts and RDKit-dependent utilities.
- DiGress post-processing optimizer scripts.
- Old model configs for DH-VAE, generic GraphER, molecular GraphER, and attribute GraphER.
- Python bytecode caches and temporary `.orig` files.
- Long legacy README instructions for the previous paper pipeline.

## Kept/reintroduced

- A minimal dataset builder for synthetic SBM graphs.
- Generic graph summary extraction.
- Coarse topology construction.
- Valid double-edge swap utilities.
- GraphER-Opt, a training-free energy-guided refiner.
- Generic evaluation metrics.
- Verification scripts.

## Rationale

The new proposal does not start by training GraphER. Instead, GraphER is initially an energy-guided, permutation-invariant rewiring optimizer. A learned GraphER student may later be trained to amortize the search, but that is not part of the initial branch.
