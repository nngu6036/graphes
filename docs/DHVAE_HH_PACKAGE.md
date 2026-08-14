# DH-VAE + Havel--Hakimi package boundary

The project-owned degree-sequence baseline is isolated under
`src/grapher/models/dhvae_hh/`. This separates the optional from-scratch base
generator from GraphER's structural-summary predictor and rewiring code.

## Canonical modules

```text
grapher.models.dhvae_hh
├── wrapper.py             Common baseline-wrapper contract
├── degree_vae.py          Ordinary degree-histogram VAE
├── typed_degree_vae.py    Typed-signature VAE
├── degree_sampler.py      Learned and empirical invariant samplers
├── havel_hakimi.py        Ordinary randomized/connected HH realization
├── typed_constructor.py   Exact typed-degree realization
├── training.py            DH-VAE training implementation and CLI
└── evaluation.py          Baseline-only diagnostics and CLI
```

New code must import these canonical modules directly. The former modules
under `grapher.generators` and `grapher.construction`, plus the two scripts,
are logic-free compatibility shims.

## Shared code that intentionally remains outside

The following modules are shared project infrastructure rather than part of
the baseline:

- prepared dataset loading under `grapher.data`;
- device and serialization helpers under `grapher.utils`;
- neutral typed-invariant records and validation in
  `grapher.rewiring_mlp.molecular.typed_invariants`; and
- molecular valence/compatibility rules in `grapher.rewiring_mlp.molecular.constraints`.

The typed VAE previously embedded in `typed_invariants.py` has moved into the
baseline package. Importing the neutral invariant module therefore no longer
loads Torch or the DH-VAE. A lazy compatibility bridge retains the former
typed-VAE names for one migration cycle.

## Dependency direction

The baseline may depend on shared data, invariant, molecular-constraint, and
utility modules. It must not import `grapher.rewiring_mlp.core`, `grapher.rewiring_mlp.generic`, or
`grapher.rewiring_mlp.attributed`.

The current topology and hybrid data modules retain lazy adapters for legacy
HH-created teacher states. Those imports occur only when the legacy source
builder is called; importing or applying the Rewiring MLP does not load the
baseline package. Post-generation experiments should instead pass completed
serialized base graphs through the common correction interface.

This refactor changes code ownership and import boundaries. It does not change
the DH-VAE objective, HH realization behavior, GraphER predictor, candidate
swaps, or correction rule.
