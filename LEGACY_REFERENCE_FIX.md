# Legacy reference sample-ID compatibility

The GraphES molecular A0/A1/A2 learning controls compare their saved
topologies against the canonical generation's `topology_graphs.pkl`.
Historical QM9 reference populations may contain no `graph.graph["sample_id"]`
fields, even though their graph list, ordered topology-state hashes, donor
indices, and manifest checksums were recorded.

`verify_attribute_population()` now permits this **one** legacy case:

1. The parent manifest has the exact recorded SHA-256, and its hashed
   artifacts (including `topology_graphs.pkl`) are independently verified.
2. The reference and control have identical per-graph topology-state hashes,
   generation seed, batch size, and ordered donor indices.
3. **All** reference topology graphs have missing/`None` sample IDs.
4. Logical reference IDs are derived from the verified generation seed and
   graph-list index *in memory*. The parent pickle is never modified.
5. A0/A1/A2 `topology_graphs.pkl` and `base_graphs.pkl` must have the explicit
   exact ID `f"{generation_seed}:{index}"` for every graph.
6. Every indexed node set and edge set must match the reference exactly.

An all-labelled modern reference is also supported. A partially labelled
reference, incorrectly labelled reference, or missing/reordered/duplicated ID
in any control remains an error.

The generation verification receipt records `reference_sample_id_policy`,
`legacy_reference_sample_ids_inferred_from_order`, and the authenticated
reference topology pickle SHA-256.

## Tests

From the repository root:

```bash
PYTHONPATH=src python -m unittest -v \
  tests.test_learning_attribute_reference_compat \
  tests.test_learning_legacy_population_ids
```

## Existing unsealed A0 generations

This compatibility fix does **not** recover an A0 generation whose validation
failed before `learning_generation_seal.json` was written. The runner refuses
to adopt an unsealed generation, and `learning_intent.json` pins the old
implementation hashes. Preserve that run unchanged. Reusing it requires a
separate audited recovery mechanism that records the old and new verifier
versions, rechecks every artifact, verifies exact topology correspondence, and
issues a recovery receipt before writing any missing seal. An alternative is
to start a new run tag and rerun the controlled experiment. Do not rewrite
the historical reference pickle, the saved intent, or the generation seal
manually.
