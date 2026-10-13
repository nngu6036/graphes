# Strict molecular rewiring

The attributed GraphER sampler now uses a stricter local-environment-preserving
same-bond double-edge swap.

For two edges of the same bond category,

\[
(a,b,r),(c,d,r),
\]

an ordinary same-bond swap may change which atom types are adjacent.  The
stricter variant rejects any candidate unless every node keeps exactly the same
number of neighbours for every `(neighbor atom type, bond type)` pair.

Equivalently, for each node \(i\) it preserves

\[
q_i(x,r)=\#\{j: x_j=x,\; r_{ij}=r\}.
\]

This contains the ordinary typed-degree invariant as a marginal and therefore
also preserves:

- node types;
- ordinary degree of every node;
- per-node degree in every bond category;
- global bond-category counts;
- per-node weighted bond valence
  \(v_i=\sum_j \operatorname{bondOrder}(r_{ij})\);
- simplicity and connectivity.

The implementation audits the weighted bond-valence vector before and after
rewiring.  On QM9 it additionally checks the projected QM9 valence envelope.
When `rdkit_candidate_filter: true`, every changed candidate must pass RDKit
sanitization. Therefore a raw-valid molecular source cannot become invalid due
to an accepted rewiring step.  Invalid sources may remain invalid when no valid
candidate is accepted.

The final QM9/ZINC configs use:

```yaml
attributed_diversification:
  enabled: true
  mode: stochastic_same_bond_diversification
  strict_same_bond: true
  preserve_neighbor_atom_type_by_bond_type: true
  verify_bond_valence_preservation: true
  rdkit_candidate_filter: true
```

Generation diagnostics now include:

```text
neighbor_atom_bond_signature_preserved
bond_valence_preserved
source_bond_valence_vector
final_bond_valence_vector
source_qm9_bond_valence_valid
final_qm9_bond_valence_valid
source_rdkit_valid
final_rdkit_valid
rdkit_validity_preserved_for_valid_source
```

The aggregate diagnostic includes preservation rates for the stronger local
signature, bond valence, and RDKit validity among raw-valid source molecules.
