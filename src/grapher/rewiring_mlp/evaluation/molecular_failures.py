"""Non-repairing diagnostics on every generated graph, including invalid draws."""
from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import networkx as nx

# Neutral QM9 representation only; no charged/aromatic generalization is implied.
_CAPACITIES = {6: 4, 7: 3, 8: 2, 9: 1}


def molecular_failure_diagnostics(
    graphs: Sequence[nx.Graph], raw_smiles: Sequence[str | None],
) -> dict[str, Any]:
    if len(graphs) != len(raw_smiles):
        raise ValueError('Raw validity records must align one-to-one with graph draws')
    rows = []; atoms: Counter[str] = Counter(); bonds: Counter[str] = Counter()
    primary: Counter[str] = Counter()
    for index, (g, smi) in enumerate(zip(graphs, raw_smiles)):
        for _, data in g.nodes(data=True):
            atoms[str(data.get('atomic_num', 'missing'))] += 1
        for _, _, data in g.edges(data=True):
            bonds[str(data.get('bond_type', 'missing'))] += 1
        applicable = (
            not g.is_directed() and not g.is_multigraph() and nx.number_of_selfloops(g) == 0
            and g.number_of_nodes() > 0
            and all(d.get('atomic_num') in _CAPACITIES and d.get('formal_charge', 0) == 0
                    for _, d in g.nodes(data=True))
            and all(d.get('bond_type') in (1, 2, 3) for _, _, d in g.edges(data=True))
        )
        row: dict[str, Any] = {
            'graph_index': index, 'raw_valid': smi is not None,
            'num_nodes': g.number_of_nodes(), 'num_edges': g.number_of_edges(),
            'disconnected': (g.number_of_nodes() > 0 and not nx.is_connected(g.to_undirected())),
            'neutral_cnof_diagnostics_applicable': applicable,
            'topology_infeasible': None, 'atom_degree_incompatible': None,
            'bond_valence_exceeded': None,
        }
        if applicable:
            topology_bad = []; atom_bad = []; bond_bad = []
            for node, data in g.nodes(data=True):
                capacity = _CAPACITIES[data['atomic_num']]
                degree = g.degree(node)
                valence = sum(d['bond_type'] for _, _, d in g.edges(node, data=True))
                if degree > max(_CAPACITIES.values()): topology_bad.append(str(node))
                if degree > capacity: atom_bad.append(str(node))
                if valence > capacity: bond_bad.append(str(node))
            row.update({
                'topology_infeasible': bool(topology_bad), 'topology_infeasible_nodes': topology_bad,
                'atom_degree_incompatible': bool(atom_bad), 'atom_degree_incompatible_nodes': atom_bad,
                'bond_valence_exceeded': bool(bond_bad), 'bond_valence_exceeded_nodes': bond_bad,
            })
        if smi is not None:
            reason = 'raw_valid'
        elif g.number_of_nodes() == 0:
            reason = 'empty_topology'
        elif not applicable:
            reason = 'outside_neutral_cnof_diagnostic_scope'
        elif row['topology_infeasible']:
            reason = 'topology_infeasible'
        elif row['atom_degree_incompatible']:
            reason = 'atom_degree_incompatible'
        elif row['bond_valence_exceeded']:
            reason = 'bond_valence_exceeded'
        else:
            reason = 'other_conversion_or_sanitization_failure'
        row['primary_classification'] = reason
        primary[reason] += 1
        rows.append(row)
    flags = {}
    for key in ('disconnected', 'topology_infeasible', 'atom_degree_incompatible', 'bond_valence_exceeded'):
        known = [row[key] for row in rows if row[key] is not None]
        flags[key] = {'count': sum(known), 'denominator': len(known),
                      'rate': sum(known)/len(known) if known else None}
    def frequencies(counts):
        total = sum(counts.values())
        return {key: count / total for key, count in sorted(counts.items())} if total else {}
    return {
        'format': 'neutral_cnof_failure_diagnostics_v1',
        'num_graphs': len(graphs),
        'scope': 'Neutral C/N/O/F with integer single/double/triple bonds; unsupported graphs are not reinterpreted.',
        'classification_rule': 'raw valid, empty, outside scope, topology infeasible, atom-degree, bond-valence, other (first applicable); disconnectedness is separate.',
        'flags_overlap': True, 'primary_counts': dict(primary), 'overlapping_flags': flags,
        'marginal_population': 'all_generated_graphs_including_invalid',
        'atom_counts': dict(sorted(atoms.items())), 'atom_frequencies': frequencies(atoms),
        'bond_counts': dict(sorted(bonds.items())), 'bond_frequencies': frequencies(bonds),
        'per_graph': rows,
    }
