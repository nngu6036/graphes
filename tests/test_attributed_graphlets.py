from __future__ import annotations

import networkx as nx
import numpy as np

from grapher.rewiring_mlp.attributed.data import (
    GRAPHLET_OVERFLOW_KEY,
    GraphCategoryVocabulary,
    GraphletBasis,
)
from grapher.rewiring_mlp.evaluation.metrics import mmd_graphlet_statistics
from grapher.properties.summary import SummaryConfig
from grapher.utils.motifs import attributed_graphlet_count_dict


def _triangle(atoms: tuple[int, int, int], bonds: tuple[int, int, int]) -> nx.Graph:
    graph = nx.Graph()
    for node, atom in enumerate(atoms):
        graph.add_node(node, atomic_num=atom)
    for edge, bond in zip(((0, 1), (1, 2), (0, 2)), bonds):
        graph.add_edge(*edge, bond_type=bond)
    return graph


def test_attributed_canonical_key_is_permutation_invariant() -> None:
    graph = _triangle((6, 7, 8), (1, 2, 1))
    relabeled = nx.relabel_nodes(graph, {0: 2, 1: 0, 2: 1})
    first = attributed_graphlet_count_dict(
        graph,
        3,
        node_label_attr="atomic_num",
        edge_label_attr="bond_type",
        backend="python",
    )
    second = attributed_graphlet_count_dict(
        relabeled,
        3,
        node_label_attr="atomic_num",
        edge_label_attr="bond_type",
        backend="python",
    )
    assert first == second
    assert sum(first.values()) == 1


def test_training_only_attributed_basis_routes_unseen_graphlet_to_overflow() -> None:
    train = _triangle((6, 6, 8), (1, 1, 1))
    unseen = _triangle((6, 7, 8), (1, 2, 1))
    vocabulary = GraphCategoryVocabulary.from_graphs(
        [train, unseen],
        {
            "node_attribute": "atomic_num",
            "node_categories": [6, 7, 8],
            "edge_attribute": "bond_type",
            "edge_categories": [1, 2],
        },
    )
    config = {
        "graphlet_history": True,
        "graphlet_k_min": 3,
        "graphlet_k_max": 3,
        "graphlet_connected_only": True,
        "graphlet_num_samples": None,
        "attributed": True,
        "attributed_backend": "python",
    }
    basis = GraphletBasis.fit_from_graphs(
        [train],
        config,
        vocabulary=vocabulary,
    )
    history, _mass = basis.statistics_for_graph(
        unseen,
        SummaryConfig.from_dict(config),
    )
    vector = basis.flatten_history(history)
    overflow_index = basis.keys_by_k["3"].index(GRAPHLET_OVERFLOW_KEY)

    assert basis.attributed
    assert basis.overflow_key == GRAPHLET_OVERFLOW_KEY
    assert vector[overflow_index] == np.float32(1.0)
    assert np.isclose(vector.sum(), 1.0)


def test_simple_cycle_filter_rejects_trees_and_chorded_cycles() -> None:
    path = nx.path_graph(4)
    cycle = nx.cycle_graph(4)
    chorded = nx.cycle_graph(4)
    chorded.add_edge(0, 2)
    for graph in (path, cycle, chorded):
        nx.set_node_attributes(graph, 6, "atomic_num")
        nx.set_edge_attributes(graph, 1, "bond_type")

    kwargs = {
        "node_label_attr": "atomic_num",
        "edge_label_attr": "bond_type",
        "backend": "python",
    }
    assert attributed_graphlet_count_dict(
        path, 4, topology_filter="simple_cycle", **kwargs
    ) == {}
    assert sum(
        attributed_graphlet_count_dict(
            cycle, 4, topology_filter="simple_cycle", **kwargs
        ).values()
    ) == 1
    assert attributed_graphlet_count_dict(
        chorded, 4, topology_filter="simple_cycle", **kwargs
    ) == {}
    assert sum(
        attributed_graphlet_count_dict(
            chorded, 4, topology_filter="cyclic", **kwargs
        ).values()
    ) == 1


def test_simple_cycle_key_and_basis_metadata_are_permutation_invariant() -> None:
    cycle = nx.cycle_graph(5)
    nx.set_node_attributes(
        cycle,
        {node: (6, 7, 8, 6, 9)[node] for node in cycle.nodes()},
        "atomic_num",
    )
    nx.set_edge_attributes(
        cycle,
        {edge: (1, 2, 1, 3, 1)[index] for index, edge in enumerate(cycle.edges())},
        "bond_type",
    )
    relabeled = nx.relabel_nodes(cycle, {0: 4, 1: 2, 2: 0, 3: 3, 4: 1})
    kwargs = {
        "node_label_attr": "atomic_num",
        "edge_label_attr": "bond_type",
        "backend": "python",
        "topology_filter": "simple_cycle",
    }
    first = attributed_graphlet_count_dict(cycle, 5, **kwargs)
    second = attributed_graphlet_count_dict(relabeled, 5, **kwargs)
    assert first == second
    assert next(iter(first)).startswith("ATTR_CYCLE_V1|")

    vocabulary = GraphCategoryVocabulary.from_graphs(
        [cycle],
        {
            "node_attribute": "atomic_num",
            "edge_attribute": "bond_type",
        },
    )
    basis = GraphletBasis.fit_from_graphs(
        [cycle],
        {
            "graphlet_history": True,
            "graphlet_k_min": 5,
            "graphlet_k_max": 5,
            "graphlet_connected_only": True,
            "graphlet_topology_filter": "simple_cycle",
            "attributed": True,
            "attributed_backend": "python",
        },
        vocabulary=vocabulary,
    )
    restored = GraphletBasis.from_dict(basis.to_dict())
    assert basis.topology_filter == "simple_cycle"
    assert restored == basis


def test_cycle_only_graphlet_mmd_uses_ring_composition_and_mass() -> None:
    cycle = nx.cycle_graph(4)
    path = nx.path_graph(4)
    for graph in (cycle, path):
        nx.set_node_attributes(graph, 6, "atomic_num")
        nx.set_edge_attributes(graph, 1, "bond_type")
    identical = mmd_graphlet_statistics(
        [cycle],
        [cycle.copy()],
        k_min=4,
        k_max=4,
        topology_filter="simple_cycle",
        node_label_attr="atomic_num",
        edge_label_attr="bond_type",
        attributed_backend="python",
    )
    different = mmd_graphlet_statistics(
        [cycle],
        [path],
        k_min=4,
        k_max=4,
        topology_filter="simple_cycle",
        node_label_attr="atomic_num",
        edge_label_attr="bond_type",
        attributed_backend="python",
    )
    assert np.allclose(identical, (0.0, 0.0))
    assert different[0] > 0.0
    assert different[1] > 0.0
