from __future__ import annotations

import networkx as nx
import numpy as np

from grapher.rewiring_mlp.attributed.data import (
    GRAPHLET_OVERFLOW_KEY,
    GraphCategoryVocabulary,
    GraphletBasis,
)
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
