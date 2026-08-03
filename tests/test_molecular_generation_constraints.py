from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from grapher.hybrid.data import GraphCategoryVocabulary, GraphletBasis
from grapher.hybrid.refiner import (
    HybridPrediction,
    HybridRefinerConfig,
    apply_hybrid_action,
    score_hybrid_candidates,
)
from grapher.molecular.constraints import (
    fit_molecular_attribute_priors,
    initialize_molecular_attributes,
    molecular_valence_errors,
)
from grapher.properties.summary import SummaryConfig


def _molecule(
    atomic_numbers: list[int],
    edges: list[tuple[int, int, int]],
) -> nx.Graph:
    graph = nx.Graph()
    for index, atomic_number in enumerate(atomic_numbers):
        graph.add_node(
            index,
            atomic_num=atomic_number,
            atom_type=atomic_number,
        )
    for u, v, bond_type in edges:
        graph.add_edge(
            u,
            v,
            bond_type=bond_type,
            bond_order=float(bond_type),
        )
    return graph


def test_empirical_initializer_writes_complete_valence_feasible_attributes() -> None:
    training = [
        _molecule([6, 8], [(0, 1, 1)]),
        _molecule([6, 7, 6], [(0, 1, 1), (1, 2, 1)]),
        _molecule([6, 6], [(0, 1, 2)]),
    ]
    priors = fit_molecular_attribute_priors(training)
    initialized = initialize_molecular_attributes(
        nx.path_graph(4),
        priors,
        rng=np.random.default_rng(42),
    )

    assert all(
        "atomic_num" in data and "atom_type" in data
        for _, data in initialized.nodes(data=True)
    )
    assert all(
        "bond_type" in data and "bond_order" in data
        for _, _, data in initialized.edges(data=True)
    )
    assert molecular_valence_errors(initialized) == []


def test_same_type_swap_preserves_bond_order_valence() -> None:
    graph = _molecule(
        [6, 6, 6, 6],
        [(0, 1, 2), (2, 3, 2)],
    )
    vocabulary = GraphCategoryVocabulary(
        node_values=(6, 7, 8, 9),
        edge_values=(1, 2, 3, 4),
        node_attribute="atomic_num",
        edge_attribute="bond_type",
    )
    prediction = HybridPrediction(
        edge_probabilities=np.full((4, 4, 5), 0.2),
        sampled_edge_labels=np.zeros((4, 4), dtype=np.int64),
        graphlet_history={},
        graphlet_connected_mass={},
        sampled_graph=nx.Graph(),
        sampled_degree_match=False,
        sampled_connected=False,
    )
    swapped = apply_hybrid_action(
        graph,
        (((0, 1), (2, 3)), ((0, 2), (1, 3))),
        prediction,
        vocabulary,
        preserve_removed_edge_type=True,
    )

    assert {data["bond_type"] for _, _, data in swapped.edges(data=True)} == {2}
    assert molecular_valence_errors(swapped) == []


def test_strict_typed_rewiring_flags_must_be_enabled_together() -> None:
    with pytest.raises(ValueError, match="Strict typed rewiring"):
        HybridRefinerConfig.from_dict({"require_same_edge_type_pair": True})
    with pytest.raises(ValueError, match="Strict typed rewiring"):
        HybridRefinerConfig.from_dict({"preserve_removed_edge_type": True})


def test_rdkit_filter_does_not_depend_on_valence_filter(monkeypatch) -> None:
    graph = _molecule([6, 6, 6, 6], [(0, 1, 1), (2, 3, 1)])
    vocabulary = GraphCategoryVocabulary(
        node_values=(6, 7, 8, 9),
        edge_values=(1, 2, 3),
        node_attribute="atomic_num",
        edge_attribute="bond_type",
    )
    prediction = HybridPrediction(
        edge_probabilities=np.full((4, 4, 4), 0.25),
        sampled_edge_labels=np.zeros((4, 4), dtype=np.int64),
        graphlet_history={},
        graphlet_connected_mass={},
        sampled_graph=nx.Graph(),
        sampled_degree_match=False,
        sampled_connected=False,
    )
    monkeypatch.setattr(
        "grapher.molecular.graph_io.is_valid_molecular_graph",
        lambda _graph: False,
    )

    rows = score_hybrid_candidates(
        graph,
        [(((0, 1), (2, 3)), ((0, 2), (1, 3)))],
        prediction,
        vocabulary=vocabulary,
        graphlet_basis=GraphletBasis(keys_by_k={}),
        summary_config=SummaryConfig(),
        config={
            "categorical_weight": 1.0,
            "graphlet_weight": 0.0,
            "require_same_edge_type_pair": True,
            "preserve_removed_edge_type": True,
            "enforce_molecular_valence": False,
            "rdkit_candidate_check": True,
        },
    )

    assert rows[0]["molecular_valid"] is False
    assert rows[0]["hybrid_score"] == float("-inf")
