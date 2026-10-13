import networkx as nx
import numpy as np

from grapher.models.grapher import attributed_diversification
from grapher.properties.summary import SummaryConfig
from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary, GraphletBasis
from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.graphlets import extract_topology_graphlet_target


def _summary_cfg(attributed: bool = False) -> SummaryConfig:
    return SummaryConfig.from_dict({
        "graphlet_history": True,
        "graphlet_k_min": 3,
        "graphlet_k_max": 4,
        "graphlet_connected_only": True,
        "graphlet_topology_filter": "all",
        "graphlet_backend": "exact",
        "graphlet_num_samples": None,
        **({
            "attributed": True,
            "node_attribute": "atomic_num",
            "edge_attribute": "bond_type",
            "attributed_backend": "python",
        } if attributed else {}),
    })


def _targets(graph: nx.Graph):
    node_values = sorted({d["atomic_num"] for _, d in graph.nodes(data=True)})
    edge_values = sorted({d["bond_type"] for _, _, d in graph.edges(data=True)})
    vocab = GraphCategoryVocabulary.from_graphs(
        [graph],
        {
            "node_attribute": "atomic_num",
            "node_categories": node_values,
            "edge_attribute": "bond_type",
            "edge_categories": edge_values,
        },
    )
    top_cfg = _summary_cfg(False)
    typed_cfg = _summary_cfg(True)
    top_basis = TopologyGraphletBasis.from_config(top_cfg)
    typed_basis = GraphletBasis.fit_from_graphs(
        [graph], typed_cfg, vocabulary=vocab, attributed=True, seed=1
    )
    top_hist, top_mass = extract_topology_graphlet_target(
        graph, graphlet_basis=top_basis, summary_config=top_cfg
    )
    history, mass = typed_basis.statistics_for_graph(graph, typed_cfg)
    return (
        vocab,
        top_cfg,
        typed_cfg,
        top_basis,
        typed_basis,
        {"graphlet_histogram": top_hist, "graphlet_mass": top_mass},
        {
            "typed_graphlet_histogram": typed_basis.flatten_history(history),
            "typed_graphlet_mass": typed_basis.flatten_mass(mass),
        },
    )


def _config():
    return attributed_diversification.resolve_options({
        "enabled": True,
        "random_steps": {
            "distribution": "uniform_integer",
            "min_steps": 1,
            "max_steps": 1,
        },
        "proposal_budget": 64,
        "valid_candidate_budget": 32,
        "preserve_connectivity": True,
        "strict_same_bond": True,
        "preserve_neighbor_atom_type_by_bond_type": True,
        "verify_bond_valence_preservation": True,
        "rdkit_candidate_filter": False,
        "guidance": {
            "topology_graphlet_weight": 0.0,
            "topology_graphlet_mass_weight": 0.0,
            "topology_orbit_weight": 0.0,
            "typed_graphlet_weight": 1.0,
            "typed_graphlet_mass_weight": 0.0,
        },
        "soft_selection": {
            "temperature_start": 1.0,
            "temperature_end": 1.0,
            "temperature_schedule": "constant",
            "score_normalization": "none",
            "max_target_worsening": 10.0,
            "max_relative_target_worsening": 10.0,
        },
        "target_trust_region": {
            "enabled": True,
            "absolute_worsening": 10.0,
            "relative_worsening": 10.0,
        },
    })


def _run(graph: nx.Graph):
    (
        vocab,
        top_cfg,
        typed_cfg,
        top_basis,
        typed_basis,
        top_prediction,
        typed_prediction,
    ) = _targets(graph)
    return attributed_diversification.diversify_same_bond_graph(
        graph,
        topology_prediction=top_prediction,
        attribute_prediction=typed_prediction,
        topology_basis=top_basis,
        typed_basis=typed_basis,
        topology_summary_config=top_cfg,
        typed_summary_config=typed_cfg,
        vocabulary=vocab,
        config=_config(),
        rng=np.random.default_rng(3),
    )


def test_strict_local_environment_rejects_atom_pair_changing_swap():
    # The only topologically valid same-single-bond swap changes the atom types
    # seen by all four endpoints, so the stricter local-environment invariant
    # must reject it.
    graph = nx.Graph()
    for node, atom in enumerate([6, 7, 8, 9]):
        graph.add_node(node, atomic_num=atom)
    graph.add_edge(0, 1, bond_type=1, bond_order=1.0)
    graph.add_edge(1, 2, bond_type=2, bond_order=2.0)
    graph.add_edge(2, 3, bond_type=1, bond_order=1.0)

    final, diag = _run(graph)

    assert diag["accepted_steps"] == 0
    assert diag["changed"] is False
    assert diag["candidate_search_totals"].get(
        "neighbor_atom_bond_signature_rejections", 0
    ) >= 1
    assert nx.utils.graphs_equal(final, graph)


def test_strict_local_environment_allows_chemically_equivalent_endpoints_and_preserves_valence():
    # Both single bonds connect C--N.  Swapping their endpoints therefore keeps
    # each node's neighbor-atom-type-by-bond-type counts unchanged.
    graph = nx.Graph()
    for node, atom in enumerate([6, 7, 7, 6]):
        graph.add_node(node, atomic_num=atom)
    graph.add_edge(0, 1, bond_type=1, bond_order=1.0)
    graph.add_edge(1, 2, bond_type=2, bond_order=2.0)
    graph.add_edge(2, 3, bond_type=1, bond_order=1.0)

    final, diag = _run(graph)

    assert diag["accepted_steps"] == 1
    assert diag["changed"] is True
    assert diag["typed_degree_preserved"] is True
    assert diag["neighbor_atom_bond_signature_preserved"] is True
    assert diag["bond_valence_preserved"] is True
    assert diag["source_bond_valence_vector"] == diag["final_bond_valence_vector"]
    assert diag["source_qm9_bond_valence_valid"] is True
    assert diag["final_qm9_bond_valence_valid"] is True
    assert diag["qm9_bond_valence_validity_preserved"] is True
    assert nx.is_connected(final)


def test_strict_options_are_mandatory_when_diversification_is_enabled():
    for key in (
        "preserve_neighbor_atom_type_by_bond_type",
        "verify_bond_valence_preservation",
    ):
        bad = {"enabled": True, key: False}
        try:
            attributed_diversification.resolve_options(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected disabling {key} to be rejected")


def test_rdkit_valid_source_remains_valid_under_strict_rewiring():
    import pytest
    pytest.importorskip("rdkit")
    from grapher.rewiring_mlp.molecular.graph_io import is_valid_molecular_graph

    graph = nx.Graph()
    for node, atom in enumerate([6, 7, 7, 6]):
        graph.add_node(node, atomic_num=atom)
    graph.add_edge(0, 1, bond_type=1, bond_order=1.0)
    graph.add_edge(1, 2, bond_type=2, bond_order=2.0)
    graph.add_edge(2, 3, bond_type=1, bond_order=1.0)
    assert is_valid_molecular_graph(graph)

    (
        vocab,
        top_cfg,
        typed_cfg,
        top_basis,
        typed_basis,
        top_prediction,
        typed_prediction,
    ) = _targets(graph)
    cfg = _config()
    cfg["rdkit_candidate_filter"] = True
    final, diag = attributed_diversification.diversify_same_bond_graph(
        graph,
        topology_prediction=top_prediction,
        attribute_prediction=typed_prediction,
        topology_basis=top_basis,
        typed_basis=typed_basis,
        topology_summary_config=top_cfg,
        typed_summary_config=typed_cfg,
        vocabulary=vocab,
        config=cfg,
        rng=np.random.default_rng(3),
    )

    assert diag["accepted_steps"] == 1
    assert diag["source_rdkit_valid"] is True
    assert diag["final_rdkit_valid"] is True
    assert diag["rdkit_validity_preserved_for_valid_source"] is True
    assert diag["bond_valence_preserved"] is True
    assert is_valid_molecular_graph(final)
