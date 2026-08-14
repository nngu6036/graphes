from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from grapher.rewiring_mlp.attributed.data import (
    GraphCategoryVocabulary,
    GraphletBasis,
    graph_to_categorical_arrays,
)
from grapher.rewiring_mlp.attributed.refiner import (
    HybridPrediction,
    refine_graph_with_hybrid_predictions,
    score_hybrid_candidates,
)
from grapher.properties.summary import SummaryConfig, extract_summary
from grapher.rewiring_mlp.core.rewiring import apply_action, make_action

CUBE_EDGES = [
    (0, 1),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 7),
    (5, 6),
    (6, 7),
]

ACTION_A = make_action(
    [(0, 1), (2, 6)],
    [(0, 2), (1, 6)],
)

ACTION_B = make_action(
    [(0, 1), (6, 7)],
    [(0, 7), (1, 6)],
)


def _cube_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(8))
    graph.add_edges_from(CUBE_EDGES)
    return graph


def _summary_config() -> SummaryConfig:
    return SummaryConfig(
        clustering_summary=False,
        spectral_summary=False,
        motif_proxy=False,
        orbit_count=False,
        graphlet_history=True,
        graphlet_k_min=3,
        graphlet_k_max=4,
        graphlet_connected_only=True,
        graphlet_num_samples=None,
        graphlet_backend="sampled",
    )


def _edge_set(graph: nx.Graph) -> set[tuple[int, int]]:
    return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges()}


def _prediction(
    current: nx.Graph,
    endpoint_graph: nx.Graph,
    graphlet_graph: nx.Graph,
    *,
    vocabulary: GraphCategoryVocabulary,
    summary_config: SummaryConfig,
) -> HybridPrediction:
    _, edge_labels = graph_to_categorical_arrays(
        endpoint_graph,
        vocabulary,
    )
    node_count = endpoint_graph.number_of_nodes()
    edge_probabilities = np.full(
        (
            node_count,
            node_count,
            vocabulary.num_edge_categories,
        ),
        0.01,
        dtype=np.float64,
    )
    for u in range(node_count):
        for v in range(node_count):
            target_category = int(edge_labels[u, v])
            edge_probabilities[u, v, target_category] = 0.99
            edge_probabilities[u, v] /= edge_probabilities[u, v].sum()

    summary = extract_summary(graphlet_graph, summary_config)
    current_degrees = [int(current.degree(node)) for node in sorted(current.nodes())]
    sampled_degrees = [
        int(endpoint_graph.degree(node)) for node in sorted(endpoint_graph.nodes())
    ]
    return HybridPrediction(
        edge_probabilities=edge_probabilities,
        sampled_edge_labels=edge_labels,
        graphlet_history=summary["graphlet_history"],
        graphlet_connected_mass=summary["graphlet_connected_mass"],
        sampled_graph=endpoint_graph.copy(),
        sampled_degree_match=current_degrees == sampled_degrees,
        sampled_connected=(
            nx.is_connected(endpoint_graph)
            if endpoint_graph.number_of_nodes() > 0
            else False
        ),
    )


def _best_action(rows: list[dict[str, Any]]):
    return max(rows, key=lambda row: float(row["hybrid_score"]))["action"]


def test_cube_weights_switch_between_categorical_and_graphlet_guidance():
    graph = _cube_graph()
    categorical_target = apply_action(graph, ACTION_A)
    graphlet_target = apply_action(graph, ACTION_B)
    vocabulary = GraphCategoryVocabulary.topology_only()
    summary_config = _summary_config()
    graphlet_basis = GraphletBasis.from_config(summary_config)
    prediction = _prediction(
        graph,
        categorical_target,
        graphlet_target,
        vocabulary=vocabulary,
        summary_config=summary_config,
    )
    candidates = [ACTION_A, ACTION_B]

    categorical_rows = score_hybrid_candidates(
        graph,
        candidates,
        prediction,
        vocabulary=vocabulary,
        graphlet_basis=graphlet_basis,
        summary_config=summary_config,
        config={
            "categorical_weight": 1.0,
            "probability_weight": 0.0,
            "graphlet_weight": 0.0,
        },
    )
    graphlet_rows = score_hybrid_candidates(
        graph,
        candidates,
        prediction,
        vocabulary=vocabulary,
        graphlet_basis=graphlet_basis,
        summary_config=summary_config,
        config={
            "categorical_weight": 0.0,
            "probability_weight": 0.0,
            "graphlet_weight": 1.0,
            "graphlet_mass_weight": 0.0,
            "graphlet_top_k": 0,
        },
    )

    assert _best_action(categorical_rows) == ACTION_A
    assert _best_action(graphlet_rows) == ACTION_B
    categorical_by_action = {row["action"]: row for row in categorical_rows}
    graphlet_by_action = {row["action"]: row for row in graphlet_rows}
    assert (
        categorical_by_action[ACTION_A]["categorical_gain"]
        > categorical_by_action[ACTION_B]["categorical_gain"]
    )
    assert (
        graphlet_by_action[ACTION_B]["graphlet_gain"]
        > graphlet_by_action[ACTION_A]["graphlet_gain"]
    )


def test_consistent_one_step_guide_selects_exact_cube_swap():
    graph = _cube_graph()
    target = apply_action(graph, ACTION_B)
    vocabulary = GraphCategoryVocabulary.topology_only()
    summary_config = _summary_config()
    graphlet_basis = GraphletBasis.from_config(summary_config)
    prediction = _prediction(
        graph,
        target,
        target,
        vocabulary=vocabulary,
        summary_config=summary_config,
    )

    def fixed_prediction(*_args, **_kwargs):
        return prediction

    refined, trace = refine_graph_with_hybrid_predictions(
        graph,
        model=None,  # The injected predictor is the model boundary in this test.
        vocabulary=vocabulary,
        graphlet_basis=graphlet_basis,
        summary_config=summary_config,
        refiner_config={
            "steps": 1,
            "candidate_budget": -1,
            "preserve_connectivity": True,
            "selection": "greedy",
            "categorical_weight": 1.0,
            "probability_weight": 0.0,
            "graphlet_weight": 1.0,
            "graphlet_mass_weight": 0.0,
            "graphlet_top_k": 0,
            "accept_only_improving": True,
            "min_improvement": 0.0,
            "sample_endpoint": False,
            "sample_graphlet": False,
        },
        device="cpu",
        rng=np.random.default_rng(0),
        return_trace=True,
        prediction_fn=fixed_prediction,
    )

    assert _edge_set(refined) == _edge_set(target)
    assert dict(refined.degree()) == dict(graph.degree())
    assert nx.is_connected(refined)
    assert len(trace) == 1
    assert trace[0]["accepted"] is True
    assert trace[0]["categorical_gain"] > 0.0
    assert trace[0]["graphlet_gain"] > 0.0


def test_infeasible_sample_is_guidance_only_and_preserves_invariants():
    graph = nx.cycle_graph(8)
    infeasible_endpoint = nx.complete_graph(8)
    vocabulary = GraphCategoryVocabulary.topology_only()
    summary_config = _summary_config()
    graphlet_basis = GraphletBasis.from_config(summary_config)
    prediction = _prediction(
        graph,
        infeasible_endpoint,
        infeasible_endpoint,
        vocabulary=vocabulary,
        summary_config=summary_config,
    )
    assert prediction.sampled_degree_match is False

    def fixed_prediction(*_args, **_kwargs):
        return prediction

    refined, trace = refine_graph_with_hybrid_predictions(
        graph,
        model=None,  # The infeasible sampled graph is never installed.
        vocabulary=vocabulary,
        graphlet_basis=graphlet_basis,
        summary_config=summary_config,
        refiner_config={
            "steps": 3,
            "candidate_budget": -1,
            "preserve_connectivity": True,
            "selection": "greedy",
            "categorical_weight": 1.0,
            "probability_weight": 0.0,
            "graphlet_weight": 0.0,
            "accept_only_improving": False,
            "patience": 10,
            "infeasible_target_policy": "guidance_only",
        },
        device="cpu",
        rng=np.random.default_rng(0),
        return_trace=True,
        prediction_fn=fixed_prediction,
    )

    assert dict(refined.degree()) == dict(graph.degree())
    assert refined.number_of_edges() == graph.number_of_edges()
    assert nx.number_of_selfloops(refined) == 0
    assert nx.is_connected(refined)
    assert _edge_set(refined) != _edge_set(infeasible_endpoint)
    assert len(trace) == 3
    assert all(item["accepted"] is True for item in trace)
    assert all(item["sampled_target_degree_match"] is False for item in trace)
