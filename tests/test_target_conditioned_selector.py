import networkx as nx
import numpy as np

from grapher.construction.coarse import construct_coarse_graph
from grapher.properties.summary import SummaryConfig, extract_summary
from grapher.refinement.features import (
    action_local_features,
    graph_context_features,
)
from grapher.refinement.rewiring import (
    enumerate_valid_double_edge_swaps,
    permute_action,
)


def test_selector_features_are_permutation_invariant_and_support_stop():
    graph = nx.cycle_graph(8)
    graph.add_edges_from([(0, 4), (2, 6)])
    cfg = SummaryConfig(
        degree_hist_max_degree=4,
        clustering_bins=8,
        spectral_bins=8,
    )
    target = extract_summary(graph, cfg)
    action = enumerate_valid_double_edge_swaps(graph)[0]
    mapping = {node: (3 * node + 1) % 8 for node in graph.nodes()}
    permuted_graph = nx.relabel_nodes(graph, mapping, copy=True)
    permuted_action = permute_action(action, mapping)
    feature_cfg = {
        "feature_version": 2,
        "degree_width": 5,
        "clustering_width": 8,
        "spectral_width": 8,
        "motif_width": 0,
        "orbit_width": 0,
        "graphlet_width": 0,
    }

    context = graph_context_features(
        graph,
        target,
        feature_cfg,
        current_summary=extract_summary(graph, cfg),
    )
    permuted_context = graph_context_features(
        permuted_graph,
        target,
        feature_cfg,
        current_summary=extract_summary(permuted_graph, cfg),
    )
    local = action_local_features(graph, action, feature_version=2)
    permuted_local = action_local_features(
        permuted_graph,
        permuted_action,
        feature_version=2,
    )
    stop = action_local_features(graph, None, feature_version=2)

    assert np.allclose(context, permuted_context)
    assert np.allclose(local, permuted_local)
    assert stop.shape == local.shape
    assert stop[-1] == 1.0
    assert local[-1] == 0.0


def test_constructor_and_target_share_exact_degree_sequence():
    target_graph = nx.cycle_graph(8)
    cfg = SummaryConfig(degree_hist_max_degree=3)
    target = extract_summary(target_graph, cfg)
    coarse = construct_coarse_graph(
        target,
        {
            "type": "havel_hakimi",
            "ensure_connected": True,
            "random_relabel": False,
        },
        np.random.default_rng(0),
    )
    assert sorted(dict(coarse.degree()).values()) == sorted(target["degree_sequence"])
