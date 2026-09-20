from collections import Counter

import networkx as nx
import numpy as np
import pytest

from grapher.utils.motifs import (
    PythonCanonicalizer,
    _iter_k_induced_subgraphs,
    _small_topology_lookup,
    _small_topology_subset_indices,
    graphlet_count_dict,
    graphlet_topology_matches,
)


def _legacy_counts(graph, k, **kwargs):
    """Original induced-NetworkX-graph path, independent of the lookup."""
    subgraphs = _iter_k_induced_subgraphs(
        graph,
        k,
        connected_only=kwargs.get("connected_only", True),
        topology_filter=kwargs.get("topology_filter", "all"),
        num_samples=kwargs.get("num_samples"),
        rng=kwargs.get("rng"),
    )
    return dict(Counter(PythonCanonicalizer().canonical_graph6_batch(subgraphs)))


@pytest.mark.parametrize("k", [3, 4, 5])
def test_every_small_labeled_topology_has_existing_key_and_predicates(k):
    keys, connected, cyclic, simple_cycle = _small_topology_lookup(k)
    pairs = [(u, v) for v in range(1, k) for u in range(v)]
    canonicalizer = PythonCanonicalizer()
    for code in range(1 << len(pairs)):
        graph = nx.Graph()
        graph.add_nodes_from(range(k))
        graph.add_edges_from(
            pair
            for position, pair in enumerate(pairs)
            if code & (1 << (len(pairs) - position - 1))
        )
        assert keys[code] == canonicalizer.canonical_graph6(graph)
        assert connected[code] == nx.is_connected(graph)
        assert cyclic[code] == graphlet_topology_matches(graph, "cyclic")
        assert simple_cycle[code] == graphlet_topology_matches(graph, "simple_cycle")
        assert graphlet_count_dict(graph, k, connected_only=False) == {keys[code]: 1}


@pytest.mark.parametrize("connected_only", [False, True])
@pytest.mark.parametrize("topology_filter", ["all", "cyclic", "simple_cycle"])
@pytest.mark.parametrize("num_samples", [None, 7])
def test_counts_match_induced_subgraphs_for_filters_and_mixed_node_labels(
    connected_only, topology_filter, num_samples
):
    graph = nx.disjoint_union(nx.complete_graph(4), nx.cycle_graph(4))
    labels = ["z", -10, (1, 2), 3.5, "a", frozenset({7}), 99, ("node",)]
    graph = nx.relabel_nodes(graph, dict(enumerate(labels)))
    # Attributes, including numerical edge weights, are excluded from topology.
    nx.set_edge_attributes(graph, 0.0, "weight")
    nx.set_node_attributes(graph, "carbon", "node_label")
    actual_rng = np.random.default_rng(41)
    expected_rng = np.random.default_rng(41)
    for k in (3, 4, 5):
        options = dict(
            connected_only=connected_only,
            topology_filter=topology_filter,
            num_samples=num_samples,
        )
        actual = graphlet_count_dict(graph, k, rng=actual_rng, **options)
        expected = _legacy_counts(graph, k, rng=expected_rng, **options)
        assert actual == expected
        assert list(actual) == list(expected)
        assert actual_rng.bit_generator.state == expected_rng.bit_generator.state


def test_report_sample_budget_and_rng_match_legacy_path():
    graph = nx.gnp_random_graph(20, 0.25, seed=42)
    actual_rng = np.random.default_rng(41)
    expected_rng = np.random.default_rng(41)
    for k in (3, 4, 5):
        actual = graphlet_count_dict(graph, k, num_samples=8192, rng=actual_rng)
        expected = _legacy_counts(graph, k, num_samples=8192, rng=expected_rng)
        assert actual == expected
        assert list(actual) == list(expected)
        assert actual_rng.bit_generator.state == expected_rng.bit_generator.state


@pytest.mark.parametrize("k,num_samples", [(4, 8192), (5, 8192), (5, None)])
def test_default_rng_cached_subsets_match_existing_samples(k, num_samples):
    _small_topology_subset_indices.cache_clear()
    graph = nx.gnp_random_graph(20, 0.25, seed=42)
    expected = _legacy_counts(graph, k, num_samples=num_samples)
    for _ in range(2):
        actual = graphlet_count_dict(graph, k, num_samples=num_samples)
        assert actual == expected
        assert list(actual) == list(expected)
    info = _small_topology_subset_indices.cache_info()
    assert info.hits == 1
    assert info.misses == 1
    rng = np.random.default_rng(0)
    assert graphlet_count_dict(graph, k, num_samples=num_samples, rng=rng) == expected
    assert _small_topology_subset_indices.cache_info() == info


def test_large_exact_default_budget_streams_without_using_subset_cache():
    info = _small_topology_subset_indices.cache_info()
    key = PythonCanonicalizer().canonical_graph6(nx.complete_graph(3))
    assert graphlet_count_dict(nx.complete_graph(48), 3) == {key: 17296}
    assert _small_topology_subset_indices.cache_info() == info


def test_subset_cache_is_bounded_and_read_only():
    _small_topology_subset_indices.cache_clear()
    for n in range(6, 46):
        indices = _small_topology_subset_indices(n, 3, 3)
        assert not indices.flags.writeable
    info = _small_topology_subset_indices.cache_info()
    assert info.maxsize == 32
    assert info.currsize == 32
    with pytest.raises(ValueError, match="limited"):
        _small_topology_subset_indices(100, 5, 16385)


def test_custom_python_canonicalizer_subclass_keeps_attributes_and_batch_size():
    class CustomCanonicalizer(PythonCanonicalizer):
        def canonical_graph6_batch(self, graphs):
            graphs = list(graphs)
            assert len(graphs) <= 2
            return [graph.nodes[0]["tag"] for graph in graphs]

    graph = nx.star_graph(4)
    graph.nodes[0]["tag"] = "custom-key"
    assert graphlet_count_dict(
        graph, 3, canonicalizer=CustomCanonicalizer(), batch_size=2
    ) == {"custom-key": 6}


def test_python_canonicalizer_node_limit_is_preserved():
    with pytest.raises(RuntimeError, match="at most 4 nodes"):
        graphlet_count_dict(nx.complete_graph(5), 5, canonicalizer=PythonCanonicalizer(4))


@pytest.mark.parametrize("k", [1, 2, 6])
def test_orders_outside_lookup_keep_existing_behavior(k):
    graph = nx.path_graph(6)
    assert graphlet_count_dict(graph, k) == _legacy_counts(graph, k)


@pytest.mark.parametrize("graph", [nx.DiGraph(), nx.MultiGraph(), nx.Graph([(0, 0)])])
def test_invalid_graphs_are_still_rejected(graph):
    with pytest.raises(ValueError):
        graphlet_count_dict(graph, 3)


def test_graph_smaller_than_order_is_empty():
    assert graphlet_count_dict(nx.path_graph(2), 3) == {}
