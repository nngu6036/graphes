from __future__ import annotations

from collections import Counter
import itertools
import json
from math import comb

import networkx as nx
import numpy as np
import pytest

from grapher.utils.motifs import (
    _canonicalize_attributed_tokens,
    attributed_graphlet_count_dict,
    canonicalize_attributed_graph_python,
)


def _legacy_key(graph: nx.Graph, *, missing_ok: bool = False) -> str:
    """Independent exhaustive oracle for the original checkpoint key format."""

    def token(value):
        return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"

    nodes = list(graph)
    node_labels = {}
    for node, data in graph.nodes(data=True):
        if "atom" not in data and not missing_ok:
            raise KeyError("atom")
        node_labels[node] = token(data.get("atom", "__MISSING__"))
    edge_labels = {}
    for left, right, data in graph.edges(data=True):
        if "bond" not in data and not missing_ok:
            raise KeyError("bond")
        edge_labels[frozenset((left, right))] = token(data.get("bond", "__MISSING__"))
    candidates = []
    for order in itertools.permutations(nodes):
        candidates.append(json.dumps(
            [[node_labels[node] for node in order],
             [edge_labels.get(frozenset((left, right)))
              for left, right in itertools.combinations(order, 2)]],
            ensure_ascii=True, separators=(",", ":"),
        ))
    return "ATTR_PY_V1|" + min(candidates)


def _key(graph: nx.Graph, **kwargs) -> str:
    return canonicalize_attributed_graph_python(
        graph, node_label_attr="atom", edge_label_attr="bond", **kwargs
    )


def _counts(graph: nx.Graph, k: int, **kwargs) -> dict[str, int]:
    return attributed_graphlet_count_dict(
        graph, k, node_label_attr="atom", edge_label_attr="bond",
        backend="python", connected_only=False, **kwargs
    )


@pytest.mark.parametrize("labels", [
    [6],
    [True, 1, 1.0, 6, 7],
    ["\u2603", "~", "\u00e9", "\\", "\""],
    [None, [1, 2], {"element": "C"}, ("N", 1), "\n"],
])
def test_canonical_keys_match_legacy_oracle_for_small_graph_atlas(labels) -> None:
    bonds = [1, None, 2.0, False, "\u00e9", [1, 2]]
    for graph in nx.graph_atlas_g():
        if len(graph) > 5:
            continue
        for index, node in enumerate(graph):
            graph.nodes[node]["atom"] = labels[index % len(labels)]
        for index, (left, right) in enumerate(graph.edges):
            graph.edges[left, right]["bond"] = bonds[index % len(bonds)]
        assert _key(graph) == _legacy_key(graph)


def test_cache_distinguishes_equal_python_values_of_different_types() -> None:
    node_keys = []
    edge_keys = []
    for value in (True, 1, 1.0):
        graph = nx.path_graph(3)
        nx.set_node_attributes(graph, value, "atom")
        nx.set_edge_attributes(graph, 1, "bond")
        node_keys.append(_key(graph))
        assert _counts(graph, 3) == {_legacy_key(graph): 1}
        nx.set_node_attributes(graph, 6, "atom")
        nx.set_edge_attributes(graph, value, "bond")
        edge_keys.append(_key(graph))
        assert _counts(graph, 3) == {_legacy_key(graph): 1}
    assert len(set(node_keys)) == len(set(edge_keys)) == 3


def test_none_bond_label_remains_distinct_from_absent_edge() -> None:
    present = nx.path_graph(3)
    nx.set_node_attributes(present, None, "atom")
    for left, right in present.edges:
        present.edges[left, right]["bond"] = None
    absent = present.copy()
    absent.remove_edge(0, 1)
    assert _key(present) != _key(absent)
    assert _counts(present, 3) == {_legacy_key(present): 1}
    assert _counts(absent, 3) == {_legacy_key(absent): 1}


@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("num_samples", [None, 7])
def test_direct_counts_match_exhaustive_or_sampled_legacy_counts(k, num_samples) -> None:
    graph = nx.gnp_random_graph(9, 0.23, seed=19)
    atoms = [6, 7, 8, 9, None]
    bonds = [1, 2, None, 3]
    for node in graph:
        graph.nodes[node]["atom"] = atoms[node % len(atoms)]
    for index, edge in enumerate(graph.edges):
        graph.edges[edge]["bond"] = bonds[index % len(bonds)]
    graph = nx.relabel_nodes(graph, {
        0: "a", 1: "b", 2: ("node", 2), 3: "\u00e9", 4: 1,
        5: False, 6: 2.5, 7: "c", 8: ("node", 8),
    })
    nodes = sorted(graph, key=lambda node: (type(node).__name__, repr(node)))
    if num_samples is None:
        subsets = list(itertools.combinations(nodes, k))
    else:
        selected = set()
        rng = np.random.default_rng(42)
        while len(selected) < num_samples:
            selected.add(tuple(sorted(int(i) for i in rng.choice(len(nodes), size=k, replace=False))))
        subsets = [tuple(nodes[index] for index in subset) for subset in selected]
    expected = Counter(_legacy_key(graph.subgraph(subset)) for subset in subsets)
    actual = _counts(graph, k, num_samples=num_samples, rng=np.random.default_rng(42))
    assert actual == dict(expected)
    assert sum(actual.values()) == (comb(len(graph), k) if num_samples is None else num_samples)


def test_direct_counting_avoids_networkx_subgraphs_and_reuses_patterns(monkeypatch) -> None:
    graph = nx.path_graph(9)
    nx.set_node_attributes(graph, 6, "atom")
    nx.set_edge_attributes(graph, 1, "bond")
    expected = Counter(_legacy_key(graph.subgraph(subset)) for subset in itertools.combinations(graph, 5))

    def unexpected_subgraph(*_args, **_kwargs):
        raise AssertionError("Unfiltered counting should use labeled adjacency patterns directly.")

    monkeypatch.setattr(nx.Graph, "subgraph", unexpected_subgraph)
    _canonicalize_attributed_tokens.cache_clear()
    assert _counts(graph, 5) == dict(expected)
    first = _canonicalize_attributed_tokens.cache_info()
    assert _counts(graph.copy(), 5) == dict(expected)
    repeated = _canonicalize_attributed_tokens.cache_info()
    assert repeated.misses == first.misses
    assert repeated.hits - first.hits == comb(9, 5)


@pytest.mark.parametrize("missing_kind", ["atom", "bond"])
def test_sampled_missing_labels_validate_only_selected_subsets(missing_kind) -> None:
    graph = nx.complete_graph(6)
    nx.set_node_attributes(graph, 6, "atom")
    nx.set_edge_attributes(graph, 1, "bond")
    chosen = tuple(sorted(int(i) for i in np.random.default_rng(42).choice(6, size=3, replace=False)))
    skipped = next(node for node in graph if node not in chosen)
    if missing_kind == "atom":
        del graph.nodes[skipped]["atom"]
    else:
        del graph.edges[skipped, chosen[0]]["bond"]
    assert _counts(graph, 3, num_samples=1, rng=np.random.default_rng(42)) == {
        _legacy_key(graph.subgraph(chosen)): 1
    }
    with pytest.raises(KeyError):
        _counts(graph, 3)
    expected = Counter(
        _legacy_key(graph.subgraph(subset), missing_ok=True)
        for subset in itertools.combinations(graph, 3)
    )
    assert _counts(graph, 3, missing_ok=True) == dict(expected)
    assert _counts(graph, 7) == {}


@pytest.mark.parametrize("kind", ["directed", "multi", "self_loop"])
def test_fast_path_preserves_graph_validation(kind) -> None:
    graph = nx.path_graph(3)
    if kind == "directed":
        graph = nx.DiGraph(graph)
    elif kind == "multi":
        graph = nx.MultiGraph(graph)
    else:
        graph.add_edge(0, 0)
    with pytest.raises(ValueError):
        _counts(graph, 3)


def test_fast_path_preserves_order_validation() -> None:
    graph = nx.path_graph(8)
    nx.set_node_attributes(graph, 6, "atom")
    nx.set_edge_attributes(graph, 1, "bond")
    with pytest.raises(ValueError, match="positive"):
        _counts(graph, 0)
    with pytest.raises(RuntimeError, match="at most 7"):
        _counts(graph, 8)
    assert _counts(graph, 9) == {}
