from __future__ import annotations

import importlib.util
from pathlib import Path

import networkx as nx
import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "run_attributed_grapher.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "run_attributed_grapher_invariant_source_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _molecule(node_types, edges):
    graph = nx.Graph()
    for idx, atom in enumerate(node_types):
        graph.add_node(idx, atomic_num=int(atom))
    for u, v, bond_type in edges:
        graph.add_edge(int(u), int(v), bond_type=int(bond_type))
    return graph


def _key(invariant):
    return (
        tuple((int(sig.node_type), tuple(int(x) for x in sig.edge_degrees)) for sig in invariant.signatures),
        tuple(int(x) for x in invariant.edge_types),
    )


def test_test_empirical_typed_source_samples_only_heldout_invariants() -> None:
    module = _load_script_module()
    train_graphs = [
        _molecule([6, 6], [(0, 1, 1)]),
    ]
    test_graphs = [
        _molecule([6, 8, 6], [(0, 1, 1), (1, 2, 2)]),
        _molecule([7, 6, 9], [(0, 1, 3), (1, 2, 1)]),
    ]
    edge_types = (1, 2, 3)

    expected = {
        _key(
            module.extract_typed_invariant(
                graph,
                edge_types=edge_types,
                node_attribute="atomic_num",
                edge_attribute="bond_type",
            )
        )
        for graph in test_graphs
    }
    train_key = _key(
        module.extract_typed_invariant(
            train_graphs[0],
            edge_types=edge_types,
            node_attribute="atomic_num",
            edge_attribute="bond_type",
        )
    )

    rng = np.random.default_rng(7)
    sampled = set()
    metadata_splits = set()
    for idx in range(64):
        invariant, metadata = module._sample_invariant(
            "test_empirical",
            index=idx,
            train_graphs=train_graphs,
            test_graphs=test_graphs,
            typed_sampler=None,
            rng=rng,
            edge_types=edge_types,
            node_attribute="atomic_num",
            edge_attribute="bond_type",
        )
        sampled.add(_key(invariant))
        metadata_splits.add(metadata["reference_split"])

    assert sampled <= expected
    assert sampled
    assert train_key not in sampled
    assert metadata_splits == {"test"}


def test_test_empirical_typed_source_requires_test_split() -> None:
    module = _load_script_module()
    with pytest.raises(ValueError, match="non-empty test split"):
        module._sample_invariant(
            "test_empirical",
            index=0,
            train_graphs=[_molecule([6, 6], [(0, 1, 1)])],
            test_graphs=[],
            typed_sampler=None,
            rng=np.random.default_rng(0),
            edge_types=(1, 2, 3),
            node_attribute="atomic_num",
            edge_attribute="bond_type",
        )


def test_test_oracle_typed_source_uses_matching_reference_index() -> None:
    module = _load_script_module()
    test_graphs = [
        _molecule([6, 8, 6], [(0, 1, 1), (1, 2, 2)]),
        _molecule([7, 6, 9], [(0, 1, 3), (1, 2, 1)]),
    ]
    invariant, metadata = module._sample_invariant(
        "test_oracle",
        index=1,
        train_graphs=[],
        test_graphs=test_graphs,
        typed_sampler=None,
        rng=np.random.default_rng(0),
        edge_types=(1, 2, 3),
        node_attribute="atomic_num",
        edge_attribute="bond_type",
    )
    expected = module.extract_typed_invariant(
        test_graphs[1],
        edge_types=(1, 2, 3),
        node_attribute="atomic_num",
        edge_attribute="bond_type",
    )
    assert _key(invariant) == _key(expected)
    assert metadata == {"reference_split": "test", "reference_index": 1}
