from __future__ import annotations

import importlib.util
from pathlib import Path

import networkx as nx
import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "run_topology_grapher.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_topology_grapher_degree_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_test_empirical_degree_source_samples_only_heldout_sequences() -> None:
    module = _load_script_module()
    train_graphs = [nx.path_graph(4)]
    test_graphs = [nx.cycle_graph(5), nx.star_graph(4)]

    sampler = module._build_generation_degree_sampler(
        "test_empirical",
        {"type": "degree_histogram_vae"},
        train_graphs=train_graphs,
        reference_graphs=test_graphs,
        seed=42,
    )

    assert sampler is not None
    expected = {
        tuple(sorted((int(d) for _, d in graph.degree()), reverse=True))
        for graph in test_graphs
    }
    train_sequence = tuple(
        sorted((int(d) for _, d in train_graphs[0].degree()), reverse=True)
    )
    rng = np.random.default_rng(7)
    sampled = {
        tuple(sampler.sample(rng)["degree_sequence"])
        for _ in range(32)
    }

    assert sampled <= expected
    assert sampled
    assert train_sequence not in sampled


def test_test_empirical_degree_source_requires_test_split() -> None:
    module = _load_script_module()
    with pytest.raises(ValueError, match="non-empty test split"):
        module._build_generation_degree_sampler(
            "test_empirical",
            {"type": "degree_histogram_vae"},
            train_graphs=[nx.path_graph(4)],
            reference_graphs=[],
            seed=0,
        )


def test_test_oracle_degree_source_is_handled_per_reference_graph() -> None:
    module = _load_script_module()
    test_graphs = [nx.cycle_graph(5)]
    sampler = module._build_generation_degree_sampler(
        "test_oracle",
        {"type": "degree_histogram_vae"},
        train_graphs=[nx.path_graph(4)],
        reference_graphs=test_graphs,
        seed=0,
    )

    assert sampler is None
    summary = module._oracle_degree_summary(test_graphs[0])
    assert summary["degree_sequence"] == [2, 2, 2, 2, 2]
