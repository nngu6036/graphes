from __future__ import annotations

import networkx as nx
import numpy as np

from grapher.rewiring_mlp.attributed.data import (
    EndpointTrajectoryIterableDataset,
    GraphCategoryVocabulary,
    GraphletBasis,
    build_aligned_teacher_states,
    build_endpoint_examples,
)
from grapher.rewiring_mlp.molecular.typed_invariants import (
    extract_typed_invariant,
    typed_invariant_matches_graph,
)
from grapher.properties.summary import SummaryConfig


def _attributed_cycle() -> nx.Graph:
    graph = nx.cycle_graph(6)
    nx.set_node_attributes(graph, 6, "atomic_num")
    nx.set_edge_attributes(graph, 1, "bond_type")
    return graph


def _setup():
    target = _attributed_cycle()
    vocabulary = GraphCategoryVocabulary.from_graphs(
        [target],
        {
            "node_attribute": "atomic_num",
            "node_categories": [6],
            "edge_attribute": "bond_type",
            "edge_categories": [1, 2, 3],
        },
    )
    config = {
        "graphlet_history": True,
        "graphlet_k_min": 3,
        "graphlet_k_max": 3,
        "graphlet_connected_only": True,
        "attributed": True,
        "attributed_backend": "python",
    }
    summary = SummaryConfig.from_dict(config)
    basis = GraphletBasis.fit_from_graphs(
        [target],
        config,
        vocabulary=vocabulary,
    )
    return target, vocabulary, summary, basis


def test_typed_soft_teacher_preserves_indexed_invariant_and_caches_stop() -> None:
    target, vocabulary, summary, basis = _setup()
    invariant = extract_typed_invariant(target, edge_types=[1, 2, 3])
    states, aligned_target, report = build_aligned_teacher_states(
        target,
        steps=5,
        candidate_budget=32,
        preserve_connectivity=True,
        vocabulary=vocabulary,
        typed_constructor_config={
            "randomize_assignment": False,
            "max_backtracks": 10_000,
        },
        source_randomization_steps=2,
        graphlet_basis=basis,
        summary_config=summary,
        teacher_mode="soft",
        teacher_temperature=0.5,
        teacher_top_k=4,
        teacher_sample_actions=True,
        rng=np.random.default_rng(11),
    )

    assert len(states) <= 6
    assert all(typed_invariant_matches_graph(state, invariant) for state in states)
    assert typed_invariant_matches_graph(aligned_target, invariant)
    assert "teacher_stop_reason" in report
    for decision in report["teacher_decisions"]:
        assert len(decision["distribution"]) == len(decision["actions"]) + 1
        assert np.isclose(sum(decision["distribution"]), 1.0)
        assert decision["stop_index"] == len(decision["actions"])


def test_multiple_paths_and_streaming_examples_are_memory_bounded() -> None:
    target, vocabulary, summary, basis = _setup()
    trajectory = {
        "steps": 3,
        "candidate_budget": 16,
        "states_per_graph": 2,
        "paths_per_graph": 2,
        "source_randomization_steps": 1,
        "teacher_mode": "hard",
        "typed_constructor": {
            "randomize_assignment": False,
            "max_backtracks": 10_000,
        },
        "shared_relabel_augmentation": False,
    }
    examples, report = build_endpoint_examples(
        [target],
        summary_config=summary,
        graphlet_basis=basis,
        vocabulary=vocabulary,
        trajectory_config=trajectory,
        seed=5,
    )
    stream = EndpointTrajectoryIterableDataset(
        [target],
        summary_config=summary,
        graphlet_basis=basis,
        vocabulary=vocabulary,
        trajectory_config=trajectory,
        seed=5,
        shuffle_graphs=False,
    )
    streamed = list(stream)

    assert report["num_paths"] == 2
    assert len(examples) <= stream.estimated_examples == 4
    assert len(streamed) == len(examples)
    assert all(0.0 <= example.time <= 1.0 for example in streamed)
