from __future__ import annotations

import hashlib
import json

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.properties.summary import SummaryConfig
from grapher.rewiring_mlp.core.rewiring import apply_action
from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.data import (
    TopologyGraphletExample,
    TopologyTrainingPair,
    build_topology_examples,
    build_topology_teacher_states,
    collate_topology_examples,
)
from grapher.rewiring_mlp.generic.graphlets import (
    TOPOLOGY_ORBIT_WIDTH,
    candidate_topology_graphlet_counts,
    extract_topology_graphlet_counts,
    extract_topology_graphlet_target,
    extract_topology_structural_target,
    topology_orbit_count_vector_from_counts,
    topology_graphlet_discrepancy,
)
from grapher.rewiring_mlp.generic.training_sources import (
    build_completed_base_training_pairs,
)
from grapher.utils.io import save_pickle
from grapher.properties.summary import python_orbit_count_vector
from grapher.rewiring_mlp.generic.model import (
    TopologyGraphletPredictor,
    load_topology_checkpoint,
    save_topology_checkpoint,
)
from grapher.rewiring_mlp.generic.refiner import (
    TopologyPrediction,
    TopologyRefinerConfig,
    refine_graph_with_topology_predictions,
)
from grapher.rewiring_mlp.generic.rewiring import propose_valid_topology_swaps


def _basis(k_max: int = 4) -> tuple[TopologyGraphletBasis, SummaryConfig]:
    config = SummaryConfig.from_dict(
        {
            "graphlet_history": True,
            "graphlet_k_min": 3,
            "graphlet_k_max": k_max,
            "graphlet_connected_only": True,
            "graphlet_num_samples": None,
        }
    )
    return TopologyGraphletBasis.from_config(config), config


def _example(
    graph: nx.Graph,
    basis: TopologyGraphletBasis,
) -> TopologyGraphletExample:
    return TopologyGraphletExample(
        current_graph=graph,
        time=0.25,
        graphlet_target=np.concatenate(
            [
                np.full(stop - start, 1.0 / (stop - start), dtype=np.float32)
                for start, stop in basis.slices
            ]
        ),
        graphlet_mass_target=np.full(len(basis.sizes), 0.5, dtype=np.float32),
    )


def test_topology_batch_and_predictor_expose_no_terminal_pair_targets() -> None:
    basis, _ = _basis(3)
    batch = collate_topology_examples([_example(nx.path_graph(5), basis)])
    model = TopologyGraphletPredictor(
        graphlet_slices=basis.slices,
        hidden_dim=16,
        edge_dim=8,
        graph_dim=16,
        num_layers=2,
    )

    outputs = model(batch)

    assert set(outputs) == {"graphlet_alpha", "graphlet_mass_ab"}
    assert not hasattr(batch, "target_edge_labels")
    assert not hasattr(batch, "target_node_labels")
    assert not hasattr(model, "edge_head")
    assert not hasattr(model, "node_head")
    assert all("edge_head" not in name for name in model.state_dict())


def test_structural_heads_attach_clustering_and_orbit_without_pair_targets() -> None:
    basis, _ = _basis(4)
    config = SummaryConfig.from_dict(
        {
            "graphlet_history": True,
            "graphlet_k_min": 3,
            "graphlet_k_max": 4,
            "graphlet_connected_only": True,
            "clustering_summary": True,
            "clustering_bins": 8,
            "orbit_count": True,
        }
    )
    graph = nx.house_graph()
    graphlet, mass, clustering, orbit = extract_topology_structural_target(
        graph,
        graphlet_basis=basis,
        summary_config=config,
    )
    batch = collate_topology_examples(
        [
            TopologyGraphletExample(
                current_graph=graph,
                time=0.0,
                graphlet_target=graphlet,
                graphlet_mass_target=mass,
                clustering_target=clustering,
                orbit_target=orbit,
            )
        ]
    )
    model = TopologyGraphletPredictor(
        graphlet_slices=basis.slices,
        clustering_width=8,
        orbit_width=TOPOLOGY_ORBIT_WIDTH,
        hidden_dim=16,
        edge_dim=8,
        graph_dim=16,
        num_layers=1,
    )
    outputs = model(batch)
    loss, metrics = model.loss(batch)
    loss.backward()

    assert set(outputs) == {
        "graphlet_alpha",
        "graphlet_mass_ab",
        "clustering_alpha",
        "orbit_log_mean",
    }
    assert batch.clustering_target.shape == (1, 8)
    assert batch.orbit_target.shape == (1, TOPOLOGY_ORBIT_WIDTH)
    assert "clustering_mae" in metrics and "orbit_log_mae" in metrics
    assert not hasattr(model, "edge_head")


def test_topology_loss_contains_only_graphlet_terms_and_backpropagates() -> None:
    basis, _ = _basis(3)
    batch = collate_topology_examples([_example(nx.house_graph(), basis)])
    model = TopologyGraphletPredictor(
        graphlet_slices=basis.slices,
        hidden_dim=16,
        edge_dim=8,
        graph_dim=16,
        num_layers=1,
    )

    loss, metrics = model.loss(
        batch,
        loss_weights={
            "graphlet_mean": 1.0,
            "graphlet_distribution": 0.1,
            "graphlet_mass": 0.1,
        },
    )
    loss.backward()

    assert set(metrics) == {
        "loss",
        "graphlet_mean_loss",
        "graphlet_distribution_loss",
        "graphlet_mass_loss",
        "graphlet_mae",
        "graphlet_mass_mae",
    }
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(
        parameter.grad is not None for parameter in model.graphlet_heads.parameters()
    )
    assert all(
        parameter.grad is not None
        for parameter in model.graphlet_mass_head.parameters()
    )


def test_topology_prediction_is_invariant_to_node_permutation() -> None:
    basis, _ = _basis(4)
    graph = nx.house_graph()
    mapping = {0: 3, 1: 0, 2: 4, 3: 1, 4: 2}
    permuted = nx.relabel_nodes(graph, mapping, copy=True)
    batch = collate_topology_examples(
        [_example(graph, basis), _example(permuted, basis)]
    )
    model = TopologyGraphletPredictor(
        graphlet_slices=basis.slices,
        hidden_dim=16,
        edge_dim=8,
        graph_dim=16,
        num_layers=2,
        dropout=0.0,
    ).eval()

    with torch.no_grad():
        outputs = model(batch)

    assert torch.allclose(
        outputs["graphlet_alpha"][0],
        outputs["graphlet_alpha"][1],
        atol=1.0e-6,
    )
    assert torch.allclose(
        outputs["graphlet_mass_ab"][0],
        outputs["graphlet_mass_ab"][1],
        atol=1.0e-6,
    )


def test_exact_graphlet_targets_are_permutation_and_insertion_invariant() -> None:
    basis, summary_config = _basis(5)
    graph = nx.grid_2d_graph(4, 4)
    graph = nx.convert_node_labels_to_integers(graph)
    mapping = {node: 15 - node for node in graph.nodes()}
    permuted = nx.relabel_nodes(graph, mapping, copy=True)
    rebuilt = nx.Graph()
    rebuilt.add_nodes_from(reversed(list(permuted.nodes())))
    rebuilt.add_edges_from(reversed(list(permuted.edges())))

    target, mass = extract_topology_graphlet_target(
        graph,
        graphlet_basis=basis,
        summary_config=summary_config,
    )
    other_target, other_mass = extract_topology_graphlet_target(
        rebuilt,
        graphlet_basis=basis,
        summary_config=summary_config,
    )

    assert np.allclose(target, other_target)
    assert np.allclose(mass, other_mass)
    assert all(
        np.isclose(target[start:stop].sum(), 1.0)
        for start, stop in basis.slices
    )


def test_topology_checkpoint_round_trip_and_endpoint_format_guard(tmp_path) -> None:
    basis, summary_config = _basis(3)
    model = TopologyGraphletPredictor(
        graphlet_slices=basis.slices,
        hidden_dim=16,
        edge_dim=8,
        graph_dim=16,
        num_layers=1,
    )
    path = tmp_path / "topology.pt"
    save_topology_checkpoint(
        model,
        path,
        graphlet_basis=basis,
        summary_config=summary_config,
    )

    restored, restored_basis, _, checkpoint = load_topology_checkpoint(
        path,
        device="cpu",
    )

    assert checkpoint["pipeline_mode"] == "topology"
    assert restored_basis.to_dict() == basis.to_dict()
    assert restored.model_config() == model.model_config()
    batch = collate_topology_examples([_example(nx.path_graph(5), basis)])
    model.eval()
    with torch.no_grad():
        expected = model(batch)
        actual = restored(batch)
    assert all(torch.allclose(expected[key], actual[key]) for key in expected)

    endpoint_path = tmp_path / "endpoint.pt"
    torch.save({"format": "hybrid_endpoint_graphlet_v2"}, endpoint_path)
    with pytest.raises(ValueError, match="cannot be migrated"):
        load_topology_checkpoint(endpoint_path, device="cpu")

    incompatible = torch.load(path, map_location="cpu")
    incompatible["topology_canonicalizer"] = "environment_dependent"
    torch.save(incompatible, path)
    with pytest.raises(ValueError, match="canonicalizer convention"):
        load_topology_checkpoint(path, device="cpu")


def test_graphlet_teacher_stops_without_terminal_adjacency_recovery() -> None:
    basis, summary_config = _basis(3)
    target = nx.cycle_graph(4)
    target_graphlet, target_mass = extract_topology_graphlet_target(
        target,
        graphlet_basis=basis,
        summary_config=summary_config,
    )

    states, report = build_topology_teacher_states(
        [2, 2, 2, 2],
        target_graphlet=target_graphlet,
        target_graphlet_mass=target_mass,
        graphlet_basis=basis,
        summary_config=summary_config,
        steps=4,
        proposal_budget=-1,
        valid_candidate_budget=-1,
        preserve_connectivity=True,
        ensure_connected_source=True,
        max_repair_trials=100,
        random_relabel_source=False,
        source_randomization_steps=0,
        teacher_mode="hard",
        teacher_temperature=1.0,
        teacher_top_k=0,
        teacher_sample_actions=False,
        teacher_graphlet_mass_weight=0.0,
        teacher_min_improvement=1.0e-8,
        target_tolerance=0.0,
        rng=np.random.default_rng(0),
    )

    assert len(states) == 1
    assert report["final_teacher_graphlet_discrepancy"] == pytest.approx(0.0)
    assert report["teacher_stop_reason"] == "target_graphlet_tolerance"
    # The teacher API receives only the degree sequence and cached histogram;
    # exact indexed terminal adjacency is not part of its inputs or report.
    assert "final_teacher_edge_disagreement" not in report


def test_all_paths_reuse_one_cached_terminal_graphlet_target() -> None:
    basis, summary_config = _basis(4)
    examples, _ = build_topology_examples(
        [nx.house_graph()],
        summary_config=summary_config,
        graphlet_basis=basis,
        trajectory_config={
            "paths_per_graph": 3,
            "states_per_graph": 2,
            "steps": 2,
            "proposal_budget": -1,
            "valid_candidate_budget": -1,
            "preserve_connectivity": True,
            "ensure_connected_source": True,
            "random_relabel_source": True,
            "teacher_mode": "hard",
        },
        seed=3,
    )
    assert examples
    expected = examples[0].graphlet_target
    expected_mass = examples[0].graphlet_mass_target
    assert all(
        np.array_equal(example.graphlet_target, expected)
        for example in examples
    )
    assert all(
        np.array_equal(example.graphlet_mass_target, expected_mass)
        for example in examples
    )


def test_soft_teacher_cannot_stop_while_positive_action_exists() -> None:
    basis, summary_config = _basis(4)
    degree_sequence = [3, 3, 2, 2, 2, 2]
    source = nx.havel_hakimi_graph(degree_sequence)
    actions, candidate_graphs, _ = propose_valid_topology_swaps(
        source,
        proposal_budget=-1,
        valid_candidate_budget=-1,
        preserve_connectivity=True,
        rng=np.random.default_rng(0),
    )
    target_graphlet = None
    target_mass = None
    for action in actions:
        vector, mass = extract_topology_graphlet_target(
            candidate_graphs[action],
            graphlet_basis=basis,
            summary_config=summary_config,
        )
        distance, _, _ = topology_graphlet_discrepancy(
            source,
            vector,
            mass,
            graphlet_basis=basis,
            summary_config=summary_config,
        )
        if distance > 1.0e-8:
            target_graphlet, target_mass = vector, mass
            break
    assert target_graphlet is not None and target_mass is not None
    _states, report = build_topology_teacher_states(
        degree_sequence,
        target_graphlet=target_graphlet,
        target_graphlet_mass=target_mass,
        graphlet_basis=basis,
        summary_config=summary_config,
        steps=1,
        proposal_budget=-1,
        valid_candidate_budget=-1,
        preserve_connectivity=True,
        ensure_connected_source=True,
        max_repair_trials=100,
        random_relabel_source=False,
        source_randomization_steps=0,
        teacher_mode="soft",
        teacher_temperature=1000.0,
        teacher_top_k=0,
        teacher_sample_actions=True,
        teacher_graphlet_mass_weight=0.0,
        teacher_min_improvement=1.0e-8,
        target_tolerance=0.0,
        rng=np.random.default_rng(7),
    )
    decision = report["teacher_decisions"][0]
    assert any(value > 0.0 for value in decision["improvements"])
    assert decision["distribution"][decision["stop_index"]] == 0.0
    assert decision["selected_index"] != decision["stop_index"]


def test_local_graphlet_delta_matches_full_candidate_recount() -> None:
    basis, _ = _basis(5)
    source = nx.havel_hakimi_graph([3, 3, 2, 2, 2, 2])
    actions, candidate_graphs, _ = propose_valid_topology_swaps(
        source,
        proposal_budget=-1,
        valid_candidate_budget=-1,
        preserve_connectivity=True,
        rng=np.random.default_rng(0),
    )
    action = actions[0]
    candidate = candidate_graphs[action]
    current_counts = extract_topology_graphlet_counts(
        source,
        graphlet_basis=basis,
    )
    delta_counts = candidate_topology_graphlet_counts(
        source,
        candidate,
        action,
        current_counts=current_counts,
        graphlet_basis=basis,
    )
    exact_counts = extract_topology_graphlet_counts(
        candidate,
        graphlet_basis=basis,
    )
    assert delta_counts == exact_counts


def test_orbit_target_from_cached_graphlets_matches_python_orbit_counter() -> None:
    basis, _ = _basis(4)
    for graph in (nx.path_graph(6), nx.house_graph(), nx.complete_graph(5)):
        counts = extract_topology_graphlet_counts(graph, graphlet_basis=basis)
        cached = topology_orbit_count_vector_from_counts(
            counts,
            num_nodes=graph.number_of_nodes(),
            num_edges=graph.number_of_edges(),
        )
        assert np.allclose(cached, python_orbit_count_vector(graph))


def test_completed_base_manifest_is_partitioned_and_explicitly_matched(tmp_path) -> None:
    sources = [
        nx.path_graph(5),
        nx.cycle_graph(5),
        nx.star_graph(4),
        nx.complete_graph(5),
        nx.house_graph(),
        nx.path_graph(6),
        nx.cycle_graph(6),
        nx.wheel_graph(6),
    ]
    source_path = tmp_path / "estimated_graphs.pkl"
    save_pickle(sources, source_path)
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "test_training_estimates_v1",
                "estimated_graphs": {
                    "path": source_path.name,
                    "count": len(sources),
                    "sha256": digest,
                },
                "pairing": {"status": "unpaired", "pair_count": 0},
            }
        )
    )
    train_targets = [
        nx.path_graph(5),
        nx.cycle_graph(5),
        nx.star_graph(4),
        nx.complete_graph(5),
        nx.house_graph(),
        nx.path_graph(6),
        nx.cycle_graph(6),
    ]
    val_targets = [nx.house_graph(), nx.wheel_graph(6), nx.path_graph(6)]
    train_pairs, val_pairs, report = build_completed_base_training_pairs(
        train_targets,
        val_targets,
        config={
            "mode": "completed_base_outputs",
            "validation_fraction": 0.25,
            "partition_seed": 7,
            "matching": {
                "method": "hungarian_degree_profile",
                "require_exact_node_count": True,
                "disconnected_policy": "error",
            },
            "generators": [
                {
                    "id": "toy_base",
                    "manifest_path": str(manifest_path),
                    "artifact": "estimated_graphs",
                }
            ],
        },
        seed=7,
    )

    assert train_pairs and val_pairs
    assert all(isinstance(pair, TopologyTrainingPair) for pair in train_pairs)
    assert all(pair.base_generator == "toy_base" for pair in train_pairs + val_pairs)
    assert all(
        pair.source_graph.number_of_nodes() == pair.target_graph.number_of_nodes()
        for pair in train_pairs + val_pairs
    )
    assert len({pair.source_index for pair in train_pairs}) == len(train_pairs)
    assert report["pool_reports"][0]["checksum_verified"] is True
    matching = report["matching_reports"][0]
    assert matching["target_features_used_for_matching"] == [
        "sorted_degree_profile"
    ]
    assert "graphlet_histogram" in matching["target_features_excluded_from_matching"]


def test_completed_training_pair_starts_from_declared_source_not_hh() -> None:
    basis, summary_config = _basis(4)
    source = nx.cycle_graph(6)
    target = nx.wheel_graph(6)
    pair = TopologyTrainingPair(
        source_graph=source,
        target_graph=target,
        base_generator="toy_base",
        source_index=3,
        target_index=1,
    )
    examples, report = build_topology_examples(
        [pair],
        summary_config=summary_config,
        graphlet_basis=basis,
        trajectory_config={
            "steps": 0,
            "states_per_graph": 1,
            "paths_per_graph": 1,
            "source_randomization_steps": 0,
            "random_relabel_source": False,
            "preserve_connectivity": True,
            "ensure_connected_source": True,
        },
    )
    assert nx.utils.graphs_equal(examples[0].current_graph, source)
    assert examples[0].base_generator == "toy_base"
    assert report["source_modes"] == ["completed_base_output"]


def test_topology_path_rejects_disconnected_inputs_and_configs() -> None:
    with pytest.raises(ValueError, match="connectivity-preserving"):
        TopologyRefinerConfig.from_dict({"preserve_connectivity": False})

    basis, summary_config = _basis(3)
    disconnected = nx.disjoint_union(nx.path_graph(3), nx.path_graph(3))
    with pytest.raises(ValueError, match="connected source"):
        refine_graph_with_topology_predictions(
            disconnected,
            model=None,
            graphlet_basis=basis,
            summary_config=summary_config,
            refiner_config={"steps": 0},
        )
    with pytest.raises(ValueError, match="targets must be connected"):
        build_topology_examples(
            [disconnected],
            summary_config=summary_config,
            graphlet_basis=basis,
        )


def test_annealed_prediction_horizon_cools_monotonically() -> None:
    config = TopologyRefinerConfig.from_dict(
        {
            "prediction_horizon": {
                "mode": "annealed",
                "initial_k": 8,
                "final_k": 1,
                "schedule": "exponential",
                "refresh_on_plateau": True,
            }
        }
    )

    horizons = [
        config.prediction_horizon_at(progress)
        for progress in np.linspace(0.0, 1.0, 21)
    ]
    assert horizons[0] == 8
    assert horizons[-1] == 1
    assert horizons[-2] == 1
    assert all(left >= right for left, right in zip(horizons, horizons[1:]))
    assert config.refresh_on_plateau is True

    with pytest.raises(ValueError, match="initial_k >= final_k"):
        TopologyRefinerConfig.from_dict(
            {
                "prediction_horizon": {
                    "mode": "annealed",
                    "initial_k": 1,
                    "final_k": 4,
                }
            }
        )


def test_annealed_horizon_reuses_one_prediction_for_multiple_swaps() -> None:
    basis, summary_config = _basis(4)
    source = nx.havel_hakimi_graph([3, 3, 2, 2, 2, 2])
    first_action = (((0, 3), (1, 5)), ((0, 5), (1, 3)))
    second_action = (((0, 1), (2, 3)), ((0, 3), (1, 2)))
    target = apply_action(apply_action(source, first_action), second_action)
    target_vector, target_mass = extract_topology_graphlet_target(
        target,
        graphlet_basis=basis,
        summary_config=summary_config,
    )
    calls: list[nx.Graph] = []

    def fake_predictor(_model, graph, **_kwargs):
        calls.append(graph.copy())
        return TopologyPrediction(
            graphlet_target=target_vector,
            graphlet_mass_target=target_mass,
            graphlet_history=basis.unflatten_history(target_vector),
            graphlet_connected_mass={
                key: float(value) for key, value in zip(basis.sizes, target_mass)
            },
        )

    _, trace = refine_graph_with_topology_predictions(
        source,
        model=None,
        graphlet_basis=basis,
        summary_config=summary_config,
        refiner_config={
            "mode": "energy",
            "steps": 2,
            "proposal_budget": -1,
            "valid_candidate_budget": -1,
            "preserve_connectivity": True,
            "selection": "greedy",
            "graphlet_weight": 1.0,
            "graphlet_mass_weight": 0.0,
            "accept_only_improving": True,
            "prediction_horizon": {
                "mode": "annealed",
                "initial_k": 3,
                "final_k": 1,
                "schedule": "linear",
                "refresh_on_plateau": True,
            },
        },
        prediction_fn=fake_predictor,
        rng=np.random.default_rng(0),
        return_trace=True,
    )

    accepted = [row for row in trace if row.get("accepted")]
    assert len(accepted) == 2
    assert len(calls) == 1
    assert [row["prediction_horizon"] for row in accepted] == [3, 3]
    assert [row["prediction_calls"] for row in accepted] == [1, 1]


def test_refiner_uses_frozen_graphlet_gain_and_refreshes_after_acceptance() -> None:
    basis, summary_config = _basis(4)
    source = nx.havel_hakimi_graph([3, 3, 2, 2, 2, 2])
    actions, candidate_graphs, _ = propose_valid_topology_swaps(
        source,
        proposal_budget=-1,
        valid_candidate_budget=-1,
        preserve_connectivity=True,
        rng=np.random.default_rng(0),
    )
    target_graph = None
    target_vector = None
    target_mass = None
    for action in actions:
        candidate = candidate_graphs[action]
        vector, mass = extract_topology_graphlet_target(
            candidate,
            graphlet_basis=basis,
            summary_config=summary_config,
        )
        distance, _, _ = topology_graphlet_discrepancy(
            source,
            vector,
            mass,
            graphlet_basis=basis,
            summary_config=summary_config,
        )
        if distance > 1.0e-8:
            target_graph = candidate
            target_vector = vector
            target_mass = mass
            break
    assert target_graph is not None
    assert target_vector is not None and target_mass is not None
    calls: list[nx.Graph] = []
    prediction_times: list[float] = []

    def fake_predictor(_model, graph, **kwargs):
        calls.append(graph.copy())
        prediction_times.append(float(kwargs["time"]))
        return TopologyPrediction(
            graphlet_target=target_vector,
            graphlet_mass_target=target_mass,
            graphlet_history=basis.unflatten_history(target_vector),
            graphlet_connected_mass={
                key: float(value) for key, value in zip(basis.sizes, target_mass)
            },
        )

    refined, trace = refine_graph_with_topology_predictions(
        source,
        model=None,  # The injected predictor owns inference in this test.
        graphlet_basis=basis,
        summary_config=summary_config,
        refiner_config={
            "mode": "energy",
            "steps": 3,
            "proposal_budget": -1,
            "valid_candidate_budget": -1,
            "preserve_connectivity": True,
            "selection": "greedy",
            "graphlet_weight": 1.0,
            "graphlet_mass_weight": 0.0,
            "accept_only_improving": True,
            "refresh_prediction_every": 1,
            "reject_revisited_states": True,
        },
        prediction_fn=fake_predictor,
        rng=np.random.default_rng(0),
        return_trace=True,
    )

    accepted = [row for row in trace if row.get("accepted")]
    assert accepted
    assert all(row["graphlet_gain"] > 0.0 for row in accepted)
    assert all(
        row["current_graphlet_discrepancy"]
        - row["candidate_graphlet_discrepancy"]
        == pytest.approx(row["graphlet_gain"])
        for row in accepted
    )
    assert len(calls) >= 2
    assert nx.utils.graphs_equal(calls[1], refined)
    assert prediction_times[:2] == pytest.approx([0.0, 1.0 / 3.0])
    assert [source.degree(node) for node in sorted(source)] == [
        refined.degree(node) for node in sorted(refined)
    ]
    assert nx.is_connected(refined)
    replay = source.copy()
    initial_degrees = [replay.degree(node) for node in sorted(replay)]
    for row in accepted:
        replay = apply_action(replay, row["action"])
        assert nx.number_of_selfloops(replay) == 0
        assert nx.is_connected(replay)
        assert [replay.degree(node) for node in sorted(replay)] == initial_degrees
    assert nx.utils.graphs_equal(replay, refined)
