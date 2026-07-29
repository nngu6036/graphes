from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pytest

from grapher.construction.coarse import construct_coarse_graph
from grapher.properties.kernel_residual import (
    KernelResidualConfig,
    KernelResidualSummarySampler,
    StructuralResidualCodec,
    degree_wasserstein_distance,
    helmert_basis,
    ilr_inverse,
    ilr_transform,
    k3_connected_mass,
    triangle_count_bounds,
    validate_degree_condition,
)
from grapher.properties.summary import (
    ORCA_EXEC,
    SummaryConfig,
    _orca_graphlet_orbit_mapping,
    extract_summary,
    python_orbit_count_vector,
)
from grapher.utils.motifs import (
    PythonCanonicalizer,
    derive_k3_graphlet_distribution,
    topology_graphlet_keys_by_size,
)


def _condition(graph: nx.Graph, cfg: SummaryConfig) -> dict:
    summary = extract_summary(graph, cfg)
    return {
        key: summary[key]
        for key in (
            "num_nodes",
            "num_edges",
            "degree_sequence",
            "degree_hist",
            "density",
        )
    }


def test_degree_wasserstein_identity_symmetry_and_shift():
    left = np.asarray([0.0, 1.0, 0.0])
    right = np.asarray([0.0, 0.0, 1.0])
    assert degree_wasserstein_distance(left, left) == 0.0
    expected = 0.5
    assert degree_wasserstein_distance(left, right) == pytest.approx(expected)
    assert degree_wasserstein_distance(right, left) == pytest.approx(expected)


def test_validation_selection_is_loaded_for_generation(tmp_path):
    selection_path = tmp_path / "target_summary_evaluation.json"
    selection_path.write_text(
        json.dumps(
            {
                "selected_kernel": {
                    "top_k": 20,
                    "bandwidth_multiplier": 0.5,
                }
            }
        ),
        encoding="utf-8",
    )
    config = KernelResidualConfig.from_dict(
        {
            "top_k": 10,
            "selection_path": str(selection_path),
            "kernel": {
                "bandwidth": "adaptive_kth",
                "bandwidth_multiplier": 1.0,
            },
            "residual": {"representation": "ilr"},
        }
    )
    assert config.top_k == 20
    assert config.bandwidth_multiplier == pytest.approx(0.5)


def test_helmert_ilr_round_trip():
    probabilities = np.asarray([0.05, 0.15, 0.3, 0.5])
    basis = helmert_basis(probabilities.size)
    assert np.allclose(basis.T @ basis, np.eye(probabilities.size - 1))
    encoded = ilr_transform(probabilities, epsilon=1.0e-12)
    decoded = ilr_inverse(encoded, dim=probabilities.size)
    assert np.allclose(decoded, probabilities, atol=1.0e-10)


def test_fixed_k3_and_k4_basis_is_complete():
    keys = topology_graphlet_keys_by_size(3, 4, connected_only=True)
    assert len(keys["3"]) == 2
    assert len(keys["4"]) == 6


def test_standard_orca_role_map_covers_every_k3_and_k4_orbit():
    k3_roles = {
        role
        for _, roles in _orca_graphlet_orbit_mapping(3)
        for role in roles
    }
    k4_roles = {
        role
        for _, roles in _orca_graphlet_orbit_mapping(4)
        for role in roles
    }
    assert k3_roles == {(1, 2), (2, 1), (3, 3)}
    assert k4_roles == {
        (4, 2),
        (5, 2),
        (6, 3),
        (7, 1),
        (8, 4),
        (9, 1),
        (10, 2),
        (11, 1),
        (12, 2),
        (13, 2),
        (14, 4),
    }


@pytest.mark.skipif(ORCA_EXEC is None, reason="ORCA executable is not configured")
def test_exact_orca_history_and_connected_mass_match_enumeration():
    shared = dict(
        clustering_summary=False,
        spectral_summary=False,
        motif_proxy=False,
        orbit_count=False,
        graphlet_history=True,
        graphlet_k_min=3,
        graphlet_k_max=4,
        graphlet_connected_only=True,
    )
    exact_cfg = SummaryConfig(**shared, graphlet_backend="exact_orca")
    enumeration_cfg = SummaryConfig(
        **shared,
        graphlet_backend="sampled",
        graphlet_num_samples=None,
    )
    graphs = [
        nx.path_graph(5),
        nx.cycle_graph(6),
        nx.complete_graph(5),
        nx.complete_bipartite_graph(3, 3),
        nx.Graph([(0, 1), (1, 2), (2, 0), (0, 3)]),
    ]
    for graph in graphs:
        exact = extract_summary(graph, exact_cfg)
        enumerated = extract_summary(graph, enumeration_cfg)
        for k in ("3", "4"):
            keys = set(exact["graphlet_history"][k]) | set(
                enumerated["graphlet_history"][k]
            )
            assert all(
                exact["graphlet_history"][k].get(key, 0.0)
                == pytest.approx(
                    enumerated["graphlet_history"][k].get(key, 0.0)
                )
                for key in keys
            )
            assert exact["graphlet_connected_mass"][k] == pytest.approx(
                enumerated["graphlet_connected_mass"][k]
            )


def test_python_orbit_fallback_uses_standard_four_node_ids():
    cycle = python_orbit_count_vector(nx.cycle_graph(4))
    claw = python_orbit_count_vector(nx.star_graph(3))
    paw = nx.Graph([(0, 1), (1, 2), (2, 0), (0, 3)])
    paw_counts = python_orbit_count_vector(paw)
    assert cycle[8] == pytest.approx(1.0)
    assert claw[6] == pytest.approx(0.75)
    assert claw[7] == pytest.approx(0.25)
    assert paw_counts[9] == pytest.approx(0.25)
    assert paw_counts[10] == pytest.approx(0.5)
    assert paw_counts[11] == pytest.approx(0.25)


def test_python_canonicalizer_is_invariant_to_node_insertion_order():
    canonicalizer = PythonCanonicalizer()
    graph = nx.path_graph(4)
    relabeled = nx.Graph()
    relabeled.add_nodes_from([3, 1, 0, 2])
    relabeled.add_edges_from([(3, 1), (1, 0), (0, 2)])
    assert canonicalizer.canonical_graph6(graph) == canonicalizer.canonical_graph6(
        relabeled
    )


def test_connected_k3_is_derived_from_degree_and_triangles():
    triangle = derive_k3_graphlet_distribution(
        [2, 2, 2],
        1,
        connected_only=True,
    )
    path = derive_k3_graphlet_distribution(
        [2, 1, 1],
        0,
        connected_only=True,
    )
    assert max(triangle.values()) == pytest.approx(1.0)
    assert max(path.values()) == pytest.approx(1.0)
    assert set(triangle) != set(path)


def test_all_induced_k3_counts_sum_to_one():
    distribution = derive_k3_graphlet_distribution(
        [2, 1, 1, 0],
        0,
        connected_only=False,
    )
    assert len(distribution) >= 2
    assert sum(distribution.values()) == pytest.approx(1.0)
    assert all(value >= 0.0 for value in distribution.values())


def test_infeasible_k3_triangle_count_is_rejected():
    with pytest.raises(ValueError, match="infeasible"):
        derive_k3_graphlet_distribution(
            [2, 2, 1, 1],
            1,
            connected_only=True,
        )


def test_connected_graphlet_mass_is_retained_and_k3_is_derived():
    cfg = SummaryConfig(
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
    cycle = extract_summary(nx.cycle_graph(4), cfg)
    matching = nx.Graph([(0, 1), (2, 3)])
    disconnected = extract_summary(matching, cfg)
    assert cycle["graphlet_connected_mass"]["4"] == pytest.approx(1.0)
    assert disconnected["graphlet_connected_mass"]["3"] == pytest.approx(0.0)
    assert k3_connected_mass([2, 2, 1, 1], 0) == pytest.approx(0.5)


def test_codec_masks_impossible_k4_block_and_rejects_unknown_keys():
    graph = nx.cycle_graph(3)
    cfg = SummaryConfig(
        degree_hist_max_degree=2,
        clustering_bins=5,
        clustering_summary=False,
        spectral_summary=False,
        motif_proxy=False,
        orbit_count=False,
        graphlet_history=True,
        graphlet_k_min=4,
        graphlet_k_max=4,
        graphlet_connected_only=True,
        graphlet_num_samples=None,
        graphlet_backend="sampled",
    )
    summary = extract_summary(graph, cfg)
    condition = _condition(graph, cfg)
    codec = StructuralResidualCodec(
        cfg,
        [summary],
        KernelResidualConfig(),
    )
    encoded = codec.encode(summary, condition)
    decoded = codec.decode(encoded, condition, template=summary)
    assert decoded["graphlet_history"]["4"] == {}

    invalid_graph = nx.path_graph(4)
    invalid = extract_summary(invalid_graph, cfg)
    invalid["graphlet_history"] = {"4": {"not-a-canonical-key": 1.0}}
    with pytest.raises(ValueError, match="outside the fixed"):
        codec.encode(invalid, _condition(invalid_graph, cfg))

    impossible = dict(summary)
    impossible["graphlet_history"] = {"4": {"not-a-canonical-key": 1.0}}
    with pytest.raises(ValueError, match="when n < k"):
        codec.encode(impossible, condition)


def test_triangle_bounds_contain_real_triangle_count():
    for graph in (
        nx.path_graph(6),
        nx.cycle_graph(6),
        nx.complete_bipartite_graph(3, 3),
        nx.complete_graph(5),
    ):
        sequence = sorted(
            [int(degree) for _, degree in graph.degree()],
            reverse=True,
        )
        condition = {
            "num_nodes": graph.number_of_nodes(),
            "degree_sequence": sequence,
        }
        lower, upper = triangle_count_bounds(condition)
        triangles = int(sum(nx.triangles(graph).values()) // 3)
        assert lower <= triangles <= upper


def test_deterministic_constructor_is_stable_and_valid():
    condition = {
        "num_nodes": 8,
        "degree_sequence": [3, 3, 2, 2, 2, 2, 1, 1],
    }
    constructor = {
        "type": "havel_hakimi",
        "ensure_connected": True,
        "random_relabel": False,
        "deterministic_seed": 17,
        "max_repair_trials": 10000,
    }
    first = construct_coarse_graph(
        condition,
        constructor,
        np.random.default_rng(1),
    )
    second = construct_coarse_graph(
        condition,
        constructor,
        np.random.default_rng(999),
    )
    assert sorted(first.edges()) == sorted(second.edges())
    assert nx.is_connected(first)
    assert sorted(
        [degree for _, degree in first.degree()],
        reverse=True,
    ) == condition["degree_sequence"]


def test_kernel_residual_samples_one_joint_donor_and_preserves_degree():
    prism = nx.Graph(
        [
            (0, 1),
            (1, 2),
            (2, 0),
            (3, 4),
            (4, 5),
            (5, 3),
            (0, 3),
            (1, 4),
            (2, 5),
        ]
    )
    bipartite = nx.complete_bipartite_graph(3, 3)
    cfg = SummaryConfig(
        degree_hist_max_degree=3,
        clustering_bins=6,
        spectral_bins=6,
        clustering_summary=True,
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
    constructor = {
        "type": "havel_hakimi",
        "ensure_connected": True,
        "random_relabel": False,
        "deterministic_seed": 0,
    }
    generator = {
        # Both donors are tied at distance zero; top_k=1 must retain the full
        # kth-boundary tie instead of deterministically keeping record zero.
        "top_k": 1,
        "distance": {
            "degree_wasserstein_weight": 1.0,
            "node_count_weight": 1.0,
            "edge_count_weight": 1.0,
        },
        "kernel": {
            "bandwidth_multiplier": 1.0,
            "min_bandwidth": 1.0e-6,
        },
        "residual": {
            "pseudocount": 1.0e-9,
            "derive_k3_from_degree_and_triangle": True,
        },
    }
    sampler = KernelResidualSummarySampler.fit(
        [prism, bipartite],
        cfg,
        constructor,
        generator,
        seed=3,
    )
    condition = _condition(prism, cfg)
    condition["degree_hist"] = np.asarray([1.0, 0.0])
    source = construct_coarse_graph(condition, constructor)
    expected = [extract_summary(prism, cfg), extract_summary(bipartite, cfg)]
    observed_pairings = set()

    for seed in range(20):
        sample, metadata = sampler.sample_conditioned(
            condition,
            np.random.default_rng(seed),
            source_graph=source,
            return_metadata=True,
        )
        assert sample["degree_sequence"] == condition["degree_sequence"]
        assert sample["num_nodes"] == condition["num_nodes"]
        assert sample["num_edges"] == condition["num_edges"]
        assert np.allclose(sample["degree_hist"], [0.0, 0.0, 0.0, 1.0])
        assert metadata["effective_neighbor_count"] == pytest.approx(2.0)

        donor_id = int(metadata["donor_record_id"])
        donor = expected[donor_id]
        assert np.allclose(
            sample["clustering_hist"],
            donor["clustering_hist"],
            atol=1.0e-6,
        )
        k4_keys = sampler.codec.graphlet_keys_by_k["4"]
        sample_k4 = np.asarray(
            [
                sample["graphlet_history"]["4"].get(key, 0.0)
                for key in k4_keys
            ]
        )
        donor_k4 = np.asarray(
            [
                donor["graphlet_history"]["4"].get(key, 0.0)
                for key in k4_keys
            ]
        )
        assert np.allclose(sample_k4, donor_k4, atol=1.0e-6)
        assert sample["graphlet_connected_mass"]["4"] == pytest.approx(
            donor["graphlet_connected_mass"]["4"],
            abs=1.0e-6,
        )
        observed_pairings.add(donor_id)

    assert observed_pairings == {0, 1}


def test_invalid_degree_condition_fails_fast():
    with pytest.raises(ValueError, match="even"):
        validate_degree_condition(
            {
                "num_nodes": 4,
                "degree_sequence": [2, 2, 2, 1],
            }
        )
