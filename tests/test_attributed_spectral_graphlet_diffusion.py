from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.properties.summary import SummaryConfig
from grapher.rewiring_mlp.attributed.data import (
    GraphCategoryVocabulary,
    GraphletBasis,
)
from grapher.rewiring_mlp.attributed.graphlet_diffusion import (
    attributed_graphlet_clr_to_simplex,
    attributed_graphlet_simplex_to_clr,
    candidate_attributed_graphlet_counts,
    extract_attributed_graphlet_counts,
    extract_attributed_graphlet_simplex,
)
from grapher.rewiring_mlp.attributed.refiner import HybridRefinerConfig
from grapher.rewiring_mlp.attributed.spectral import attributed_laplacian_spectra
from grapher.rewiring_mlp.attributed.spectral_data import (
    AttributedTrainingPair,
    build_attributed_spectral_diffusion_examples,
    collate_attributed_spectral_examples,
    resolve_attributed_diffusion_endpoints,
)
from grapher.rewiring_mlp.attributed.spectral_graphlet_refiner import (
    AttributedSpectralGraphletRefinerConfig,
    _apply_attributed_action,
    _graphlet_probability_and_mass_distance,
    _normalized_conserved_l1,
    _typed_degree_matrix,
    _typed_actions_for_topology_action,
    _weighted_valence_vector,
    refine_attributed_graph_with_spectral_graphlet_diffusion,
)
from grapher.rewiring_mlp.attributed.spectral_model import (
    AttributedSpectralGraphletTransformerPredictor,
    load_attributed_spectral_graphlet_checkpoint,
    save_attributed_spectral_graphlet_checkpoint,
)
from grapher.rewiring_mlp.core.rewiring import (
    apply_action,
    candidate_actions_from_edge_pair,
    make_action,
)
from grapher.rewiring_mlp.molecular.constraints import QM9_PROJECTED_MAX_VALENCE
from grapher.rewiring_mlp.molecular.graph_io import is_valid_molecular_graph
from grapher.rewiring_mlp.molecular.typed_invariants import (
    attributed_rewiring_invariant_matches_graph,
    extract_attributed_rewiring_invariant,
    extract_typed_invariant,
    typed_invariant_matches_graph,
)
from grapher.utils.io import load_yaml
from scripts.run_attributed_grapher import _generation_rdkit_valid


def _cycle_graph(edges: list[tuple[int, int]] | None = None) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from((node, {"atomic_num": 6, "atom_type": 6}) for node in range(6))
    for u, v in edges or [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]:
        graph.add_edge(u, v, bond_type=1, bond_order=1.0)
    return graph


def _source_target_action():
    source = _cycle_graph()
    action = make_action([(0, 1), (3, 4)], [(0, 3), (1, 4)])
    target = apply_action(source, action)
    for u, v in target.edges():
        target.edges[u, v]["bond_type"] = 1
        target.edges[u, v]["bond_order"] = 1.0
    return source, target, action


def _projected_tetravalent_nitrogen_graph() -> nx.Graph:
    graph = nx.Graph()
    for node, atomic_num in enumerate((7, 6, 8, 8)):
        graph.add_node(node, atomic_num=atomic_num, atom_type=atomic_num)
    graph.add_edge(0, 1, bond_type=1, bond_order=1.0)
    graph.add_edge(0, 2, bond_type=2, bond_order=2.0)
    graph.add_edge(0, 3, bond_type=1, bond_order=1.0)
    graph.graph.update(
        qm9_source_state_projection_policy=(
            "audit_and_project_from_categorical_graph_state_v1"
        ),
        projected_formal_charge_atoms=[[0, 1], [3, -1]],
        projected_chiral_atoms=[],
        projected_stereo_bonds=[],
    )
    return graph


def _vocabulary(graphs: list[nx.Graph]) -> GraphCategoryVocabulary:
    return GraphCategoryVocabulary.from_graphs(
        graphs,
        {
            "node_attribute": "atomic_num",
            "node_categories": [6],
            "edge_attribute": "bond_type",
            "edge_categories": [1],
        },
    )


def _basis(
    graphs: list[nx.Graph],
    vocabulary: GraphCategoryVocabulary,
    *,
    topology_filter: str = "all",
) -> GraphletBasis:
    return GraphletBasis.fit_from_graphs(
        graphs,
        {
            "graphlet_history": True,
            "graphlet_k_min": 3,
            "graphlet_k_max": 4,
            "graphlet_connected_only": True,
            "graphlet_topology_filter": topology_filter,
            "graphlet_num_samples": None,
            "attributed": True,
            "node_attribute": "atomic_num",
            "edge_attribute": "bond_type",
            "attributed_backend": "python",
        },
        vocabulary=vocabulary,
        attributed=True,
        seed=7,
    )


def _small_model(vocabulary: GraphCategoryVocabulary, basis: GraphletBasis):
    return AttributedSpectralGraphletTransformerPredictor(
        num_node_categories=vocabulary.num_node_categories,
        num_edge_categories=vocabulary.num_edge_categories,
        graphlet_block_widths=basis.simplex_block_widths,
        hidden_dim=32,
        edge_dim=16,
        graph_dim=32,
        num_layers=2,
        spectral_dim=32,
        spectral_layers=2,
        spectral_heads=4,
        spectral_ff_dim=64,
        graphlet_dim=48,
        dropout=0.0,
        graphlet_dropout=0.0,
    )


def test_dual_spectra_have_expected_traces() -> None:
    graph = _cycle_graph()
    spectra = attributed_laplacian_spectra(graph)
    assert spectra.shape == (2, 6)
    assert np.allclose(spectra[:, 0], 0.0, atol=1.0e-8)
    assert np.allclose(spectra.sum(axis=1), [12.0, 12.0], atol=1.0e-8)


def test_attributed_graphlet_simplex_clr_roundtrip() -> None:
    source, target, _ = _source_target_action()
    vocabulary = _vocabulary([source, target])
    basis = _basis([source, target], vocabulary)
    probabilities, mask, _ = extract_attributed_graphlet_simplex(
        source, graphlet_basis=basis
    )
    for start, stop in basis.simplex_slices:
        assert np.isclose(probabilities[start:stop].sum(), 1.0)
        assert mask[start:stop].all()
    logits = attributed_graphlet_simplex_to_clr(
        probabilities,
        graphlet_basis=basis,
        epsilon=1.0e-4,
        coordinate_mask=mask,
    )
    reconstructed = attributed_graphlet_clr_to_simplex(
        logits,
        graphlet_basis=basis,
        coordinate_mask=mask,
    )
    assert np.allclose(reconstructed, probabilities, atol=1.1e-4)


def test_stateful_attributed_graphlet_delta_matches_full_recount() -> None:
    source, target, action = _source_target_action()
    vocabulary = _vocabulary([source, target])
    basis = _basis([source, target], vocabulary)
    source_counts = extract_attributed_graphlet_counts(source, graphlet_basis=basis)
    updated = candidate_attributed_graphlet_counts(
        source,
        target,
        action,
        current_counts=source_counts,
        graphlet_basis=basis,
    )
    exact = extract_attributed_graphlet_counts(target, graphlet_basis=basis)
    assert updated == exact


def test_stateful_cycle_only_delta_matches_full_recount() -> None:
    source, target, action = _source_target_action()
    vocabulary = _vocabulary([source, target])
    basis = _basis(
        [source, target],
        vocabulary,
        topology_filter="simple_cycle",
    )
    source_counts = extract_attributed_graphlet_counts(source, graphlet_basis=basis)
    updated = candidate_attributed_graphlet_counts(
        source,
        target,
        action,
        current_counts=source_counts,
        graphlet_basis=basis,
    )
    exact = extract_attributed_graphlet_counts(target, graphlet_basis=basis)
    assert updated == exact
    assert basis.topology_filter == "simple_cycle"


def test_cycle_only_delta_handles_ring_destroyed_by_inserted_chord() -> None:
    source = nx.Graph()
    source.add_nodes_from(
        (node, {"atomic_num": 6, "atom_type": 6}) for node in range(7)
    )
    source.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (0, 4),
            (2, 5),
            (1, 6),
            (4, 6),
            (5, 6),
        ]
    )
    for u, v in source.edges():
        source.edges[u, v].update(bond_type=1, bond_order=1.0)
    action = make_action([(0, 4), (2, 5)], [(0, 2), (4, 5)])
    target = apply_action(source, action)
    for u, v in target.edges():
        target.edges[u, v].update(bond_type=1, bond_order=1.0)

    vocabulary = _vocabulary([source, target])
    basis = GraphletBasis.fit_from_graphs(
        [source, target],
        {
            "graphlet_history": True,
            "graphlet_k_min": 4,
            "graphlet_k_max": 4,
            "graphlet_connected_only": True,
            "graphlet_topology_filter": "simple_cycle",
            "graphlet_num_samples": None,
            "attributed": True,
            "node_attribute": "atomic_num",
            "edge_attribute": "bond_type",
            "attributed_backend": "python",
        },
        vocabulary=vocabulary,
        attributed=True,
        seed=7,
    )
    source_counts = extract_attributed_graphlet_counts(source, graphlet_basis=basis)
    updated = candidate_attributed_graphlet_counts(
        source,
        target,
        action,
        current_counts=source_counts,
        graphlet_basis=basis,
    )
    exact = extract_attributed_graphlet_counts(target, graphlet_basis=basis)
    assert updated == exact

    # The reverse move removes the chord and creates induced rings.
    reverse_action = make_action([(0, 2), (4, 5)], [(0, 4), (2, 5)])
    target_counts = extract_attributed_graphlet_counts(target, graphlet_basis=basis)
    reverse_updated = candidate_attributed_graphlet_counts(
        target,
        source,
        reverse_action,
        current_counts=target_counts,
        graphlet_basis=basis,
    )
    assert reverse_updated == source_counts


def test_training_samples_continuous_diffusion_not_rewiring_states() -> None:
    source, target, _ = _source_target_action()
    vocabulary = _vocabulary([source, target])
    basis = _basis([source, target], vocabulary)
    examples, diagnostics = build_attributed_spectral_diffusion_examples(
        [AttributedTrainingPair(source, target)],
        vocabulary=vocabulary,
        graphlet_basis=basis,
        diffusion_config={
            "samples_per_graph": 3,
            "paths_per_graph": 1,
            "time_sampling": "stratified",
            "schedule": "cosine",
            "spectral_sigma": 0.4,
            "graphlet_sigma": 0.4,
            "preserve_spectral_trace": True,
            "fix_spectral_lambda1": True,
        },
        spectral_config={"normalization": "mean_degree"},
        seed=13,
    )
    assert diagnostics["training_state_source"] == "continuous_summary_diffusion"
    assert diagnostics["rewiring_used_for_training_states"] is False
    assert len(examples) == 3
    assert all(example.conditioning_graph.number_of_nodes() == 6 for example in examples)
    assert any(
        not np.allclose(example.current_spectra, example.source_spectra)
        for example in examples
    )
    assert all(
        np.allclose(
            example.current_spectra.sum(axis=1),
            example.source_spectra.sum(axis=1),
            atol=1.0e-6,
        )
        for example in examples
    )


@pytest.mark.parametrize(
    ("config_path", "expected_source_gate"),
    [
        (
            "configs/experiments/grapher/"
            "qm9_attributed_spectral_graphlet_light.yaml",
            False,
        ),
        (
            "configs/experiments/grapher/"
            "qm9_attributed_spectral_graphlet.yaml",
            True,
        ),
    ],
)
def test_qm9_projected_valence_config_constructs_training_endpoint(
    config_path: str,
    expected_source_gate: bool,
) -> None:
    graph = _projected_tetravalent_nitrogen_graph()
    vocabulary = GraphCategoryVocabulary.from_graphs(
        [graph],
        {
            "node_attribute": "atomic_num",
            "node_categories": [6, 7, 8, 9],
            "edge_attribute": "bond_type",
            "edge_categories": [1, 2, 3],
        },
    )
    config = load_yaml(config_path)
    projected_limits = {
        key: value for key, value in QM9_PROJECTED_MAX_VALENCE.items() if key != 1
    }

    training_limits = config["training_sources"]["typed_constructor"][
        "max_weighted_valence"
    ]
    generation_limits = config["constructor"]["max_weighted_valence"]
    refiner_limits = config["attributed_refiner"]["molecular"]["max_valence"]
    assert training_limits == projected_limits
    assert generation_limits == projected_limits
    assert refiner_limits == projected_limits
    refiner_config = AttributedSpectralGraphletRefinerConfig.from_dict(
        config["attributed_refiner"]
    )
    assert refiner_config.rdkit_infer_projected_formal_charges is True
    assert (
        config["generation"]["require_rdkit_source_validity"]
        is expected_source_gate
    )
    assert refiner_config.require_rdkit_source_validity is expected_source_gate

    source, target, metadata = resolve_attributed_diffusion_endpoints(
        graph,
        vocabulary=vocabulary,
        source_config=config["training_sources"],
        rng=np.random.default_rng(42),
    )
    invariant = extract_typed_invariant(
        target,
        edge_types=vocabulary.edge_values,
        node_attribute="atomic_num",
        edge_attribute="bond_type",
    )
    nitrogen = next(
        signature for signature in invariant.signatures if signature.node_type == 7
    )

    assert metadata["source_mode"] == "target_typed_constructor"
    assert nitrogen.degree == 3
    assert nitrogen.weighted_degree(invariant.edge_types) == 4.0
    assert typed_invariant_matches_graph(source, invariant)


def test_all_qm9_typed_generation_configs_use_projected_valence_policy() -> None:
    projected_limits = {
        key: value for key, value in QM9_PROJECTED_MAX_VALENCE.items() if key != 1
    }
    dhvae = load_yaml("configs/experiments/dhvae/qm9_typed.yaml")
    hybrid = load_yaml(
        "configs/experiments/grapher/qm9_attributed_hybrid_endpoint_graphlet.yaml"
    )

    assert dhvae["typed_signature"]["max_weighted_valence"] == projected_limits
    assert (
        hybrid["endpoint_trajectory"]["typed_constructor"][
            "max_weighted_valence"
        ]
        == projected_limits
    )
    assert hybrid["typed_signature"]["max_weighted_valence"] == projected_limits

    refiner_config = HybridRefinerConfig.from_dict(hybrid["hybrid_refiner"])
    assert refiner_config.molecular_max_valence == projected_limits
    assert refiner_config.rdkit_infer_projected_formal_charges is True


def test_projected_charge_inference_is_opt_in_for_rdkit_validity() -> None:
    graph = _projected_tetravalent_nitrogen_graph()

    assert is_valid_molecular_graph(graph) is False
    assert (
        is_valid_molecular_graph(graph, infer_projected_formal_charges=True) is True
    )
    assert (
        _generation_rdkit_valid(
            graph,
            infer_projected_formal_charges=True,
        )
        is True
    )
    assert (
        _generation_rdkit_valid(
            graph,
            infer_projected_formal_charges=False,
        )
        is False
    )


def test_model_forward_loss_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    source, target, _ = _source_target_action()
    vocabulary = _vocabulary([source, target])
    basis = _basis([source, target], vocabulary)
    examples, _ = build_attributed_spectral_diffusion_examples(
        [AttributedTrainingPair(source, target)],
        vocabulary=vocabulary,
        graphlet_basis=basis,
        diffusion_config={
            "samples_per_graph": 1,
            "paths_per_graph": 1,
            "spectral_sigma": 0.2,
            "graphlet_sigma": 0.2,
        },
        seed=17,
    )
    batch = collate_attributed_spectral_examples(examples, vocabulary)
    model = _small_model(vocabulary, basis)
    outputs = model(batch)
    assert outputs["clean_spectra"].shape == (1, 2, 6)
    assert outputs["clean_graphlet_logits"].shape[-1] == basis.simplex_width
    assert torch.allclose(
        outputs["clean_spectra"].sum(dim=-1),
        batch.source_spectra.sum(dim=-1),
        atol=1.0e-4,
    )
    loss, metrics = model.loss(
        batch,
        loss_weights={
            "spectrum": 1.0,
            "moment2": 0.25,
            "graphlet_logit": 1.0,
            "graphlet_probability": 0.25,
            "graphlet_selected_mass": 1.0,
        },
    )
    assert torch.isfinite(loss)
    assert "topology_moment2_relative_error" in metrics
    assert "graphlet_selected_mass_loss" in metrics
    assert "graphlet_selected_mass_mae" in metrics

    checkpoint = tmp_path / "checkpoint.pt"
    save_attributed_spectral_graphlet_checkpoint(
        model,
        checkpoint,
        vocabulary=vocabulary,
        graphlet_basis=basis,
        summary_config=SummaryConfig.from_dict(
            {
                "graphlet_history": True,
                "graphlet_k_min": 3,
                "graphlet_k_max": 4,
                "graphlet_connected_only": True,
            }
        ),
    )
    loaded, loaded_vocabulary, loaded_basis, _, payload = (
        load_attributed_spectral_graphlet_checkpoint(checkpoint, device="cpu")
    )
    assert payload["format"] == "attributed_spectral_graphlet_transformer_v1"
    assert loaded_vocabulary == vocabulary
    assert loaded_basis.to_dict() == basis.to_dict()
    loaded_outputs = loaded(batch)
    assert torch.allclose(
        outputs["clean_spectra"], loaded_outputs["clean_spectra"], atol=1.0e-6
    )


def test_generation_refiner_preserves_indexed_typed_invariant() -> None:
    source, target, _ = _source_target_action()
    vocabulary = _vocabulary([source, target])
    basis = _basis([source, target], vocabulary)
    model = _small_model(vocabulary, basis)
    invariant = extract_typed_invariant(
        source,
        edge_types=vocabulary.edge_values,
        node_attribute="atomic_num",
        edge_attribute="bond_type",
    )
    config = AttributedSpectralGraphletRefinerConfig.from_dict(
        {
            "mode": "attributed_spectral_graphlet",
            "steps": 1,
            "proposal_budget": -1,
            "valid_candidate_budget": -1,
            "min_improvement": -1.0e6,
            "min_relative_improvement": -1.0e6,
            "molecular": {
                "require_same_edge_type_pair": True,
                "preserve_removed_edge_type": True,
                "enforce_molecular_valence": True,
                "allowed_bond_types": [1],
                "max_valence": {6: 4.0},
                "rdkit_candidate_check": False,
            },
            "debug": {"enabled": False},
        }
    )
    refined, trace = refine_attributed_graph_with_spectral_graphlet_diffusion(
        model,
        source,
        vocabulary=vocabulary,
        graphlet_basis=basis,
        config=config,
        device="cpu",
        rng=np.random.default_rng(23),
        return_trace=True,
    )
    assert typed_invariant_matches_graph(refined, invariant)
    assert refined.number_of_edges() == source.number_of_edges()
    assert nx.is_connected(refined)
    assert trace


def _mixed_bond_cycle_graph() -> nx.Graph:
    graph = _cycle_graph()
    graph.edges[0, 1]["bond_type"] = 1
    graph.edges[0, 1]["bond_order"] = 1.0
    graph.edges[3, 4]["bond_type"] = 2
    graph.edges[3, 4]["bond_order"] = 2.0
    return graph


def test_cross_type_kernel_enumerates_both_type_assignments_per_orientation() -> None:
    graph = _mixed_bond_cycle_graph()
    vocabulary = GraphCategoryVocabulary.from_graphs(
        [graph],
        {
            "node_attribute": "atomic_num",
            "node_categories": [6],
            "edge_attribute": "bond_type",
            "edge_categories": [1, 2],
        },
    )
    config = AttributedSpectralGraphletRefinerConfig.from_dict(
        {
            "molecular": {
                "require_same_edge_type_pair": False,
                "preserve_global_edge_type_counts": True,
                "enumerate_edge_type_permutations": True,
                "preserve_node_types": True,
                "preserve_ordinary_degree": True,
                "preserve_typed_degree": False,
                "preserve_weighted_valence": False,
                "rdkit_candidate_check": False,
            }
        }
    )
    actions = candidate_actions_from_edge_pair((0, 1), (3, 4))
    assert len(actions) == 2
    typed = [
        candidate
        for topology_action in actions
        for candidate in _typed_actions_for_topology_action(
            graph, topology_action, vocabulary=vocabulary, config=config
        )
    ]
    assert len(typed) == 4
    assert {tuple(action.added_edge_categories) for action in typed} == {(1, 2), (2, 1)}


def test_cross_type_action_preserves_relaxed_invariant_but_can_change_typed_degree() -> None:
    graph = _mixed_bond_cycle_graph()
    vocabulary = GraphCategoryVocabulary.from_graphs(
        [graph],
        {
            "node_attribute": "atomic_num",
            "node_categories": [6],
            "edge_attribute": "bond_type",
            "edge_categories": [1, 2],
        },
    )
    config = AttributedSpectralGraphletRefinerConfig.from_dict(
        {
            "molecular": {
                "require_same_edge_type_pair": False,
                "preserve_global_edge_type_counts": True,
                "enumerate_edge_type_permutations": True,
                "preserve_node_types": True,
                "preserve_ordinary_degree": True,
                "preserve_typed_degree": False,
                "preserve_weighted_valence": False,
                "enforce_molecular_valence": True,
                "allowed_bond_types": [1, 2],
                "max_valence": {6: 4.0},
                "rdkit_candidate_check": False,
            }
        }
    )
    topology_action = next(
        action
        for action in candidate_actions_from_edge_pair((0, 1), (3, 4))
        if set(action[1]) == {(0, 3), (1, 4)}
    )
    typed_actions = _typed_actions_for_topology_action(
        graph, topology_action, vocabulary=vocabulary, config=config
    )
    # The first assignment aligns the removed categories with the canonical
    # added-edge order and changes the local bond-type incidences of nodes 1/3.
    candidate = _apply_attributed_action(graph, typed_actions[0], vocabulary)

    relaxed = extract_attributed_rewiring_invariant(
        graph,
        edge_types=vocabulary.edge_values,
        node_attribute="atomic_num",
        edge_attribute="bond_type",
    )
    strict = extract_typed_invariant(
        graph,
        edge_types=vocabulary.edge_values,
        node_attribute="atomic_num",
        edge_attribute="bond_type",
    )
    assert attributed_rewiring_invariant_matches_graph(candidate, relaxed)
    assert not typed_invariant_matches_graph(candidate, strict)
    assert sorted(data["bond_type"] for _, _, data in candidate.edges(data=True)) == sorted(
        data["bond_type"] for _, _, data in graph.edges(data=True)
    )
    assert [candidate.degree(i) for i in range(6)] == [graph.degree(i) for i in range(6)]


def test_chemistry_drift_scores_detect_cross_type_reassignment() -> None:
    graph = _mixed_bond_cycle_graph()
    vocabulary = GraphCategoryVocabulary.from_graphs(
        [graph],
        {
            "node_attribute": "atomic_num",
            "node_categories": [6],
            "edge_attribute": "bond_type",
            "edge_categories": [1, 2],
        },
    )
    config = AttributedSpectralGraphletRefinerConfig.from_dict(
        {
            "molecular": {
                "require_same_edge_type_pair": False,
                "preserve_global_edge_type_counts": True,
                "enumerate_edge_type_permutations": True,
                "typed_degree_drift_weight": 1.0,
                "weighted_valence_drift_weight": 1.0,
                "rdkit_candidate_check": False,
            }
        }
    )
    topology_action = next(
        action
        for action in candidate_actions_from_edge_pair((0, 1), (3, 4))
        if set(action[1]) == {(0, 3), (1, 4)}
    )
    action = _typed_actions_for_topology_action(
        graph,
        topology_action,
        vocabulary=vocabulary,
        config=config,
    )[0]
    candidate = _apply_attributed_action(graph, action, vocabulary)

    typed_reference = _typed_degree_matrix(graph, vocabulary=vocabulary)
    valence_reference = _weighted_valence_vector(graph, vocabulary=vocabulary)
    typed_drift = _normalized_conserved_l1(
        _typed_degree_matrix(candidate, vocabulary=vocabulary), typed_reference
    )
    valence_drift = _normalized_conserved_l1(
        _weighted_valence_vector(candidate, vocabulary=vocabulary),
        valence_reference,
    )
    assert typed_drift > 0.0
    assert valence_drift > 0.0
    assert typed_drift <= 1.0
    assert valence_drift <= 1.0


def test_cycle_guidance_separates_class_composition_and_selected_mass() -> None:
    basis = GraphletBasis(
        keys_by_k={"3": ("ring_a", "ring_b", "__overflow__")},
        connected_only=True,
        topology_filter="simple_cycle",
        attributed=True,
        node_attribute="atomic_num",
        edge_attribute="bond_type",
        overflow_key="__overflow__",
        attributed_backend="python",
    )
    # Same selected mass (0.20), different distribution over selected classes.
    left = np.asarray([0.15, 0.04, 0.01, 0.80], dtype=np.float64)
    right = np.asarray([0.04, 0.15, 0.01, 0.80], dtype=np.float64)
    probability, mass = _graphlet_probability_and_mass_distance(
        left,
        right,
        graphlet_basis=basis,
        coordinate_mask=np.ones(4, dtype=np.bool_),
        metric="rmse",
        size_weights={"3": 1.0},
    )
    assert probability > 0.0
    assert mass == pytest.approx(0.0)

    lower_mass = np.asarray([0.03, 0.01, 0.01, 0.95], dtype=np.float64)
    _, mass = _graphlet_probability_and_mass_distance(
        left,
        lower_mass,
        graphlet_basis=basis,
        coordinate_mask=np.ones(4, dtype=np.bool_),
        metric="rmse",
        size_weights={"3": 1.0},
    )
    assert mass == pytest.approx(0.15)


def test_v2_cycle_config_uses_raw_validity_and_balanced_guidance() -> None:
    config = load_yaml(
        "configs/experiments/grapher/"
        "qm9_attributed_spectral_cycle_graphlet_v2.yaml"
    )
    refiner = AttributedSpectralGraphletRefinerConfig.from_dict(
        config["attributed_refiner"]
    )
    assert refiner.rdkit_validation_mode == "raw"
    assert refiner.rdkit_infer_projected_formal_charges is False
    assert refiner.graphlet_probability_distance_weight > 0.0
    assert refiner.graphlet_selected_mass_weight > 0.0
    assert refiner.typed_degree_drift_weight > 0.0
    assert refiner.weighted_valence_drift_weight > 0.0
    assert config["generation"]["require_rdkit_final_validity"] is True


def test_revised_qm9_configs_enable_bond_reassigning_kernel() -> None:
    for config_path in (
        "configs/experiments/grapher/qm9_attributed_spectral_graphlet_light.yaml",
        "configs/experiments/grapher/qm9_attributed_spectral_graphlet.yaml",
    ):
        config = load_yaml(config_path)
        molecular = config["attributed_refiner"]["molecular"]
        assert molecular["require_same_edge_type_pair"] is False
        assert molecular["preserve_global_edge_type_counts"] is True
        assert molecular["enumerate_edge_type_permutations"] is True
        assert molecular["preserve_typed_degree"] is False
        refiner = AttributedSpectralGraphletRefinerConfig.from_dict(
            config["attributed_refiner"]
        )
        assert refiner.require_same_edge_type_pair is False
        assert refiner.preserve_typed_degree is False


def test_attributed_hogdiff_ou_bridge_preserves_both_spectral_traces() -> None:
    source, target, _ = _source_target_action()
    vocabulary = _vocabulary([source, target])
    basis = _basis([source, target], vocabulary)
    examples, diagnostics = build_attributed_spectral_diffusion_examples(
        [AttributedTrainingPair(source, target)],
        vocabulary=vocabulary,
        graphlet_basis=basis,
        diffusion_config={
            "bridge": "ou_bridge",
            "ou_num_scales": 32,
            "ou_schedule": "linear",
            "ou_eps": 0.005,
            "samples_per_graph": 3,
            "paths_per_graph": 1,
            "spectral_sigma": 0.2,
            "graphlet_sigma": 0.2,
            "preserve_spectral_trace": True,
            "fix_spectral_lambda1": True,
        },
        spectral_config={"normalization": "mean_degree"},
        seed=29,
    )
    assert diagnostics["bridge"] == "ou_bridge"
    assert diagnostics["ou_num_scales"] == 32
    assert len(examples) == 3
    for example in examples:
        assert np.allclose(
            example.current_spectra.sum(axis=1),
            example.source_spectra.sum(axis=1),
            atol=1.0e-6,
        )
        assert np.allclose(example.current_spectra[:, 0], 0.0, atol=1.0e-8)
