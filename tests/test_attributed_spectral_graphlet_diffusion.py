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
    refine_attributed_graph_with_spectral_graphlet_diffusion,
)
from grapher.rewiring_mlp.attributed.spectral_model import (
    AttributedSpectralGraphletTransformerPredictor,
    load_attributed_spectral_graphlet_checkpoint,
    save_attributed_spectral_graphlet_checkpoint,
)
from grapher.rewiring_mlp.core.rewiring import apply_action, make_action
from grapher.rewiring_mlp.molecular.constraints import QM9_PROJECTED_MAX_VALENCE
from grapher.rewiring_mlp.molecular.graph_io import is_valid_molecular_graph
from grapher.rewiring_mlp.molecular.typed_invariants import (
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


def _basis(graphs: list[nx.Graph], vocabulary: GraphCategoryVocabulary) -> GraphletBasis:
    return GraphletBasis.fit_from_graphs(
        graphs,
        {
            "graphlet_history": True,
            "graphlet_k_min": 3,
            "graphlet_k_max": 4,
            "graphlet_connected_only": True,
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
        },
    )
    assert torch.isfinite(loss)
    assert "topology_moment2_relative_error" in metrics

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
