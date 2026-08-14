from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.rewiring_mlp.attributed.data import (
    GraphCategoryVocabulary,
    GraphletBasis,
    HybridEndpointExample,
    aligned_havel_hakimi_source,
    build_endpoint_examples,
    collate_endpoint_examples,
)
from grapher.rewiring_mlp.attributed.model import (
    HybridEndpointPredictor,
    load_hybrid_endpoint_checkpoint,
    save_hybrid_endpoint_checkpoint,
)
from grapher.properties.summary import SummaryConfig, extract_summary
from grapher.rewiring_mlp.core.rewiring import apply_action, make_action

CUBE_EDGES = [
    (0, 1),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 7),
    (5, 6),
    (6, 7),
]

CUBE_ACTION = make_action(
    [(0, 1), (6, 7)],
    [(0, 7), (1, 6)],
)


def _cube_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(8))
    graph.add_edges_from(CUBE_EDGES)
    return graph


def _summary_config() -> SummaryConfig:
    return SummaryConfig(
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


def _example(
    current: nx.Graph,
    target: nx.Graph,
    *,
    summary_config: SummaryConfig,
    graphlet_basis: GraphletBasis,
    time: float = 0.25,
) -> HybridEndpointExample:
    summary = extract_summary(target, summary_config)
    return HybridEndpointExample(
        current_graph=current,
        target_graph=target,
        time=time,
        graphlet_target=graphlet_basis.flatten_history(summary["graphlet_history"]),
        graphlet_mass_target=graphlet_basis.flatten_mass(
            summary["graphlet_connected_mass"]
        ),
    )


def _small_model(
    vocabulary: GraphCategoryVocabulary,
    graphlet_basis: GraphletBasis,
) -> HybridEndpointPredictor:
    return HybridEndpointPredictor(
        num_node_categories=vocabulary.num_node_categories,
        num_edge_categories=vocabulary.num_edge_categories,
        graphlet_slices=graphlet_basis.slices,
        hidden_dim=16,
        edge_dim=12,
        graph_dim=14,
        num_layers=1,
        dropout=0.0,
        min_concentration=0.1,
    )


def _edge_set(graph: nx.Graph) -> set[tuple[int, int]]:
    return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges()}


def test_aligned_source_and_batch_require_per_node_degree_alignment():
    target = nx.path_graph(6)
    source, normalized_target = aligned_havel_hakimi_source(
        target,
        ensure_connected=True,
        rng=np.random.default_rng(0),
    )

    assert dict(source.degree()) == dict(normalized_target.degree())
    assert nx.is_connected(source)

    # This relabeling preserves the sorted degree multiset, but changes which
    # labelled node owns each degree. It is not a valid endpoint-training pair.
    current = nx.path_graph(4)
    misaligned = nx.relabel_nodes(
        current,
        {0: 1, 1: 0, 2: 2, 3: 3},
        copy=True,
    )
    assert sorted(dict(current.degree()).values()) == sorted(
        dict(misaligned.degree()).values()
    )
    assert dict(current.degree()) != dict(misaligned.degree())

    cfg = _summary_config()
    basis = GraphletBasis.from_config(cfg)
    example = _example(
        current,
        misaligned,
        summary_config=cfg,
        graphlet_basis=basis,
    )
    with pytest.raises(ValueError, match="each labelled node degree"):
        collate_endpoint_examples(
            [example],
            GraphCategoryVocabulary.topology_only(),
        )


def test_teacher_examples_share_clean_endpoint_and_graphlet_targets():
    target = _cube_graph()
    cfg = _summary_config()
    basis = GraphletBasis.from_config(cfg)
    examples, diagnostics = build_endpoint_examples(
        [target],
        summary_config=cfg,
        graphlet_basis=basis,
        trajectory_config={
            "steps": 2,
            "candidate_budget": 32,
            "states_per_graph": 3,
            "preserve_connectivity": True,
            "ensure_connected_source": True,
            "shared_relabel_augmentation": False,
        },
        seed=0,
    )

    target_summary = extract_summary(target, cfg)
    expected_graphlets = basis.flatten_history(target_summary["graphlet_history"])
    expected_mass = basis.flatten_mass(target_summary["graphlet_connected_mass"])

    assert diagnostics["num_graphs"] == 1
    assert len(examples) >= 2
    for example in examples:
        assert _edge_set(example.target_graph) == _edge_set(target)
        assert dict(example.current_graph.degree()) == dict(
            example.target_graph.degree()
        )
        assert np.allclose(example.graphlet_target, expected_graphlets)
        assert np.allclose(example.graphlet_mass_target, expected_mass)
        assert 0.0 <= example.time <= 1.0


def test_forward_simplexes_symmetry_equivariance_and_loss():
    torch.manual_seed(0)
    vocabulary = GraphCategoryVocabulary.topology_only()
    cfg = _summary_config()
    basis = GraphletBasis.from_config(cfg)
    current = _cube_graph()
    target = apply_action(current, CUBE_ACTION)
    example = _example(
        current,
        target,
        summary_config=cfg,
        graphlet_basis=basis,
    )
    batch = collate_endpoint_examples([example], vocabulary)
    model = _small_model(vocabulary, basis)
    model.eval()

    outputs = model(batch)
    node_probabilities, edge_probabilities = model.endpoint_probabilities(outputs)
    graphlet_means, mass_means = model.graphlet_means(outputs)

    assert outputs["node_logits"].shape == (1, 8, 1)
    assert outputs["edge_logits"].shape == (1, 8, 8, 2)
    assert outputs["graphlet_alpha"].shape == (1, basis.width)
    assert outputs["graphlet_mass_ab"].shape == (
        1,
        len(basis.sizes),
        2,
    )
    assert torch.allclose(
        node_probabilities.sum(dim=-1),
        torch.ones_like(node_probabilities[..., 0]),
    )
    assert torch.allclose(
        edge_probabilities.sum(dim=-1),
        torch.ones_like(edge_probabilities[..., 0]),
    )
    assert torch.allclose(
        edge_probabilities,
        edge_probabilities.transpose(1, 2),
        atol=1.0e-7,
    )
    assert torch.all(outputs["graphlet_alpha"] > 0.0)
    assert torch.all(outputs["graphlet_mass_ab"] > 0.0)
    for start, stop in basis.slices:
        assert torch.allclose(
            graphlet_means[:, start:stop].sum(dim=-1),
            torch.ones(1),
            atol=1.0e-6,
        )
    assert torch.all((mass_means > 0.0) & (mass_means < 1.0))

    loss, metrics = model.loss(batch)
    assert torch.isfinite(loss)
    assert all(np.isfinite(float(value)) for value in metrics.values())
    loss.backward()
    assert model.edge_head.weight.grad is not None
    assert float(model.edge_head.weight.grad.norm()) > 0.0
    assert model.graphlet_heads[0].weight.grad is not None
    assert float(model.graphlet_heads[0].weight.grad.norm()) > 0.0
    assert model.node_in.weight.grad is not None
    assert float(model.node_in.weight.grad.norm()) > 0.0

    mapping = {node: (3 * node + 1) % 8 for node in range(8)}
    permuted_example = _example(
        nx.relabel_nodes(current, mapping, copy=True),
        nx.relabel_nodes(target, mapping, copy=True),
        summary_config=cfg,
        graphlet_basis=basis,
    )
    permuted_batch = collate_endpoint_examples(
        [permuted_example],
        vocabulary,
    )
    with torch.no_grad():
        permuted_outputs = model(permuted_batch)
        permuted_loss, _ = model.loss(permuted_batch)

    old_to_new = torch.tensor(
        [mapping[node] for node in range(8)],
        dtype=torch.long,
    )
    new_to_old = torch.argsort(old_to_new)
    expected_node_logits = outputs["node_logits"][:, new_to_old, :]
    expected_edge_logits = outputs["edge_logits"][:, new_to_old, :, :][
        :, :, new_to_old, :
    ]
    assert torch.allclose(
        permuted_outputs["node_logits"],
        expected_node_logits,
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert torch.allclose(
        permuted_outputs["edge_logits"],
        expected_edge_logits,
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert torch.allclose(
        permuted_outputs["graphlet_alpha"],
        outputs["graphlet_alpha"],
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert torch.allclose(
        permuted_outputs["graphlet_mass_ab"],
        outputs["graphlet_mass_ab"],
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert torch.allclose(
        permuted_loss,
        loss.detach(),
        atol=1.0e-5,
        rtol=1.0e-5,
    )


def test_hybrid_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(1)
    vocabulary = GraphCategoryVocabulary.topology_only()
    cfg = _summary_config()
    basis = GraphletBasis.from_config(cfg)
    current = _cube_graph()
    target = apply_action(current, CUBE_ACTION)
    batch = collate_endpoint_examples(
        [
            _example(
                current,
                target,
                summary_config=cfg,
                graphlet_basis=basis,
            )
        ],
        vocabulary,
    )
    model = _small_model(vocabulary, basis)
    model.eval()
    with torch.no_grad():
        expected = model(batch)

    checkpoint_path = tmp_path / "hybrid_endpoint.pt"
    save_hybrid_endpoint_checkpoint(
        model,
        checkpoint_path,
        vocabulary=vocabulary,
        graphlet_basis=basis,
        summary_config=cfg,
        config={"experiment": "unit_test"},
        report={"status": "ok"},
    )
    (
        loaded_model,
        loaded_vocabulary,
        loaded_basis,
        loaded_summary_config,
        checkpoint,
    ) = load_hybrid_endpoint_checkpoint(
        checkpoint_path,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        actual = loaded_model(batch)

    assert loaded_vocabulary == vocabulary
    assert loaded_basis == basis
    assert loaded_summary_config.graphlet_k_min == cfg.graphlet_k_min
    assert loaded_summary_config.graphlet_k_max == cfg.graphlet_k_max
    assert loaded_summary_config.graphlet_connected_only == cfg.graphlet_connected_only
    assert checkpoint["config"]["experiment"] == "unit_test"
    assert checkpoint["report"]["status"] == "ok"
    for key in expected:
        assert torch.allclose(actual[key], expected[key], atol=1.0e-7)
