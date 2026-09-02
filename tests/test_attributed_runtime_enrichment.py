from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import torch

from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary, GraphletBasis
from grapher.rewiring_mlp.attributed.spectral import (
    attributed_laplacian_spectra,
    batched_attributed_laplacian_spectra,
)
from grapher.rewiring_mlp.attributed.spectral_data import (
    AttributedSpectralDiffusionIterableDataset,
    AttributedTrainingPair,
    build_attributed_spectral_diffusion_examples,
    collate_attributed_spectral_examples,
)
from grapher.rewiring_mlp.attributed.spectral_model import (
    AttributedSpectralGraphletTransformerPredictor,
)


def _graph() -> nx.Graph:
    g = nx.Graph()
    atoms = [6, 6, 7, 8, 6, 6]
    for i, atom in enumerate(atoms):
        g.add_node(i, atomic_num=atom, atom_type=atom)
    for u, v, bond in [
        (0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 0, 1),
    ]:
        g.add_edge(u, v, bond_type=bond, bond_order=float(bond))
    return g


def _vocab_basis(graphs: list[nx.Graph]):
    vocab = GraphCategoryVocabulary.from_graphs(
        graphs,
        {
            "node_attribute": "atomic_num",
            "node_categories": [6, 7, 8, 9],
            "edge_attribute": "bond_type",
            "edge_categories": [1, 2, 3],
        },
    )
    basis = GraphletBasis.fit_from_graphs(
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
        vocabulary=vocab,
        attributed=True,
        seed=7,
    )
    return vocab, basis


def _model(vocab, basis):
    return AttributedSpectralGraphletTransformerPredictor(
        num_node_categories=vocab.num_node_categories,
        num_edge_categories=vocab.num_edge_categories,
        graphlet_block_widths=basis.simplex_block_widths,
        hidden_dim=32,
        edge_dim=16,
        graph_dim=32,
        num_layers=1,
        spectral_dim=32,
        spectral_layers=1,
        spectral_heads=4,
        spectral_ff_dim=64,
        graphlet_dim=48,
        dropout=0.0,
        graphlet_dropout=0.0,
        invariant_summary_enabled=True,
        invariant_summary_dim=32,
        invariant_summary_layers=1,
        invariant_summary_dropout=0.0,
    ).eval()


def test_batched_dual_spectra_match_scalar() -> None:
    g1 = _graph()
    g2 = _graph()
    g2.edges[0, 1]["bond_type"] = 2
    g2.edges[0, 1]["bond_order"] = 2.0
    batched = batched_attributed_laplacian_spectra(
        [g1, g2], backend="numpy", device="cpu"
    )
    for graph, values in zip([g1, g2], batched):
        assert np.allclose(values, attributed_laplacian_spectra(graph), atol=1.0e-10)


def test_hard_invariant_summary_is_permutation_invariant() -> None:
    source = _graph()
    target = _graph()
    vocab, basis = _vocab_basis([source, target])
    examples, _ = build_attributed_spectral_diffusion_examples(
        [AttributedTrainingPair(source, target)],
        vocabulary=vocab,
        graphlet_basis=basis,
        diffusion_config={"samples_per_graph": 1, "paths_per_graph": 1},
        spectral_config={"normalization": "mean_degree"},
        seed=3,
    )
    model = _model(vocab, basis)
    batch1 = collate_attributed_spectral_examples(examples, vocab)
    with torch.no_grad():
        out1 = model.invariant_summary(batch1)

    mapping = {i: (i * 5 + 1) % source.number_of_nodes() for i in source.nodes()}
    relabeled = nx.relabel_nodes(source, mapping, copy=True)
    relabeled_target = nx.relabel_nodes(target, mapping, copy=True)
    examples2, _ = build_attributed_spectral_diffusion_examples(
        [AttributedTrainingPair(relabeled, relabeled_target)],
        vocabulary=vocab,
        graphlet_basis=basis,
        diffusion_config={"samples_per_graph": 1, "paths_per_graph": 1},
        spectral_config={"normalization": "mean_degree"},
        seed=3,
    )
    batch2 = collate_attributed_spectral_examples(examples2, vocab)
    with torch.no_grad():
        out2 = model.invariant_summary(batch2)
    assert torch.allclose(
        out1["invariant_clean_spectra"], out2["invariant_clean_spectra"], atol=1.0e-6
    )
    assert torch.allclose(
        out1["invariant_clean_graphlet_logits"],
        out2["invariant_clean_graphlet_logits"],
        atol=1.0e-6,
    )


def test_streaming_endpoint_sqlite_cache_hits_second_epoch(tmp_path: Path) -> None:
    source = _graph()
    target = _graph()
    vocab, basis = _vocab_basis([source, target])
    ds = AttributedSpectralDiffusionIterableDataset(
        [AttributedTrainingPair(source, target)],
        vocabulary=vocab,
        graphlet_basis=basis,
        diffusion_config={
            "samples_per_graph": 2,
            "paths_per_graph": 1,
            "cache_endpoints": True,
            "endpoint_cache_path": str(tmp_path / "endpoint.sqlite"),
        },
        source_config={"shared_relabel_augmentation": True},
        spectral_config={"normalization": "mean_degree"},
        seed=11,
        shuffle_graphs=False,
        cache_namespace="train",
    )
    assert len(list(ds)) == 2
    assert ds.last_diagnostics[0]["endpoint_cache_hit"] is False
    ds.set_epoch(1)
    assert len(list(ds)) == 2
    assert ds.last_diagnostics[0]["endpoint_cache_hit"] is True


def test_invariant_summary_auxiliary_loss_backpropagates() -> None:
    source = _graph()
    target = _graph()
    vocab, basis = _vocab_basis([source, target])
    examples, _ = build_attributed_spectral_diffusion_examples(
        [AttributedTrainingPair(source, target)],
        vocabulary=vocab,
        graphlet_basis=basis,
        diffusion_config={"samples_per_graph": 1, "paths_per_graph": 1},
        spectral_config={"normalization": "mean_degree"},
        seed=5,
    )
    batch = collate_attributed_spectral_examples(examples, vocab)
    model = _model(vocab, basis)
    loss, metrics = model.loss(
        batch,
        loss_weights={
            "spectrum": 1.0,
            "topology_spectrum": 1.0,
            "bond_spectrum": 1.25,
            "moment2": 0.2,
            "low_frequency": 0.75,
            "low_frequency_k": 4,
            "graphlet_logit": 2.0,
            "graphlet_probability": 0.75,
            "invariant_summary": 0.35,
        },
    )
    loss.backward()
    assert "invariant_summary_loss" in metrics
    assert any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if name.startswith("invariant_")
    )
