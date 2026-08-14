from __future__ import annotations

import json

import networkx as nx
import numpy as np
import torch

from grapher.models.dhvae_hh.typed_constructor import construct_typed_graph
from grapher.rewiring_mlp.molecular.typed_invariants import (
    TypedInvariant,
    extract_typed_invariant,
    typed_invariant_errors,
    typed_invariant_matches_graph,
)
from grapher.models.dhvae_hh.typed_degree_vae import (
    TypedSignatureVectorizer,
    build_typed_signature_vae,
    load_typed_signature_checkpoint,
    save_typed_signature_checkpoint,
)


def _typed_cycle(atom_types: tuple[int, ...] = (6, 6, 8, 8)) -> nx.Graph:
    graph = nx.cycle_graph(len(atom_types))
    nx.set_node_attributes(
        graph,
        {node: atom for node, atom in enumerate(atom_types)},
        "atomic_num",
    )
    nx.set_edge_attributes(graph, 1, "bond_type")
    return graph


def test_typed_vectorizer_json_round_trip_and_exact_construction() -> None:
    graph = _typed_cycle()
    vectorizer = TypedSignatureVectorizer.fit(
        [graph],
        edge_types=[1, 2, 3],
        max_ordinary_degree=4,
        max_weighted_valence={6: 4.0, 8: 2.0},
    )
    restored = TypedSignatureVectorizer.from_dict(
        json.loads(json.dumps(vectorizer.to_dict()))
    )
    invariant = extract_typed_invariant(graph, edge_types=[1, 2, 3])

    assert restored.max_weighted_valence == {6: 4.0, 8: 2.0}
    assert (
        typed_invariant_errors(
            invariant,
            max_ordinary_degree=restored.max_ordinary_degree,
            max_weighted_valence=restored.max_weighted_valence,
        )
        == []
    )
    realized, diagnostics = construct_typed_graph(
        invariant,
        {"randomize_assignment": False, "max_backtracks": 10_000},
        np.random.default_rng(7),
    )
    assert diagnostics["success"]
    assert typed_invariant_matches_graph(realized, invariant)


def test_typed_vae_checkpoint_and_feasible_decode(tmp_path) -> None:
    graphs = [_typed_cycle(), _typed_cycle((6, 7, 8, 6))]
    vectorizer = TypedSignatureVectorizer.fit(
        graphs,
        edge_types=[1, 2, 3],
        max_ordinary_degree=4,
        max_weighted_valence={6: 4.0, 7: 3.0, 8: 2.0},
    )
    model = build_typed_signature_vae(
        vectorizer,
        latent_dim=4,
        hidden_dim=8,
        size_condition_dim=4,
        prior_type="conditional_gmm",
        prior_components=2,
        num_layers=1,
    )
    checkpoint = tmp_path / "typed.pt"
    save_typed_signature_checkpoint(checkpoint, model, vectorizer)
    loaded, restored, metadata = load_typed_signature_checkpoint(
        checkpoint,
        device="cpu",
    )
    with torch.no_grad():
        outputs = loaded.sample_outputs(2, node_counts=[4, 4], device="cpu")
    summaries = restored.outputs_to_summaries(
        outputs,
        rng=np.random.default_rng(3),
        max_resample=100,
        fallback="empirical_nearest_n",
        include_diagnostics=True,
    )

    assert metadata["format"] == "typed_signature_histogram_vae_v1"
    assert len(summaries) == 2
    for summary in summaries:
        invariant = TypedInvariant.from_dict(summary["typed_invariant"])
        assert invariant.num_nodes == 4
        assert summary["sampling_diagnostics"]["attempts_used"] >= 1
        assert (
            typed_invariant_errors(
                invariant,
                max_ordinary_degree=restored.max_ordinary_degree,
                max_weighted_valence=restored.max_weighted_valence,
            )
            == []
        )
