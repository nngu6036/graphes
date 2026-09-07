from __future__ import annotations

import json
import sys

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.utils.io import load_pickle, save_yaml
from grapher.rewiring_mlp.core.rewiring import enumerate_valid_double_edge_swaps
from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.graphlet_diffusion import (
    extract_topology_graphlet_simplex,
    graphlet_simplex_to_clr,
)
from grapher.rewiring_mlp.generic.rewiring import propose_valid_topology_swaps
from grapher.rewiring_mlp.generic.spectral import (
    batched_laplacian_eigenvalues,
    laplacian_eigenvalues,
)
from grapher.rewiring_mlp.generic.spectral_data import (
    TopologySpectralDiffusionIterableDataset,
    TopologySpectralExample,
    collate_spectral_examples,
)
from grapher.rewiring_mlp.generic.spectral_graphlet_refiner import (
    DegreeConditionedSummaryPrediction,
    SpectralGraphletRefinerConfig,
    enrich_graph_with_degree_summary,
    predict_degree_conditioned_summary,
)
from grapher.rewiring_mlp.generic.spectral_model import (
    TopologySpectralGraphletTransformerPredictor,
    load_topology_spectral_graphlet_checkpoint,
    save_topology_spectral_graphlet_checkpoint,
)


def _graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (3, 5), (4, 5)]
    )
    return graph


def _basis() -> TopologyGraphletBasis:
    return TopologyGraphletBasis.from_config(
        {
            "graphlet_history": True,
            "graphlet_k_min": 3,
            "graphlet_k_max": 4,
            "graphlet_connected_only": True,
        }
    )


def _model(basis: TopologyGraphletBasis) -> TopologySpectralGraphletTransformerPredictor:
    return TopologySpectralGraphletTransformerPredictor(
        graphlet_block_widths=basis.simplex_block_widths,
        graphlet_dim=24,
        graphlet_dropout=0.0,
        hidden_dim=16,
        edge_dim=8,
        graph_dim=16,
        num_layers=1,
        spectral_dim=16,
        spectral_layers=1,
        spectral_heads=4,
        spectral_ff_dim=32,
        dropout=0.0,
        degree_summary_enabled=True,
        degree_summary_dim=16,
        degree_summary_layers=1,
        degree_summary_dropout=0.0,
    )


def test_batched_candidate_spectra_match_scalar() -> None:
    source = _graph()
    actions, candidates, _ = propose_valid_topology_swaps(
        source,
        proposal_budget=-1,
        valid_candidate_budget=-1,
        preserve_connectivity=True,
        rng=np.random.default_rng(0),
    )
    graphs = [candidates[action] for action in actions]
    batch = batched_laplacian_eigenvalues(graphs, device="cpu", backend="numpy")
    assert len(batch) == len(graphs)
    for graph, spectrum in zip(graphs, batch):
        np.testing.assert_allclose(spectrum, laplacian_eigenvalues(graph), atol=1.0e-10)


def test_fast_proposal_exhaustive_matches_reference_validity() -> None:
    graph = _graph()
    reference = set(enumerate_valid_double_edge_swaps(graph, preserve_connectivity=True))
    proposed, _graphs, _diag = propose_valid_topology_swaps(
        graph,
        proposal_budget=-1,
        valid_candidate_budget=-1,
        preserve_connectivity=True,
        rng=np.random.default_rng(0),
    )
    assert set(proposed) == reference


def test_degree_summary_is_permutation_invariant_and_trace_preserving() -> None:
    graph = _graph()
    basis = _basis()
    model = _model(basis).eval()
    prediction = predict_degree_conditioned_summary(
        model,
        graph,
        graphlet_basis=basis,
        device="cpu",
        graphlet_logit_epsilon=1.0e-5,
    )
    mapping = {0: 3, 1: 5, 2: 1, 3: 4, 4: 0, 5: 2}
    relabeled = nx.relabel_nodes(graph, mapping, copy=True)
    prediction_relabel = predict_degree_conditioned_summary(
        model,
        relabeled,
        graphlet_basis=basis,
        device="cpu",
        graphlet_logit_epsilon=1.0e-5,
    )
    np.testing.assert_allclose(
        prediction.clean_spectrum,
        prediction_relabel.clean_spectrum,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        prediction.clean_graphlet_logits,
        prediction_relabel.clean_graphlet_logits,
        atol=1.0e-6,
    )
    assert prediction.clean_spectrum[0] == pytest.approx(0.0, abs=1.0e-7)
    assert float(prediction.clean_spectrum.sum()) == pytest.approx(
        2.0 * graph.number_of_edges(), rel=1.0e-5, abs=1.0e-5
    )


def test_joint_degree_summary_loss_backpropagates() -> None:
    graph = _graph()
    basis = _basis()
    prob, mask, _ = extract_topology_graphlet_simplex(graph, graphlet_basis=basis)
    logits = graphlet_simplex_to_clr(
        prob, graphlet_basis=basis, coordinate_mask=mask
    )
    spectrum = laplacian_eigenvalues(graph).astype(np.float32)
    batch = collate_spectral_examples(
        [
            TopologySpectralExample(
                current_graph=graph,
                time=0.5,
                current_spectrum=spectrum,
                source_spectrum=spectrum,
                clean_spectrum_target=spectrum,
                current_graphlet_probabilities=prob.astype(np.float32),
                source_graphlet_probabilities=prob.astype(np.float32),
                clean_graphlet_probabilities_target=prob.astype(np.float32),
                current_graphlet_logits=logits.astype(np.float32),
                source_graphlet_logits=logits.astype(np.float32),
                clean_graphlet_logits_target=logits.astype(np.float32),
                graphlet_coordinate_mask=mask,
            )
        ]
    )
    model = _model(basis)
    loss, metrics = model.loss(
        batch,
        loss_weights={
            "spectrum": 0.75,
            "moment2": 0.25,
            "low_frequency": 0.75,
            "low_frequency_k": 4,
            "graphlet_logit": 2.0,
            "graphlet_probability": 1.0,
            "degree_summary": 0.35,
        },
    )
    loss.backward()
    assert metrics["degree_summary_loss"] >= 0.0
    assert model.degree_spectral_head[-1].weight.grad is not None
    assert model.degree_graphlet_heads[0][-1].weight.grad is not None


def test_degree_summary_checkpoint_roundtrip(tmp_path) -> None:
    basis = _basis()
    model = _model(basis)
    path = tmp_path / "enriched.pt"
    save_topology_spectral_graphlet_checkpoint(
        model,
        path,
        graphlet_basis=basis,
    )
    loaded, loaded_basis, _summary, _checkpoint = (
        load_topology_spectral_graphlet_checkpoint(path, device="cpu")
    )
    assert loaded.degree_summary_enabled is True
    assert loaded.degree_summary_dim == model.degree_summary_dim
    assert loaded_basis.simplex_block_widths == basis.simplex_block_widths


def test_source_enrichment_preserves_degree_and_improves_fixed_summary() -> None:
    source = _graph()
    basis = _basis()
    actions, candidate_graphs, _ = propose_valid_topology_swaps(
        source,
        proposal_budget=-1,
        valid_candidate_budget=-1,
        preserve_connectivity=True,
        rng=np.random.default_rng(0),
    )
    target_graph = candidate_graphs[actions[0]]
    prob, mask, _ = extract_topology_graphlet_simplex(target_graph, graphlet_basis=basis)
    logits = graphlet_simplex_to_clr(prob, graphlet_basis=basis, coordinate_mask=mask)
    spectrum = laplacian_eigenvalues(target_graph)
    target = DegreeConditionedSummaryPrediction(
        clean_spectrum=spectrum,
        clean_graphlet_logits=logits,
        clean_graphlet_probabilities=prob,
        graphlet_coordinate_mask=mask,
        trace=float(spectrum.sum()),
        second_moment=float(np.square(spectrum).sum()),
    )
    cfg = SpectralGraphletRefinerConfig.from_dict(
        {
            "mode": "spectral_graphlet",
            "steps": 1,
            "proposal_budget": -1,
            "valid_candidate_budget": -1,
            "preserve_connectivity": True,
            "selection": "greedy",
            "accept_only_improving": True,
            "min_improvement": 1.0e-12,
            "min_relative_improvement": 0.0,
            "spectral_guidance": {
                "distance": "rmse",
                "normalization": "mean_degree",
                "low_frequency_weight": 1.0,
                "low_frequency_cutoff": 0,
                "expand_on_plateau": False,
            },
            "graphlet_guidance": {
                "distance": "clr_rmse",
                "logit_epsilon": 1.0e-5,
                "size_weights": {"3": 1.0, "4": 1.0},
            },
            "global_to_local": {
                "schedule": "linear",
                "spectral_initial": 1.0,
                "spectral_final": 1.0,
                "graphlet_initial": 1.0,
                "graphlet_final": 1.0,
            },
        }
    )
    enriched, trace = enrich_graph_with_degree_summary(
        source,
        target=target,
        graphlet_basis=basis,
        refiner_config=cfg,
        device="cpu",
        rng=np.random.default_rng(0),
        return_trace=True,
    )
    assert trace and trace[0]["accepted"]
    assert trace[0]["energy_improvement"] > 0.0
    assert [enriched.degree(i) for i in sorted(enriched)] == [
        source.degree(i) for i in sorted(source)
    ]
    assert nx.is_connected(enriched)


def test_streaming_diffusion_caches_fixed_endpoints() -> None:
    graph = _graph()
    basis = _basis()
    dataset = TopologySpectralDiffusionIterableDataset(
        [graph],
        diffusion_config={
            "storage": "streaming",
            "cache_endpoints": True,
            "samples_per_graph": 2,
            "paths_per_graph": 1,
            "spectral_sigma": 0.2,
            "graphlet_sigma": 0.35,
        },
        source_config={
            "ensure_connected_source": True,
            "random_relabel_source": True,
            "source_randomization_steps": 0,
        },
        spectral_config={"require_same_degree_sequence": True},
        graphlet_basis=basis,
        seed=42,
    )
    assert dataset._endpoint_cache is not None
    first_source = dataset._endpoint_cache[0].source_spectrum.copy()
    list(iter(dataset))
    dataset.set_epoch(1)
    list(iter(dataset))
    np.testing.assert_allclose(dataset._endpoint_cache[0].source_spectrum, first_source)
    assert dataset.last_diagnostics[-1]["endpoint_cache"] is True


def test_enriched_generation_cli_has_reproducible_independent_rng_streams(tmp_path, monkeypatch) -> None:
    from scripts import run_topology_grapher as run

    basis = _basis()
    checkpoint = tmp_path / "checkpoint.pt"
    save_topology_spectral_graphlet_checkpoint(
        _model(basis).eval(), checkpoint, graphlet_basis=basis,
        report={"val_spectral_normalized_rmse": 0.1, "val_graphlet_logit_rmse": 0.1},
    )
    config_path = tmp_path / "config.yaml"
    save_yaml({
        "generation": {"degree_source": "test_oracle"},
        "topology_refiner": {"steps": 1},
        "source_enrichment": {"enabled": True, "rewiring": {"steps": 1}},
        "evaluation": {"inline_during_generation": False},
    }, config_path)
    monkeypatch.setattr(run, "load_dataset_splits", lambda *args, **kwargs: {
        "train": [_graph()], "test": [_graph()],
    })

    draws = {"enrichment": [], "refiner": []}

    def record_rng(name, function):
        def wrapped(*args, **kwargs):
            draws[name].append(kwargs["rng"].random(4))
            return function(*args, **kwargs)
        return wrapped

    monkeypatch.setattr(run, "enrich_graph_with_degree_summary", record_rng(
        "enrichment", run.enrich_graph_with_degree_summary,
    ))
    monkeypatch.setattr(run, "refine_graph_with_spectral_graphlet_predictions", record_rng(
        "refiner", run.refine_graph_with_spectral_graphlet_predictions,
    ))

    # Increasing the requested count must preserve the prefix of every stream.
    for count in (2, 3):
        output = tmp_path / f"generation_{count}"
        monkeypatch.setattr(sys, "argv", [
            "run_topology_grapher.py", "--config", str(config_path),
            "--checkpoint", str(checkpoint), "--output-dir", str(output),
            "--num-generate", str(count), "--seed", "42", "--device", "cpu",
        ])
        run.main()
        report = json.loads((output / "report.json").read_text())
        assert report["num_generated"] == count
        assert report["diagnostics"]["source_enrichment_enabled"] is True
        assert report["diagnostics"]["final_target_degree_match_rate"] == 1.0

    for values in draws.values():
        np.testing.assert_array_equal(values[:2], values[2:4])
        assert not np.array_equal(values[0], values[1])
    assert not np.array_equal(draws["enrichment"][0], draws["refiner"][0])
    # Refinement keeps the same seed stream used before enrichment was added.
    old_refiner_seed = np.random.SeedSequence(42).spawn(2)[1]
    for actual, seed in zip(draws["refiner"][:2], old_refiner_seed.spawn(2)):
        np.testing.assert_array_equal(actual, np.random.default_rng(seed).random(4))
    for filename in ("coarse_graphs.pkl", "enriched_base_graphs.pkl", "topology_refined_graphs.pkl"):
        first = load_pickle(tmp_path / "generation_2" / filename)
        second = load_pickle(tmp_path / "generation_3" / filename)
        assert all(nx.utils.graphs_equal(a, b) for a, b in zip(first, second))
