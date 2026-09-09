from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

from grapher.data.io import save_dataset_splits
from grapher.rewiring_mlp.evaluation.studies import aggregate_pipeline_diagnostics
from grapher.rewiring_mlp.generic.rewiring import propose_valid_topology_swaps
from grapher.rewiring_mlp.generic.spectral import (
    laplacian_eigenvalues,
    spectrum_moments,
)
from grapher.rewiring_mlp.generic.spectral_data import (
    TopologySpectralExample,
    assert_same_degree_fibre,
    build_spectral_examples,
    collate_spectral_examples,
)
from grapher.rewiring_mlp.generic.spectral_model import (
    TOPOLOGY_SPECTRAL_CHECKPOINT_FORMAT,
    TopologySpectralTransformerPredictor,
    load_topology_spectral_checkpoint,
    save_topology_spectral_checkpoint,
)
from grapher.rewiring_mlp.generic.spectral_refiner import (
    SpectralPrediction,
    SpectralRefinerConfig,
    refine_graph_with_spectral_predictions,
)
from grapher.utils.io import load_pickle, save_yaml


def _small_model() -> TopologySpectralTransformerPredictor:
    return TopologySpectralTransformerPredictor(
        hidden_dim=16,
        edge_dim=8,
        graph_dim=16,
        num_layers=2,
        spectral_dim=16,
        spectral_layers=2,
        spectral_heads=4,
        spectral_ff_dim=32,
        dropout=0.0,
        min_gap=1.0e-6,
        input_normalization="mean_degree",
    )


def _nontrivial_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from(
        [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (2, 4),
            (3, 5),
            (4, 5),
        ]
    )
    return graph


def test_spectral_transformer_variable_size_joint_output_constraints() -> None:
    graphs = [nx.path_graph(5), nx.cycle_graph(7)]
    examples = [
        TopologySpectralExample(
            current_graph=graph,
            time=0.4,
            clean_spectrum_target=laplacian_eigenvalues(graph),
        )
        for graph in graphs
    ]
    batch = collate_spectral_examples(examples)
    model = _small_model().eval()

    with torch.no_grad():
        output = model(batch)["clean_spectrum"]

    assert output.shape == (2, 7)
    for row, graph in zip(output, graphs):
        n = graph.number_of_nodes()
        valid = row[:n]
        assert float(valid[0]) == pytest.approx(0.0, abs=1.0e-8)
        assert torch.all(valid[1:] >= valid[:-1] - 1.0e-7)
        assert float(valid.sum()) == pytest.approx(
            2.0 * graph.number_of_edges(), rel=1.0e-5, abs=1.0e-5
        )
        assert torch.all(row[n:] == 0.0)


def test_spectral_transformer_is_permutation_invariant() -> None:
    graph = _nontrivial_graph()
    permutation = {0: 4, 1: 0, 2: 5, 3: 2, 4: 1, 5: 3}
    relabelled = nx.relabel_nodes(graph, permutation, copy=True)
    relabelled = nx.convert_node_labels_to_integers(relabelled, ordering="sorted")
    target = laplacian_eigenvalues(graph)
    batch = collate_spectral_examples(
        [
            TopologySpectralExample(graph, 0.25, target),
            TopologySpectralExample(relabelled, 0.25, target),
        ]
    )
    model = _small_model().eval()
    with torch.no_grad():
        outputs = model(batch)["clean_spectrum"]
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1.0e-5, atol=1.0e-5)


def test_spectral_loss_backpropagates_through_transformer() -> None:
    graph = _nontrivial_graph()
    batch = collate_spectral_examples(
        [TopologySpectralExample(graph, 0.5, laplacian_eigenvalues(graph))]
    )
    model = _small_model()
    loss, metrics = model.loss(batch, loss_weights={"spectrum": 1.0, "moment2": 0.1})
    loss.backward()
    assert metrics["spectral_trace_mae"] < 1.0e-4
    assert model.gap_head[-1].weight.grad is not None
    transformer_grad = next(model.spectral_transformer.parameters()).grad
    assert transformer_grad is not None


def test_spectral_checkpoint_roundtrip(tmp_path) -> None:
    model = _small_model().eval()
    path = tmp_path / "spectral.pt"
    save_topology_spectral_checkpoint(
        model,
        path,
        report={"val_spectral_normalized_rmse": 0.123},
    )
    loaded, _summary, checkpoint = load_topology_spectral_checkpoint(path, device="cpu")
    assert checkpoint["format"] == TOPOLOGY_SPECTRAL_CHECKPOINT_FORMAT
    assert isinstance(loaded, TopologySpectralTransformerPredictor)
    assert checkpoint["report"]["val_spectral_normalized_rmse"] == pytest.approx(0.123)


def test_same_degree_fibre_guard() -> None:
    assert_same_degree_fibre(nx.path_graph(5), nx.path_graph(5))
    with pytest.raises(ValueError, match="same degree sequence"):
        assert_same_degree_fibre(nx.path_graph(5), nx.cycle_graph(5))


def test_spectral_teacher_states_are_real_degree_preserving_graphs() -> None:
    target = _nontrivial_graph()
    examples, report = build_spectral_examples(
        [target],
        trajectory_config={
            "steps": 3,
            "proposal_budget": -1,
            "valid_candidate_budget": -1,
            "states_per_graph": 3,
            "paths_per_graph": 1,
            "preserve_connectivity": True,
            "ensure_connected_source": True,
            "random_relabel_source": False,
            "source_randomization_steps": 0,
            "teacher_mode": "hard",
            "teacher_temperature": 0.2,
            "teacher_top_k": 0,
            "teacher_sample_actions": False,
        },
        spectral_config={"require_same_degree_sequence": True},
        seed=3,
    )
    target_degree = sorted(dict(target.degree()).values())
    assert examples
    for example in examples:
        assert sorted(dict(example.current_graph.degree()).values()) == target_degree
        assert nx.is_connected(example.current_graph)
        assert len(example.clean_spectrum_target) == target.number_of_nodes()
    assert report["mean_final_teacher_spectral_discrepancy"] <= report[
        "mean_initial_spectral_discrepancy"
    ] + 1.0e-12


def test_spectral_refiner_projects_to_valid_swap_and_prints_debug(capsys) -> None:
    source = _nontrivial_graph()
    candidates, candidate_graphs, _ = propose_valid_topology_swaps(
        source,
        proposal_budget=-1,
        valid_candidate_budget=-1,
        preserve_connectivity=True,
        rng=np.random.default_rng(4),
    )
    source_spectrum = laplacian_eigenvalues(source)
    target_action = next(
        action
        for action in candidates
        if not np.allclose(laplacian_eigenvalues(candidate_graphs[action]), source_spectrum)
    )
    target_spectrum = laplacian_eigenvalues(candidate_graphs[target_action])

    def fake_predictor(_model, graph, *, time, device):
        del time, device
        current = laplacian_eigenvalues(graph)
        first, second = spectrum_moments(target_spectrum)
        return SpectralPrediction(
            clean_spectrum=target_spectrum,
            current_spectrum=current,
            trace=first,
            second_moment=second,
        )

    config = SpectralRefinerConfig.from_dict(
        {
            "mode": "spectral",
            "steps": 1,
            "proposal_budget": -1,
            "valid_candidate_budget": -1,
            "preserve_connectivity": True,
            "reject_revisited_states": False,
            "selection": "greedy",
            "min_improvement": 0.0,
            "min_relative_improvement": 0.0,
            "spectral_guidance": {
                "distance": "rmse",
                "normalization": "mean_degree",
                "min_clean_mix": 1.0,
                "max_clean_mix": 1.0,
                "expand_on_plateau": False,
            },
            "debug": {
                "enabled": True,
                "print_every": 1,
                "top_candidates": 2,
                "spectrum_values": 8,
            },
        }
    )
    model = _small_model()
    refined, trace = refine_graph_with_spectral_predictions(
        source,
        model=model,
        refiner_config=config,
        device="cpu",
        rng=np.random.default_rng(5),
        return_trace=True,
        prediction_fn=fake_predictor,
        debug_context="unit-test",
    )
    captured = capsys.readouterr().out
    assert "[GraphER/Spectral]" in captured
    assert "lambda_current=" in captured
    assert "lambda_next_target=" in captured
    assert "clean_hat_moments" in captured
    assert "candidate_rank=" in captured
    assert "ACCEPT" in captured
    assert "lambda_after_accept=" in captured
    assert sorted(dict(refined.degree()).values()) == sorted(dict(source.degree()).values())
    assert nx.is_connected(refined)
    accepted = [row for row in trace if row.get("accepted")]
    assert len(accepted) == 1
    assert accepted[0]["spectral_gain"] > 0.0


def test_pipeline_diagnostics_accept_spectral_guidance() -> None:
    result = aggregate_pipeline_diagnostics(
        [
            {
                "pipeline_mode": "topology",
                "guidance_mode": "spectral",
                "spectral_error": 0.2,
                "invariant_feasible": 1.0,
                "constructor_success": 1.0,
                "accepted_swaps": 2,
                "runtime_seconds": 1.5,
                "fallback_used": 0.0,
                "candidate_proposals": 4,
                "candidate_passes": 3,
                "candidate_pass_rate": 0.75,
                "proposals_per_accepted_swap": 2.0,
                "stopped": 1.0,
                "stop_opportunities": 1,
                "stop_rate": 1.0,
                "generation_attempts": 1,
                "generation_successes": 1,
                "end_to_end_yield": 1.0,
                "rejection_reasons": {},
            }
        ],
        require_complete=True,
        allow_fallback=False,
    )
    assert result["pipeline_mode"] == "topology"
    assert result["metrics"]["spectral_error"]["mean"] == pytest.approx(0.2)


def test_spectral_only_mode_ignores_graph_topology_context() -> None:
    graph_a = nx.cycle_graph(6)
    graph_b = nx.Graph()
    graph_b.add_nodes_from(range(6))
    graph_b.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5)])
    assert graph_a.number_of_edges() == graph_b.number_of_edges()
    assert nx.is_connected(graph_b)

    source = laplacian_eigenvalues(graph_a).astype(np.float32)
    current = (source + np.linspace(0.0, 0.05, source.size)).astype(np.float32)
    current[0] = 0.0
    target = source.copy()
    batch = collate_spectral_examples(
        [
            TopologySpectralExample(
                current_graph=graph_a,
                time=0.4,
                current_spectrum=current,
                source_spectrum=source,
                clean_spectrum_target=target,
            ),
            TopologySpectralExample(
                current_graph=graph_b,
                time=0.4,
                current_spectrum=current,
                source_spectrum=source,
                clean_spectrum_target=target,
            ),
        ]
    )
    model = TopologySpectralTransformerPredictor(
        hidden_dim=8,
        edge_dim=8,
        graph_dim=8,
        num_layers=1,
        spectral_dim=16,
        spectral_layers=1,
        spectral_heads=4,
        spectral_ff_dim=32,
        dropout=0.0,
        use_graph_context=False,
    ).eval()
    with torch.no_grad():
        output = model(batch)["clean_spectrum"]
    torch.testing.assert_close(output[0], output[1], rtol=0.0, atol=1.0e-7)


def test_spectral_only_checkpoint_preserves_mode(tmp_path) -> None:
    model = TopologySpectralTransformerPredictor(
        hidden_dim=8,
        edge_dim=8,
        graph_dim=8,
        num_layers=1,
        spectral_dim=16,
        spectral_layers=1,
        spectral_heads=4,
        spectral_ff_dim=32,
        dropout=0.0,
        use_graph_context=False,
    ).eval()
    path = tmp_path / "spectral_only.pt"
    save_topology_spectral_checkpoint(model, path)
    loaded, _summary, checkpoint = load_topology_spectral_checkpoint(path, device="cpu")
    assert checkpoint["model_config"]["use_graph_context"] is False
    assert loaded.use_graph_context is False


def test_topology_generation_without_python310_imports(tmp_path) -> None:
    repository = Path(__file__).resolve().parents[1]
    root = tmp_path / "datasets"
    graph = _nontrivial_graph()
    save_dataset_splits(
        "tiny", {split: [graph] for split in ("train", "val", "test")}, {}, root,
    )
    config = tmp_path / "config.yaml"
    save_yaml({
        "dataset": {"name": "tiny", "root": str(root), "build_if_missing": False},
        "generation": {"degree_source": "empirical"},
        "topology_refiner": {"steps": 1},
        "evaluation": {"inline_during_generation": False},
    }, config)
    checkpoint = tmp_path / "checkpoint.pt"
    model = _small_model().eval()
    model.use_graph_context = False
    save_topology_spectral_checkpoint(
        model, checkpoint, report={"val_spectral_normalized_rmse": 0.1},
    )
    output = tmp_path / "generated"
    # A fresh process ensures the imports cannot pass via sys.modules caching.
    # Limit the missing-API simulation to GraphER: host dependencies may target
    # newer Python versions than those installed in the baseline environment.
    code = """
import builtins
import runpy
import sys

original_import = builtins.__import__
def compatible_import(name, globals=None, locals=None, fromlist=(), level=0):
    caller = (globals or {}).get('__name__', '')
    missing = {'itertools': {'pairwise'}, 'typing': {'TypeAlias'}}
    unavailable = missing.get(name, set()).intersection(fromlist or ())
    if caller.startswith('grapher.') and unavailable:
        raise ImportError(f'Python 3.9 does not provide {name}.{sorted(unavailable)}')
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = compatible_import
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, "scripts/run_topology_grapher.py",
         "--config", str(config), "--checkpoint", str(checkpoint),
         "--output-dir", str(output), "--num-generate", "2", "--seed", "42",
         "--device", "cpu"],
        cwd=repository, env={**os.environ, "PYTHONPATH": str(repository / "src")},
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["num_generated"] == 2
    assert report["degree_source"] == "empirical"
    assert report["diagnostics"]["final_target_degree_match_rate"] == 1.0
    assert len(load_pickle(output / "topology_refined_graphs.pkl")) == 2
