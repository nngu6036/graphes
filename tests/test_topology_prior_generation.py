"""Exercise topology CLI routing and report persistence with a stub predictor."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import networkx as nx
import pytest
import torch
import yaml

from grapher.models.dhvae_hh.degree_perturbation import DegreePerturbationError


@pytest.mark.parametrize("outcome", ["complete", "parent_budget", "constructor_budget"])
def test_topology_cli_retries_and_keeps_completed_graphs_auditable(tmp_path, monkeypatch, outcome):
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_topology_grapher.py"
    spec = importlib.util.spec_from_file_location("_topology_prior_generation", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    train = [nx.complete_graph(3), nx.star_graph(3)]
    if outcome == "constructor_budget":
        train = train[1:]
    config = {
        "dataset": {"name": "toy", "build_if_missing": False},
        "generation": {"degree_source": "edge_relocation", "degree_rng_mode": "independent",
                       "max_attempts_per_graph": 3},
        "evaluation": {"inline_during_generation": False},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"fixture predictor checkpoint")
    output = tmp_path / "generation"
    model = torch.nn.Linear(1, 1)
    checkpoint = {"format": runner.TOPOLOGY_CHECKPOINT_FORMAT, "report": {"val_graphlet_mae": 0.1}}
    monkeypatch.setattr(runner, "_checkpoint_format", lambda _: runner.TOPOLOGY_CHECKPOINT_FORMAT)
    monkeypatch.setattr(runner, "load_topology_checkpoint", lambda *a, **k: (model, object(), None, checkpoint))
    monkeypatch.setattr(runner, "load_dataset_splits", lambda *a, **k: {"train": train, "test": [nx.path_graph(4)]})
    monkeypatch.setattr(runner, "refine_graph_with_topology_predictions", lambda g, **k: (g.copy(), []))
    original_builder = runner._build_generation_degree_sampler
    captured = []

    def build(*args, **kwargs):
        sampler = original_builder(*args, **kwargs)
        captured.append(sampler)
        if outcome == "parent_budget":
            # Complete one output, then exhaust three immutable parent draws.
            indices = iter([1, 0, 0, 0])
            original_sample = sampler.sample
            forced_rng = SimpleNamespace(integers=lambda _: next(indices))
            sampler.sample = lambda ignored_rng: original_sample(forced_rng)
        return sampler

    monkeypatch.setattr(runner, "_build_generation_degree_sampler", build)
    if outcome == "constructor_budget":
        def reject(*args, **kwargs):
            raise RuntimeError("fixture final HH constructor failure")
        monkeypatch.setattr(runner, "construct_coarse_graph", reject)

    monkeypatch.setattr(sys, "argv", [str(path), "--config", str(config_path), "--checkpoint", str(checkpoint_path),
        "--output-dir", str(output), "--num-generate", "3", "--seed", "5", "--device", "cpu",
        "--set", "generation.degree_failure_policy=resample_parent",
        "--set", "generation.max_degree_parent_attempts=" + ("32" if outcome == "complete" else "3"),
        "--set", "generation.degree_perturbation.require_novel=false"])
    if outcome == "complete":
        runner.main()
    else:
        expected = DegreePerturbationError if outcome == "parent_budget" else RuntimeError
        with pytest.raises(expected):
            runner.main()

    prior = json.loads((output / "degree_prior_report.json").read_text())
    partial = outcome != "complete"
    report = json.loads((output / ("partial_report.json" if partial else "report.json")).read_text())
    prefix = "partial_" if partial else ""
    with (output / (prefix + "topology_refined_graphs.pkl")).open("rb") as handle:
        graphs = pickle.load(handle)
    sequences = json.loads((output / (prefix + "sampled_degree_sequences.json")).read_text())
    expected_count = {"complete": 3, "parent_budget": 1, "constructor_budget": 0}[outcome]
    assert len(graphs) == len(sequences) == report["num_generated"] == prior["num_returned"] == expected_count
    assert len(prior["returned_records"]) == expected_count
    assert prior["parent_failure_policy"] == "resample_parent"
    assert prior["num_parent_draws"] == len(captured[0].records)
    assert prior["num_accepted_samples"] == len(captured[0].returned_records)
    assert report["parent_degree_fingerprint"] == prior["returned_parent_degree_fingerprint"]
    assert report["sampled_degree_fingerprint"] == prior["returned_degree_fingerprint"]
    assert all(nx.is_connected(g) for g in graphs)
    assert all(sorted(dict(g.degree()).values(), reverse=True) == seq for g, seq in zip(graphs, sequences))
    if outcome == "complete":
        assert prior["num_rejected_parent_draws"] > 0
        assert prior["num_accepted_samples"] == 3
    else:
        assert report["complete"] is False and prior["generation_aborted"] is True
        assert not (output / "topology_refined_graphs.pkl").exists()
        assert not (output / "report.json").exists()
        if outcome == "parent_budget":
            assert prior["num_parent_draws"] == 4
            assert prior["num_rejected_parent_draws"] == 3
        else:
            assert prior["num_accepted_samples"] == 3
            assert report["generation_rejections"] == {"constructor_rejected": 3}
