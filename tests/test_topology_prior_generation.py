"""Exercise topology CLI routing and report persistence with a stub predictor."""
from __future__ import annotations

from collections import Counter
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


def _setup_cli(tmp_path, monkeypatch, *, train=None, retry=True, budget=3,
               forced_parents=None, inline_evaluation=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_topology_grapher.py"
    spec = importlib.util.spec_from_file_location("_topology_prior_generation", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    train = [nx.complete_graph(3), nx.star_graph(3)] if train is None else train
    config = {
        "dataset": {"name": "toy", "build_if_missing": False},
        "generation": {"degree_source": "edge_relocation", "degree_rng_mode": "independent",
                       "max_attempts_per_graph": 3},
        "evaluation": {"inline_during_generation": inline_evaluation},
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
    def refine(graph, **kwargs):
        result = graph.copy()
        result.graph["refiner_random_value"] = float(kwargs["rng"].random())
        return result, []

    monkeypatch.setattr(runner, "refine_graph_with_topology_predictions", refine)
    original_builder = runner._build_generation_degree_sampler
    captured = []

    def build(*args, **kwargs):
        sampler = original_builder(*args, **kwargs)
        captured.append(sampler)
        if forced_parents is not None:
            indices = iter(forced_parents)
            original_sample = sampler.sample
            forced_rng = SimpleNamespace(integers=lambda _: next(indices))
            sampler.sample = lambda ignored_rng: original_sample(forced_rng)
        return sampler

    monkeypatch.setattr(runner, "_build_generation_degree_sampler", build)
    argv = [str(path), "--config", str(config_path), "--checkpoint", str(checkpoint_path),
        "--output-dir", str(output), "--num-generate", "3", "--seed", "5", "--device", "cpu",
        "--set", "generation.degree_perturbation.require_novel=false"]
    if retry:
        argv.extend(["--set", "generation.degree_failure_policy=resample_parent",
                     "--set", f"generation.max_degree_parent_attempts={budget}"])
    monkeypatch.setattr(sys, "argv", argv)
    return SimpleNamespace(runner=runner, output=output, captured=captured)


def _read_json(path):
    def invalid_constant(value):
        raise AssertionError(f"Non-finite JSON value in {path}: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid_constant)


def _read_completed_output(context):
    prior = _read_json(context.output / "degree_prior_report.json")
    report = _read_json(context.output / "report.json")
    with (context.output / "topology_refined_graphs.pkl").open("rb") as handle:
        graphs = pickle.load(handle)
    sequences = _read_json(context.output / "sampled_degree_sequences.json")
    assert not list(context.output.glob("partial_*"))
    return prior, report, graphs, sequences


@pytest.mark.parametrize("outcome", ["complete", "parent_budget", "constructor_budget"])
def test_topology_cli_retries_and_keeps_completed_graphs_auditable(tmp_path, monkeypatch, outcome):
    context = _setup_cli(
        tmp_path, monkeypatch,
        train=[nx.star_graph(3)] if outcome == "constructor_budget" else None,
        budget=32 if outcome == "complete" else 3,
        # Slot0 succeeds; slot1 exhausts its budget; slot2 must still run.
        forced_parents=[1, 0, 0, 0, 1] if outcome == "parent_budget" else None,
        inline_evaluation=outcome == "constructor_budget",
    )
    if outcome == "constructor_budget":
        def reject(*args, **kwargs):
            raise RuntimeError("fixture final HH constructor failure")

        def forbidden_evaluation(*args, **kwargs):
            raise AssertionError("No graph metrics or ORCA calls are valid for an empty generated batch.")

        monkeypatch.setattr(context.runner, "construct_coarse_graph", reject)
        monkeypatch.setattr(context.runner, "evaluate_graph_sets", forbidden_evaluation)
    context.runner.main()

    prior, report, graphs, sequences = _read_completed_output(context)
    expected_count = {"complete": 3, "parent_budget": 2, "constructor_budget": 0}[outcome]
    assert len(graphs) == len(sequences) == report["num_generated"] == prior["num_returned"] == expected_count
    assert len(prior["returned_records"]) == expected_count
    for audit in (prior, report):
        assert audit["num_requested"] == audit["num_attempted"] == 3
        assert audit["num_generated"] == expected_count
        assert audit["num_skipped"] == len(audit["skipped_graphs"]) == 3 - expected_count
    assert report["complete"] is True
    assert report["requested_count_reached"] is (expected_count == 3)
    assert not prior.get("generation_aborted", False)
    assert prior["parent_failure_policy"] == "resample_parent"
    assert prior["num_parent_draws"] == len(context.captured[0].records)
    assert prior["num_accepted_samples"] == len(context.captured[0].returned_records)
    assert report["parent_degree_fingerprint"] == prior["returned_parent_degree_fingerprint"]
    assert report["sampled_degree_fingerprint"] == prior["returned_degree_fingerprint"]
    assert all(nx.is_connected(g) for g in graphs)
    assert all(sorted(dict(g.degree()).values(), reverse=True) == seq for g, seq in zip(graphs, sequences))
    if outcome == "complete":
        assert prior["num_rejected_parent_draws"] > 0
        assert prior["num_accepted_samples"] == 3
        assert prior["skipped_graphs"] == report["skipped_graphs"] == []
    elif outcome == "parent_budget":
        assert [row["generation_index"] for row in prior["returned_records"]] == [0, 2]
        assert prior["num_parent_draws"] == 5
        assert prior["num_rejected_parent_draws"] == 3
        skipped = report["skipped_graphs"][0]
        assert skipped["generation_index"] == 1 and skipped["stage"] == "degree_prior"
        assert skipped["attempts_used"] == 3
        assert "exhaust" in skipped["reason"].lower()
    else:
        assert prior["num_accepted_samples"] == 9
        total_rejections = Counter()
        for skipped in report["skipped_graphs"]:
            total_rejections.update(skipped["generation_rejections"])
        assert total_rejections == {"constructor_rejected": 9}
        assert report["generation_rejections"] == prior["generation_rejections"] == dict(total_rejections)
        assert [row["generation_index"] for row in report["skipped_graphs"]] == [0, 1, 2]
        for skipped in report["skipped_graphs"]:
            assert skipped["stage"] == "source_construction"
            assert skipped["attempts_used"] == 3
            assert skipped["generation_rejections"] == {"constructor_rejected": 3}
        # Check every JSON artifact, including diagnostics, for NaN/Infinity.
        for path in context.output.glob("*.json"):
            _read_json(path)


def test_strict_sampler_failure_skips_only_current_generation_slot(tmp_path, monkeypatch):
    context = _setup_cli(tmp_path, monkeypatch, retry=False, forced_parents=[0, 1, 1])
    context.runner.main()
    prior, report, graphs, _sequences = _read_completed_output(context)
    assert context.captured[0].parent_failure_policy == "error"
    assert prior["num_parent_draws"] == 3 and prior["num_rejected_parent_draws"] == 1
    assert len(graphs) == report["num_generated"] == 2
    assert [row["generation_index"] for row in prior["returned_records"]] == [1, 2]
    assert report["num_attempted"] == 3 and report["num_skipped"] == 1
    skipped = report["skipped_graphs"][0]
    assert skipped["generation_index"] == 0 and skipped["stage"] == "degree_prior"
    assert skipped["attempts_used"] == 1


def test_constructor_budget_skip_recovers_without_reusing_previous_graph_or_refiner_seed(tmp_path, monkeypatch):
    baseline = _setup_cli(tmp_path / "baseline", monkeypatch, train=[nx.star_graph(3)])
    baseline.runner.main()
    _, _, baseline_graphs, _ = _read_completed_output(baseline)

    context = _setup_cli(tmp_path / "skipped", monkeypatch, train=[nx.star_graph(3)])
    original_construct = context.runner.construct_coarse_graph
    calls = []

    def construct(*args, **kwargs):
        call = len(calls) + 1
        calls.append(call)
        if 2 <= call <= 4:
            raise RuntimeError("fixture middle-slot constructor failures")
        graph = original_construct(*args, **kwargs)
        graph.graph["constructed_on_call"] = call
        return graph

    monkeypatch.setattr(context.runner, "construct_coarse_graph", construct)
    context.runner.main()
    prior, report, graphs, _sequences = _read_completed_output(context)
    assert calls == [1, 2, 3, 4, 5]
    assert [graph.graph["constructed_on_call"] for graph in graphs] == [1, 5]
    assert [row["generation_index"] for row in prior["returned_records"]] == [0, 2]
    assert [graph.graph["refiner_random_value"] for graph in graphs] == [
        baseline_graphs[index].graph["refiner_random_value"] for index in (0, 2)
    ]
    assert prior["num_accepted_samples"] == 5 and prior["num_returned"] == 2
    assert report["num_skipped"] == 1 and report["skipped_graphs"][0]["generation_index"] == 1


def test_unrecorded_degree_error_remains_fatal_after_a_completed_output(tmp_path, monkeypatch):
    context = _setup_cli(tmp_path, monkeypatch, train=[nx.star_graph(3)])
    original_build = context.runner._build_generation_degree_sampler

    def build(*args, **kwargs):
        sampler = original_build(*args, **kwargs)
        original_sample = sampler.sample
        calls = []

        def sample(rng):
            calls.append(1)
            if len(calls) == 2:
                raise DegreePerturbationError("fixture unrecorded internal degree error")
            return original_sample(rng)

        sampler.sample = sample
        return sampler

    monkeypatch.setattr(context.runner, "_build_generation_degree_sampler", build)
    with pytest.raises(DegreePerturbationError, match="unrecorded internal"):
        context.runner.main()
    assert len(context.captured[0].records) == 1
    assert not (context.output / "report.json").exists()
    partial = _read_json(context.output / "partial_report.json")
    assert partial["complete"] is False and partial["num_generated"] == 1


def test_unexpected_refiner_error_is_not_treated_as_a_skipped_source(tmp_path, monkeypatch):
    context = _setup_cli(tmp_path, monkeypatch, train=[nx.star_graph(3)])
    calls = []

    def broken_refiner(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("fixture unexpected refiner failure")

    monkeypatch.setattr(context.runner, "refine_graph_with_topology_predictions", broken_refiner)
    with pytest.raises(RuntimeError, match="unexpected refiner failure"):
        context.runner.main()
    assert len(calls) == len(context.captured[0].records) == 1
    assert not (context.output / "report.json").exists()
