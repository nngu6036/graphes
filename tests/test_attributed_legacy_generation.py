"""Bounded legacy molecular replacements, using real typed construction."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import torch
import yaml


def _carbon_graph(graph):
    nx.set_node_attributes(graph, 6, "atomic_num")
    nx.set_edge_attributes(graph, 1, "bond_type")
    return graph


def _setup(tmp_path, monkeypatch, *, train=None, source="train_empirical", total_budget=3):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_attributed_grapher.py"
    spec = importlib.util.spec_from_file_location("_legacy_generation", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = {
        "dataset": {"name": "qm9_attributed"},
        "generation": {"invariant_source": source, "max_attempts_per_graph": 3,
                       "checkpoint_every": 1, "require_rdkit_source_validity": True,
                       "require_rdkit_final_validity": True},
        "constructor": {"randomize_assignment": False, "max_ordinary_degree": 4},
    }
    if total_budget is not None:
        config["generation"]["max_total_graph_attempts"] = total_budget
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "generation"
    train = [_carbon_graph(nx.cycle_graph(3))] if train is None else train
    model = torch.nn.Linear(1, 1)
    vocabulary = SimpleNamespace(node_attribute="atomic_num", edge_attribute="bond_type",
                                 node_values=(6,), edge_values=(1,), to_dict=lambda: {})
    basis = SimpleNamespace(topology_filter="simple_cycle", sizes=("3",))
    checkpoint = {"format": runner.ATTRIBUTED_SPECTRAL_GRAPHLET_CHECKPOINT_FORMAT}
    monkeypatch.setattr(runner, "load_dataset_splits", lambda *a, **k: {"train": train, "test": []})
    monkeypatch.setattr(runner, "load_attributed_spectral_graphlet_checkpoint",
                        lambda *a, **k: (model, vocabulary, basis, SimpleNamespace(), checkpoint))

    def refine(model, graph, **kwargs):
        result = graph.copy()
        result.graph["refiner_random_value"] = float(kwargs["rng"].random())
        result.graph["refined"] = True
        return result, []

    monkeypatch.setattr(runner, "refine_attributed_graph_with_spectral_graphlet_diffusion", refine)
    monkeypatch.setattr(sys, "argv", [str(path), "--config", str(config_path), "--checkpoint", "fixture.pt",
                                    "--output-dir", str(output), "--num-generate", "3", "--seed", "5",
                                    "--device", "cpu"])
    return SimpleNamespace(runner=runner, output=output)


def _json(path):
    def reject_nonfinite(value):
        raise AssertionError(f"Non-finite JSON value {value} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)


def _outputs(context):
    report = _json(context.output / "report.json")
    with (context.output / "molecular_graphs.pkl").open("rb") as handle:
        graphs = pickle.load(handle)
    with (context.output / "typed_source_graphs.pkl").open("rb") as handle:
        sources = pickle.load(handle)
    with (context.output / "generated_graphs.pkl").open("rb") as handle:
        alias = pickle.load(handle)
    assert len(graphs) == len(sources) == len(alias) == report["num_generated"]
    for path in context.output.glob("*.json"):
        _json(path)
    return report, graphs


def test_legacy_constructor_budget_skips_slot_and_preserves_later_seed(tmp_path, monkeypatch):
    baseline = _setup(tmp_path / "baseline", monkeypatch)
    baseline.runner.main()
    _, baseline_graphs = _outputs(baseline)

    context = _setup(tmp_path / "skip", monkeypatch)
    construct = context.runner.construct_typed_graph
    calls = []

    def reject_middle(*args, **kwargs):
        calls.append(len(calls) + 1)
        if 2 <= calls[-1] <= 4:
            raise context.runner.TypedConstructionError("fixture budget", {"failure_reason": "search_budget_exhausted"})
        graph, diagnostics = construct(*args, **kwargs)
        graph.graph["constructor_call"] = calls[-1]
        return graph, diagnostics

    monkeypatch.setattr(context.runner, "construct_typed_graph", reject_middle)
    context.runner.main()
    report, graphs = _outputs(context)
    assert calls == [1, 2, 3, 4, 5]
    assert report["complete"] is True and report["requested_count_reached"] is False
    assert report["num_requested"] == report["num_attempted"] == 3
    assert report["num_generated"] == 2 and report["num_skipped"] == 1
    assert report["max_total_graph_attempts"] == 3 and report["replacement_budget_exhausted"] is True
    assert report["replacement_attempts"] == 0
    assert report["generated_indices"] == [0, 2]
    assert [row["generation_index"] for row in report["source_metadata"]] == [0, 2]
    assert [graph.graph["constructor_call"] for graph in graphs] == [1, 5]
    assert [graph.graph["refiner_random_value"] for graph in graphs] == [
        baseline_graphs[index].graph["refiner_random_value"] for index in (0, 2)
    ]
    assert report["generation_attempts"] == report["diagnostics"]["generation_attempts"] == 5
    assert report["end_to_end_yield"] == 2 / 5
    assert report["generation_success_fraction"] == 2 / 3
    assert report["generation_rejections"] == {"constructor:search_budget_exhausted": 3}
    skipped = report["skipped_graphs"][0]
    assert skipped["generation_index"] == 1 and skipped["attempts_used"] == 3
    assert skipped["stage"] == "source_construction"
    partial = _json(context.output / "partial_report.json")
    assert partial["num_attempted"] == 3 and partial["num_skipped"] == 1
    assert partial["generation_attempts"] == 5 and partial["num_generated"] == 2


def test_legacy_all_infeasible_parents_save_empty_outputs_and_finite_reports(tmp_path, monkeypatch):
    context = _setup(tmp_path, monkeypatch, train=[_carbon_graph(nx.star_graph(5))])
    context.runner.main()
    report, graphs = _outputs(context)
    assert graphs == []
    assert report["num_requested"] == report["num_attempted"] == report["num_skipped"] == 3
    assert report["num_generated"] == 0 and report["generation_attempts"] == 9
    assert report["generation_success_fraction"] == report["end_to_end_yield"] == 0.0
    assert report["complete"] is True and report["requested_count_reached"] is False
    assert report["generation_rejections"] == {"constructor:precheck_failed": 9}
    assert [row["generation_index"] for row in report["skipped_graphs"]] == [0, 1, 2]
    assert (context.output / "generated.smi").read_text(encoding="utf-8") == ""
    partial = _json(context.output / "partial_report.json")
    assert partial["num_attempted"] == partial["num_skipped"] == 3
    assert partial["generation_attempts"] == 9 and partial["num_generated"] == 0


@pytest.mark.parametrize("reject_stage", ["source", "final"])
def test_configured_chemical_invalidity_uses_budget_then_skips(tmp_path, monkeypatch, reject_stage):
    context = _setup(tmp_path, monkeypatch)
    validate = context.runner._generation_rdkit_valid

    def invalid_at_stage(graph, **kwargs):
        refined = graph.graph.get("refined", False)
        if (reject_stage == "final") == refined:
            return False
        return validate(graph, **kwargs)

    monkeypatch.setattr(context.runner, "_generation_rdkit_valid", invalid_at_stage)
    context.runner.main()
    report, graphs = _outputs(context)
    assert graphs == [] and report["generation_attempts"] == 9
    reason = f"rdkit_invalid_{reject_stage}"
    assert report["generation_rejections"] == {reason: 9}
    assert all(row["reason"] == reason and row["attempts_used"] == 3 for row in report["skipped_graphs"])


def test_learned_typed_prior_budget_failure_skips_and_continues(tmp_path, monkeypatch):
    context = _setup(tmp_path, monkeypatch, source="learned")
    invariant = context.runner.extract_typed_invariant(_carbon_graph(nx.cycle_graph(3)), edge_types=(1,))
    calls = []

    def sample(rng):
        calls.append(len(calls) + 1)
        if 2 <= calls[-1] <= 4:
            raise RuntimeError("Typed invariant sampling exhausted its feasibility budget: fixture infeasible")
        return {"typed_invariant": invariant.to_dict()}

    monkeypatch.setattr(context.runner.TypedDegreeVAESampler, "from_config",
                        lambda *a, **k: SimpleNamespace(sample=sample))
    context.runner.main()
    report, graphs = _outputs(context)
    assert len(graphs) == 2 and calls == [1, 2, 3, 4, 5]
    assert report["generated_indices"] == [0, 2]
    assert report["generation_rejections"] == {"typed_prior_feasibility_budget": 3}
    assert report["skipped_graphs"][0]["stage"] == "invariant_sampling"


@pytest.mark.parametrize("error", [AssertionError("invariant drift"), ValueError("bad vocabulary"),
                                   RuntimeError("unexpected compute failure"), KeyError("missing attribute")])
def test_internal_errors_remain_fatal_and_save_accurate_partial_state(tmp_path, monkeypatch, error):
    context = _setup(tmp_path, monkeypatch)
    refine = context.runner.refine_attributed_graph_with_spectral_graphlet_diffusion
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise error
        return refine(*args, **kwargs)

    monkeypatch.setattr(context.runner, "refine_attributed_graph_with_spectral_graphlet_diffusion", fail_second)
    with pytest.raises(type(error)) as caught:
        context.runner.main()
    assert caught.value is error and len(calls) == 2
    assert not (context.output / "report.json").exists()
    partial = _json(context.output / "partial_report.json")
    assert partial["complete"] is False and partial["num_generated"] == 1
    assert partial["num_attempted"] == partial["generation_attempts"] == 2
    assert partial["num_skipped"] == 0 and partial["generated_indices"] == [0]
    assert partial["failure_stage"] == "refinement" and partial["failed_generation_index"] == 1


def test_incorrect_constructor_config_is_fatal_without_retry(tmp_path, monkeypatch):
    context = _setup(tmp_path, monkeypatch)
    path = tmp_path / "config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["constructor"]["candidate_ranking"] = "empirical"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="initialization_score"):
        context.runner.main()
    partial = _json(context.output / "partial_report.json")
    assert partial["num_attempted"] == partial["generation_attempts"] == 1
    assert partial["num_generated"] == partial["num_skipped"] == 0


@pytest.mark.parametrize("failed_trials", [1, 2])
def test_default_budget_resamples_replacements_until_requested_count(tmp_path, monkeypatch, failed_trials):
    context = _setup(tmp_path, monkeypatch, total_budget=None)
    construct = context.runner.construct_typed_graph
    sample = context.runner._sample_invariant
    construction_calls = []
    sampled_trials = []

    def track_sample(*args, **kwargs):
        sampled_trials.append(kwargs["index"])
        return sample(*args, **kwargs)

    def reject_then_recover(*args, **kwargs):
        construction_calls.append(len(construction_calls) + 1)
        if 2 <= construction_calls[-1] <= 1 + 3 * failed_trials:
            raise context.runner.TypedConstructionError("fixture budget", {"failure_reason": "search_budget_exhausted"})
        return construct(*args, **kwargs)

    monkeypatch.setattr(context.runner, "_sample_invariant", track_sample)
    monkeypatch.setattr(context.runner, "construct_typed_graph", reject_then_recover)
    context.runner.main()
    report, graphs = _outputs(context)
    expected_indices = [0, failed_trials + 1, failed_trials + 2]
    assert len(graphs) == report["num_generated"] == report["num_requested"] == 3
    assert report["num_attempted"] == 3 + failed_trials
    assert report["num_skipped"] == report["replacement_attempts"] == failed_trials
    assert report["max_total_graph_attempts"] == 30
    assert report["complete"] is True and report["requested_count_reached"] is True
    assert report["replacement_budget_exhausted"] is False
    assert report["generated_indices"] == expected_indices
    assert len(construction_calls) == report["generation_attempts"] == 3 + 3 * failed_trials
    assert sampled_trials == [0] + [trial for trial in range(1, failed_trials + 1) for _ in range(3)] + expected_indices[1:]
    assert report["generation_success_fraction"] == 3 / (3 + failed_trials)
    assert [graph.graph["refiner_random_value"] for graph in graphs] == [
        float(np.random.default_rng(np.random.SeedSequence([5, index, 1, 7919])).random())
        for index in expected_indices
    ]
    partial = _json(context.output / "partial_report.json")
    assert partial["num_generated"] == 3 and partial["num_attempted"] == 3 + failed_trials
    assert partial["replacement_attempts"] == failed_trials
    assert partial["max_total_graph_attempts"] == 30 and partial["replacement_budget_exhausted"] is False


def test_total_replacement_budget_exhaustion_saves_completed_outputs(tmp_path, monkeypatch, capsys):
    context = _setup(tmp_path, monkeypatch, total_budget=5)
    construct = context.runner.construct_typed_graph
    calls = []

    def only_first_succeeds(*args, **kwargs):
        calls.append(len(calls) + 1)
        if len(calls) > 1:
            raise context.runner.TypedConstructionError("fixture budget", {"failure_reason": "search_budget_exhausted"})
        return construct(*args, **kwargs)

    monkeypatch.setattr(context.runner, "construct_typed_graph", only_first_succeeds)
    context.runner.main()
    report, graphs = _outputs(context)
    assert len(graphs) == 1 and report["generated_indices"] == [0]
    assert report["num_attempted"] == report["max_total_graph_attempts"] == 5
    assert report["num_requested"] == 3 and report["replacement_attempts"] == 2
    assert report["complete"] is True and report["requested_count_reached"] is False
    assert report["replacement_budget_exhausted"] is True
    assert report["num_skipped"] == 4 and report["generation_attempts"] == len(calls) == 13
    assert [row["generation_index"] for row in report["skipped_graphs"]] == [1, 2, 3, 4]
    assert report["generation_rejections"] == {"constructor:search_budget_exhausted": 12}
    partial = _json(context.output / "partial_report.json")
    assert partial["replacement_budget_exhausted"] is True and partial["num_generated"] == 1
    assert partial["num_attempted"] == 5 and partial["generation_attempts"] == 13
    assert "total graph-attempt budget exhausted (5); saved 1/3" in capsys.readouterr().out


def test_default_replacement_budget_does_not_change_successful_rng_stream(tmp_path, monkeypatch):
    bounded = _setup(tmp_path / "bounded", monkeypatch)
    bounded.runner.main()
    _, original_graphs = _outputs(bounded)
    default = _setup(tmp_path / "default", monkeypatch, total_budget=None)
    default.runner.main()
    report, graphs = _outputs(default)
    assert report["num_attempted"] == report["num_generated"] == 3
    assert report["replacement_attempts"] == 0 and report["replacement_budget_exhausted"] is False
    assert [nx.to_dict_of_dicts(graph) for graph in graphs] == [nx.to_dict_of_dicts(graph) for graph in original_graphs]
    assert [graph.graph["refiner_random_value"] for graph in graphs] == [
        graph.graph["refiner_random_value"] for graph in original_graphs
    ]


@pytest.mark.parametrize("budget", [0, 2, True, 3.0, "6"])
def test_total_replacement_budget_requires_integer_at_least_requested(tmp_path, monkeypatch, budget):
    context = _setup(tmp_path, monkeypatch, total_budget=budget)
    with pytest.raises(ValueError, match="max_total_graph_attempts.*integer"):
        context.runner.main()
    assert not context.output.exists()
