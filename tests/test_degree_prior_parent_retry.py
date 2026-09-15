from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from grapher.models.dhvae_hh.degree_perturbation import (
    DegreePerturbationError,
    PerturbedEmpiricalDegreeSampler,
    sequence_fingerprint,
)
from grapher.models.dhvae_hh.degree_sampler import EmpiricalDegreeSampler
from grapher.models.dhvae_hh.havel_hakimi import construct_coarse_graph


def _bank():
    # The triangle cannot relocate an edge. A four-node star can become P4.
    return [nx.complete_graph(3), nx.star_graph(3)]


def _sampler(*, graphs=None, retry=False, budget=8, config=None, seed=1, support=None):
    return PerturbedEmpiricalDegreeSampler.fit_from_graphs(
        _bank() if graphs is None else graphs,
        {
            "method": "edge_relocation", "probability": 1.0, "steps": 1,
            "failure_policy": "error", "require_novel": True,
            **(config or {}),
        },
        seed=seed, support_max_degree=support,
        parent_failure_policy="resample_parent" if retry else "error",
        max_parent_attempts=budget,
    )


def test_default_policy_raises_after_exactly_one_parent_draw() -> None:
    sampler = _sampler(config={"require_novel": False})
    rng = np.random.default_rng(1)
    expected = np.random.default_rng(1)
    parent = int(expected.integers(2))
    with pytest.raises(DegreePerturbationError, match="no_valid_edge_relocation"):
        sampler.sample(rng)
    assert len(sampler.records) == 1
    row = sampler.records[0]
    assert row["parent_train_index"] == parent == 0
    assert not row["returned"] and not row["fallback_used"]
    assert row["requested"] and not row["changed"]
    assert sampler.returned_records == []
    np.testing.assert_array_equal(rng.integers(100, size=8), expected.integers(100, size=8))


def test_retry_records_each_draw_and_reports_only_accepted_output_fingerprints() -> None:
    sampler = _sampler(retry=True)
    rng = np.random.default_rng(1)
    expected_rng = np.random.default_rng(1)
    summary = sampler.sample(rng)
    expected_parents = [int(expected_rng.integers(2)), int(expected_rng.integers(2))]

    assert expected_parents == [0, 1]
    assert [row["parent_train_index"] for row in sampler.records] == expected_parents
    assert [row["parent_attempt"] for row in sampler.records] == [1, 2]
    assert [row["output_index"] for row in sampler.records] == [0, 0]
    assert [row["returned"] for row in sampler.records] == [False, True]
    assert sampler.returned_records == [sampler.records[1]]
    row = summary["sampling_diagnostics"]
    assert row["changed"] and row["novel_vs_training"] and row["returned"]
    assert row["accepted_steps"] == 1
    assert not row["fallback_used"] and not row["repair_used"]
    assert summary["degree_sequence"] == [2, 2, 1, 1]
    np.testing.assert_array_equal(rng.integers(100, size=8), expected_rng.integers(100, size=8))

    report = sampler.report()
    assert report["num_samples"] == report["num_parent_draws"] == 2
    assert report["num_rejected_parent_draws"] == report["num_accepted_samples"] == 1
    assert report["accepted_record_indices"] == [1]
    assert report["failure_reasons"] == {"no_valid_edge_relocation": 1}
    assert report["changed_fraction"] == 0.5
    assert report["accepted_novel_degree_fraction"] == report["accepted_changed_fraction"] == 1.0
    assert report["accepted_parent_degree_fingerprint"] == sequence_fingerprint([row["parent_degree_sequence"]])
    assert report["accepted_degree_fingerprint"] == sequence_fingerprint([summary["degree_sequence"]])


@pytest.mark.parametrize("graph", [nx.complete_graph(3), nx.path_graph(3)])
def test_retry_budget_exhausts_for_immovable_parents_even_without_novelty(graph) -> None:
    sampler = _sampler(graphs=[graph], retry=True, budget=3, config={"require_novel": False})
    with pytest.raises(DegreePerturbationError):
        sampler.sample()
    assert [row["parent_attempt"] for row in sampler.records] == [1, 2, 3]
    assert {row["output_index"] for row in sampler.records} == {0}
    assert all(not row["returned"] and not row["fallback_used"] and not row["changed"] for row in sampler.records)
    report = sampler.report()
    assert report["num_parent_draws"] == report["num_rejected_parent_draws"] == 3
    assert report["num_accepted_samples"] == 0
    assert report["accepted_record_indices"] == []
    assert sampler.returned_records == []


@pytest.mark.parametrize("graphs, config, support, rejection", [
    ([nx.path_graph(4), nx.star_graph(3)], {"require_novel": True}, None, "already_in_training_degree_support"),
    ([nx.star_graph(3)], {"max_distance": 0.5}, None, "distance_limit"),
    ([nx.path_graph(4)], {"require_novel": False}, 2, "non_graphical_or_outside_connected_support"),
])
def test_retries_never_relax_novelty_distance_or_degree_support(graphs, config, support, rejection) -> None:
    sampler = _sampler(graphs=graphs, retry=True, budget=3, config=config, support=support)
    with pytest.raises(DegreePerturbationError):
        sampler.sample()
    assert len(sampler.records) == 3
    assert all(not row["returned"] and not row["changed"] for row in sampler.records)
    assert sampler.report()["proposal_rejections"][rejection] > 0


def test_failed_multiple_steps_roll_back_before_parent_rejection() -> None:
    sampler = _sampler(graphs=[nx.star_graph(3)], retry=True, budget=2,
                       config={"steps": 2, "require_novel": False})
    with pytest.raises(DegreePerturbationError):
        sampler.sample()
    assert len(sampler.records) == 2
    for row in sampler.records:
        assert row["rolled_back_steps"] == 1
        assert row["accepted_steps"] == 0
        assert row["degree_sequence"] == row["parent_degree_sequence"]
        assert not row["returned"] and not row["changed"]


def test_keep_original_remains_an_explicit_identity_transition() -> None:
    sampler = _sampler(graphs=[nx.complete_graph(3)], config={"failure_policy": "keep_original"})
    summary = sampler.sample()
    row = summary["sampling_diagnostics"]
    assert row["returned"] and row["fallback_used"] and not row["changed"]
    assert summary["degree_sequence"] == row["parent_degree_sequence"]
    assert len(sampler.records) == 1
    report = sampler.report()
    assert report["num_identity_fallbacks"] == report["num_accepted_samples"] == 1
    assert report["num_rejected_parent_draws"] == 0
    with pytest.raises(ValueError, match="failure_policy"):
        _sampler(retry=True, config={"failure_policy": "keep_original"})


def test_outputs_preserve_n_m_and_construct_fresh_connected_graphs_without_training_adjacency() -> None:
    bank = [nx.star_graph(3)]
    sampler = _sampler(graphs=bank, retry=True, support=3)
    bank[0].clear()
    for index in range(3):
        summary = sampler.sample()
        row = summary["sampling_diagnostics"]
        assert row["output_index"] == index
        assert summary["num_nodes"] == 4 and summary["num_edges"] == 3
        assert row["preserved_n_m"] and row["connected_feasible"]
        assert row["distance_half_l1"] == 1.0
        assert row["temporary_witness_discarded"]
        assert row["actual_grapher_source_reconstructed_from_degrees"]
        source = construct_coarse_graph(summary, {"ensure_connected": True}, np.random.default_rng(100 + index))
        assert nx.is_connected(source) and nx.number_of_selfloops(source) == 0
        assert source.number_of_nodes() == 4 and source.number_of_edges() == 3
        assert sorted((degree for _, degree in source.degree()), reverse=True) == summary["degree_sequence"]
        assert max(dict(source.degree()).values()) <= 3
    assert len(sampler.returned_records) == 3


def test_seeded_retry_stream_is_reproducible_and_independent_of_refiner_rng() -> None:
    first = _sampler(retry=True, budget=32)
    second = _sampler(retry=True, budget=32)
    global_before = np.random.get_state()
    first_outputs = [first.sample()["degree_sequence"] for _ in range(4)]
    global_after = np.random.get_state()
    assert global_before[0] == global_after[0]
    np.testing.assert_array_equal(global_before[1], global_after[1])
    assert global_before[2:] == global_after[2:]
    refiner_rng = np.random.default_rng(923)
    second_outputs = []
    for _ in range(4):
        refiner_rng.normal(size=500)
        second_outputs.append(second.sample()["degree_sequence"])
    assert first_outputs == second_outputs
    assert first.records == second.records
    assert first.report() == second.report()


@pytest.mark.parametrize("error", [RuntimeError("internal"), KeyError("field"), DegreePerturbationError("unrecorded")])
def test_retry_does_not_swallow_unrelated_or_unrecorded_errors(monkeypatch, error) -> None:
    sampler = _sampler(retry=True)
    calls = []

    def broken(parent_index, **_kwargs):
        calls.append(parent_index)
        raise error

    monkeypatch.setattr(sampler, "perturb_parent", broken)
    with pytest.raises(type(error), match=str(error).strip("'")):
        sampler.sample(np.random.default_rng(1))
    assert calls == [0]
    assert sampler.records == []


def test_unmarked_error_after_rejected_parent_propagates_without_sampling_failure_marker(monkeypatch) -> None:
    sampler = _sampler(retry=True)
    perturb_parent = sampler.perturb_parent
    error = DegreePerturbationError("Moment-block catalogue exceeds max_block_alternatives.")
    calls = []

    def fail_after_rejection(parent_index, **kwargs):
        calls.append(parent_index)
        if len(calls) == 2:
            raise error
        return perturb_parent(parent_index, **kwargs)

    monkeypatch.setattr(sampler, "perturb_parent", fail_after_rejection)
    rng = np.random.default_rng(1)
    expected = np.random.default_rng(1)
    expected.integers(2, size=2)
    with pytest.raises(DegreePerturbationError) as caught:
        sampler.sample(rng)
    assert caught.value is error
    assert not caught.value.sampling_failure
    assert calls == [0, 1]
    assert len(sampler.records) == 1 and not sampler.records[0]["returned"]
    assert sampler.records[0]["failure_reason"] == "no_valid_edge_relocation"
    assert sampler.returned_records == []
    np.testing.assert_array_equal(rng.integers(100, size=8), expected.integers(100, size=8))


@pytest.mark.parametrize("retry, expected_records", [(False, 1), (True, 3)])
def test_expected_terminal_sampling_errors_are_explicitly_marked(retry, expected_records) -> None:
    sampler = _sampler(graphs=[nx.complete_graph(3)], retry=retry, budget=3)
    with pytest.raises(DegreePerturbationError) as caught:
        sampler.sample()
    assert caught.value.sampling_failure is True
    assert len(sampler.records) == expected_records
    assert all(not row["returned"] for row in sampler.records)
    if retry:
        assert isinstance(caught.value.__cause__, DegreePerturbationError)
        assert caught.value.__cause__.sampling_failure is True


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_parent_retry_budget_requires_a_positive_integer(budget) -> None:
    with pytest.raises(ValueError):
        _sampler(retry=True, budget=budget)


@pytest.fixture(scope="module")
def topology_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_topology_grapher.py"
    spec = importlib.util.spec_from_file_location("_degree_retry_topology_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build(runner, source="edge_relocation", **kwargs):
    return runner._build_generation_degree_sampler(
        source, {}, train_graphs=_bank(), reference_graphs=[nx.path_graph(5)], seed=1,
        **kwargs,
    )


def test_topology_builder_forwards_retry_policy_and_keeps_alias_constraints(topology_runner) -> None:
    settings = {"require_novel": True, "max_distance": 2.0}
    original = deepcopy(settings)
    sampler = _build(topology_runner, perturbation_cfg=settings, support_max_degree=3,
                     parent_failure_policy="resample_parent", max_parent_attempts=7)
    assert settings == original
    assert sampler.parent_failure_policy == "resample_parent" and sampler.max_parent_attempts == 7
    assert sampler.config.method == "edge_relocation" and sampler.config.probability == 1.0
    assert sampler.config.steps == 1 and sampler.config.failure_policy == "error"
    assert sampler.config.require_novel and sampler.max_degree == 3
    assert sampler.sample(np.random.default_rng(1))["sampling_diagnostics"]["changed"]


@pytest.mark.parametrize("conflict", [{"probability": 0.5}, {"failure_policy": "keep_original"}])
def test_direct_alias_still_rejects_weakened_constraints_with_retry(topology_runner, conflict) -> None:
    with pytest.raises(ValueError, match="fixes"):
        _build(topology_runner, perturbation_cfg=conflict, parent_failure_policy="resample_parent")


@pytest.mark.parametrize("source", ["train_empirical", "test_empirical", "test_oracle", "learned"])
@pytest.mark.parametrize("options", [{"parent_failure_policy": "resample_parent"}, {"max_parent_attempts": 2}])
def test_topology_builder_rejects_retry_controls_for_nonperturbed_sources(topology_runner, source, options) -> None:
    with pytest.raises(ValueError, match="perturb|edge_relocation"):
        _build(topology_runner, source, **options)


def test_topology_builder_keeps_default_empirical_source_compatible(topology_runner) -> None:
    sampler = _build(topology_runner, "train_empirical")
    assert isinstance(sampler, EmpiricalDegreeSampler)
