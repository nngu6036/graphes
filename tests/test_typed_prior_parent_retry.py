from __future__ import annotations

from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from grapher.models.dhvae_hh.degree_perturbation import (
    DegreePerturbationConfig,
    DegreePerturbationError,
)
from grapher.models.dhvae_hh.typed_constructor import construct_typed_graph
from grapher.models.dhvae_hh.typed_degree_perturbation import (
    PerturbedEmpiricalTypedDegreeSampler,
    typed_fingerprint,
    typed_key,
    typed_totals,
)
from grapher.rewiring_mlp.attributed.typed_prior import build_typed_empirical_sampler
from grapher.rewiring_mlp.molecular.typed_invariants import (
    TypedDegreeSignature,
    TypedInvariant,
    extract_typed_invariant,
    typed_invariant_matches_graph,
)


def _graph(graph: nx.Graph) -> nx.Graph:
    nx.set_node_attributes(graph, 6, "atomic_num")
    nx.set_edge_attributes(graph, 1, "bond_type")
    return graph


def _bank() -> list[nx.Graph]:
    # A complete triangle has no absent edge to receive a relocated endpoint.
    # Relocating a star edge onto another leaf produces a connected path of 4.
    return [_graph(nx.complete_graph(3)), _graph(nx.star_graph(3))]


def _constructor() -> dict:
    return {
        "ensure_connected": True,
        "max_restarts": 4,
        "max_backtracks": 1000,
        "max_ordinary_degree": 4,
        "max_weighted_valence": {6: 4.0},
        "candidate_ranking": "uniform",
    }


def _supported_signatures() -> tuple[TypedDegreeSignature, ...]:
    return tuple(TypedDegreeSignature(6, (degree,)) for degree in (1, 2, 3))


def _valid_carbon_graph(graph: nx.Graph) -> bool:
    return (
        nx.is_connected(graph)
        and not graph.is_directed()
        and not graph.is_multigraph()
        and nx.number_of_selfloops(graph) == 0
        and all(data["atomic_num"] == 6 for _, data in graph.nodes(data=True))
        and all(degree <= 4 for _, degree in graph.degree())
        and all(data["bond_type"] == 1 for _, _, data in graph.edges(data=True))
    )


def _sampler(*, graphs=None, retry=False, max_parent_attempts=8, seed=0, **kwargs):
    options = {"parent_failure_policy": "resample_parent"} if retry else {}
    return PerturbedEmpiricalTypedDegreeSampler.fit(
        _bank() if graphs is None else graphs,
        DegreePerturbationConfig(
            method="edge_relocation", probability=1.0, steps=1,
            failure_policy="error", require_novel=True,
        ),
        edge_types=(1,), seed=seed, constructor_config=_constructor(),
        allowed_signatures=_supported_signatures(),
        graph_validator=_valid_carbon_graph,
        max_parent_attempts=max_parent_attempts,
        **options, **kwargs,
    )


def _builder_config() -> dict:
    return {
        "joint_typed_degree": {"enabled": True},
        "constructor": _constructor(),
        "typed_signature": {"max_ordinary_degree": 4, "max_weighted_valence": {6: 4.0}},
        "generation": {
            "invariant_source": "edge_relocation",
            "invariant_failure_policy": "resample_parent",
            "max_invariant_parent_attempts": 8,
            "degree_perturbation": {"require_novel": True},
        },
    }


def test_default_policy_still_raises_for_first_immovable_parent() -> None:
    sampler = _sampler()
    with pytest.raises(DegreePerturbationError, match="edge_relocation"):
        sampler.sample()
    assert len(sampler.records) == 1
    record = sampler.records[0]
    assert record["parent_train_index"] == 0
    assert record["failure_reason"] == "no_valid_edge_relocation"
    assert record["requested"] and not record["changed"]
    assert not record["fallback_used"] and not record["returned"]
    assert sampler.returned_records == []
    assert sampler.report()["num_returned_samples"] == 0


def test_opt_in_retry_returns_changed_parent_and_records_rejection() -> None:
    sampler = _sampler(retry=True)
    summary = sampler.sample()
    assert [row["parent_train_index"] for row in sampler.records] == [0, 1]
    assert [row["parent_attempt"] for row in sampler.records] == [1, 2]
    assert [row["output_index"] for row in sampler.records] == [0, 0]
    assert [row["returned"] for row in sampler.records] == [False, True]
    assert sampler.returned_records == [sampler.records[1]]
    returned = summary["sampling_diagnostics"]
    assert returned["returned"] and returned["changed"] and returned["novel_vs_training"]
    assert returned["failure_reason"] is None and returned["output_failure"] is None
    assert not returned["fallback_used"] and not returned["repair_used"]
    assert returned["accepted_steps"] == 1
    report = sampler.report()
    assert report["num_samples"] == report["num_parent_draws"] == 2
    assert report["num_returned_samples"] == report["num_rejected_parent_draws"] == 1
    assert report["returned_record_indices"] == [1]
    assert report["failure_reasons"] == {"no_valid_edge_relocation": 1}
    assert report["changed_fraction"] == 0.5
    assert report["returned_novel_typed_fraction"] == 1.0
    assert report["returned_parent_typed_fingerprint"] == typed_fingerprint([
        TypedInvariant.from_dict(returned["parent_typed_invariant"]),
    ])
    assert report["returned_sampled_typed_fingerprint"] == typed_fingerprint([
        TypedInvariant.from_dict(summary["typed_invariant"]),
    ])


def test_retries_are_bounded_when_all_parents_are_immovable() -> None:
    sampler = _sampler(graphs=[_bank()[0]], retry=True, max_parent_attempts=3)
    with pytest.raises(DegreePerturbationError):
        sampler.sample()
    assert len(sampler.records) == 3
    assert [row["parent_attempt"] for row in sampler.records] == [1, 2, 3]
    assert {row["output_index"] for row in sampler.records} == {0}
    assert all(not row["returned"] and not row["fallback_used"] for row in sampler.records)
    assert sampler.returned_records == []
    report = sampler.report()
    assert report["num_parent_draws"] == report["num_rejected_parent_draws"] == 3
    assert report["num_returned_samples"] == 0
    assert report["returned_record_indices"] == []
    assert report["failure_reasons"] == {"no_valid_edge_relocation": 3}


def test_existing_keep_original_policy_remains_an_explicit_identity_transition() -> None:
    sampler = PerturbedEmpiricalTypedDegreeSampler.fit(
        [_bank()[0]],
        DegreePerturbationConfig(
            method="edge_relocation", probability=1.0, failure_policy="keep_original",
        ),
        edge_types=(1,), seed=0, constructor_config=_constructor(),
        allowed_signatures=_supported_signatures(), graph_validator=_valid_carbon_graph,
    )
    summary = sampler.sample()
    row = summary["sampling_diagnostics"]
    assert row["returned"] and row["fallback_used"] and not row["changed"]
    assert row["failure_reason"] == "no_valid_edge_relocation"
    assert summary["typed_invariant"] == row["parent_typed_invariant"]
    assert row["joint_realization_verified"]
    assert len(sampler.records) == len(sampler.returned_records) == 1
    report = sampler.report()
    assert report["num_identity_fallbacks"] == report["num_returned_samples"] == 1
    assert report["num_rejected_parent_draws"] == 0


def test_returned_invariants_preserve_constraints_and_are_independently_realizable() -> None:
    sampler = _sampler(retry=True, max_parent_attempts=32)
    summaries = [sampler.sample() for _ in range(5)]
    assert [row["output_index"] for row in sampler.returned_records] == list(range(5))
    assert len(sampler.returned_records) == 5
    for index, summary in enumerate(summaries):
        row = summary["sampling_diagnostics"]
        parent = TypedInvariant.from_dict(row["parent_typed_invariant"])
        output = TypedInvariant.from_dict(summary["typed_invariant"])
        assert parent.num_nodes == output.num_nodes == 4
        assert Counter(s.node_type for s in output.signatures) == Counter(s.node_type for s in parent.signatures)
        assert tuple(s.node_type for s in output.signatures) == tuple(s.node_type for s in parent.signatures)
        assert typed_totals(output) == typed_totals(parent)
        assert typed_key(output) not in sampler.training_set
        assert set(output.signatures) <= set(_supported_signatures())
        assert sorted(output.degree_sequence) == [1, 1, 2, 2]
        assert all(row[key] for key in (
            "joint_realization_verified", "preserved_node_categories", "preserved_edge_type_counts",
        ))
        witness, _ = construct_typed_graph(output, sampler.constructor_config, np.random.default_rng(100 + index))
        assert _valid_carbon_graph(witness)
        assert typed_invariant_matches_graph(witness, output)


def test_retry_random_stream_is_reproducible_and_does_not_use_global_numpy_rng() -> None:
    first = _sampler(retry=True, max_parent_attempts=32)
    second = _sampler(retry=True, max_parent_attempts=32)
    global_before = np.random.get_state()
    first_summaries = [first.sample() for _ in range(4)]
    global_after = np.random.get_state()
    assert global_before[0] == global_after[0]
    np.testing.assert_array_equal(global_before[1], global_after[1])
    assert global_before[2:] == global_after[2:]
    unrelated = np.random.default_rng(912)
    second_summaries = []
    for _ in range(4):
        unrelated.normal(size=500)
        second_summaries.append(second.sample())
    assert [row["typed_invariant"] for row in first_summaries] == [row["typed_invariant"] for row in second_summaries]
    assert first.records == second.records
    assert first.report() == second.report()


@pytest.mark.parametrize("error", [RuntimeError("internal failure"), KeyError("unexpected field")])
def test_retry_does_not_swallow_unrelated_exceptions(monkeypatch, error) -> None:
    sampler = _sampler(retry=True)
    calls = []

    def broken(parent_index, **kwargs):
        calls.append(parent_index)
        raise error

    monkeypatch.setattr(sampler, "perturb_parent", broken)
    with pytest.raises(type(error)):
        sampler.sample()
    assert calls == [0]
    assert sampler.records == []


def test_builder_forwards_retry_policy_and_keeps_strict_relocation_settings() -> None:
    config = _builder_config()
    original = deepcopy(config)
    vectorizer = SimpleNamespace(
        vocabulary=SimpleNamespace(signatures=_supported_signatures()),
        max_ordinary_degree=3, max_weighted_valence={6: 4.0},
    )
    sampler = build_typed_empirical_sampler(
        config, _bank(), seed=0, edge_types=(1,), vectorizer=vectorizer,
        graph_validator=_valid_carbon_graph,
    )
    assert config == original
    assert sampler.config.method == "edge_relocation"
    assert sampler.config.probability == 1.0 and sampler.config.steps == 1
    assert sampler.config.failure_policy == "error"
    assert sampler.constructor_config.max_ordinary_degree == 3
    assert sampler.sample()["sampling_diagnostics"]["changed"]
    assert len(sampler.records) == 2


@pytest.mark.parametrize("policy", ["keep_original", "unknown", ""])
def test_builder_rejects_unknown_parent_failure_policy(policy) -> None:
    config = _builder_config()
    config["generation"]["invariant_failure_policy"] = policy
    with pytest.raises(ValueError):
        build_typed_empirical_sampler(config, _bank(), seed=0, edge_types=(1,))


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_builder_rejects_invalid_parent_retry_budget(budget) -> None:
    config = _builder_config()
    config["generation"]["max_invariant_parent_attempts"] = budget
    with pytest.raises(ValueError):
        build_typed_empirical_sampler(config, _bank(), seed=0, edge_types=(1,))


def test_builder_rejects_parent_retry_combined_with_identity_fallback() -> None:
    config = _builder_config()
    config["generation"]["invariant_source"] = "train_empirical_perturbed"
    config["generation"]["degree_perturbation"] = {
        "method": "edge_relocation", "probability": 1.0, "failure_policy": "keep_original",
    }
    with pytest.raises(ValueError):
        build_typed_empirical_sampler(config, _bank(), seed=0, edge_types=(1,))


@pytest.mark.parametrize("source", ["learned", "train_empirical"])
def test_builder_does_not_silently_ignore_retry_controls_on_inactive_sources(source) -> None:
    config = _builder_config()
    config["generation"]["invariant_source"] = source
    config["generation"]["invariant_rng_mode"] = "legacy"
    config["generation"]["degree_perturbation"] = {}
    with pytest.raises(ValueError):
        build_typed_empirical_sampler(config, _bank(), seed=0, edge_types=(1,))


def _generation_model():
    import torch

    return SimpleNamespace(
        degree_model=None, vectorizer=None, edge_types=(1,),
        parameters=lambda: iter([torch.zeros(1)]),
    )


def _generation_config() -> dict:
    config = _builder_config()
    config["generation"].update(max_attempts_per_graph=3, require_rdkit_source_validity=True)
    config["attributed_refiner"] = {"rdkit_candidate_filter": True}
    return config


def test_generation_reaches_requested_count_and_ignores_refiner_rng_consumption() -> None:
    pytest.importorskip("rdkit")
    import torch
    from grapher.rewiring_mlp.attributed import joint_typed_edge_generation as generation

    config = _generation_config()
    first_sampler = _sampler(retry=True, max_parent_attempts=32)
    first = list(generation.generation_sources(
        _generation_model(), _bank(), config, seed=42, num_generate=5,
        empirical_sampler=first_sampler,
    ))
    assert len(first) == len(first_sampler.returned_records) == 5
    assert len(first_sampler.records) > 5
    for (graph, report, counts), sampled in zip(first, first_sampler.returned_records):
        assert _valid_carbon_graph(graph)
        observed = extract_typed_invariant(graph, edge_types=(1,))
        expected = TypedInvariant.from_dict(sampled["typed_invariant"])
        assert typed_key(observed) == typed_key(expected)
        assert report["sampling"] == sampled
        assert report["constructor_target_typed_match"]
        assert counts["requested_graphs"] == 5
    totals = first[-1][2]
    assert totals["constructed_sources"] == 5
    assert totals["invariant_proposals"] == len(first_sampler.records)
    assert totals["invariant_sampling_failures"] == len(first_sampler.records) - 5
    assert totals.get("constructor_failures", 0) == totals.get("rdkit_source_rejections", 0) == 0

    second_sampler = _sampler(retry=True, max_parent_attempts=32)
    second = []
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        for item in generation.generation_sources(
            _generation_model(), _bank(), config, seed=42, num_generate=5,
            empirical_sampler=second_sampler,
        ):
            second.append(item)
            # Mimic an unrelated stochastic refiner between source requests.
            np.random.normal(size=1000)
            torch.rand(1000)
    finally:
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
    assert second_sampler.records == first_sampler.records
    for (left, left_report, left_counts), (right, right_report, right_counts) in zip(first, second):
        assert generation.graph_record(left) == generation.graph_record(right)
        assert left_report["sampling"] == right_report["sampling"]
        assert left_counts == right_counts


def test_later_source_constructor_failure_retries_the_fixed_invariant_only(monkeypatch) -> None:
    pytest.importorskip("rdkit")
    from grapher.models.dhvae_hh.typed_constructor import TypedConstructionError
    from grapher.rewiring_mlp.attributed import joint_typed_edge_generation as generation

    sampler = _sampler(retry=True, max_parent_attempts=32)
    attempted_invariants = []

    def fail_construction(invariant, constructor, rng):
        attempted_invariants.append(invariant)
        raise TypedConstructionError("synthetic constructor exhaustion", {"failure_reason": "budget_exhausted"})

    # Patch only final source construction; the sampler still builds real witnesses.
    monkeypatch.setattr(generation, "construct_typed_graph", fail_construction)
    sampling_counts = Counter()
    with pytest.raises(RuntimeError, match="Failed to realize source 0"):
        list(generation.generation_sources(
            _generation_model(), _bank(), _generation_config(), seed=42, num_generate=5,
            empirical_sampler=sampler, sampling_counts=sampling_counts,
        ))
    assert len(attempted_invariants) == 3
    assert [row["parent_train_index"] for row in sampler.records] == [0, 1]
    assert len(sampler.returned_records) == 1
    expected = TypedInvariant.from_dict(sampler.returned_records[0]["typed_invariant"])
    assert all(typed_key(invariant) == typed_key(expected) for invariant in attempted_invariants)
    assert sampler.report()["num_parent_draws"] == 2
    assert sampling_counts["constructor_failures"] == 3
    assert sampling_counts["invariant_proposals"] == 2
    assert sampling_counts["invariant_sampling_failures"] == 1
    assert sampling_counts["requested_graphs"] == 5
