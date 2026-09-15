"""Exercise bounded source failures through the joint molecular generation runner."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import pickle
from types import SimpleNamespace

import networkx as nx
import pytest
import torch

from grapher.models.dhvae_hh.degree_perturbation import DegreePerturbationError
from grapher.models.dhvae_hh.typed_constructor import TypedConstructionError
from grapher.models.dhvae_hh.typed_degree_perturbation import typed_key
from grapher.rewiring_mlp.attributed import joint_typed_edge_generation as generation
from grapher.rewiring_mlp.molecular.typed_invariants import (
    TypedDegreeSignature, TypedInvariant, extract_typed_invariant,
)


def _carbon(graph):
    nx.set_node_attributes(graph, 6, "atomic_num")
    nx.set_edge_attributes(graph, 1, "bond_type")
    return graph


def _setup(tmp_path, monkeypatch, *, train=None, parents=None, retry=False):
    train = [_carbon(nx.complete_graph(3)), _carbon(nx.star_graph(3))] if train is None else train
    config = {
        "dataset": {"name": "toy"},
        "joint_typed_degree": {"enabled": True},
        "categorical_state": {"node_categories": [6], "edge_categories": [1]},
        "attributed_predictor": {"loss_weights": {"edge_ce": 1.0, "edge_logit": 1.0}},
        "structure_summary_prediction": {"clustering_histogram": False, "orbit_summary": False},
        "edge_diffusion": {"sampling_steps": 2, "spectral_enabled": False},
        "constructor": {
            "ensure_connected": True, "max_restarts": 4, "max_backtracks": 1000,
            "max_ordinary_degree": 4, "max_weighted_valence": {6: 4.0},
            "candidate_ranking": "uniform",
        },
        "attributed_refiner": {"steps": 0, "weights": {"edge": 1.0}, "rdkit_candidate_filter": False},
        "generation": {
            "invariant_source": "edge_relocation", "invariant_rng_mode": "independent",
            "require_rdkit_source_validity": False, "max_attempts_per_graph": 3,
            "checkpoint_every": 16, "degree_perturbation": {"require_novel": True},
        },
    }
    if retry:
        config["generation"].update(invariant_failure_policy="resample_parent", max_invariant_parent_attempts=3)
    model = torch.nn.Linear(1, 1)
    model.degree_model = None
    model.vectorizer = SimpleNamespace(
        vocabulary=SimpleNamespace(signatures=tuple(TypedDegreeSignature(6, (d,)) for d in (1, 2, 3))),
        max_ordinary_degree=4, max_weighted_valence={6: 4.0},
    )
    model.edge_types = (1,)
    model.atom_types = (6,)
    model.spectral_mode = generation.LEGACY_MODE
    model.spectral_enabled = False
    model.smoothing = 0.01
    model.histogram_bins = 0
    model.orbit_enabled = False
    model.induced_graphlet_basis = model.induced_graphlet_spec = None
    model.diffusion_metadata = lambda: {"mode": generation.LEGACY_MODE}
    model.induced_graphlet_metadata = lambda: None
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"stub predictor checkpoint")
    checkpoint = {"format": "fixture_joint_typed_edge", "config": deepcopy(config),
                  "dataset_provenance": {"fingerprint": "fixture_train"}}
    monkeypatch.setattr(generation, "load_checkpoint", lambda *a, **k: (model, checkpoint))
    monkeypatch.setattr(generation, "load_splits", lambda _: ({"train": train}, {"fingerprint": "fixture_train"}))
    captured = []
    original_build = generation.build_typed_empirical_sampler

    def build(*args, **kwargs):
        sampler = original_build(*args, **kwargs)
        if sampler is not None:
            captured.append(sampler)
            if parents is not None:
                indices = iter(parents)
                sampler.parent_rng = SimpleNamespace(integers=lambda _: next(indices))
        return sampler

    monkeypatch.setattr(generation, "build_typed_empirical_sampler", build)
    bridge_seeds, refiner_seeds = [], []

    def bridge(model, source, config, *, seed):
        bridge_seeds.append(seed)
        return {"seed": seed}, {"sampling_steps": 2, "prediction_calls": 3}

    def refine(source, targets, model, config, *, seed):
        refiner_seeds.append(seed)
        graph = source.copy()
        graph.graph["refiner_seed"] = seed
        return graph, {"accepted_steps": 0, "typed_degree_preserved": True, "connected": True}

    monkeypatch.setattr(generation, "sample_soft_endpoint", bridge)
    monkeypatch.setattr(generation, "refine_typed_graph", refine)
    output = tmp_path / "generation"
    args = SimpleNamespace(seed=42, checkpoint=str(checkpoint_path), device="cpu",
                           num_generate=3, output_dir=str(output))
    return SimpleNamespace(config=config, args=args, model=model, train=train, output=output,
                           captured=captured, bridge_seeds=bridge_seeds, refiner_seeds=refiner_seeds)


def _json(path):
    def invalid(value):
        raise AssertionError(f"Non-finite JSON value {value} in {path}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def _outputs(context, *, partial=False):
    report = _json(context.output / "report.json")
    prior = _json(context.output / "typed_degree_prior_report.json")
    invariants = _json(context.output / "sampled_typed_invariants.json")
    with (context.output / ("partial_molecular_graphs.pkl" if partial else "molecular_graphs.pkl")).open("rb") as handle:
        graphs = pickle.load(handle)
    assert len(graphs) == len(invariants) == report["num_generated"] == prior["num_returned"]
    assert len(prior["completed_records"]) == len(graphs)
    assert report["parent_typed_fingerprint"] == prior["completed_parent_typed_fingerprint"]
    assert report["sampled_typed_fingerprint"] == prior["completed_sampled_typed_fingerprint"]
    for graph, invariant in zip(graphs, invariants):
        assert nx.is_connected(graph)
        assert typed_key(extract_typed_invariant(graph, edge_types=(1,))) == typed_key(TypedInvariant.from_dict(invariant))
    for path in context.output.glob("*.json"):
        _json(path)
    return report, prior, graphs


@pytest.mark.parametrize("retry", [False, True])
def test_parent_exhaustion_skips_middle_slot_and_preserves_later_seed(tmp_path, monkeypatch, retry):
    parents = [1, 0, 0, 0, 1] if retry else [1, 0, 1]
    context = _setup(tmp_path, monkeypatch, parents=parents, retry=retry)
    generation.generate_joint_typed_edge(context.config, context.args)
    report, prior, graphs = _outputs(context)
    assert report["num_requested"] == report["num_attempted"] == 3
    assert report["num_generated"] == len(graphs) == 2
    assert report["num_skipped"] == 1 and report["complete"] is True
    assert report["requested_count_reached"] is False
    assert [row["source_index"] for row in report["records"]] == [0, 2]
    assert context.bridge_seeds == [42 + 7043 + i * 1009 for i in (0, 2)]
    assert context.refiner_seeds == [42 + 9049 + i * 1009 for i in (0, 2)]
    skipped = report["skipped_graphs"][0]
    assert skipped["generation_index"] == 1 and skipped["stage"] == "invariant_sampling"
    assert skipped["attempts_used"] == (3 if retry else 1)
    assert prior["num_parent_draws"] == len(parents)
    assert prior["num_rejected_parent_draws"] == len(parents) - 2
    assert prior["completed_record_indices"] == [0, len(parents) - 1]
    assert report["sampling_counts"]["attempted_graphs"] == 3
    assert report["sampling_counts"]["skipped_graphs"] == 1
    assert not list(context.output.glob("partial_*"))


@pytest.mark.parametrize("stage", ["invariant_sampling", "source_construction"])
def test_all_skipped_outputs_are_empty_complete_and_finite(tmp_path, monkeypatch, stage):
    context = _setup(tmp_path, monkeypatch, parents=([0] * 9 if stage == "invariant_sampling" else [1] * 3), retry=True)
    if stage == "source_construction":
        def reject(*args, **kwargs):
            raise TypedConstructionError("fixture source budget exhausted", {"failure_reason": "budget_exhausted"})
        monkeypatch.setattr(generation, "construct_typed_graph", reject)
    generation.generate_joint_typed_edge(context.config, context.args)
    report, prior, graphs = _outputs(context)
    assert graphs == [] and report["num_generated"] == 0
    assert report["num_requested"] == report["num_attempted"] == report["num_skipped"] == 3
    assert report["complete"] is True and report["requested_count_reached"] is False
    assert [row["generation_index"] for row in report["skipped_graphs"]] == [0, 1, 2]
    assert all(row["stage"] == stage and row["attempts_used"] == 3 for row in report["skipped_graphs"])
    assert context.bridge_seeds == context.refiner_seeds == []
    assert prior["completed_record_indices"] == []
    # Successful prior draws are not completed outputs if construction later fails.
    assert prior["num_returned_samples"] == (3 if stage == "source_construction" else 0)
    assert prior["num_parent_draws"] == (3 if stage == "source_construction" else 9)
    assert (context.output / "coarse_graphs.pkl").exists()
    assert (context.output / "soft_endpoints.pkl").exists()
    assert not list(context.output.glob("partial_*"))


def test_source_budget_skip_exports_only_completed_invariants(tmp_path, monkeypatch):
    context = _setup(tmp_path, monkeypatch, parents=[1, 1, 1])
    original = generation.construct_typed_graph
    invariants = []

    def construct(invariant, constructor, rng):
        invariants.append(invariant)
        if 2 <= len(invariants) <= 4:
            raise TypedConstructionError("fixture middle source failed", {"failure_reason": "budget_exhausted"})
        return original(invariant, constructor, rng)

    monkeypatch.setattr(generation, "construct_typed_graph", construct)
    generation.generate_joint_typed_edge(context.config, context.args)
    report, prior, graphs = _outputs(context)
    assert len(invariants) == 5 and len(graphs) == 2
    assert len(context.captured[0].returned_records) == 3
    assert prior["num_returned_samples"] == 3 and prior["num_returned"] == 2
    assert prior["completed_record_indices"] == [0, 2]
    rejected = TypedInvariant.from_dict(context.captured[0].returned_records[1]["typed_invariant"])
    assert all(invariant == rejected for invariant in invariants[1:4])
    assert [row["source_index"] for row in report["records"]] == [0, 2]
    skipped = report["skipped_graphs"][0]
    assert skipped["generation_index"] == 1 and skipped["stage"] == "source_construction"
    assert skipped["attempts_used"] == 3
    assert report["sampling_counts"]["constructor_failures"] == 3
    assert context.refiner_seeds == [42 + 9049 + i * 1009 for i in (0, 2)]


def test_rdkit_source_budget_skips_and_attempts_later_slots(tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    context = _setup(tmp_path, monkeypatch)
    context.config["generation"] = {
        "invariant_source": "train_empirical", "require_rdkit_source_validity": True,
        "max_attempts_per_graph": 3,
    }
    calls = []

    def valid(graph):
        calls.append(graph)
        return len(calls) > 3

    monkeypatch.setattr(generation, "is_valid_molecular_graph", valid)
    skipped, counts = [], Counter()
    outputs = list(generation.generation_sources(
        context.model, context.train, context.config, seed=42, num_generate=3,
        skipped_graphs=skipped, sampling_counts=counts,
    ))
    assert len(calls) == 5 and len(outputs) == 2
    assert [report["source_index"] for _, report, _ in outputs] == [1, 2]
    assert len(skipped) == 1 and skipped[0]["generation_index"] == 0
    assert skipped[0]["stage"] == "source_construction" and skipped[0]["attempts_used"] == 3
    assert counts["rdkit_source_rejections"] == 3 and counts["constructed_sources"] == 5
    assert counts["attempted_graphs"] == 3 and counts["skipped_graphs"] == 1


@pytest.mark.parametrize("expected_exhaustion", [True, False])
def test_learned_histogram_budget_is_skippable_but_internal_errors_are_not(tmp_path, monkeypatch, expected_exhaustion):
    context = _setup(tmp_path, monkeypatch)
    context.config["generation"] = {
        "invariant_source": "learned", "require_rdkit_source_validity": False,
        "max_attempts_per_graph": 3, "max_invariant_resample": 5,
    }
    calls = []
    message = ("Typed invariant sampling exhausted its feasibility budget: fixture infeasible"
               if expected_exhaustion else "fixture unexpected tensor shape")

    def summaries(outputs, **kwargs):
        calls.append(kwargs)
        raise RuntimeError(message)

    context.model.degree_model = SimpleNamespace(sample_outputs=lambda *a, **k: object())
    context.model.vectorizer.sample_empirical_node_count = lambda _: 4
    context.model.vectorizer.outputs_to_summaries = summaries
    counts, skipped = Counter(), []
    iterator = generation.generation_sources(
        context.model, context.train, context.config, seed=42, num_generate=3,
        sampling_counts=counts, skipped_graphs=skipped,
    )
    if expected_exhaustion:
        assert list(iterator) == []
        assert len(calls) == 9 and counts["histogram_draws"] == 45
        assert counts["invariant_sampling_failures"] == 9
        assert counts["attempted_graphs"] == counts["skipped_graphs"] == 3
        assert [row["generation_index"] for row in skipped] == [0, 1, 2]
        assert all(row["stage"] == "invariant_sampling" and row["attempts_used"] == 3 for row in skipped)
    else:
        with pytest.raises(RuntimeError, match="unexpected tensor shape"):
            list(iterator)
        assert len(calls) == 1 and counts["attempted_graphs"] == 1
        assert skipped == [] and counts["skipped_graphs"] == 0


@pytest.mark.parametrize("failure", ["unmarked_degree_error", "refiner_error"])
def test_unexpected_errors_remain_fatal_and_completed_exports_stay_aligned(tmp_path, monkeypatch, failure):
    context = _setup(tmp_path, monkeypatch, parents=[1, 1, 1])
    if failure == "unmarked_degree_error":
        original_build = generation.build_typed_empirical_sampler

        def build(*args, **kwargs):
            sampler = original_build(*args, **kwargs)
            original_sample = sampler.sample
            calls = []

            def sample():
                calls.append(1)
                if len(calls) == 2:
                    raise DegreePerturbationError("fixture unmarked internal error")
                return original_sample()

            sampler.sample = sample
            return sampler

        monkeypatch.setattr(generation, "build_typed_empirical_sampler", build)
        expected = DegreePerturbationError
    else:
        original_refine = generation.refine_typed_graph
        calls = []

        def refine(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("fixture unexpected refiner error")
            return original_refine(*args, **kwargs)

        monkeypatch.setattr(generation, "refine_typed_graph", refine)
        expected = RuntimeError
    with pytest.raises(expected, match="fixture"):
        generation.generate_joint_typed_edge(context.config, context.args)
    report, prior, graphs = _outputs(context, partial=True)
    assert len(graphs) == 1 and report["num_attempted"] == 2
    assert report["complete"] is False and report["num_skipped"] == 0
    assert report["skipped_graphs"] == [] and "failure" in report
    assert prior["completed_record_indices"] == [0]
    assert prior["num_returned_samples"] == (1 if failure == "unmarked_degree_error" else 2)
    assert not (context.output / "molecular_graphs.pkl").exists()
