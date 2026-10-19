import csv
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import networkx as nx
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_graph_generation_report.py"
spec = spec_from_file_location("graph_generation_report", SCRIPT)
report = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(report)


@pytest.mark.parametrize("protocol", ["graphrnn", "graphes_adaptive"])
def test_generic_metrics_distinguish_graphs_and_reach_outputs(protocol, tmp_path, capsys):
    reference = [nx.path_graph(5), nx.cycle_graph(5)]
    options = {"compute_orbit": False, "metric_protocol": protocol, "train": reference}
    identical = report._paper_mmd(reference, reference, **options)
    different = report._paper_mmd(reference, [nx.complete_graph(5)], **options)
    for key in ("spectral_mmd", "graphlet_history_mmd"):
        assert identical[key] == pytest.approx(0.0, abs=1e-12)
        assert different[key] > 0.0

    rows = [{"comparison": "generated_to_test", **different}]
    path = tmp_path / "metrics.csv"
    report._write_csv(rows, path)
    with path.open(newline="") as handle:
        stored = next(csv.DictReader(handle))
    assert set(stored) == {"comparison", *report.GENERIC_REPORT_METRICS}
    assert float(stored["spectral_mmd"]) == different["spectral_mmd"]
    assert float(stored["graphlet_history_mmd"]) == different["graphlet_history_mmd"]
    assert identical["novelty_rate"] == 0.0
    assert float(stored["novelty_rate"]) == different["novelty_rate"] == 1.0
    report._print_table(rows)
    output = capsys.readouterr().out
    assert "Spectral MMD" in output
    assert "Graphlet MMD" in output
    assert "Novelty" in output
    assert "higher is better" in output


def test_generic_novelty_uses_training_topology_and_counts_duplicates():
    train = [nx.path_graph(5)]
    relabelled = nx.relabel_nodes(train[0], lambda node: f"node_{node}")
    nx.set_node_attributes(relabelled, "ignored", "label")
    candidate = [relabelled, nx.cycle_graph(5), nx.cycle_graph(5)]
    metrics = report._paper_mmd(
        [nx.cycle_graph(5)], candidate, train=train, compute_orbit=False,
    )
    assert metrics["novelty_rate"] == pytest.approx(2 / 3)


def test_reference_row_novelty_uses_heldout_population_against_training():
    train = [nx.path_graph(5)]
    heldout = [nx.cycle_graph(5)]

    # MMD semantics for the reference row remain heldout-vs-train, while the
    # novelty cell must describe the held-out population relative to train.
    metrics = report._paper_mmd(
        heldout,
        train,
        train=train,
        novelty_candidate=heldout,
        compute_orbit=False,
    )

    assert metrics["novelty_rate"] == pytest.approx(1.0)


def test_generated_row_novelty_still_uses_generated_population_by_default():
    train = [nx.path_graph(5)]
    generated = [nx.path_graph(5), nx.cycle_graph(5)]

    metrics = report._paper_mmd(
        [nx.cycle_graph(5)],
        generated,
        train=train,
        compute_orbit=False,
    )

    assert metrics["novelty_rate"] == pytest.approx(0.5)


@pytest.mark.parametrize("train", [None, []])
def test_generic_novelty_without_training_is_unavailable(train):
    import math

    graphs = [nx.path_graph(4)]
    metrics = report._paper_mmd(graphs, graphs, train=train, compute_orbit=False)
    assert math.isnan(metrics["novelty_rate"])


def test_molecular_report_keeps_existing_metrics(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Generic-only descriptors should not be computed")

    monkeypatch.setattr(report, "mmd_graphlet_statistics", unexpected)
    monkeypatch.setattr(report, "spectral_histogram", unexpected)
    graphs = [nx.path_graph(4)]
    metrics = report._paper_mmd(
        graphs, graphs, compute_orbit=False, include_generic_metrics=False,
    )
    assert set(metrics) == set(report.REPORT_METRICS)
    assert report._report_metrics([metrics]) == report.REPORT_METRICS


def test_spectral_mmd_ignores_node_order():
    graph = nx.cycle_graph(6)
    order = [4, 1, 5, 0, 3, 2]
    permuted = nx.from_numpy_array(nx.to_numpy_array(graph, nodelist=order))
    metrics = report._paper_mmd([graph], [permuted], compute_orbit=False)
    assert metrics["spectral_mmd"] == pytest.approx(0.0, abs=1e-12)


def test_comparison_reports_descriptor_progress_and_preserves_metrics(capsys):
    reference = [nx.path_graph(5)]
    generated = [nx.cycle_graph(5), nx.complete_graph(5)]
    expected = report._paper_mmd(reference, generated, compute_orbit=False)
    row = report._evaluate_comparison(
        reference, generated, "generated_to_test", compute_orbit=False,
    )
    assert row["comparison"] == "generated_to_test"
    for key in report.GENERIC_REPORT_METRICS:
        assert row[key] == pytest.approx(expected[key], nan_ok=True)
    output = capsys.readouterr().out
    assert "Evaluating generated_to_test: reference=1 candidate=2" in output
    assert "graphlet descriptors: 0/3" in output
    assert "graphlet descriptors: 3/3" in output
    assert "spectral MMD: 1/1" in output
    assert "degree MMD: 1/1" in output
    assert "clustering MMD: 1/1" in output
    assert "Completed generated_to_test" in output


def test_auxiliary_degree_sequence_artifact_reports_graphrnn_degree_mmd(tmp_path, capsys):
    import pickle

    sequences = [[2, 2, 2, 2], [1, 2, 2, 1]]
    path = tmp_path / "sampled_degree_sequences.pkl"
    with path.open("wb") as handle:
        pickle.dump(sequences, handle)

    loaded = report._load_degree_sequence_list(path)
    assert loaded == [[2, 2, 2, 2], [2, 2, 1, 1]]
    reference_graphs = [nx.cycle_graph(4), nx.path_graph(4)]
    metrics = report.evaluate_degree_sequence_sets(
        report._graph_degree_sequences(reference_graphs),
        loaded,
        train=report._graph_degree_sequences(reference_graphs),
    )
    assert metrics["degree_histogram_mmd_graphrnn"] == pytest.approx(0.0, abs=1e-12)
    report._print_degree_sequence_prior(metrics, reference_split="test")
    output = capsys.readouterr().out
    assert "Auxiliary generated degree-sequence MMD" in output
    assert "dhvae_degree_sequences_to_test" in output


def test_exact_isomorphism_novelty_detects_wl_collision():
    # C6 and two disjoint triangles are both 2-regular on six nodes and share
    # NetworkX's topology-only WL hash, but they are not isomorphic.
    train_graph = nx.cycle_graph(6)
    wl_collision = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))
    assert nx.weisfeiler_lehman_graph_hash(train_graph) == nx.weisfeiler_lehman_graph_hash(wl_collision)
    assert not nx.is_isomorphic(train_graph, wl_collision)

    diagnostics = report.exact_isomorphism_support_diagnostics(
        {
            "train": [train_graph],
            "heldout_test": [train_graph.copy()],
            "generated": [wl_collision],
        },
        reference_name="heldout_test",
    )
    rows = {row["population"]: row for row in diagnostics["population_metrics"]}
    assert rows["generated"]["wl_novelty_rate_against_train"] == pytest.approx(0.0)
    assert rows["generated"]["exact_isomorphism_novelty_rate_against_train"] == pytest.approx(1.0)
    assert rows["heldout_test"]["exact_isomorphism_novelty_rate_against_train"] == pytest.approx(0.0)


def test_exact_isomorphism_class_frequency_matching_reports_tv_and_overlap():
    path = nx.path_graph(4)
    cycle = nx.cycle_graph(4)
    diagnostics = report.exact_isomorphism_support_diagnostics(
        {
            "train": [path.copy(), path.copy(), cycle.copy(), cycle.copy()],
            "heldout_test": [path.copy(), cycle.copy()],
            "generated": [path.copy(), path.copy()],
        },
        reference_name="heldout_test",
    )
    rows = {row["population"]: row for row in diagnostics["population_metrics"]}

    assert rows["train"]["class_frequency_tv_to_train"] == pytest.approx(0.0)
    assert rows["heldout_test"]["class_frequency_tv_to_train"] == pytest.approx(0.0)
    assert rows["generated"]["class_frequency_tv_to_train"] == pytest.approx(0.5)
    assert rows["generated"]["class_frequency_overlap_to_train"] == pytest.approx(0.5)
    assert rows["generated"]["class_frequency_tv_to_reference"] == pytest.approx(0.5)
    assert rows["generated"]["exact_isomorphism_novelty_rate_against_train"] == pytest.approx(0.0)
    assert rows["generated"]["train_isomorphism_class_coverage_rate"] == pytest.approx(0.5)


def test_exact_isomorphism_support_csv_contains_per_class_frequencies(tmp_path):
    diagnostics = report.exact_isomorphism_support_diagnostics(
        {
            "train": [nx.path_graph(4)],
            "heldout_test": [nx.cycle_graph(4)],
            "generated": [nx.path_graph(4), nx.cycle_graph(4)],
        },
        reference_name="heldout_test",
    )
    path = tmp_path / "classes.csv"
    report._write_isomorphism_support_csv(diagnostics["class_frequencies"], path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["is_training_class"] for row in rows} == {"True", "False"}
    assert all("generated_count" in row for row in rows)
    assert all("heldout_test_count" in row for row in rows)


def test_diversification_artifacts_use_semantic_stage_labels(tmp_path):
    import pickle

    generated_dir = tmp_path / "generations" / "seed_42_n_4"
    generated_dir.mkdir(parents=True)
    final_path = generated_dir / "base_graphs.pkl"
    source_path = generated_dir / "pre_refinement_graphs.pkl"
    graphs = [nx.path_graph(4)]
    for path in (final_path, source_path):
        with path.open("wb") as handle:
            pickle.dump(graphs, handle)

    manifest = {
        "model_id": "grapher",
        "sampling": {"stochastic_diversification": {"random_steps": {"enabled": True}}},
    }
    assert report.resolve_generated_stage(
        {}, final_path, generation_manifest=manifest
    ) == "grapher_diversified"
    assert report.resolve_generated_stage(
        {}, source_path, generation_manifest=manifest
    ) == "grapher_spectral_pre_rewire"

    base_path, base_stage = report.resolve_base_graph_path(
        generated_dir,
        generated_path=final_path,
        generation_manifest=manifest,
    )
    assert base_path == source_path
    assert base_stage == "grapher_spectral_pre_rewire"


def test_graded_topology_distance_is_permutation_invariant_and_bounded():
    graph = nx.path_graph(6)
    relabelled = nx.relabel_nodes(graph, {node: f"v{5-node}" for node in graph.nodes()})
    distance = report.graded_topology_distance(graph, relabelled, wl_iterations=3)
    assert distance == pytest.approx(0.0, abs=1e-12)

    modified = graph.copy()
    modified.add_edge(0, 5)
    modified_distance = report.graded_topology_distance(graph, modified, wl_iterations=3)
    assert 0.0 < modified_distance <= 1.0


def test_graded_diversity_exposes_near_copy_despite_exact_novelty():
    train_graph = nx.path_graph(12)
    near_copy = train_graph.copy()
    near_copy.add_edge(0, 11)
    heldout = nx.cycle_graph(10)

    exact = report.exact_isomorphism_support_diagnostics(
        {
            "train": [train_graph],
            "heldout_test": [heldout],
            "generated": [near_copy],
        },
        reference_name="heldout_test",
    )
    exact_rows = {row["population"]: row for row in exact["population_metrics"]}
    assert exact_rows["generated"]["exact_isomorphism_novelty_rate_against_train"] == pytest.approx(1.0)

    graded = report.graded_diversity_diagnostics(
        {
            "train": [train_graph],
            "heldout_test": [heldout],
            "generated": [near_copy],
        },
        reference_name="heldout_test",
        epsilons=(0.0, 0.05, 0.3),
        wl_iterations=3,
    )
    graded_rows = {row["population"]: row for row in graded["population_metrics"]}
    distance = graded_rows["generated"]["mean_nearest_train_structural_distance"]
    assert 0.0 < distance < 0.5
    curves = {
        (row["population"], row["epsilon"]): row
        for row in graded["epsilon_curve"]
    }
    assert curves[("generated", 0.0)]["epsilon_novelty_rate_against_train"] == pytest.approx(1.0)
    # A sufficiently loose tolerance should correctly classify the one-edge
    # perturbation as close to the training support rather than maximally novel.
    assert curves[("generated", 0.3)]["epsilon_novelty_rate_against_train"] == pytest.approx(0.0)


def test_graded_diversity_coverage_curve_uses_heldout_as_target():
    train = [nx.path_graph(6)]
    heldout = [nx.cycle_graph(6), nx.path_graph(7)]
    generated = [nx.cycle_graph(6)]
    diagnostics = report.graded_diversity_diagnostics(
        {
            "train": train,
            "heldout_test": heldout,
            "generated": generated,
        },
        reference_name="heldout_test",
        epsilons=(0.0, 1.0),
        wl_iterations=2,
    )
    curve = {
        (row["population"], row["epsilon"]): row
        for row in diagnostics["epsilon_curve"]
    }
    assert curve[("generated", 0.0)]["heldout_coverage_rate"] == pytest.approx(0.5)
    assert curve[("generated", 1.0)]["heldout_coverage_rate"] == pytest.approx(1.0)
    assert curve[("heldout_test", 0.0)]["heldout_coverage_rate"] == pytest.approx(1.0)


def test_graded_distance_is_explicitly_a_pseudodistance_for_wl_collisions():
    # Connected-component mass distinguishes this classic C6 vs 2*C3 WL
    # collision, so the graded distance is positive even though the degree/WL
    # layers alone would match.
    cycle = nx.cycle_graph(6)
    two_triangles = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))
    assert report.graded_topology_distance(cycle, two_triangles, wl_iterations=3) > 0.0


def test_degree_constrained_realization_artifacts_use_semantic_stage_labels(tmp_path):
    import pickle

    generated_dir = tmp_path / "generations" / "seed_42_n_4"
    generated_dir.mkdir(parents=True)
    final_path = generated_dir / "base_graphs.pkl"
    control_path = generated_dir / "pre_realization_graphs.pkl"
    graphs = [nx.path_graph(4)]
    for path in (final_path, control_path):
        with path.open("wb") as handle:
            pickle.dump(graphs, handle)

    manifest = {
        "model_id": "grapher",
        "sampling": {
            "degree_constrained_realization": {
                "enabled": True,
                "method": "gumbel_milp",
            }
        },
    }
    assert report.resolve_generated_stage(
        {}, final_path, generation_manifest=manifest
    ) == "grapher_degree_constrained_realization"
    assert report.resolve_generated_stage(
        {}, control_path, generation_manifest=manifest
    ) == "grapher_threshold_control"

    base_path, base_stage = report.resolve_base_graph_path(
        generated_dir,
        generated_path=final_path,
        generation_manifest=manifest,
    )
    assert base_path == control_path
    assert base_stage == "grapher_threshold_control"


def test_summary_selected_degree_realization_uses_distinct_semantic_stage_label(tmp_path):
    import pickle

    generated_dir = tmp_path / "generations" / "summary_selected"
    generated_dir.mkdir(parents=True)
    final_path = generated_dir / "base_graphs.pkl"
    control_path = generated_dir / "pre_realization_graphs.pkl"
    graphs = [nx.path_graph(4)]
    for path in (final_path, control_path):
        with path.open("wb") as handle:
            pickle.dump(graphs, handle)

    manifest = {
        "model_id": "grapher",
        "sampling": {
            "degree_constrained_realization": {
                "enabled": True,
                "method": "gumbel_milp",
                "candidate_selection_enabled": True,
            }
        },
    }
    assert report.resolve_generated_stage(
        {}, final_path, generation_manifest=manifest
    ) == "grapher_degree_constrained_summary_selected_realization"
    assert report.resolve_generated_stage(
        {}, control_path, generation_manifest=manifest
    ) == "grapher_threshold_control"
