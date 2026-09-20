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
    options = {"compute_orbit": False, "metric_protocol": protocol}
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
    report._print_table(rows)
    output = capsys.readouterr().out
    assert "Spectral MMD" in output
    assert "Graphlet MMD" in output


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
