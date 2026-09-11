from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import networkx as nx
import pytest

from grapher.utils.io import save_pickle
from scripts import draw_dataset as draw
from scripts import draw_generated_dataset as generated


def _molecular_cycle(atoms: tuple[int, ...] = (6, 6, 6)) -> nx.Graph:
    graph = nx.cycle_graph(len(atoms))
    nx.set_node_attributes(graph, dict(enumerate(atoms)), "atomic_num")
    nx.set_edge_attributes(graph, 1, "bond_type")
    return graph


def _nitrogen_star(*, charge: int | None = None) -> nx.Graph:
    graph = nx.star_graph(4)
    nx.set_node_attributes(graph, 6, "atomic_num")
    nx.set_edge_attributes(graph, 1, "bond_type")
    graph.nodes[0]["atomic_num"] = 7
    if charge is not None:
        graph.nodes[0]["formal_charge"] = charge
    return graph


def _pdf_page_count(path: Path) -> int:
    pytest.importorskip("PIL")
    from PIL import PdfParser

    assert path.read_bytes().startswith(b"%PDF-")
    with PdfParser.PdfParser(filename=str(path)) as document:
        return len(document.pages)


def _report(output: Path, suffix: str) -> dict:
    return json.loads(output.with_name(f"{output.stem}_{suffix}.json").read_text())


def test_parser_requires_generated_graphs_and_supports_shared_options() -> None:
    with pytest.raises(SystemExit, match="2"):
        generated.build_parser().parse_args([])
    args = generated.build_parser().parse_args([
        "--generated-graphs", "run/molecular_graphs.pkl", "--dataset", "qm9_attributed",
        "--count", "1024", "--seed", "7", "--row", "4", "--col", "4",
        "--k-min", "3", "--k-max", "5", "--output", "molecules.pdf",
        "--graphlet-output", "cycles.pdf",
    ])
    assert args.generated_graphs == Path("run/molecular_graphs.pkl")
    assert args.dataset == "qm9_attributed"
    assert args.count == 1024
    assert args.seed == 7
    assert (args.row, args.col, args.k_min, args.k_max) == (4, 4, 3, 5)
    assert args.output == Path("molecules.pdf")
    assert args.graphlet_output == Path("cycles.pdf")
    with pytest.raises(SystemExit, match="2"):
        generated.build_parser().parse_args([
            "--generated-graphs", "graphs.pkl", "--count", "4", "--all",
        ])


@pytest.mark.parametrize("wrapper", [None, "graphs", "molecular_graphs", "generated_graphs"])
@pytest.mark.parametrize("container", [list, tuple])
def test_loader_accepts_evaluator_pickle_layouts(tmp_path: Path, wrapper, container) -> None:
    graphs = container([_molecular_cycle(), _molecular_cycle((6, 6, 8))])
    path = tmp_path / "molecular_graphs.pkl"
    save_pickle(graphs if wrapper is None else {wrapper: graphs, "metadata": {}}, path)
    loaded = generated._load_generated_graphs(path)
    assert isinstance(loaded, list)
    assert len(loaded) == 2
    assert dict(loaded[1].nodes(data=True)) == dict(graphs[1].nodes(data=True))
    assert list(loaded[1].edges(data=True)) == list(graphs[1].edges(data=True))


@pytest.mark.parametrize("payload", [[], (), {}, {"graphs": []}, ["not a graph"], 7])
def test_loader_rejects_empty_or_malformed_payload(tmp_path: Path, payload) -> None:
    path = tmp_path / "graphs.pkl"
    save_pickle(payload, path)
    with pytest.raises((TypeError, ValueError)):
        generated._load_generated_graphs(path)


def test_loader_handles_networkx_cached_view_incompatibility(tmp_path: Path, monkeypatch) -> None:
    graph = _molecular_cycle()
    graph.__dict__["edges"] = graph.edges
    graph.__dict__["degree"] = graph.degree
    graph.__dict__["nodes"] = graph.nodes
    path = tmp_path / "molecular_graphs.pkl"
    save_pickle({"molecular_graphs": [graph]}, path)

    def incompatible_setstate(self, state):
        raise AttributeError("'Graph' object has no attribute '_adj'")

    monkeypatch.setattr(type(graph.edges), "__setstate__", incompatible_setstate)
    loaded = generated._load_generated_graphs(path)
    assert dict(loaded[0].nodes(data=True)) == dict(graph.nodes(data=True))
    assert list(loaded[0].edges(data=True)) == list(graph.edges(data=True))
    loaded[0].add_edge(0, 3, bond_type=1)
    assert loaded[0].degree[0] == 3


@pytest.mark.parametrize("standalone_suffix", [".pdf", ".png"])
def test_typed_graphlets_use_entire_valid_generated_file_and_paginate(
    tmp_path: Path, monkeypatch, standalone_suffix: str,
) -> None:
    pytest.importorskip("rdkit")
    pytest.importorskip("PIL")
    types = list(itertools.combinations_with_replacement((6, 7, 8), 3))[:7]
    graphs = [_molecular_cycle(atoms) for atoms in types]
    graphs += [graphs[0].copy(), _molecular_cycle((6, 6, 9))]
    source = tmp_path / "molecular_graphs.pkl"
    save_pickle(graphs, source)
    output = tmp_path / "generated.pdf"
    standalone = tmp_path / f"cycles{standalone_suffix}"

    def forbidden_base_dataset_access(*args, **kwargs):
        raise AssertionError("Generated drawing must not read a base dataset.")

    monkeypatch.setattr(draw, "_load_prepared_dataset_selection", forbidden_base_dataset_access)
    assert generated.main([
        "--generated-graphs", str(source), "--dataset", "qm9_attributed",
        "--count", "1", "--seed", "42", "--row", "1", "--col", "2",
        "--panel-width", "220", "--panel-height", "220", "--output", str(output),
        "--k-min", "3", "--k-max", "5", "--graphlet-output", str(standalone),
    ]) == 0
    assert _pdf_page_count(output) == 3
    if standalone_suffix == ".pdf":
        assert _pdf_page_count(standalone) == 2
    else:
        pages = sorted(tmp_path.glob("cycles_page_*.png"))
        assert len(pages) == 2
        assert all(page.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for page in pages)
    report_path = standalone.with_suffix(".json")
    report = json.loads(report_path.read_text())
    assert report["generated_graphs"] == str(source.resolve())
    assert report["split"] == "generated"
    assert report["metric_molecule_source"] == "raw_valid"
    assert report["definition"] == "induced_atom_bond_typed_simple_cycle_Ck"
    assert report["total_dataset_graphs"] == 9
    assert report["selected_graphs"] == 8
    assert report["drawn_graphs"] == 1
    assert len(report["drawn_indices"]) == 1
    assert report["drawn_indices"][0] in range(8)
    assert report["excluded_invalid_molecular_graphs"] == 1
    assert report["total_cycle_graphlets"] == 8
    assert report["graphlet_pages"] == 2
    rows = report["graphlets"]
    assert len(rows) == 7
    assert [row["count"] for row in rows] == [2, 1, 1, 1, 1, 1, 1]
    assert sum(row["frequency"] for row in rows) == pytest.approx(1.0)
    assert {tuple(sorted(row["node_types"])) for row in rows} == set(types)
    assert all(row["edge_types"] == [1, 1, 1] for row in rows)
    metadata = _report(output, "drawing")
    assert metadata["generated_graphs"] == str(source.resolve())
    assert metadata["total_graphs"] == 9
    assert metadata["eligible_graphs"] == 8
    assert metadata["excluded_invalid_molecular_graphs"] == 1
    assert metadata["selected_indices"] == report["drawn_indices"]
    assert metadata["metric_molecule_source"] == "raw_valid"


def test_raw_valid_filter_preserves_explicit_charges_and_original_input_indices(
    tmp_path: Path, monkeypatch,
) -> None:
    pytest.importorskip("rdkit")
    pytest.importorskip("PIL")
    graphs = [_nitrogen_star(), _nitrogen_star(charge=1), nx.Graph(), _molecular_cycle()]
    for index, graph in enumerate(graphs):
        graph.graph["source_index"] = 100 + index
    source = tmp_path / "molecular_graphs.pkl"
    save_pickle(graphs, source)
    output = tmp_path / "raw.pdf"
    rendered = []
    compose = draw._compose_page

    def capture(items, **kwargs):
        rendered.extend(items)
        return compose(items, **kwargs)

    monkeypatch.setattr(draw, "_compose_page", capture)
    assert generated.main([
        "--generated-graphs", str(source), "--all", "--row", "1", "--col", "2",
        "--panel-width", "220", "--panel-height", "220", "--output", str(output),
        "--k-min", "3", "--k-max", "5",
    ]) == 0
    assert [item.info.dataset_index for item in rendered] == [1, 3]
    assert [item.info.source_index for item in rendered] == [101, 103]
    assert all(item.info.index_label.endswith(f"[{item.info.dataset_index}]") for item in rendered)
    assert all(item.error is None and item.render_mode == "molecule" for item in rendered)
    assert rendered[0].mol.GetAtomWithIdx(0).GetFormalCharge() == 1
    assert _pdf_page_count(output) == 2
    metadata = _report(output, "drawing")
    assert metadata["selected_indices"] == [1, 3]
    assert metadata["eligible_graphs"] == 2
    assert metadata["excluded_invalid_molecular_graphs"] == 2
    assert metadata["metric_molecule_source"] == "raw_valid"
    histogram = _report(output, "graphlet_histogram")
    assert histogram["selected_graphs"] == 2
    assert histogram["total_cycle_graphlets"] == 1


def test_generated_molecular_graphlets_distinguish_bond_types(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    pytest.importorskip("PIL")
    single = _molecular_cycle()
    double = single.copy()
    double[0][1]["bond_type"] = 2
    source = tmp_path / "molecular_graphs.pkl"
    save_pickle([single, double], source)
    output = tmp_path / "typed.pdf"
    assert generated.main([
        "--generated-graphs", str(source), "--count", "1", "--output", str(output),
        "--k-min", "3", "--k-max", "3",
    ]) == 0
    report = _report(output, "graphlet_histogram")
    assert report["total_cycle_graphlets"] == 2
    assert {tuple(sorted(row["edge_types"])) for row in report["graphlets"]} == {
        (1, 1, 1), (1, 1, 2),
    }
    assert [row["frequency"] for row in report["graphlets"]] == [0.5, 0.5]


@pytest.mark.parametrize(
    "kind", ["directed", "multigraph", "fractional_atom", "wildcard_atom", "fractional_bond"],
)
def test_malformed_molecular_schema_is_excluded_before_drawing_and_counting(
    tmp_path: Path, kind: str,
) -> None:
    pytest.importorskip("rdkit")
    pytest.importorskip("PIL")
    malformed = _molecular_cycle()
    if kind == "directed":
        malformed = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
        nx.set_node_attributes(malformed, 6, "atomic_num")
        nx.set_edge_attributes(malformed, 1, "bond_type")
    elif kind == "multigraph":
        malformed = nx.MultiGraph(malformed)
    elif kind == "fractional_atom":
        malformed.nodes[0]["atomic_num"] = 6.5
    elif kind == "wildcard_atom":
        malformed.nodes[0]["atomic_num"] = 0
    else:
        malformed[0][1]["bond_type"] = 1.5
    source = tmp_path / "molecular_graphs.pkl"
    save_pickle([malformed, _molecular_cycle()], source)
    output = tmp_path / "schema.pdf"
    assert generated.main([
        "--generated-graphs", str(source), "--count", "1", "--output", str(output),
        "--row", "1", "--col", "1", "--panel-width", "220", "--panel-height", "220",
        "--k-min", "3", "--k-max", "5",
    ]) == 0
    assert _pdf_page_count(output) == 2
    metadata = _report(output, "drawing")
    assert metadata["total_graphs"] == 2
    assert metadata["eligible_graphs"] == 1
    assert metadata["excluded_invalid_molecular_graphs"] == 1
    assert metadata["selected_indices"] == [1]
    report = _report(output, "graphlet_histogram")
    assert report["total_dataset_graphs"] == 2
    assert report["selected_graphs"] == 1
    assert report["excluded_invalid_molecular_graphs"] == 1
    assert report["total_cycle_graphlets"] == 1
    assert report["graphlets"][0]["node_types"] == [6, 6, 6]
    assert report["graphlets"][0]["edge_types"] == [1, 1, 1]


@pytest.mark.parametrize("all_invalid", [True, False])
def test_sampling_fails_if_too_few_raw_valid_molecules(tmp_path: Path, all_invalid: bool) -> None:
    pytest.importorskip("rdkit")
    graphs = [_nitrogen_star()]
    if not all_invalid:
        graphs.append(_molecular_cycle())
    source = tmp_path / "molecular_graphs.pkl"
    save_pickle(graphs, source)
    output = tmp_path / "not_written.pdf"
    with pytest.raises(ValueError, match="(?i)valid"):
        generated.main([
            "--generated-graphs", str(source), "--dataset", "qm9_attributed",
            *( ["--all"] if all_invalid else ["--count", "2"] ),
            "--output", str(output),
        ])
    assert not output.exists()


def test_generic_sampling_is_repeatable_without_replacement(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    source = tmp_path / "generated_graphs.pkl"
    save_pickle([nx.path_graph(order) for order in range(1, 21)], source)
    selections = []
    for run, seed in enumerate((42, 42, 43)):
        output = tmp_path / f"sample_{run}.pdf"
        assert generated.main([
            "--generated-graphs", str(source), "--count", "4", "--seed", str(seed),
            "--row", "2", "--col", "2", "--panel-width", "220", "--panel-height", "220",
            "--output", str(output),
        ]) == 0
        metadata = _report(output, "drawing")
        selections.append(metadata["selected_indices"])
        assert metadata["eligible_graphs"] == 20
        assert metadata["excluded_invalid_molecular_graphs"] == 0
        assert len(selections[-1]) == len(set(selections[-1])) == 4
        assert _pdf_page_count(output) == 1
    assert selections[0] == selections[1]
    assert selections[0] != selections[2]


def test_generic_graphlets_include_all_input_graphs_and_empty_graph_is_drawable(
    tmp_path: Path, monkeypatch,
) -> None:
    pytest.importorskip("PIL")
    source = tmp_path / "generated_graphs.pkl"
    save_pickle([nx.Graph(), nx.cycle_graph(3), nx.cycle_graph(3), nx.cycle_graph(4)], source)
    output = tmp_path / "generic.pdf"
    rendered = []
    compose = draw._compose_page

    def capture(items, **kwargs):
        rendered.extend(items)
        return compose(items, **kwargs)

    monkeypatch.setattr(draw, "_compose_page", capture)
    assert generated.main([
        "--generated-graphs", str(source), "--all", "--row", "1", "--col", "2",
        "--panel-width", "220", "--panel-height", "220", "--output", str(output),
        "--k-min", "3", "--k-max", "5",
    ]) == 0
    assert _pdf_page_count(output) == 3
    assert [item.info.dataset_index for item in rendered] == [0, 1, 2, 3]
    assert all(item.render_mode == "generic" for item in rendered)
    assert rendered[0].graph.number_of_nodes() == 0
    report = _report(output, "graphlet_histogram")
    assert report["definition"] == "induced_simple_cycle_Ck"
    assert report["selected_graphs"] == report["total_dataset_graphs"] == 4
    assert report["total_cycle_graphlets"] == 3
    assert [(row["graphlet"], row["count"]) for row in report["graphlets"]] == [
        ("C3", 2), ("C4", 1), ("C5", 0),
    ]


@pytest.mark.parametrize("draw_all", [False, True])
def test_default_output_is_pdf_next_to_generated_pickle(tmp_path: Path, draw_all: bool) -> None:
    pytest.importorskip("PIL")
    source = tmp_path / "generated_graphs.pkl"
    save_pickle([nx.path_graph(3), nx.path_graph(4)], source)
    assert generated.main([
        "--generated-graphs", str(source), *( ["--all"] if draw_all else [] ),
    ]) == 0
    output = tmp_path / (
        "generated_graphs_all.pdf" if draw_all else "generated_graphs_sample_n1_seed42.pdf"
    )
    assert _pdf_page_count(output) == 1
    assert _report(output, "drawing")["total_graphs"] == 2


def test_direct_cli_help_works_outside_repository(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src")] + ([existing] if existing else [])
    )
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "draw_generated_dataset.py"), "--help"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--generated-graphs" in completed.stdout
    assert "--k-min" in completed.stdout
    assert "--graphlet-output" in completed.stdout
