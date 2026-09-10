from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

from grapher.utils.io import save_pickle
from scripts import draw_dataset as draw


def _write_split_files(
    root: Path,
    dataset: str,
    *,
    selected_split: str,
    selected_graphs: list[Any],
) -> Path:
    """Write one valid split and intentionally unreadable unused splits."""

    directory = root / dataset
    directory.mkdir(parents=True)
    for split in ("train", "val", "test"):
        path = directory / f"{split}.pkl"
        if split == selected_split:
            save_pickle(selected_graphs, path)
        else:
            path.write_bytes(b"unused split must not be unpickled")
    return directory


def _molecular_graph(*, source_index: int | None = None) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(0, atomic_num=6, atom_type=6)
    graph.add_node(1, atomic_num=8, atom_type=8)
    graph.add_edge(0, 1, bond_type=2, bond_order=2.0)
    if source_index is not None:
        graph.graph["source_index"] = source_index
    return graph


def _generic_graphs() -> list[nx.Graph]:
    empty = nx.Graph()
    empty.graph["source_index"] = 10

    singleton = nx.empty_graph(1)
    singleton.graph["source_index"] = 20

    disconnected = nx.disjoint_union(nx.path_graph(2), nx.path_graph(3))
    disconnected.graph["source_index"] = 30
    return [empty, singleton, disconnected]


def test_parser_requires_dataset_and_has_sampling_defaults() -> None:
    with pytest.raises(SystemExit, match="2"):
        draw.build_parser().parse_args([])

    args = draw.build_parser().parse_args(["--dataset", "qm9_attributed"])

    assert args.dataset == "qm9_attributed"
    assert Path(args.root) == Path("outputs/datasets")
    assert args.split == "test"
    assert args.count == 1
    assert args.seed == 42
    assert args.all is False


def test_parser_supports_all_graphs_across_all_splits() -> None:
    args = draw.build_parser().parse_args(
        ["--dataset", "community_small", "--split", "all", "--all"]
    )

    assert args.split == "all"
    assert args.all is True

    with pytest.raises(SystemExit, match="2"):
        draw.build_parser().parse_args(
            ["--dataset", "community_small", "--count", "4", "--all"]
        )


def test_parser_supports_induced_cycle_graphlet_range_and_output() -> None:
    args = draw.build_parser().parse_args(
        [
            "--dataset",
            "community_small",
            "--k-min",
            "3",
            "--graphlet-k-max",
            "6",
            "--graphlet-output",
            "outputs/cycles.png",
        ]
    )

    assert args.k_min == 3
    assert args.k_max == 6
    assert args.graphlet_output == Path("outputs/cycles.png")


@pytest.mark.parametrize(
    "options",
    [
        ["--k-min", "3"],
        ["--k-max", "5"],
        ["--k-min", "2", "--k-max", "5"],
        ["--k-min", "5", "--k-max", "4"],
        ["--graphlet-output", "cycles.png"],
    ],
)
def test_cli_validates_cycle_graphlet_arguments(options: list[str]) -> None:
    with pytest.raises(ValueError):
        draw.main(["--dataset", "community_small", *options])


@pytest.mark.parametrize("option", ["--index", "--index-from", "--index-to"])
def test_parser_rejects_removed_index_options(option: str) -> None:
    with pytest.raises(SystemExit, match="2"):
        draw.build_parser().parse_args(
            ["--dataset", "qm9_attributed", option, "3"]
        )


def test_default_output_identifies_sample_count_and_seed() -> None:
    assert draw._default_output(
        16, 42, "qm9_attributed_test"
    ) == Path("outputs/qm9_attributed_test_sample_n16_seed42.png")


def test_sample_graph_indices_is_deterministic_and_without_replacement() -> None:
    first = draw._sample_graph_indices(dataset_size=20, count=8, seed=42)
    repeated = draw._sample_graph_indices(dataset_size=20, count=8, seed=42)
    different_seed = draw._sample_graph_indices(dataset_size=20, count=8, seed=43)

    assert first == repeated
    assert first != different_seed
    assert len(first) == 8
    assert len(set(first)) == 8
    assert all(0 <= index < 20 for index in first)


def test_select_graph_indices_returns_every_graph_in_dataset_order() -> None:
    assert draw._select_graph_indices(
        5,
        count=1,
        seed=42,
        draw_all=True,
    ) == [0, 1, 2, 3, 4]


def test_load_all_splits_preserves_split_local_locations(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    directory = root / "generic"
    directory.mkdir(parents=True)
    save_pickle([nx.path_graph(2), nx.path_graph(3)], directory / "train.pkl")
    save_pickle([nx.path_graph(4)], directory / "val.pkl")
    save_pickle([nx.path_graph(5), nx.path_graph(6)], directory / "test.pkl")

    graphs, dataset_path, dataset_name, locations = (
        draw._load_prepared_dataset_selection("generic", root, "all")
    )

    assert dataset_path == directory
    assert dataset_name == "generic"
    assert [graph.number_of_nodes() for graph in graphs] == [2, 3, 4, 5, 6]
    assert locations == [
        ("train", 0),
        ("train", 1),
        ("val", 0),
        ("test", 0),
        ("test", 1),
    ]


@pytest.mark.parametrize(
    ("dataset_size", "count"),
    [
        (5, 0),
        (5, -1),
        (5, 6),
        (0, 1),
    ],
)
def test_sample_graph_indices_validates_count(
    dataset_size: int,
    count: int,
) -> None:
    with pytest.raises(ValueError, match="count"):
        draw._sample_graph_indices(
            dataset_size=dataset_size,
            count=count,
            seed=42,
        )


def test_load_prepared_split_resolves_qm9_alias_and_only_reads_selected_split(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    expected = _molecular_graph(source_index=91)
    directory = _write_split_files(
        root,
        "qm9_attributed",
        selected_split="val",
        selected_graphs=[expected],
    )

    graphs, split_path, serialized_name = draw._load_prepared_dataset_split(
        "qm9",
        root,
        "val",
    )

    assert serialized_name == "qm9_attributed"
    assert split_path == directory / "val.pkl"
    assert len(graphs) == 1
    assert graphs[0].graph["source_index"] == 91


def test_load_prepared_split_uses_requested_split_local_order(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    expected = [
        _molecular_graph(source_index=700),
        _molecular_graph(source_index=300),
    ]
    directory = _write_split_files(
        root,
        "custom_molecules",
        selected_split="test",
        selected_graphs=expected,
    )

    graphs, split_path, serialized_name = draw._load_prepared_dataset_split(
        "custom_molecules",
        root,
        "test",
    )

    assert serialized_name == "custom_molecules"
    assert split_path == directory / "test.pkl"
    assert [graph.graph["source_index"] for graph in graphs] == [700, 300]


def test_load_prepared_split_rejects_an_unknown_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="split"):
        draw._load_prepared_dataset_split("qm9_attributed", tmp_path, "holdout")


def test_molecular_detection_requires_complete_node_and_edge_evidence() -> None:
    empty = nx.Graph()
    generic_singleton = nx.empty_graph(1)

    molecular_singleton = nx.Graph()
    molecular_singleton.add_node("carbon", atomic_num=6)

    partial_nodes = nx.Graph()
    partial_nodes.add_node(0, atomic_num=6)
    partial_nodes.add_node(1)
    partial_nodes.add_edge(0, 1, bond_type=1)

    missing_bond_type = nx.Graph()
    missing_bond_type.add_node(0, atomic_num=6)
    missing_bond_type.add_node(1, atomic_num=8)
    missing_bond_type.add_edge(0, 1)

    assert draw._is_molecular_graph(_molecular_graph()) is True
    assert draw._is_molecular_graph(molecular_singleton) is True
    assert draw._is_molecular_graph(empty) is False
    assert draw._is_molecular_graph(generic_singleton) is False
    assert draw._is_molecular_graph(partial_nodes) is False
    assert draw._is_molecular_graph(missing_bond_type) is False
    assert draw._is_molecular_graph(object()) is False


def test_load_from_prepared_graph_converts_attributes_and_propagates_source_index(
) -> None:
    pytest.importorskip("rdkit")

    molecule, info = draw._load_from_prepared_graph(
        _molecular_graph(source_index=1234),
        7,
        "qm9_attributed",
        "test",
    )

    assert molecule.GetNumAtoms() == 2
    assert molecule.GetBondBetweenAtoms(
        0, 1
    ).GetBondTypeAsDouble() == pytest.approx(2.0)
    assert info.dataset_index == 7
    assert info.source_index == 1234
    assert "qm9_attributed" in info.source
    assert "test" in info.source
    assert info.smiles != "<unavailable>"


def test_load_from_prepared_graph_leaves_absent_source_index_unknown() -> None:
    pytest.importorskip("rdkit")

    _molecule, info = draw._load_from_prepared_graph(
        _molecular_graph(),
        2,
        "qm9_attributed",
        "train",
    )

    assert info.dataset_index == 2
    assert info.source_index is None


@pytest.mark.parametrize(
    ("graph", "message"),
    [
        (nx.empty_graph(1), "atomic_num/atom_type"),
        (
            nx.Graph([(0, 1)]),
            "atomic_num/atom_type",
        ),
    ],
)
def test_load_from_prepared_graph_reports_missing_atom_attributes(
    graph: nx.Graph,
    message: str,
) -> None:
    pytest.importorskip("rdkit")

    with pytest.raises(ValueError, match=message):
        draw._load_from_prepared_graph(graph, 0, "qm9_topology", "test")


def test_load_from_prepared_graph_reports_missing_bond_type() -> None:
    pytest.importorskip("rdkit")
    graph = nx.Graph()
    graph.add_node(0, atomic_num=6)
    graph.add_node(1, atomic_num=8)
    graph.add_edge(0, 1)

    with pytest.raises(ValueError, match="bond_type"):
        draw._load_from_prepared_graph(graph, 0, "broken", "test")


@pytest.mark.parametrize("graph", _generic_graphs())
def test_generic_renderer_handles_empty_singleton_and_disconnected_graphs(
    graph: nx.Graph,
) -> None:
    pytest.importorskip("PIL")
    info = draw._generic_graph_info(graph, 4, "generic", "test")

    panel = draw._render_generic_graph_panel(
        graph,
        info,
        panel_width=260,
        panel_height=220,
        show_title=True,
        layout_seed=19,
    )

    assert panel.size == (260, 220)
    assert info.dataset_index == 4
    assert info.source_index == graph.graph["source_index"]
    assert f"nodes={graph.number_of_nodes()}" in info.smiles
    assert f"edges={graph.number_of_edges()}" in info.smiles


def test_generic_renderer_layout_is_deterministic_for_a_fixed_seed() -> None:
    pytest.importorskip("PIL")
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.path_graph(3))
    info = draw._generic_graph_info(graph, 2, "generic", "val")

    first = draw._render_generic_graph_panel(
        graph,
        info,
        panel_width=300,
        panel_height=240,
        show_title=True,
        layout_seed=73,
    )
    repeated = draw._render_generic_graph_panel(
        graph,
        info,
        panel_width=300,
        panel_height=240,
        show_title=True,
        layout_seed=73,
    )

    assert first.mode == repeated.mode
    assert first.size == repeated.size
    assert first.tobytes() == repeated.tobytes()


def test_induced_cycle_requires_the_subgraph_to_be_exactly_a_cycle() -> None:
    triangle = nx.cycle_graph(3)
    square = nx.cycle_graph(4)
    square_with_chord = square.copy()
    square_with_chord.add_edge(0, 2)

    assert draw._is_induced_cycle(triangle, tuple(triangle.nodes())) is True
    assert draw._is_induced_cycle(square, tuple(square.nodes())) is True
    assert (
        draw._is_induced_cycle(
            square_with_chord,
            tuple(square_with_chord.nodes()),
        )
        is False
    )
    assert draw._is_induced_cycle(nx.path_graph(4), (0, 1, 2, 3)) is False


def test_cycle_graphlet_histogram_counts_normalizes_and_sorts() -> None:
    graphs = [nx.cycle_graph(3), nx.cycle_graph(3), nx.cycle_graph(4)]

    rows = draw._cycle_graphlet_histogram(graphs, k_min=3, k_max=4)

    assert [(row.order, row.count) for row in rows] == [(3, 2), (4, 1)]
    assert [row.frequency for row in rows] == pytest.approx([2 / 3, 1 / 3])
    assert rows[0].possible_subsets == 6
    assert rows[0].subset_rate == pytest.approx(2 / 6)
    assert rows[1].possible_subsets == 1
    assert rows[1].subset_rate == pytest.approx(1.0)


def test_chorded_cycle_is_not_counted_as_a_cycle_graphlet() -> None:
    complete = nx.complete_graph(4)

    rows = draw._cycle_graphlet_histogram([complete], k_min=3, k_max=4)
    by_order = {row.order: row for row in rows}

    assert by_order[3].count == 4
    assert by_order[4].count == 0


class _FakeCanvas:
    def __init__(self) -> None:
        self.saved: list[Path] = []

    def save(self, path: str | Path) -> None:
        self.saved.append(Path(path))


def test_cli_samples_and_propagates_actual_split_local_indices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graphs = [_molecular_graph(source_index=100 + index) for index in range(12)]
    loaded_call: dict[str, Any] = {}
    converted: list[tuple[Any, int, str, str]] = []
    rendered_items: list[draw.LoadedItem] = []
    canvas = _FakeCanvas()

    def fake_load(dataset: str, root: str | Path, split: str):
        loaded_call.update(dataset=dataset, root=Path(root), split=split)
        return (
            graphs,
            Path(root) / "qm9_attributed" / f"{split}.pkl",
            "qm9_attributed",
            [(split, index) for index in range(len(graphs))],
        )

    def fake_convert(graph: Any, index: int, dataset_name: str, split: str):
        converted.append((graph, index, dataset_name, split))
        return object(), draw.MoleculeInfo(
            source=f"{dataset_name}/{split}",
            name=f"molecule {index}",
            smiles="CO",
            dataset_index=index,
            source_index=100 + index,
        )

    monkeypatch.setattr(draw, "_load_prepared_dataset_selection", fake_load)
    monkeypatch.setattr(draw, "_load_from_prepared_graph", fake_convert)
    monkeypatch.setattr(draw, "_prepare_molecule", lambda molecule, **_kwargs: molecule)

    def fake_compose(items: list[draw.LoadedItem], *_args, **_kwargs):
        rendered_items.extend(items)
        return canvas

    monkeypatch.setattr(draw, "_compose_page", fake_compose)

    output = tmp_path / "prepared.png"
    result = draw.main(
        [
            "--dataset",
            "qm9",
            "--root",
            str(tmp_path / "datasets"),
            "--count",
            "5",
            "--seed",
            "17",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert loaded_call == {
        "dataset": "qm9",
        "root": tmp_path / "datasets",
        "split": "test",
    }
    expected_indices = draw._sample_graph_indices(
        dataset_size=len(graphs),
        count=5,
        seed=17,
    )
    assert [row[1] for row in converted] == expected_indices
    assert len(set(expected_indices)) == 5
    assert all(
        graph is graphs[index]
        and dataset_name == "qm9_attributed"
        and split == "test"
        for graph, index, dataset_name, split in converted
    )
    assert [item.info.dataset_index for item in rendered_items] == expected_indices
    assert [item.info.source_index for item in rendered_items] == [
        100 + index for index in expected_indices
    ]
    assert canvas.saved == [output.resolve()]


def test_cli_dispatches_generic_graphs_without_rdkit_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graphs = [nx.path_graph(size) for size in range(2, 8)]
    for source_index, graph in enumerate(graphs, start=100):
        graph.graph["source_index"] = source_index
    rendered_items: list[draw.LoadedItem] = []
    canvas = _FakeCanvas()

    monkeypatch.setattr(
        draw,
        "_load_prepared_dataset_selection",
        lambda dataset, root, split: (
            graphs,
            Path(root) / dataset / f"{split}.pkl",
            dataset,
            [(split, index) for index in range(len(graphs))],
        ),
    )
    monkeypatch.setattr(
        draw,
        "_load_from_prepared_graph",
        lambda *_args, **_kwargs: pytest.fail(
            "generic graphs must bypass RDKit conversion"
        ),
    )
    monkeypatch.setattr(
        draw,
        "_prepare_molecule",
        lambda *_args, **_kwargs: pytest.fail(
            "generic graphs must bypass molecule preparation"
        ),
    )

    def fake_compose(items: list[draw.LoadedItem], *_args, **_kwargs):
        rendered_items.extend(items)
        return canvas

    monkeypatch.setattr(draw, "_compose_page", fake_compose)

    output = tmp_path / "generic.png"
    result = draw.main(
        [
            "--dataset",
            "generic",
            "--root",
            str(tmp_path / "datasets"),
            "--count",
            "4",
            "--seed",
            "29",
            "--output",
            str(output),
        ]
    )

    expected_indices = draw._sample_graph_indices(
        dataset_size=len(graphs),
        count=4,
        seed=29,
    )
    assert result == 0
    assert [item.info.dataset_index for item in rendered_items] == expected_indices
    assert [item.info.source_index for item in rendered_items] == [
        graphs[index].graph["source_index"] for index in expected_indices
    ]
    assert all(item.render_mode == "generic" for item in rendered_items)
    assert all(item.mol is None for item in rendered_items)
    assert all(
        item.graph is graphs[index]
        for item, index in zip(rendered_items, expected_indices)
    )
    assert all(item.error is None for item in rendered_items)
    assert canvas.saved == [output.resolve()]


def test_cli_draws_empty_singleton_and_disconnected_generic_graphs_end_to_end(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("PIL")
    root = tmp_path / "datasets"
    _write_split_files(
        root,
        "generic",
        selected_split="test",
        selected_graphs=_generic_graphs(),
    )
    output = tmp_path / "generic.png"

    result = draw.main(
        [
            "--dataset",
            "generic",
            "--dataset-root",
            str(root),
            "--split",
            "test",
            "--count",
            "3",
            "--seed",
            "42",
            "--row",
            "1",
            "--col",
            "3",
            "--panel-width",
            "220",
            "--panel-height",
            "220",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    captured = capsys.readouterr()
    assert "failed to load" not in captured.err.lower()


def test_cli_draws_all_generic_graphs_across_all_splits(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    root = tmp_path / "datasets"
    directory = root / "generic"
    directory.mkdir(parents=True)
    save_pickle([nx.path_graph(2), nx.path_graph(3)], directory / "train.pkl")
    save_pickle([nx.path_graph(4)], directory / "val.pkl")
    save_pickle([nx.path_graph(5), nx.path_graph(6)], directory / "test.pkl")
    output = tmp_path / "all.png"

    result = draw.main(
        [
            "--dataset",
            "generic",
            "--dataset-root",
            str(root),
            "--split",
            "all",
            "--all",
            "--row",
            "1",
            "--col",
            "2",
            "--panel-width",
            "220",
            "--panel-height",
            "220",
            "--k-min",
            "3",
            "--k-max",
            "4",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    pages = sorted(tmp_path.glob("all_page_*.png"))
    assert [path.name for path in pages] == [
        "all_page_001.png",
        "all_page_002.png",
        "all_page_003.png",
    ]
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in pages)
    histogram = tmp_path / "all_graphlet_histogram.png"
    report = tmp_path / "all_graphlet_histogram.json"
    assert histogram.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["definition"] == "induced_simple_cycle_Ck"
    assert payload["k_min"] == 3
    assert payload["k_max"] == 4
    assert payload["total_cycle_graphlets"] == 0
    assert [row["graphlet"] for row in payload["graphlets"]] == ["C3", "C4"]


def test_graphlet_frequencies_use_full_dataset_when_drawing_one_graph(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    directory = tmp_path / "datasets" / "generic"
    directory.mkdir(parents=True)
    save_pickle([nx.cycle_graph(3), nx.cycle_graph(3)], directory / "train.pkl")
    save_pickle([nx.cycle_graph(4)], directory / "val.pkl")
    save_pickle([nx.path_graph(3), nx.cycle_graph(4)], directory / "test.pkl")

    for seed in (1, 42):
        output = tmp_path / f"sample_{seed}.png"
        assert draw.main([
            "--dataset", "generic", "--root", str(directory.parent),
            "--split", "test", "--count", "1", "--seed", str(seed),
            "--k-min", "3", "--k-max", "4", "--output", str(output),
        ]) == 0
        payload = json.loads(
            output.with_name(f"sample_{seed}_graphlet_histogram.json").read_text()
        )
        assert payload["split"] == "all"
        assert payload["selected_graphs"] == 5
        assert payload["drawn_graphs"] == 1
        assert payload["drawn_split"] == "test"
        assert payload["total_cycle_graphlets"] == 4
        assert [(row["count"], row["frequency"]) for row in payload["graphlets"]] == [
            (2, 0.5), (2, 0.5)
        ]


def test_cli_draws_a_prepared_molecule_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    pytest.importorskip("PIL")
    root = tmp_path / "datasets"
    _write_split_files(
        root,
        "molecules",
        selected_split="test",
        selected_graphs=[_molecular_graph(source_index=42)],
    )
    output = tmp_path / "molecule.png"

    result = draw.main(
        [
            "--dataset",
            "molecules",
            "--dataset-root",
            str(root),
            "--split",
            "test",
            "--row",
            "1",
            "--col",
            "1",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def _pdf_page_count(path: Path) -> int:
    pytest.importorskip("PIL")
    from PIL import PdfParser

    assert path.read_bytes().startswith(b"%PDF-")
    with PdfParser.PdfParser(filename=str(path)) as document:
        return len(document.pages)


def test_pdf_includes_all_graph_pages_and_full_dataset_graphlets(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    directory = tmp_path / "datasets" / "generic"
    directory.mkdir(parents=True)
    save_pickle([nx.cycle_graph(3), nx.cycle_graph(3)], directory / "train.pkl")
    save_pickle([nx.cycle_graph(4)], directory / "val.pkl")
    save_pickle([nx.cycle_graph(5), nx.path_graph(3), nx.path_graph(4)], directory / "test.pkl")
    output = tmp_path / "drawings.pdf"
    args = [
        "--dataset", "generic", "--root", str(directory.parent),
        "--split", "test", "--count", "3", "--seed", "42",
        "--row", "1", "--col", "2", "--k-min", "3", "--k-max", "5",
        "--output", str(output),
    ]
    # Rerunning replaces the document instead of appending a second copy.
    for _ in range(2):
        assert draw.main(args) == 0
        assert _pdf_page_count(output) == 3  # two graph pages, then graphlets
    assert list(tmp_path.glob("*.pdf")) == [output]
    assert not list(tmp_path.glob("*.png"))
    report = json.loads((tmp_path / "drawings_graphlet_histogram.json").read_text())
    assert report["split"] == "all"
    assert report["selected_graphs"] == 6
    assert report["drawn_graphs"] == 3
    assert report["total_cycle_graphlets"] == 4
    assert [(row["k"], row["count"]) for row in report["graphlets"]] == [(3, 2), (4, 1), (5, 1)]


@pytest.mark.parametrize("graphlet_filename", ["drawings.pdf", "cycles.pdf", "cycles.png"])
def test_pdf_can_also_export_separate_graphlet_drawing(tmp_path: Path, graphlet_filename: str) -> None:
    pytest.importorskip("PIL")
    directory = tmp_path / "datasets" / "generic"
    directory.mkdir(parents=True)
    for split in ("train", "val", "test"):
        save_pickle([nx.cycle_graph(3)], directory / f"{split}.pkl")
    output = tmp_path / "drawings.pdf"
    graphlet_output = tmp_path / graphlet_filename
    assert draw.main([
        "--dataset", "generic", "--root", str(directory.parent),
        "--count", "1", "--row", "1", "--col", "1",
        "--k-min", "3", "--k-max", "3", "--output", str(output),
        "--graphlet-output", str(graphlet_output),
    ]) == 0
    assert _pdf_page_count(output) == 2
    if graphlet_output != output:
        if graphlet_output.suffix == ".pdf":
            assert _pdf_page_count(graphlet_output) == 1
        else:
            assert graphlet_output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_pdf_draws_molecules_without_graphlet_page(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    pytest.importorskip("rdkit")
    root = tmp_path / "datasets"
    _write_split_files(
        root, "molecules", selected_split="test",
        selected_graphs=[_molecular_graph(source_index=42)],
    )
    output = tmp_path / "molecule.PDF"
    assert draw.main([
        "--dataset", "molecules", "--root", str(root),
        "--row", "1", "--col", "1", "--output", str(output),
    ]) == 0
    assert _pdf_page_count(output) == 1
    assert not list(tmp_path.glob("*.json"))


@pytest.mark.parametrize("option", ["--output", "--graphlet-output"])
def test_invalid_output_format_rejected_before_loading_dataset(option: str) -> None:
    with pytest.raises(ValueError, match=r"must use a .png or .pdf extension"):
        draw.main([
            "--dataset", "missing", "--k-min", "3", "--k-max", "5",
            option, "unsupported.svg",
        ])


def _molecular_cycle(order: int, *, invalid: bool = False) -> nx.Graph:
    graph = nx.cycle_graph(order)
    nx.set_node_attributes(graph, 6, "atomic_num")
    nx.set_edge_attributes(graph, 1, "bond_type")
    if invalid:
        graph.nodes[0]["atomic_num"] = 9  # Fluorine cannot have these two bonds.
    return graph


@pytest.mark.parametrize("split", ["test", "all"])
@pytest.mark.parametrize("draw_all", [False, True])
def test_molecular_sampling_and_graphlets_share_valid_pool(
    tmp_path: Path, monkeypatch, split: str, draw_all: bool,
) -> None:
    pytest.importorskip("rdkit")
    pytest.importorskip("PIL")
    directory = tmp_path / "datasets" / "molecules"
    directory.mkdir(parents=True)
    missing_bond = _molecular_cycle(3)
    del missing_bond.edges[0, 1]["bond_type"]
    splits = {
        "train": [_molecular_cycle(3), _molecular_cycle(3, invalid=True), nx.Graph()],
        "val": [_molecular_cycle(4), missing_bond],
        "test": [_molecular_cycle(3, invalid=True), _molecular_cycle(5),
                 _molecular_cycle(3), nx.cycle_graph(3)],
    }
    for name, graphs in splits.items():
        for index, graph in enumerate(graphs):
            graph.graph["source_index"] = 100 + index
        save_pickle(graphs, directory / f"{name}.pkl")
    rendered = []
    compose = draw._compose_page

    def capture(items, **kwargs):
        rendered.extend(items)
        return compose(items, **kwargs)

    monkeypatch.setattr(draw, "_compose_page", capture)
    output = tmp_path / "valid.pdf"
    assert draw.main([
        "--dataset", "molecules", "--root", str(directory.parent), "--split", split,
        *( ["--all"] if draw_all else ["--count", "1"] ),
        "--seed", "42", "--row", "1", "--col", "2",
        "--k-min", "3", "--k-max", "5", "--output", str(output),
    ]) == 0
    allowed = {"molecules/test[1]", "molecules/test[2]"}
    if split == "all":
        allowed.update({"molecules/train[0]", "molecules/val[0]"})
    assert len(rendered) == (len(allowed) if draw_all else 1)
    assert {item.info.index_label for item in rendered} <= allowed
    assert all(item.render_mode == "molecule" and item.error is None for item in rendered)
    assert all(item.info.source_index == 100 + item.info.dataset_index for item in rendered)
    assert _pdf_page_count(output) == (len(rendered) + 1) // 2 + 1
    report = json.loads((tmp_path / "valid_graphlet_histogram.json").read_text())
    assert report["valid_molecular_graphs_only"] is True
    assert report["total_dataset_graphs"] == 9
    assert report["selected_graphs"] == 4
    assert report["excluded_invalid_molecular_graphs"] == 5
    assert report["total_cycle_graphlets"] == 4
    assert report["total_eligible_node_subsets"] == 23
    assert [(row["k"], row["count"], row["frequency"]) for row in report["graphlets"]] == [
        (3, 2, 0.5), (4, 1, 0.25), (5, 1, 0.25),
    ]


@pytest.mark.parametrize("empty_pool", [False, True])
def test_molecular_sampling_rejects_insufficient_valid_pool(tmp_path: Path, empty_pool: bool) -> None:
    pytest.importorskip("rdkit")
    graphs = [_molecular_cycle(3, invalid=True)]
    if not empty_pool:
        graphs.append(_molecular_cycle(3))
    root = tmp_path / "datasets"
    _write_split_files(root, "qm9_attributed", selected_split="test", selected_graphs=graphs)
    output = tmp_path / "not_written.pdf"
    message = "No valid molecular graphs" if empty_pool else "exceeds the 1 valid molecular"
    with pytest.raises(ValueError, match=message):
        draw.main([
            "--dataset", "qm9_attributed", "--root", str(root),
            *( ["--all"] if empty_pool else ["--count", "2"] ),
            "--output", str(output),
        ])
    assert not output.exists()


def test_molecular_validation_preserves_explicit_and_projected_formal_charges() -> None:
    pytest.importorskip("rdkit")
    charged = _molecular_graph()
    charged.nodes[1]["formal_charge"] = -1
    charged.edges[0, 1].update(bond_type=1, bond_order=1.0)
    molecule = draw._validated_molecule(charged, label="charged")
    assert molecule.GetAtomWithIdx(1).GetFormalCharge() == -1

    projected = nx.star_graph(4)
    nx.set_node_attributes(projected, 6, "atomic_num")
    nx.set_edge_attributes(projected, 1, "bond_type")
    projected.nodes[0]["atomic_num"] = 7
    molecule = draw._validated_molecule(projected, label="projected")
    assert molecule.GetAtomWithIdx(0).GetFormalCharge() == 1


@pytest.mark.parametrize("kind", ["empty", "directed", "multi", "self_loop", "fractional_atom", "valence"])
def test_invalid_molecular_graphs_are_not_drawable(kind: str) -> None:
    pytest.importorskip("rdkit")
    graph = _molecular_cycle(3, invalid=kind == "valence")
    if kind == "empty":
        graph.clear()
    elif kind == "directed":
        graph = nx.DiGraph(graph)
    elif kind == "multi":
        graph = nx.MultiGraph(graph)
    elif kind == "self_loop":
        graph.add_edge(0, 0, bond_type=1)
    elif kind == "fractional_atom":
        graph.nodes[0]["atomic_num"] = 6.5
    with pytest.raises(ValueError):
        draw._load_from_prepared_graph(graph, 0, "molecules", "test")


def test_molecular_cycle_histogram_separates_atom_and_bond_types() -> None:
    carbon = _molecular_cycle(3)
    oxygen = _molecular_cycle(3)
    oxygen.nodes[0]["atomic_num"] = 8
    double = _molecular_cycle(3)
    double.edges[0, 1]["bond_type"] = 2
    rows = draw._cycle_graphlet_histogram(
        [carbon, carbon.copy(), oxygen, double], k_min=3, k_max=3, molecular=True,
    )
    assert len(rows) == 3
    assert [row.count for row in rows] == [2, 1, 1]
    assert [row.frequency for row in rows] == [0.5, 0.25, 0.25]
    assert len({row.canonical_key for row in rows}) == 3
    assert all(row.possible_subsets == 4 for row in rows)
    assert {row.node_types for row in rows} == {(6, 6, 6), (6, 6, 8)}
    assert {tuple(sorted(row.edge_types)) for row in rows} == {(1, 1, 1), (1, 1, 2)}


def test_typed_cycle_keys_and_sequences_are_invariant_to_node_relabeling() -> None:
    graph = _molecular_cycle(4)
    graph.nodes[1]["atomic_num"] = 7
    graph.nodes[3]["atomic_num"] = 8
    graph.edges[0, 1]["bond_type"] = 2
    graphs = []
    for order in itertools.permutations(range(4)):
        relabeled = nx.Graph()
        relabeled.add_nodes_from((f"node-{node}", graph.nodes[order[node]]) for node in range(4))
        inverse = {old: f"node-{new}" for new, old in enumerate(order)}
        relabeled.add_edges_from((inverse[u], inverse[v], data) for u, v, data in graph.edges(data=True))
        graphs.append(relabeled)
    rows = draw._cycle_graphlet_histogram(graphs, k_min=4, k_max=4, molecular=True)
    assert len(rows) == 1
    assert rows[0].count == 24
    row = rows[0]
    # The exported sequences reconstruct exactly the canonical typed cycle.
    reconstructed = nx.cycle_graph(4)
    nx.set_node_attributes(reconstructed, dict(enumerate(row.node_types)), "atomic_num")
    for index, edge_type in enumerate(row.edge_types):
        reconstructed.edges[index, (index + 1) % row.order]["bond_type"] = edge_type
    assert draw.canonicalize_attributed_simple_cycle(
        reconstructed, node_label_attr="atomic_num", edge_label_attr="bond_type",
    ) == row.canonical_key


@pytest.mark.parametrize("different_nodes", [False, True])
def test_typed_cycles_distinguish_arrangements_with_the_same_type_counts(different_nodes: bool) -> None:
    adjacent, opposite = _molecular_cycle(4), _molecular_cycle(4)
    if different_nodes:
        for index in (0, 1):
            adjacent.nodes[index]["atomic_num"] = 7
        for index in (0, 2):
            opposite.nodes[index]["atomic_num"] = 7
    else:
        adjacent.edges[0, 1]["bond_type"] = adjacent.edges[1, 2]["bond_type"] = 2
        opposite.edges[0, 1]["bond_type"] = opposite.edges[2, 3]["bond_type"] = 2
    rows = draw._cycle_graphlet_histogram([adjacent, opposite], k_min=4, k_max=4, molecular=True)
    assert len(rows) == 2
    assert rows[0].canonical_key != rows[1].canonical_key
    assert sorted(rows[0].node_types) == sorted(rows[1].node_types)
    assert sorted(rows[0].edge_types) == sorted(rows[1].edge_types)


def test_typed_cycles_normalize_attribute_aliases_and_aromatic_bond_encoding() -> None:
    aromatic = _molecular_cycle(6)
    nx.set_edge_attributes(aromatic, 4, "bond_type")
    alias = nx.cycle_graph(6)
    nx.set_node_attributes(alias, 6.0, "atomic_number")
    nx.set_edge_attributes(alias, 1.5, "bond_order")
    rows = draw._cycle_graphlet_histogram(
        [aromatic, alias, _molecular_cycle(6)], k_min=6, k_max=6, molecular=True,
    )
    assert len(rows) == 2
    assert rows[0].count == 2
    assert rows[0].edge_types == (4,) * 6
    assert rows[1].edge_types == (1,) * 6


def test_typed_cycles_exclude_chords_and_disconnected_subsets() -> None:
    graph = _molecular_cycle(4)
    graph.add_edge(0, 2, bond_type=1)
    rows = draw._cycle_graphlet_histogram(
        [nx.disjoint_union(graph, _molecular_cycle(3))], k_min=3, k_max=5, molecular=True,
    )
    assert [(row.order, row.count) for row in rows] == [(3, 3)]
    assert draw._cycle_graphlet_histogram([graph], k_min=4, k_max=5, molecular=True) == []


@pytest.mark.parametrize("graphlet_extension", [".pdf", ".png"])
def test_typed_cycle_pages_and_report_cover_all_observed_types(tmp_path: Path, graphlet_extension: str) -> None:
    pytest.importorskip("rdkit")
    pytest.importorskip("PIL")
    graphs = []
    for atoms in list(itertools.combinations_with_replacement((6, 7, 8), 3))[:7]:
        graph = _molecular_cycle(3)
        nx.set_node_attributes(graph, dict(enumerate(atoms)), "atomic_num")
        graphs.append(graph)
    directory = tmp_path / "datasets" / "molecules"
    directory.mkdir(parents=True)
    save_pickle(graphs[:3], directory / "train.pkl")
    save_pickle(graphs[3:5], directory / "val.pkl")
    save_pickle(graphs[5:] + [graphs[0], _molecular_cycle(3, invalid=True)], directory / "test.pkl")
    output = tmp_path / "molecules.pdf"
    extra = tmp_path / f"cycles{graphlet_extension}"
    assert draw.main([
        "--dataset", "molecules", "--root", str(directory.parent),
        "--count", "1", "--row", "1", "--col", "1",
        "--k-min", "3", "--k-max", "5",
        "--output", str(output), "--graphlet-output", str(extra),
    ]) == 0
    assert _pdf_page_count(output) == 3
    if graphlet_extension == ".pdf":
        assert _pdf_page_count(extra) == 2
    else:
        pages = sorted(tmp_path.glob("cycles_page_*.png"))
        assert len(pages) == 2
        assert all(page.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for page in pages)
    report = json.loads(extra.with_suffix(".json").read_text())
    assert report["definition"] == "induced_atom_bond_typed_simple_cycle_Ck"
    assert report["graphlet_pages"] == 2
    assert report["total_cycle_graphlets"] == 8
    assert report["excluded_invalid_molecular_graphs"] == 1
    assert report["selected_graphs"] == 8
    rows = report["graphlets"]
    assert len(rows) == 7
    assert rows[0]["count"] == 2
    assert sum(row["frequency"] for row in rows) == pytest.approx(1.0)
    assert {tuple(row["node_types"]) for row in rows} == {
        tuple(graph.nodes[i]["atomic_num"] for i in range(3)) for graph in graphs
    }
