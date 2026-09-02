from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import pytest

from grapher.utils.io import save_pickle
from scripts import draw_qm9_molecule as draw


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
        return graphs, Path(root) / "qm9_attributed" / f"{split}.pkl", "qm9_attributed"

    def fake_convert(graph: Any, index: int, dataset_name: str, split: str):
        converted.append((graph, index, dataset_name, split))
        return object(), draw.MoleculeInfo(
            source=f"{dataset_name}/{split}",
            name=f"molecule {index}",
            smiles="CO",
            dataset_index=index,
            source_index=100 + index,
        )

    monkeypatch.setattr(draw, "_load_prepared_dataset_split", fake_load)
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
        "_load_prepared_dataset_split",
        lambda dataset, root, split: (
            graphs,
            Path(root) / dataset / f"{split}.pkl",
            dataset,
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
