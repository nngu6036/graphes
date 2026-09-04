from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from grapher.utils.io import save_pickle
from scripts import draw_generated_qm9_outliers as outliers


def _graph(index: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(0, atomic_num=6, atom_type=6)
    graph.add_node(1, atomic_num=8, atom_type=8)
    graph.add_edge(0, 1, bond_type=1, bond_order=1.0)
    graph.graph["source_index"] = index
    return graph


def test_parser_requires_exactly_one_generated_source() -> None:
    with pytest.raises(SystemExit, match="2"):
        outliers.build_parser().parse_args([])
    with pytest.raises(SystemExit, match="2"):
        outliers.build_parser().parse_args(
            ["--generated-dir", "a", "--generated-graphs", "b"]
        )

    args = outliers.build_parser().parse_args(["--generated-dir", "run"])
    assert args.dataset == "qm9_attributed"
    assert args.split == "train"
    assert args.count == 16
    assert args.ranking == "joint"


def test_load_generated_graphs_prefers_molecular_graphs(tmp_path: Path) -> None:
    directory = tmp_path / "generated"
    directory.mkdir()
    expected = [_graph(7)]
    save_pickle(expected, directory / "molecular_graphs.pkl")
    save_pickle([_graph(9)], directory / "generated_graphs.pkl")

    graphs, path = outliers._load_generated_graphs(
        generated_dir=directory,
        generated_graphs=None,
    )

    assert path == directory / "molecular_graphs.pkl"
    assert graphs[0].graph["source_index"] == 7


def test_fcd_mean_distances_use_chemnet_reference_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activations = {
        "ref-a": [0.0, 0.0],
        "ref-b": [2.0, 0.0],
        "near": [1.0, 0.0],
        "far": [4.0, 0.0],
    }

    class FakeFCD:
        def __init__(self, **_kwargs) -> None:
            pass

        def get_predictions(self, smiles):
            return np.asarray([activations[value] for value in smiles])

    monkeypatch.setattr(outliers, "_load_fcd_class", lambda: FakeFCD)
    monkeypatch.setattr(outliers, "resolve_torch_device", lambda _device: "cpu")

    scores = outliers.fcd_mean_distances(
        ["ref-a", "ref-b"],
        ["near", "far"],
        device="cpu",
    )

    np.testing.assert_allclose(scores, [0.0, 9.0])


def test_joint_ranking_requires_each_metric_to_be_extreme() -> None:
    graphs = [_graph(index) for index in range(4)]
    ranked = outliers.rank_outliers(
        [0, 1, 2, 3],
        graphs,
        ["a", "b", "c", "d"],
        fcd_scores=[100.0, 80.0, 20.0, 0.0],
        nspdk_scores=[0.0, 80.0, 100.0, 20.0],
        ranking="joint",
    )

    assert ranked[0].generated_index == 1
    assert ranked[0].ranking_score == pytest.approx(2.0 / 3.0)
    assert ranked[0].fcd_percentile == pytest.approx(2.0 / 3.0)
    assert ranked[0].nspdk_percentile == pytest.approx(2.0 / 3.0)


def test_ranking_falls_back_to_nspdk_when_fcd_is_skipped() -> None:
    graphs = [_graph(index) for index in range(3)]
    ranked = outliers.rank_outliers(
        [0, 1, 2],
        graphs,
        ["a", "b", "c"],
        fcd_scores=None,
        nspdk_scores=[1.0, 3.0, 2.0],
        ranking="joint",
    )

    assert [row.generated_index for row in ranked] == [1, 2, 0]
    assert all(row.fcd_mean_distance is None for row in ranked)
