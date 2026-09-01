from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from grapher.data.statistics import (
    compute_prepared_dataset_statistics,
    resolve_prepared_dataset,
)
from grapher.utils.io import save_pickle
from scripts.print_dataset_statistics import main


def _write_splits(
    root: Path,
    dataset: str,
    *,
    train: list[nx.Graph] | None = None,
    val: list[nx.Graph] | None = None,
    test: list[nx.Graph] | None = None,
) -> Path:
    directory = root / dataset
    save_pickle(train or [], directory / "train.pkl")
    save_pickle(val or [], directory / "val.pkl")
    save_pickle(test or [], directory / "test.pkl")
    return directory


def test_reports_split_and_aggregate_graph_statistics(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    _write_splits(
        root,
        "toy",
        train=[nx.path_graph(3), nx.empty_graph(2)],
        val=[nx.cycle_graph(4)],
    )

    dataset = resolve_prepared_dataset("toy", root=root)
    report = compute_prepared_dataset_statistics(dataset)
    overall = report["statistics"]["all"]

    assert report["split_sizes"] == {"train": 2, "val": 1, "test": 0}
    assert overall["num_graphs"] == 3
    assert overall["total_nodes"] == 9
    assert overall["total_edges"] == 6
    assert overall["node_count"]["min"] == 2
    assert overall["node_count"]["max"] == 4
    assert overall["node_count"]["mean"] == pytest.approx(3.0)
    assert overall["edge_count"]["mean"] == pytest.approx(2.0)
    assert overall["degree_histogram"] == {"0": 2, "1": 2, "2": 5}
    assert overall["max_degree"] == 2
    assert overall["connected_rate"] == pytest.approx(2.0 / 3.0)
    assert overall["isolated_nodes"] == 2

    empty = report["statistics"]["test"]
    assert empty["num_graphs"] == 0
    assert empty["node_count"]["min"] is None
    assert empty["node_count"]["mean"] is None
    assert empty["connected_rate"] is None


def test_reports_molecular_categories_and_attribute_coverage(tmp_path: Path) -> None:
    molecule = nx.Graph()
    molecule.add_node(0, atomic_num=6, atom_type=6)
    molecule.add_node(1, atomic_num=8, atom_type=8)
    molecule.add_edge(0, 1, bond_type=2, bond_order=2.0)
    root = tmp_path / "datasets"
    _write_splits(root, "molecules", train=[molecule])

    dataset = resolve_prepared_dataset("molecules", root=root)
    overall = compute_prepared_dataset_statistics(dataset)["statistics"]["all"]
    molecular = overall["molecular_attributes"]

    assert molecular["atom_type_counts"] == {"6": 1, "8": 1}
    assert molecular["bond_type_counts"] == {"2": 1}
    assert molecular["atom_attribute_keys"] == {"atomic_num": 2}
    assert molecular["bond_attribute_keys"] == {"bond_type": 1}
    assert molecular["atom_attribute_coverage"] == 1.0
    assert molecular["bond_attribute_coverage"] == 1.0
    assert overall["node_attributes"]["fields"]["atomic_num"]["coverage"] == 1.0


def test_resolver_uses_config_alias_when_direct_dataset_is_absent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "community_small.yaml").write_text(
        "name: sbm\n", encoding="utf-8"
    )
    _write_splits(root, "sbm", train=[nx.path_graph(2)])

    resolved = resolve_prepared_dataset(
        "community_small", root=root, config_dir=config_dir
    )

    assert resolved.serialized_name == "sbm"
    assert resolved.resolution == "config_alias"
    assert resolved.config_path == config_dir / "community_small.yaml"


def test_resolver_prefers_an_exact_prepared_directory(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "qm9.yaml").write_text(
        "name: qm9_attributed\n", encoding="utf-8"
    )
    _write_splits(root, "qm9", train=[nx.path_graph(2)])
    _write_splits(root, "qm9_attributed", train=[nx.path_graph(3)])

    resolved = resolve_prepared_dataset("qm9", root=root, config_dir=config_dir)

    assert resolved.serialized_name == "qm9"
    assert resolved.resolution == "direct"


def test_missing_dataset_is_read_only_and_has_a_clear_error(tmp_path: Path) -> None:
    root = tmp_path / "datasets"

    with pytest.raises(FileNotFoundError, match="Prepared dataset 'missing'"):
        resolve_prepared_dataset("missing", root=root)

    assert not root.exists()


def test_cli_prints_table_and_optionally_saves_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "datasets"
    output = tmp_path / "statistics.json"
    _write_splits(root, "toy", train=[nx.path_graph(3)])

    assert (
        main(
            [
                "--dataset",
                "toy",
                "--root",
                str(root),
                "--json-out",
                str(output),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    assert "Dataset:    toy" in stdout
    assert "Processing train split:" in stdout
    assert "Processed test: 0 graphs" in stdout
    assert "nodes min/mean/max" in stdout
    assert "Degree histogram (all splits):" in stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["statistics"]["all"]["num_graphs"] == 1


def test_cli_cannot_overwrite_a_dataset_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "datasets"
    directory = _write_splits(root, "toy", train=[nx.path_graph(3)])
    train_path = directory / "train.pkl"
    original = train_path.read_bytes()

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--dataset",
                "toy",
                "--root",
                str(root),
                "--json-out",
                str(train_path),
                "--force",
            ]
        )

    assert "must be outside the prepared dataset directory" in capsys.readouterr().err
    assert train_path.read_bytes() == original


def test_split_payload_must_contain_networkx_graphs(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    _write_splits(root, "bad")
    save_pickle(["not-a-graph"], root / "bad" / "train.pkl")
    dataset = resolve_prepared_dataset("bad", root=root)

    with pytest.raises(TypeError, match="is not a NetworkX graph"):
        compute_prepared_dataset_statistics(dataset)


@pytest.mark.parametrize("dataset", ["../escape", "with/slash", ".", ""])
def test_dataset_name_must_be_one_safe_identifier(
    tmp_path: Path, dataset: str
) -> None:
    with pytest.raises(ValueError, match="dataset must be one identifier"):
        resolve_prepared_dataset(dataset, root=tmp_path)
