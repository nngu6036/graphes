from pathlib import Path

import networkx as nx
import pytest
import yaml

from grapher.data.io import save_dataset_splits, validate_prepared_dataset_contract


def _write_config(path: Path, *, train: float, val: float, test: float) -> None:
    payload = {
        "name": "fixture",
        "protocol_id": "fixture_protocol",
        "num_graphs": 10,
        "split": {"seed": 0, "train": train, "val": val, "test": test},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_prepared_dataset_contract_rejects_stale_split_sizes(tmp_path):
    root = tmp_path / "datasets"
    config_path = tmp_path / "fixture.yaml"
    _write_config(config_path, train=0.6, val=0.2, test=0.2)

    stale_config = {
        "name": "fixture",
        "protocol_id": "fixture_protocol",
        "num_graphs": 10,
        "split": {"seed": 0, "train": 0.7, "val": 0.1, "test": 0.2},
    }
    graphs = [nx.path_graph(4) for _ in range(10)]
    save_dataset_splits(
        "fixture",
        {"train": graphs[:7], "val": graphs[7:8], "test": graphs[8:]},
        stale_config,
        root,
    )

    with pytest.raises(RuntimeError, match="expected train/val/test=6/2/2"):
        validate_prepared_dataset_contract(
            "fixture", root=root, config_path=config_path
        )


def test_prepared_dataset_contract_accepts_matching_split_contract(tmp_path):
    root = tmp_path / "datasets"
    config_path = tmp_path / "fixture.yaml"
    _write_config(config_path, train=0.6, val=0.2, test=0.2)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    graphs = [nx.path_graph(4) for _ in range(10)]
    splits = {"train": graphs[:6], "val": graphs[6:8], "test": graphs[8:]}
    save_dataset_splits("fixture", splits, config, root)
    assert validate_prepared_dataset_contract(
        "fixture", root=root, config_path=config_path, splits=splits
    ) == []
