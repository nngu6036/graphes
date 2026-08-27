from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from grapher.models.graphrnn.codec import (
    DATASET_EXPORT_FORMAT,
    GENERATION_EXPORT_FORMAT,
    export_graphrnn_dataset,
    load_graphrnn_export,
    write_generation_export,
)


def test_dataset_export_preserves_split_graphs_and_padding(tmp_path: Path) -> None:
    splits = {
        "train": [nx.path_graph(4), nx.cycle_graph(5)],
        "val": [nx.star_graph(3)],
        "test": [nx.complete_graph(3)],
    }
    output = tmp_path / "dataset.npz"
    manifest_path = tmp_path / "manifest.json"
    manifest = export_graphrnn_dataset(
        splits,
        output_path=output,
        manifest_path=manifest_path,
        benchmark_id="toy",
        max_num_node=6,
    )

    assert manifest["format"] == DATASET_EXPORT_FORMAT
    assert manifest["max_num_node"] == 6
    with np.load(output, allow_pickle=False) as payload:
        assert payload["train_adjacency"].shape == (2, 6, 6)
        assert payload["train_num_nodes"].tolist() == [4, 5]
        assert np.all(payload["train_adjacency"][0, 4:, :] == 0)
    assert json.loads(manifest_path.read_text())["format"] == DATASET_EXPORT_FORMAT


def test_generation_export_round_trip_keeps_order_and_empty_graph(tmp_path: Path) -> None:
    adjacency = np.zeros((3, 5, 5), dtype=np.uint8)
    adjacency[0, 0, 1] = adjacency[0, 1, 0] = 1
    triangle = nx.to_numpy_array(nx.complete_graph(3), dtype=np.uint8)
    adjacency[1, :3, :3] = triangle
    sizes = np.asarray([2, 3, 0], dtype=np.int64)
    output = tmp_path / "samples.npz"
    manifest_path = tmp_path / "samples.json"
    manifest = write_generation_export(
        output_path=output,
        manifest_path=manifest_path,
        adjacency=adjacency,
        num_nodes=sizes,
        metadata={"generation_seed": 7},
    )

    assert manifest["format"] == GENERATION_EXPORT_FORMAT
    graphs = load_graphrnn_export(output)
    assert [graph.number_of_nodes() for graph in graphs] == [2, 3, 0]
    assert [graph.number_of_edges() for graph in graphs] == [1, 3, 0]
    assert [graph.graph["graphrnn_sample_index"] for graph in graphs] == [0, 1, 2]


def test_generation_export_rejects_nonzero_padding(tmp_path: Path) -> None:
    adjacency = np.zeros((1, 4, 4), dtype=np.uint8)
    adjacency[0, 2, 3] = adjacency[0, 3, 2] = 1
    output = tmp_path / "bad.npz"
    with output.open("wb") as handle:
        np.savez_compressed(
            handle,
            adjacency=adjacency,
            num_nodes=np.asarray([2]),
            sample_index=np.asarray([0]),
        )
    with pytest.raises(ValueError, match="outside num_nodes"):
        load_graphrnn_export(output)
