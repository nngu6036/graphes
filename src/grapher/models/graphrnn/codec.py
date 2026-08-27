"""Neutral numeric codecs used by the GraphRNN subprocess boundary.

The attached GraphRNN checkout predates current NetworkX and PyTorch releases.
GraphES therefore does not pass NetworkX pickles into the external interpreter.
Prepared graphs are converted to padded binary adjacency tensors, and generated
adjacency tensors are decoded back into NetworkX graphs only after validation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

DATASET_EXPORT_FORMAT = "grapher_graphrnn_dataset_v1"
GENERATION_EXPORT_FORMAT = "grapher_graphrnn_export_v1"
_SPLITS = ("train", "val", "test")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _as_simple_adjacency(graph: nx.Graph, *, split: str, index: int) -> np.ndarray:
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"{split}[{index}] is not a NetworkX graph.")
    if graph.is_directed():
        raise ValueError(f"{split}[{index}] is directed; GraphRNN expects undirected graphs.")
    if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError(f"{split}[{index}] is a multigraph; GraphRNN expects simple graphs.")
    if nx.number_of_selfloops(graph):
        raise ValueError(f"{split}[{index}] contains self-loops.")
    if graph.number_of_nodes() <= 0:
        raise ValueError(f"{split}[{index}] contains no nodes.")
    nodes = list(graph.nodes())
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, dtype=np.uint8, weight=None)
    adjacency = (adjacency != 0).astype(np.uint8, copy=False)
    np.fill_diagonal(adjacency, 0)
    if not np.array_equal(adjacency, adjacency.T):
        raise ValueError(f"{split}[{index}] produced a non-symmetric adjacency matrix.")
    return adjacency


def _split_summary(graphs: Sequence[nx.Graph]) -> dict[str, Any]:
    node_counts = [int(graph.number_of_nodes()) for graph in graphs]
    edge_counts = [int(graph.number_of_edges()) for graph in graphs]
    connected = [
        bool(nx.is_connected(graph)) if graph.number_of_nodes() > 0 else False
        for graph in graphs
    ]
    return {
        "graph_count": len(graphs),
        "min_nodes": min(node_counts) if node_counts else None,
        "max_nodes": max(node_counts) if node_counts else None,
        "min_edges": min(edge_counts) if edge_counts else None,
        "max_edges": max(edge_counts) if edge_counts else None,
        "connected_rate": (
            float(sum(connected) / len(connected)) if connected else None
        ),
    }


def export_graphrnn_dataset(
    splits: Mapping[str, Sequence[nx.Graph]],
    *,
    output_path: str | Path,
    manifest_path: str | Path,
    benchmark_id: str,
    max_num_node: int | None = None,
) -> dict[str, Any]:
    """Serialize prepared GraphES splits into one neutral padded NPZ file."""

    normalized: dict[str, list[nx.Graph]] = {}
    observed_max = 0
    for split in _SPLITS:
        values = splits.get(split)
        if values is None:
            raise KeyError(f"Missing prepared split {split!r}.")
        graphs = list(values)
        if not graphs:
            raise ValueError(f"Prepared split {split!r} is empty.")
        normalized[split] = graphs
        observed_max = max(
            observed_max,
            max(int(graph.number_of_nodes()) for graph in graphs),
        )

    resolved_max = observed_max if max_num_node is None else int(max_num_node)
    if resolved_max <= 0:
        raise ValueError("max_num_node must be positive.")
    if observed_max > resolved_max:
        raise ValueError(
            f"Observed a {observed_max}-node graph, exceeding max_num_node={resolved_max}."
        )

    arrays: dict[str, np.ndarray] = {}
    summaries: dict[str, Any] = {}
    for split, graphs in normalized.items():
        adjacency = np.zeros(
            (len(graphs), resolved_max, resolved_max), dtype=np.uint8
        )
        sizes = np.zeros(len(graphs), dtype=np.int64)
        edge_counts = np.zeros(len(graphs), dtype=np.int64)
        for index, graph in enumerate(graphs):
            matrix = _as_simple_adjacency(graph, split=split, index=index)
            n = int(matrix.shape[0])
            adjacency[index, :n, :n] = matrix
            sizes[index] = n
            edge_counts[index] = int(matrix.sum() // 2)
        arrays[f"{split}_adjacency"] = adjacency
        arrays[f"{split}_num_nodes"] = sizes
        arrays[f"{split}_num_edges"] = edge_counts
        summaries[split] = _split_summary(graphs)

    destination = Path(output_path)
    manifest_destination = Path(manifest_path)
    _atomic_npz(destination, **arrays)
    manifest = {
        "format": DATASET_EXPORT_FORMAT,
        "benchmark_id": str(benchmark_id),
        "representation": "simple_undirected_binary_adjacency",
        "node_order": "split_pickle_iteration_order_then_graph_node_iteration_order",
        "attributes": "topology_only",
        "max_num_node": resolved_max,
        "observed_max_num_node": observed_max,
        "splits": summaries,
        "output": {
            "path": destination.name,
            "sha256": sha256(destination),
        },
    }
    _atomic_json(manifest_destination, manifest)
    return manifest


def read_dataset_manifest(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    if value.get("format") != DATASET_EXPORT_FORMAT:
        raise RuntimeError(f"Unsupported GraphRNN dataset manifest: {value.get('format')!r}.")
    return value


def write_generation_export(
    *,
    output_path: str | Path,
    manifest_path: str | Path,
    adjacency: np.ndarray,
    num_nodes: np.ndarray,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish a validated numeric GraphRNN generation batch."""

    matrix = np.asarray(adjacency, dtype=np.uint8)
    sizes = np.asarray(num_nodes, dtype=np.int64)
    if matrix.ndim != 3 or matrix.shape[1] != matrix.shape[2]:
        raise ValueError("adjacency must have shape [num_graphs, max_nodes, max_nodes].")
    if sizes.shape != (matrix.shape[0],):
        raise ValueError("num_nodes must have one entry per generated graph.")
    destination = Path(output_path)
    manifest_destination = Path(manifest_path)
    _atomic_npz(
        destination,
        adjacency=matrix,
        num_nodes=sizes,
        sample_index=np.arange(matrix.shape[0], dtype=np.int64),
    )
    manifest = {
        "format": GENERATION_EXPORT_FORMAT,
        "num_generated": int(matrix.shape[0]),
        "max_num_node": int(matrix.shape[1]),
        "output": {"path": destination.name, "sha256": sha256(destination)},
        **dict(metadata),
    }
    _atomic_json(manifest_destination, manifest)
    return manifest


def load_graphrnn_export(path: str | Path) -> list[nx.Graph]:
    """Decode a neutral GraphRNN generation export into ordered graphs."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        required = {"adjacency", "num_nodes", "sample_index"}
        missing = required.difference(payload.files)
        if missing:
            raise RuntimeError(f"GraphRNN export is missing arrays: {sorted(missing)}.")
        adjacency = np.asarray(payload["adjacency"])
        sizes = np.asarray(payload["num_nodes"])
        sample_index = np.asarray(payload["sample_index"])

    if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2]:
        raise ValueError("Invalid GraphRNN adjacency tensor shape.")
    count, max_nodes, _ = adjacency.shape
    if sizes.shape != (count,) or sample_index.shape != (count,):
        raise ValueError("Invalid GraphRNN export vector shape.")
    if not np.array_equal(sample_index, np.arange(count)):
        raise ValueError("GraphRNN sample order is not contiguous and ascending.")

    graphs: list[nx.Graph] = []
    for index in range(count):
        n = int(sizes[index])
        if n < 0 or n > max_nodes:
            raise ValueError(f"Invalid num_nodes={n} for generated graph {index}.")
        raw = np.asarray(adjacency[index])
        if not np.all((raw == 0) | (raw == 1)):
            raise ValueError(f"Generated graph {index} has non-binary adjacency entries.")
        if not np.array_equal(raw, raw.T):
            raise ValueError(f"Generated graph {index} has non-symmetric adjacency.")
        if np.any(np.diag(raw) != 0):
            raise ValueError(f"Generated graph {index} contains a self-loop.")
        if n < max_nodes and (np.any(raw[n:, :]) or np.any(raw[:, n:])):
            raise ValueError(
                f"Generated graph {index} has non-zero entries outside num_nodes={n}."
            )
        graph = nx.from_numpy_array(raw[:n, :n], create_using=nx.Graph)
        graph.graph.update(
            {
                "generator": "graphrnn",
                "graphrnn_sample_index": index,
                "graphrnn_raw_num_nodes": n,
            }
        )
        graphs.append(graph)
    return graphs
