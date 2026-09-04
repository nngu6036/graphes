"""Neutral dataset and generated-sample codecs for the GDSS wrapper.

The GraphER process never imports the legacy GDSS PyTorch stack.  Prepared
NetworkX graphs are projected to numeric NPZ files, isolated workers train and
sample GDSS, and generated numeric arrays are validated here before publishing
GraphER-facing NetworkX graphs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class GDSSDatasetProfile:
    benchmark_id: str
    native_id: str
    config_name: str
    domain: str
    max_nodes: int
    max_feat_num: int
    sampling_config_name: str | None = None
    atomic_numbers: tuple[int, ...] = ()


PROFILES: dict[str, GDSSDatasetProfile] = {
    "community_small": GDSSDatasetProfile(
        "community_small", "community_small", "community_small.yaml", "generic", 20, 10
    ),
    "ego_small": GDSSDatasetProfile(
        "ego_small", "ego_small", "ego_small.yaml", "generic", 18, 17
    ),
    "grid": GDSSDatasetProfile("grid", "grid", "grid.yaml", "generic", 361, 5),
    "qm9": GDSSDatasetProfile(
        "qm9", "QM9", "qm9.yaml", "attributed", 9, 4,
        sampling_config_name="sample_qm9.yaml", atomic_numbers=(6, 7, 8, 9)
    ),
    "zinc": GDSSDatasetProfile(
        "zinc", "ZINC250k", "zinc250k.yaml", "attributed", 38, 9,
        sampling_config_name="sample_zinc250k.yaml",
        atomic_numbers=(6, 7, 8, 9, 15, 16, 17, 35, 53),
    ),
}


def profile_for(benchmark_id: str) -> GDSSDatasetProfile:
    try:
        return PROFILES[str(benchmark_id)]
    except KeyError as exc:
        raise ValueError(f"GDSS supports {sorted(PROFILES)}; got {benchmark_id!r}.") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_simple(graph: nx.Graph, *, label: str, max_nodes: int) -> None:
    if graph.is_directed():
        raise ValueError(f"{label} must be undirected.")
    if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError(f"{label} must be simple, not a multigraph.")
    if nx.number_of_selfloops(graph):
        raise ValueError(f"{label} contains a self-loop.")
    n = graph.number_of_nodes()
    if n < 1 or n > max_nodes:
        raise ValueError(f"{label} has {n} nodes; GDSS profile permits 1..{max_nodes}.")


def _atomic_number(data: Mapping[str, Any], *, label: str) -> int:
    raw = data.get("atomic_num", data.get("atom_type"))
    if raw is None:
        raise ValueError(f"{label} is missing atomic_num/atom_type.")
    return int(raw)


def _bond_type(data: Mapping[str, Any], *, label: str) -> int:
    if "bond_type" in data:
        value = int(data["bond_type"])
    elif "bond_order" in data:
        order = float(data["bond_order"])
        if abs(order - round(order)) > 1.0e-8:
            raise ValueError(f"{label} has unsupported non-integral bond order {order}.")
        value = int(round(order))
    else:
        raise ValueError(f"{label} is missing bond_type/bond_order.")
    if value not in {1, 2, 3}:
        raise ValueError(f"{label} has GDSS-unsupported bond type {value}; expected 1, 2, or 3.")
    return value


def _encode_split(
    graphs: Sequence[nx.Graph], *, profile: GDSSDatasetProfile, split: str, path: Path
) -> dict[str, Any]:
    count = len(graphs)
    adjacency = np.zeros((count, profile.max_nodes, profile.max_nodes), dtype=np.int8)
    num_nodes = np.zeros((count,), dtype=np.int64)
    node_types = (
        np.full((count, profile.max_nodes), -1, dtype=np.int16)
        if profile.domain == "attributed"
        else None
    )
    atom_to_channel = {value: index for index, value in enumerate(profile.atomic_numbers)}

    for graph_index, graph in enumerate(graphs):
        label = f"{profile.benchmark_id}:{split}[{graph_index}]"
        _validate_simple(graph, label=label, max_nodes=profile.max_nodes)
        nodes = list(graph.nodes())
        local = {node: index for index, node in enumerate(nodes)}
        num_nodes[graph_index] = len(nodes)
        if profile.domain == "attributed":
            assert node_types is not None
            for node, index in local.items():
                atomic_number = _atomic_number(graph.nodes[node], label=f"{label} node {node!r}")
                if atomic_number not in atom_to_channel:
                    raise ValueError(
                        f"{label} has atomic number {atomic_number}; allowed atoms are {profile.atomic_numbers}."
                    )
                node_types[graph_index, index] = atom_to_channel[atomic_number]
        for u, v, data in graph.edges(data=True):
            i, j = local[u], local[v]
            category = 1 if profile.domain == "generic" else _bond_type(data, label=f"{label} edge {(u, v)!r}")
            adjacency[graph_index, i, j] = adjacency[graph_index, j, i] = category

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "adjacency": adjacency,
        "num_nodes": num_nodes,
        "sample_index": np.arange(count, dtype=np.int64),
    }
    if node_types is not None:
        payload["node_types"] = node_types
    np.savez_compressed(path, **payload)
    return {"path": path.name, "sha256": _sha256(path), "count": count}


def export_dataset(
    *,
    train_graphs: Sequence[nx.Graph],
    val_graphs: Sequence[nx.Graph],
    test_graphs: Sequence[nx.Graph],
    profile: GDSSDatasetProfile,
    output_dir: str | Path,
) -> dict[str, Any]:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    split_records: dict[str, Any] = {}
    for split, graphs in (
        ("train", train_graphs), ("val", val_graphs), ("test", test_graphs)
    ):
        split_records[split] = _encode_split(
            graphs, profile=profile, split=split, path=destination / f"{split}.npz"
        )
    manifest = {
        "format": "grapher_gdss_dataset_v1",
        "benchmark_id": profile.benchmark_id,
        "native_id": profile.native_id,
        "domain": profile.domain,
        "max_nodes": profile.max_nodes,
        "max_feat_num": profile.max_feat_num,
        "atomic_numbers": list(profile.atomic_numbers),
        "bond_types": [1, 2, 3] if profile.domain == "attributed" else [],
        "splits": split_records,
        "training_projection": {
            "optimizer_split": "train",
            "monitor_split": "val",
            "test_used_during_training": False,
        },
    }
    _atomic_json(destination / "manifest.json", manifest)
    return manifest


def load_generated_export(path: str | Path, *, profile: GDSSDatasetProfile) -> list[nx.Graph]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing GDSS neutral export: {source}")
    with np.load(source, allow_pickle=False) as payload:
        required = {"adjacency", "num_nodes", "sample_index"}
        if profile.domain == "attributed":
            required.add("node_types")
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"GDSS export is missing arrays: {sorted(missing)}.")
        adjacency = np.asarray(payload["adjacency"])
        num_nodes = np.asarray(payload["num_nodes"])
        sample_index = np.asarray(payload["sample_index"])
        node_types = np.asarray(payload["node_types"]) if "node_types" in payload.files else None

    if adjacency.ndim != 3 or adjacency.shape[1:] != (profile.max_nodes, profile.max_nodes):
        raise ValueError(
            f"Invalid GDSS adjacency shape {adjacency.shape}; expected [B,{profile.max_nodes},{profile.max_nodes}]."
        )
    count = int(adjacency.shape[0])
    if num_nodes.shape != (count,) or sample_index.shape != (count,):
        raise ValueError("Invalid GDSS generated vector shapes.")
    if not np.array_equal(sample_index, np.arange(count, dtype=sample_index.dtype)):
        raise ValueError("GDSS generated sample order is not contiguous.")
    if profile.domain == "attributed" and (
        node_types is None or node_types.shape != (count, profile.max_nodes)
    ):
        raise ValueError("Invalid GDSS generated node_types shape.")

    graphs: list[nx.Graph] = []
    for graph_index in range(count):
        n = int(num_nodes[graph_index])
        min_nodes = 1 if profile.domain == "generic" else 0
        if n < min_nodes or n > profile.max_nodes:
            raise ValueError(f"GDSS sample {graph_index} has invalid num_nodes={n}.")
        raw = np.asarray(adjacency[graph_index])
        allowed = (0, 1) if profile.domain == "generic" else (0, 1, 2, 3)
        if not np.all(np.isin(raw, allowed)):
            raise ValueError(f"GDSS sample {graph_index} contains unsupported edge categories.")
        if not np.array_equal(raw, raw.T):
            raise ValueError(f"GDSS sample {graph_index} adjacency is not symmetric.")
        if np.any(np.diag(raw) != 0):
            raise ValueError(f"GDSS sample {graph_index} contains a self-loop.")
        if n < profile.max_nodes and (np.any(raw[n:, :]) or np.any(raw[:, n:])):
            raise ValueError(f"GDSS sample {graph_index} has edges outside num_nodes={n}.")

        graph = nx.Graph()
        if profile.domain == "generic":
            graph.add_nodes_from(range(n))
        else:
            assert node_types is not None
            for node in range(n):
                channel = int(node_types[graph_index, node])
                if channel < 0 or channel >= len(profile.atomic_numbers):
                    raise ValueError(
                        f"GDSS sample {graph_index} node {node} has invalid atom channel {channel}."
                    )
                atomic_number = int(profile.atomic_numbers[channel])
                graph.add_node(node, atomic_num=atomic_number, atom_type=atomic_number)
        for u in range(n):
            for v in range(u + 1, n):
                edge_type = int(raw[u, v])
                if not edge_type:
                    continue
                if profile.domain == "generic":
                    graph.add_edge(u, v)
                else:
                    graph.add_edge(u, v, bond_type=edge_type, bond_order=float(edge_type))
        graph.graph.update(
            {
                "generator": "gdss",
                "gdss_sample_index": graph_index,
                "gdss_native_dataset": profile.native_id,
                "gdss_raw_num_nodes": n,
                "molecular_representation": (
                    "heavy_atom_graph" if profile.domain == "attributed" else None
                ),
            }
        )
        graphs.append(graph)
    return graphs
