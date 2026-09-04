"""Neutral dataset/sample codecs for the external HOG-Diff wrapper.

GraphER never imports torch or RDKit from the HOG-Diff environment.  Prepared
NetworkX graphs are converted to simple pickle/NPZ representations in the
GraphER process; isolated workers materialize the native tensors expected by
HOG-Diff and export generated tensors back to NPZ.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class HOGDiffDatasetProfile:
    benchmark_id: str
    native_id: str
    config_name: str
    domain: str
    max_nodes: int
    atomic_numbers: tuple[int, ...] = ()


PROFILES: dict[str, HOGDiffDatasetProfile] = {
    "community_small": HOGDiffDatasetProfile(
        benchmark_id="community_small",
        native_id="community_small",
        config_name="cs.yaml",
        domain="generic",
        max_nodes=20,
    ),
    "ego_small": HOGDiffDatasetProfile(
        benchmark_id="ego_small",
        native_id="ego_small",
        config_name="ego.yaml",
        domain="generic",
        max_nodes=18,
    ),
    "qm9": HOGDiffDatasetProfile(
        benchmark_id="qm9",
        native_id="qm9",
        config_name="qm9.yaml",
        domain="attributed",
        max_nodes=9,
        atomic_numbers=(6, 7, 8, 9),
    ),
    "zinc": HOGDiffDatasetProfile(
        benchmark_id="zinc",
        native_id="zinc250k",
        config_name="zinc250k.yaml",
        domain="attributed",
        max_nodes=38,
        atomic_numbers=(6, 7, 8, 9, 15, 16, 17, 35, 53),
    ),
}


def profile_for(benchmark_id: str) -> HOGDiffDatasetProfile:
    try:
        return PROFILES[str(benchmark_id)]
    except KeyError as exc:
        raise ValueError(
            f"HOG-Diff supports {sorted(PROFILES)}; got {benchmark_id!r}."
        ) from exc


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


def _graph_nodes(graph: nx.Graph) -> list[Any]:
    # Preserve serialized node iteration order.  HOG-Diff is permutation
    # equivariant; no semantic node alignment is introduced by the codec.
    return list(graph.nodes())


def _validate_simple(graph: nx.Graph, *, label: str, max_nodes: int) -> None:
    if graph.is_directed():
        raise ValueError(f"{label} must be undirected.")
    if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError(f"{label} must be simple, not a multigraph.")
    if nx.number_of_selfloops(graph):
        raise ValueError(f"{label} contains a self-loop.")
    if graph.number_of_nodes() > int(max_nodes):
        raise ValueError(
            f"{label} has {graph.number_of_nodes()} nodes; HOG-Diff profile allows at most {max_nodes}."
        )


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
            raise ValueError(f"{label} has unsupported non-integral HOG-Diff bond order {order}.")
        value = int(round(order))
    else:
        raise ValueError(f"{label} is missing bond_type/bond_order.")
    if value not in {1, 2, 3}:
        raise ValueError(
            f"{label} has bond type {value}; the supplied HOG-Diff QM9/ZINC codec supports single/double/triple bonds only."
        )
    if "bond_order" in data and abs(float(data["bond_order"]) - float(value)) > 1.0e-8:
        raise ValueError(f"{label} has inconsistent bond_type and bond_order.")
    return value


def _test_split_fraction(test_count: int, train_count: int) -> float:
    total = int(test_count) + int(train_count)
    if total <= 0:
        raise ValueError("HOG-Diff generic export requires a non-empty train/test pool.")
    # HOG-Diff computes int(test_split * total).  A midpoint in the target
    # integer bin is robust to floating-point representation and makes the
    # first test_count graphs exactly the held-out prefix.
    fraction = (float(test_count) + 0.5) / float(total)
    if int(fraction * total) != int(test_count):
        fraction = float(test_count) / float(total)
        while int(fraction * total) < int(test_count):
            fraction = np.nextafter(fraction, 1.0)
    if int(fraction * total) != int(test_count):
        raise RuntimeError("Could not encode an exact HOG-Diff generic test split.")
    return float(fraction)


def export_generic_dataset(
    *,
    train_graphs: Sequence[nx.Graph],
    val_graphs: Sequence[nx.Graph],
    test_graphs: Sequence[nx.Graph],
    profile: HOGDiffDatasetProfile,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write HOG-Diff's native generic ``<name>.pkl`` without split leakage.

    The upstream generic dataloader treats the first ``int(test_split*N)``
    entries as test and every remaining entry as training.  GraphER therefore
    exports ``test + train`` and intentionally omits validation graphs from the
    optimization pool.
    """

    if profile.domain != "generic":
        raise ValueError("export_generic_dataset requires a generic profile.")
    destination = Path(output_dir).expanduser().resolve()
    native_dir = destination / profile.native_id
    native_dir.mkdir(parents=True, exist_ok=True)
    all_splits = {
        "train": list(train_graphs),
        "val": list(val_graphs),
        "test": list(test_graphs),
    }
    for split, graphs in all_splits.items():
        for index, graph in enumerate(graphs):
            _validate_simple(
                graph, label=f"{profile.benchmark_id}:{split}[{index}]", max_nodes=profile.max_nodes
            )
    upstream_pool = list(test_graphs) + list(train_graphs)
    raw_path = native_dir / f"{profile.native_id}.pkl"
    with raw_path.open("wb") as handle:
        pickle.dump(upstream_pool, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Preserve GraphER's exact three splits alongside the upstream projection.
    split_hashes: dict[str, str] = {}
    split_dir = native_dir / "grapher_splits"
    split_dir.mkdir(exist_ok=True)
    for split, graphs in all_splits.items():
        path = split_dir / f"{split}.pkl"
        with path.open("wb") as handle:
            pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        split_hashes[split] = _sha256(path)

    test_split = _test_split_fraction(len(test_graphs), len(train_graphs))
    manifest = {
        "format": "grapher_hogdiff_dataset_v1",
        "benchmark_id": profile.benchmark_id,
        "native_id": profile.native_id,
        "domain": profile.domain,
        "max_nodes": profile.max_nodes,
        "counts": {key: len(value) for key, value in all_splits.items()},
        "upstream_training_projection": {
            "path": f"{profile.native_id}/{profile.native_id}.pkl",
            "order": ["test", "train"],
            "validation_excluded": True,
            "test_split": test_split,
            "sha256": _sha256(raw_path),
        },
        "grapher_split_sha256": split_hashes,
    }
    _atomic_json(destination / "manifest.json", manifest)
    return manifest


def export_molecular_dataset(
    *,
    train_graphs: Sequence[nx.Graph],
    val_graphs: Sequence[nx.Graph],
    test_graphs: Sequence[nx.Graph],
    profile: HOGDiffDatasetProfile,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Export the GraphER training split as neutral HOG-Diff atom/bond tensors."""

    if profile.domain != "attributed":
        raise ValueError("export_molecular_dataset requires an attributed profile.")
    destination = Path(output_dir).expanduser().resolve()
    native_dir = destination / profile.native_id
    processed_dir = native_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    atom_to_channel = {value: index for index, value in enumerate(profile.atomic_numbers)}

    x = np.zeros(
        (len(train_graphs), profile.max_nodes, len(profile.atomic_numbers)), dtype=np.float32
    )
    adjacency = np.zeros(
        (len(train_graphs), profile.max_nodes, profile.max_nodes), dtype=np.float32
    )
    num_nodes = np.zeros((len(train_graphs),), dtype=np.int64)
    for graph_index, graph in enumerate(train_graphs):
        label = f"{profile.benchmark_id}:train[{graph_index}]"
        _validate_simple(graph, label=label, max_nodes=profile.max_nodes)
        nodes = _graph_nodes(graph)
        node_to_index = {node: index for index, node in enumerate(nodes)}
        num_nodes[graph_index] = len(nodes)
        for node, local in node_to_index.items():
            atomic_number = _atomic_number(graph.nodes[node], label=f"{label} node {node!r}")
            try:
                channel = atom_to_channel[atomic_number]
            except KeyError as exc:
                raise ValueError(
                    f"{label} has atomic number {atomic_number}; allowed HOG-Diff atoms are {profile.atomic_numbers}."
                ) from exc
            x[graph_index, local, channel] = 1.0
        for u, v, data in graph.edges(data=True):
            bond = _bond_type(data, label=f"{label} edge {(u, v)!r}")
            i, j = node_to_index[u], node_to_index[v]
            adjacency[graph_index, i, j] = adjacency[graph_index, j, i] = float(bond) / 3.0

    neutral_path = processed_dir / "grapher_atom_bond.npz"
    np.savez_compressed(
        neutral_path,
        x=x,
        adjacency=adjacency,
        num_nodes=num_nodes,
        sample_index=np.arange(len(train_graphs), dtype=np.int64),
    )

    # Preserve all three source splits for provenance but do not feed val/test to
    # HOG-Diff's optimizer.  This avoids the official raw CSV/index path and its
    # incompatible split semantics.
    split_hashes: dict[str, str] = {}
    split_dir = native_dir / "grapher_splits"
    split_dir.mkdir(exist_ok=True)
    for split, graphs in {
        "train": list(train_graphs),
        "val": list(val_graphs),
        "test": list(test_graphs),
    }.items():
        for index, graph in enumerate(graphs):
            _validate_simple(graph, label=f"{profile.benchmark_id}:{split}[{index}]", max_nodes=profile.max_nodes)
        path = split_dir / f"{split}.pkl"
        with path.open("wb") as handle:
            pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        split_hashes[split] = _sha256(path)

    manifest = {
        "format": "grapher_hogdiff_dataset_v1",
        "benchmark_id": profile.benchmark_id,
        "native_id": profile.native_id,
        "domain": profile.domain,
        "max_nodes": profile.max_nodes,
        "atomic_numbers": list(profile.atomic_numbers),
        "bond_types": [1, 2, 3],
        "counts": {
            "train": len(train_graphs),
            "val": len(val_graphs),
            "test": len(test_graphs),
        },
        "upstream_training_projection": {
            "path": f"{profile.native_id}/processed/grapher_atom_bond.npz",
            "training_split_only": True,
            "sha256": _sha256(neutral_path),
        },
        "grapher_split_sha256": split_hashes,
    }
    _atomic_json(destination / "manifest.json", manifest)
    return manifest


def export_dataset(
    *,
    train_graphs: Sequence[nx.Graph],
    val_graphs: Sequence[nx.Graph],
    test_graphs: Sequence[nx.Graph],
    profile: HOGDiffDatasetProfile,
    output_dir: str | Path,
) -> dict[str, Any]:
    if profile.domain == "generic":
        return export_generic_dataset(
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            test_graphs=test_graphs,
            profile=profile,
            output_dir=output_dir,
        )
    return export_molecular_dataset(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        test_graphs=test_graphs,
        profile=profile,
        output_dir=output_dir,
    )


def load_generated_export(
    path: str | Path,
    *,
    profile: HOGDiffDatasetProfile,
) -> list[nx.Graph]:
    """Decode a neutral HOG-Diff worker NPZ into GraphER NetworkX graphs."""

    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as payload:
        required = {"adjacency", "num_nodes", "sample_index"}
        if profile.domain == "attributed":
            required.add("node_types")
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"HOG-Diff export is missing arrays: {sorted(missing)}.")
        adjacency = np.asarray(payload["adjacency"])
        num_nodes = np.asarray(payload["num_nodes"])
        sample_index = np.asarray(payload["sample_index"])
        node_types = np.asarray(payload["node_types"]) if "node_types" in payload.files else None

    if adjacency.ndim != 3 or adjacency.shape[1:] != (profile.max_nodes, profile.max_nodes):
        raise ValueError(
            f"Invalid HOG-Diff adjacency shape {adjacency.shape}; expected [B,{profile.max_nodes},{profile.max_nodes}]."
        )
    count = adjacency.shape[0]
    if num_nodes.shape != (count,) or sample_index.shape != (count,):
        raise ValueError("Invalid HOG-Diff generated vector shapes.")
    if not np.array_equal(sample_index, np.arange(count)):
        raise ValueError("HOG-Diff generated sample order is not contiguous.")
    if profile.domain == "attributed" and (
        node_types is None or node_types.shape != (count, profile.max_nodes)
    ):
        raise ValueError("Invalid HOG-Diff generated node_types shape.")

    graphs: list[nx.Graph] = []
    for index in range(count):
        n = int(num_nodes[index])
        if n < 1 or n > profile.max_nodes:
            raise ValueError(f"Invalid HOG-Diff num_nodes={n} for sample {index}.")
        raw = np.asarray(adjacency[index])
        if profile.domain == "generic":
            if not np.all(np.isin(raw, (0, 1))):
                raise ValueError(f"Generic HOG-Diff sample {index} is not binary.")
        else:
            if not np.all(np.isin(raw, (0, 1, 2, 3))):
                raise ValueError(f"Molecular HOG-Diff sample {index} has unsupported bond classes.")
        if not np.array_equal(raw, raw.T):
            raise ValueError(f"HOG-Diff sample {index} adjacency is not symmetric.")
        if np.any(np.diag(raw) != 0):
            raise ValueError(f"HOG-Diff sample {index} contains a self-loop.")
        if n < profile.max_nodes and (np.any(raw[n:, :]) or np.any(raw[:, n:])):
            raise ValueError(f"HOG-Diff sample {index} has edges outside num_nodes={n}.")

        graph = nx.Graph()
        if profile.domain == "generic":
            graph.add_nodes_from(range(n))
        else:
            assert node_types is not None
            for node in range(n):
                channel = int(node_types[index, node])
                if channel < 0 or channel >= len(profile.atomic_numbers):
                    raise ValueError(
                        f"HOG-Diff sample {index} node {node} has atom channel {channel}."
                    )
                atomic_number = int(profile.atomic_numbers[channel])
                graph.add_node(node, atomic_num=atomic_number, atom_type=atomic_number)
        for u in range(n):
            for v in range(u + 1, n):
                category = int(raw[u, v])
                if not category:
                    continue
                if profile.domain == "generic":
                    graph.add_edge(u, v)
                else:
                    graph.add_edge(u, v, bond_type=category, bond_order=float(category))
        graph.graph.update(
            {
                "generator": "hog_diff",
                "hog_diff_sample_index": index,
                "hog_diff_native_dataset": profile.native_id,
                "hog_diff_raw_num_nodes": n,
            }
        )
        graphs.append(graph)
    return graphs
