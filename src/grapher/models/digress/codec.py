"""Neutral, pickle-free DiGress graph-batch codec.

The external DiGress process exports compact numeric arrays.  GraphER validates
those arrays in its own environment before constructing NetworkX graphs.  This
keeps the incompatible upstream dependency stack outside the GraphER process
and avoids loading third-party pickle artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import networkx as nx
import numpy as np

DIGRESS_EXPORT_FORMAT: Final[str] = "digress_graph_batch_v1"
SUPPORTED_DATASETS: Final[frozenset[str]] = frozenset(
    {"comm20", "planar", "sbm", "qm9", "zinc"}
)
GENERIC_DATASETS: Final[frozenset[str]] = frozenset(
    {"comm20", "planar", "sbm"}
)
QM9_ATOMIC_NUMBERS: Final[tuple[int, ...]] = (6, 7, 8, 9)
ZINC_ATOMIC_NUMBERS: Final[tuple[int, ...]] = (
    6,
    7,
    8,
    9,
    15,
    16,
    17,
    35,
    53,
)
MOLECULAR_ATOMIC_NUMBERS: Final[dict[str, tuple[int, ...]]] = {
    "qm9": QM9_ATOMIC_NUMBERS,
    "zinc": ZINC_ATOMIC_NUMBERS,
}
MOLECULAR_BOND_TYPES: Final[dict[str, frozenset[int]]] = {
    "qm9": frozenset({1, 2, 3, 4}),
    "zinc": frozenset({1, 2, 3}),
}
BOND_ORDERS: Final[dict[int, float]] = {1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5}


def _integer_array(
    arrays: np.lib.npyio.NpzFile,
    name: str,
    *,
    ndim: int,
) -> np.ndarray:
    if name not in arrays.files:
        raise ValueError(f"DiGress export is missing array {name!r}.")
    value = np.asarray(arrays[name])
    if value.ndim != ndim:
        raise ValueError(
            f"DiGress export array {name!r} has shape {value.shape}; "
            f"expected {ndim} dimensions."
        )
    if value.dtype.kind not in {"i", "u"}:
        raise TypeError(
            f"DiGress export array {name!r} must be integral, got {value.dtype}."
        )
    return value.astype(np.int64, copy=False)


def _validate_offsets(offsets: np.ndarray, *, total: int, name: str) -> None:
    if offsets.ndim != 1 or offsets.size < 2:
        raise ValueError(f"{name} must contain at least [0, total].")
    if int(offsets[0]) != 0 or int(offsets[-1]) != int(total):
        raise ValueError(
            f"{name} endpoints are invalid: first={offsets[0]}, "
            f"last={offsets[-1]}, total={total}."
        )
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"{name} must be non-decreasing.")


def load_digress_export(path: str | Path, *, dataset: str) -> list[nx.Graph]:
    """Load one validated neutral export as ordered NetworkX graphs."""

    dataset_name = str(dataset).lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported DiGress dataset {dataset!r}; supported datasets are "
            f"{sorted(SUPPORTED_DATASETS)}."
        )
    export_path = Path(path)
    if not export_path.is_file():
        raise FileNotFoundError(f"Missing DiGress export: {export_path}")

    with np.load(export_path, allow_pickle=False) as arrays:
        node_offsets = _integer_array(arrays, "node_offsets", ndim=1)
        node_types = _integer_array(arrays, "node_types", ndim=1)
        edge_offsets = _integer_array(arrays, "edge_offsets", ndim=1)
        endpoints = _integer_array(arrays, "edge_endpoints", ndim=2)
        edge_types = _integer_array(arrays, "edge_types", ndim=1)
        if endpoints.shape[1:] != (2,):
            raise ValueError(
                "DiGress edge_endpoints must have shape (num_edges, 2), got "
                f"{endpoints.shape}."
            )
        if endpoints.shape[0] != edge_types.shape[0]:
            raise ValueError(
                "DiGress export has mismatched endpoint and edge-type counts."
            )
        _validate_offsets(node_offsets, total=node_types.size, name="node_offsets")
        _validate_offsets(edge_offsets, total=edge_types.size, name="edge_offsets")
        if node_offsets.size != edge_offsets.size:
            raise ValueError(
                "DiGress node_offsets and edge_offsets describe different graph counts."
            )

    graphs: list[nx.Graph] = []
    graph_count = int(node_offsets.size - 1)
    for graph_index in range(graph_count):
        node_start, node_end = (
            int(node_offsets[graph_index]),
            int(node_offsets[graph_index + 1]),
        )
        edge_start, edge_end = (
            int(edge_offsets[graph_index]),
            int(edge_offsets[graph_index + 1]),
        )
        local_node_types = node_types[node_start:node_end]
        local_endpoints = endpoints[edge_start:edge_end]
        local_edge_types = edge_types[edge_start:edge_end]
        n = int(local_node_types.size)
        if n <= 0:
            raise ValueError(f"DiGress graph {graph_index} has no nodes.")

        graph = nx.Graph()
        if dataset_name in GENERIC_DATASETS:
            if np.any(local_node_types != 0):
                raise ValueError(
                    f"Generic DiGress graph {graph_index} contains a nonzero "
                    "node category."
                )
            graph.add_nodes_from(range(n))
        else:
            atomic_numbers = MOLECULAR_ATOMIC_NUMBERS[dataset_name]
            if np.any(local_node_types < 0) or np.any(
                local_node_types >= len(atomic_numbers)
            ):
                raise ValueError(
                    f"{dataset_name.upper()} DiGress graph {graph_index} "
                    "contains an invalid atom class."
                )
            for node, atom_class in enumerate(local_node_types.tolist()):
                atomic_number = atomic_numbers[int(atom_class)]
                graph.add_node(
                    node,
                    atomic_num=atomic_number,
                    atom_type=atomic_number,
                )

        for (source, target), edge_type in zip(
            local_endpoints.tolist(),
            local_edge_types.tolist(),
        ):
            source = int(source)
            target = int(target)
            semantic_edge_type = int(edge_type)
            if not (0 <= source < n and 0 <= target < n):
                raise ValueError(
                    f"DiGress graph {graph_index} has endpoint outside [0, {n})."
                )
            if source >= target:
                raise ValueError(
                    f"DiGress graph {graph_index} edges must be stored once with "
                    "source < target."
                )
            if graph.has_edge(source, target):
                raise ValueError(
                    f"DiGress graph {graph_index} contains a duplicate edge."
                )
            if dataset_name in GENERIC_DATASETS:
                if semantic_edge_type != 1:
                    raise ValueError(
                        f"Generic DiGress graph {graph_index} has edge class "
                        f"{semantic_edge_type}; expected 1."
                    )
                graph.add_edge(source, target)
            else:
                if semantic_edge_type not in MOLECULAR_BOND_TYPES[dataset_name]:
                    raise ValueError(
                        f"{dataset_name.upper()} DiGress graph {graph_index} "
                        "has bond class "
                        f"{semantic_edge_type}."
                    )
                graph.add_edge(
                    source,
                    target,
                    bond_type=semantic_edge_type,
                    bond_order=BOND_ORDERS[semantic_edge_type],
                )

        if nx.number_of_selfloops(graph):
            raise ValueError(f"DiGress graph {graph_index} contains a self-loop.")
        graph.graph.update(
            {
                "generator": "digress",
                "digress_sample_index": graph_index,
                "molecular_dataset": (
                    dataset_name
                    if dataset_name in MOLECULAR_ATOMIC_NUMBERS
                    else None
                ),
                "molecular_representation": (
                    "heavy_atom_graph"
                    if dataset_name in MOLECULAR_ATOMIC_NUMBERS
                    else None
                ),
            }
        )
        graphs.append(graph)
    return graphs
