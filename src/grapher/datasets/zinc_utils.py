from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import networkx as nx


def is_zinc_dataset(dataset: str | None) -> bool:
    return str(dataset or "").lower() == "zinc"


def zinc_preparation_hint() -> str:
    return (
        "Prepare ZINC with scripts/prepare_zinc_from_smiles.py using an RDKit-readable "
        "SMILES CSV so node atomic numbers are persisted."
    )


def zinc_preparation_error(action: str) -> RuntimeError:
    return RuntimeError(f"Cannot {action} through the generic dataset builder. {zinc_preparation_hint()}")


def _iter_graphs(payload: Any) -> list[nx.Graph]:
    if isinstance(payload, Mapping):
        graphs: list[nx.Graph] = []
        for value in payload.values():
            graphs.extend(_iter_graphs(value))
        return graphs
    if isinstance(payload, nx.Graph):
        return [payload]
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return [g for g in payload if isinstance(g, nx.Graph)]
    return []


def assert_zinc_atomic_numbers(payload: Any, *, context: str = "ZINC preparation") -> dict[str, int]:
    graphs = _iter_graphs(payload)
    missing_nodes = 0
    total_nodes = 0
    for graph in graphs:
        for _, data in graph.nodes(data=True):
            total_nodes += 1
            if data.get("atomic_number", data.get("z")) is None:
                missing_nodes += 1
    if missing_nodes:
        raise ValueError(
            f"{context}: {missing_nodes}/{total_nodes} nodes are missing atomic_number or z. "
            + zinc_preparation_hint()
        )
    return {"num_graphs": len(graphs), "num_nodes": total_nodes, "missing_atomic_number_nodes": 0}
