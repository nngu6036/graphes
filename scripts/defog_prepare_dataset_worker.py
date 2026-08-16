#!/usr/bin/env python
"""Convert trusted GraphER generic-graph splits into DeFoG raw tensors.

This worker runs with the DeFoG interpreter.  It intentionally imports no
GraphER modules, which prevents DeFoG's bare ``models`` and ``datasets``
imports from colliding with the GraphER package.  The input files are project-
owned pickle artifacts and must therefore be treated as trusted inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FORMAT = "grapher_to_defog_generic_dataset_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_graphs(path: Path, *, split: str) -> list[Any]:
    # Pickle can execute code while loading.  The parent wrapper only supplies
    # prepared GraphER dataset files, never an arbitrary third-party pickle.
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, (list, tuple)) or not value:
        raise TypeError(f"{split}.pkl must contain a non-empty graph sequence.")
    return list(value)


def _convert_split(source: Path, destination: Path, *, split: str) -> dict[str, Any]:
    import networkx as nx
    import numpy as np
    import torch

    graphs = _load_graphs(source, split=split)
    tensors = []
    node_counts: list[int] = []
    edge_counts: list[int] = []
    for index, graph in enumerate(graphs):
        if not isinstance(graph, nx.Graph):
            raise TypeError(f"{split}[{index}] is not a NetworkX graph.")
        if graph.is_directed() or graph.is_multigraph():
            raise ValueError(f"{split}[{index}] must be simple and undirected.")
        if graph.number_of_nodes() <= 0:
            raise ValueError(f"{split}[{index}] has no nodes.")
        if nx.number_of_selfloops(graph):
            raise ValueError(f"{split}[{index}] contains a self-loop.")

        # Supplying nodelist explicitly preserves isolates and fixes the tensor
        # order to the graph's serialized node iteration order.
        nodes = list(graph.nodes())
        adjacency = nx.to_numpy_array(
            graph,
            nodelist=nodes,
            dtype=np.float32,
            weight=None,
        )
        if adjacency.shape != (len(nodes), len(nodes)):
            raise AssertionError("NetworkX returned an invalid adjacency shape.")
        if not np.array_equal(adjacency, adjacency.T):
            raise ValueError(f"{split}[{index}] adjacency is not symmetric.")
        if np.any(np.diag(adjacency) != 0) or not np.all(np.isin(adjacency, (0, 1))):
            raise ValueError(f"{split}[{index}] is not a binary loop-free graph.")
        tensors.append(torch.from_numpy(adjacency))
        node_counts.append(graph.number_of_nodes())
        edge_counts.append(graph.number_of_edges())

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(tensors, temporary)
    temporary.replace(destination)
    return {
        "source": {"path": str(source), "sha256": _sha256(source)},
        "output": {"path": str(destination), "sha256": _sha256(destination)},
        "graph_count": len(graphs),
        "node_count": {"min": min(node_counts), "max": max(node_counts)},
        "edge_count": {"min": min(edge_counts), "max": max(edge_counts)},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare GraphER generic splits for isolated DeFoG training."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = datetime.now(timezone.utc)
    output_root = args.output_root.expanduser().resolve()
    records = {}
    for split in ("train", "val", "test"):
        source = getattr(args, split).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Missing GraphER split: {source}")
        records[split] = _convert_split(
            source,
            output_root / "raw" / f"{split}.pt",
            split=split,
        )

    import networkx as nx
    import torch

    finished = datetime.now(timezone.utc)
    _atomic_json(
        args.manifest.expanduser().resolve(),
        {
            "format": FORMAT,
            "dataset": args.dataset,
            "splits": records,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
            "runtime": {
                "python": platform.python_version(),
                "networkx": nx.__version__,
                "torch": torch.__version__,
            },
        },
    )


if __name__ == "__main__":
    main()
