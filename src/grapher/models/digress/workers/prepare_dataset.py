#!/usr/bin/env python
"""Convert trusted GraphER generic splits into DiGress PyG artifacts.

This worker is executed with the isolated DiGress interpreter.  It writes both
raw adjacency tensors and processed ``InMemoryDataset`` files so the attached
DiGress ``SpectreGraphDataset.process`` implementation is never invoked.  That
upstream implementation appends each graph twice in the provided snapshot.
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

FORMAT = "grapher_to_digress_generic_dataset_v1"
SPLITS = ("train", "val", "test")


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
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_torch_save(path: Path, value: Any, torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _load_graphs(path: Path, *, split: str) -> list[Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, (list, tuple)) or not value:
        raise TypeError(f"{split}.pkl must contain a non-empty graph sequence.")
    return list(value)


def _convert_split(
    source: Path,
    *,
    raw_destination: Path,
    processed_destination: Path,
    split: str,
) -> dict[str, Any]:
    import networkx as nx
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data, InMemoryDataset

    graphs = _load_graphs(source, split=split)
    adjacency_tensors: list[Any] = []
    data_list: list[Any] = []
    node_counts: list[int] = []
    edge_counts: list[int] = []

    for index, graph in enumerate(graphs):
        label = f"{split}[{index}]"
        if not isinstance(graph, nx.Graph):
            raise TypeError(f"{label} is not a NetworkX graph.")
        if graph.is_directed() or graph.is_multigraph():
            raise ValueError(f"{label} must be simple and undirected.")
        if graph.number_of_nodes() <= 0:
            raise ValueError(f"{label} has no nodes.")
        if nx.number_of_selfloops(graph):
            raise ValueError(f"{label} contains a self-loop.")

        nodes = list(graph.nodes())
        adjacency = nx.to_numpy_array(
            graph, nodelist=nodes, dtype=np.float32, weight=None
        )
        if adjacency.shape != (len(nodes), len(nodes)):
            raise AssertionError(f"{label} has an invalid adjacency shape.")
        if not np.array_equal(adjacency, adjacency.T):
            raise ValueError(f"{label} adjacency is not symmetric.")
        if np.any(np.diag(adjacency) != 0) or not np.all(
            np.isin(adjacency, (0, 1))
        ):
            raise ValueError(f"{label} is not binary and loop-free.")

        adjacency_tensor = torch.from_numpy(adjacency)
        adjacency_tensors.append(adjacency_tensor)
        edge_index = adjacency_tensor.nonzero(as_tuple=False).t().contiguous()
        edge_labels = torch.ones(edge_index.shape[1], dtype=torch.long)
        edge_attr = F.one_hot(edge_labels, num_classes=2).to(torch.float)
        data_list.append(
            Data(
                x=torch.ones((len(nodes), 1), dtype=torch.float),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.zeros((1, 0), dtype=torch.float),
                n_nodes=torch.tensor([len(nodes)], dtype=torch.long),
                idx=torch.tensor([index], dtype=torch.long),
            )
        )
        node_counts.append(len(nodes))
        edge_counts.append(graph.number_of_edges())

    _atomic_torch_save(raw_destination, adjacency_tensors, torch)
    _atomic_torch_save(
        processed_destination,
        InMemoryDataset.collate(data_list),
        torch,
    )
    return {
        "source": {"path": str(source), "sha256": _sha256(source)},
        "raw": {
            "path": str(raw_destination),
            "sha256": _sha256(raw_destination),
        },
        "processed": {
            "path": str(processed_destination),
            "sha256": _sha256(processed_destination),
        },
        "graph_count": len(graphs),
        "source_to_processed_index": "identity",
        "node_count": {"min": min(node_counts), "max": max(node_counts)},
        "edge_count": {"min": min(edge_counts), "max": max(edge_counts)},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare GraphER generic splits for isolated DiGress training."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    for split in SPLITS:
        parser.add_argument(f"--{split}", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    started = datetime.now(timezone.utc)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    for split in SPLITS:
        source = getattr(args, split).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Missing GraphER split: {source}")
        records[split] = _convert_split(
            source,
            raw_destination=output_root / "raw" / f"{split}.pt",
            processed_destination=output_root / "processed" / f"{split}.pt",
            split=split,
        )

    import networkx as nx
    import torch
    import torch_geometric

    finished = datetime.now(timezone.utc)
    _atomic_json(
        args.manifest.expanduser().resolve(),
        {
            "format": FORMAT,
            "dataset": args.dataset,
            "representation": "generic_simple_undirected",
            "split_order_preserved": True,
            "graphs_dropped": 0,
            "upstream_process_bypassed": True,
            "reason": (
                "The attached SpectreGraphDataset.process appends every graph "
                "twice; GraphER writes validated processed files directly."
            ),
            "splits": records,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
            "runtime": {
                "python": platform.python_version(),
                "networkx": str(nx.__version__),
                "torch": str(torch.__version__),
                "torch_geometric": str(torch_geometric.__version__),
            },
        },
    )


if __name__ == "__main__":
    main()
