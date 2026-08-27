#!/usr/bin/env python
"""Convert trusted GraphER QM9 splits into DiGress processed artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

FORMAT = "grapher_to_digress_qm9_dataset_v1"
SPLITS = ("train", "val", "test")
ATOM_VOCABULARY = (6, 7, 8, 9)
EDGE_VOCABULARY = (1, 2, 3, 4)
BOND_ORDERS = {1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5}
PROCESSED_FILES = {
    "train": "proc_tr_no_h.pt",
    "val": "proc_val_no_h.pt",
    "test": "proc_test_no_h.pt",
}
RAW_PLACEHOLDERS = ("gdb9.sdf", "gdb9.sdf.csv", "uncharacterized.txt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
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


def _atomic_number(data: Mapping[str, Any], *, label: str) -> int:
    value = data.get("atomic_num", data.get("atom_type"))
    if value is None:
        raise ValueError(f"{label} contains a node without atomic_num/atom_type.")
    atomic_number = int(value)
    if "atomic_num" in data and "atom_type" in data:
        if int(data["atomic_num"]) != int(data["atom_type"]):
            raise ValueError(f"{label} has inconsistent atom attributes.")
    if atomic_number not in ATOM_VOCABULARY:
        raise ValueError(
            f"{label} atomic number {atomic_number} is outside "
            f"{ATOM_VOCABULARY}."
        )
    return atomic_number


def _bond_type(data: Mapping[str, Any], *, label: str) -> int:
    value = data.get("bond_type")
    if value is None:
        raw_order = data.get("bond_order")
        if raw_order is None:
            raise ValueError(f"{label} contains an edge without bond_type.")
        order = float(raw_order)
        value = 4 if abs(order - 1.5) < 1e-6 else int(round(order))
    bond_type = int(value)
    if bond_type not in EDGE_VOCABULARY:
        raise ValueError(
            f"{label} bond type {bond_type} is outside {EDGE_VOCABULARY}."
        )
    if "bond_order" in data:
        expected = BOND_ORDERS[bond_type]
        if abs(float(data["bond_order"]) - expected) > 1e-6:
            raise ValueError(f"{label} has inconsistent bond_type/bond_order.")
    return bond_type


def _convert_split(
    source: Path,
    destination: Path,
    *,
    model_view_destination: Path,
    split: str,
) -> dict[str, Any]:
    import networkx as nx
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data, InMemoryDataset

    graphs = _load_graphs(source, split=split)
    data_list: list[Any] = []
    model_view_graphs: list[Any] = []
    node_counts: list[int] = []
    edge_counts: list[int] = []
    atom_counts: Counter[int] = Counter()
    bond_counts: Counter[int] = Counter()
    projected_charge_graphs = 0
    projected_chirality_graphs = 0

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
        positions = {node: position for position, node in enumerate(nodes)}
        atomic_numbers = [
            _atomic_number(graph.nodes[node], label=label) for node in nodes
        ]
        atom_classes = [ATOM_VOCABULARY.index(value) for value in atomic_numbers]

        row: list[int] = []
        col: list[int] = []
        edge_classes: list[int] = []
        model_edges: list[tuple[int, int, int]] = []
        for source_node, target_node, data in graph.edges(data=True):
            source_index = positions[source_node]
            target_index = positions[target_node]
            if source_index == target_index:
                raise ValueError(f"{label} contains a self-loop.")
            bond_type = _bond_type(data, label=label)
            row.extend((source_index, target_index))
            col.extend((target_index, source_index))
            edge_classes.extend((bond_type, bond_type))
            model_edges.append(
                (min(source_index, target_index), max(source_index, target_index), bond_type)
            )

        if row:
            order = sorted(
                range(len(row)), key=lambda item: row[item] * len(nodes) + col[item]
            )
            edge_index = torch.tensor(
                [[row[item] for item in order], [col[item] for item in order]],
                dtype=torch.long,
            )
            edge_labels = torch.tensor(
                [edge_classes[item] for item in order], dtype=torch.long
            )
            edge_attr = F.one_hot(edge_labels, num_classes=5).to(torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 5), dtype=torch.float)

        x = F.one_hot(
            torch.tensor(atom_classes, dtype=torch.long), num_classes=4
        ).to(torch.float)
        data_list.append(
            Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.zeros((1, 0), dtype=torch.float),
                idx=torch.tensor([index], dtype=torch.long),
            )
        )

        model_view = nx.Graph()
        for node_index, atomic_number in enumerate(atomic_numbers):
            model_view.add_node(
                node_index,
                atomic_num=int(atomic_number),
                atom_type=int(atomic_number),
            )
        for source_index, target_index, bond_type in model_edges:
            model_view.add_edge(
                source_index,
                target_index,
                bond_type=int(bond_type),
                bond_order=float(BOND_ORDERS[bond_type]),
            )
        model_view.graph.update(
            {
                "source_graph_index": index,
                "molecular_dataset": "qm9",
                "molecular_representation": "digress_model",
                "qm9_source_state_projection_policy": graph.graph.get(
                    "qm9_source_state_projection_policy"
                ),
                "projected_formal_charge_atoms": graph.graph.get(
                    "projected_formal_charge_atoms", []
                ),
                "projected_chiral_atoms": graph.graph.get(
                    "projected_chiral_atoms", []
                ),
                "projected_stereo_bonds": graph.graph.get(
                    "projected_stereo_bonds", []
                ),
            }
        )
        model_view_graphs.append(model_view)

        projected_charge_graphs += int(
            bool(graph.graph.get("projected_formal_charge_atoms"))
        )
        projected_chirality_graphs += int(
            bool(graph.graph.get("projected_chiral_atoms"))
            or bool(graph.graph.get("projected_stereo_bonds"))
        )
        node_counts.append(len(nodes))
        edge_counts.append(len(model_edges))
        atom_counts.update(atomic_numbers)
        bond_counts.update(bond_type for _, _, bond_type in model_edges)

    _atomic_torch_save(
        destination,
        InMemoryDataset.collate(data_list),
        torch,
    )
    _atomic_pickle(model_view_destination, model_view_graphs)
    return {
        "source": {"path": str(source), "sha256": _sha256(source)},
        "processed": {"path": str(destination), "sha256": _sha256(destination)},
        "model_view": {
            "path": str(model_view_destination),
            "sha256": _sha256(model_view_destination),
            "graph_count": len(model_view_graphs),
        },
        "graph_count": len(graphs),
        "source_to_processed_index": "identity",
        "node_count": {"min": min(node_counts), "max": max(node_counts)},
        "edge_count": {"min": min(edge_counts), "max": max(edge_counts)},
        "atomic_number_counts": {
            str(key): int(value) for key, value in sorted(atom_counts.items())
        },
        "bond_type_counts": {
            str(key): int(value) for key, value in sorted(bond_counts.items())
        },
        "projected_source_state": {
            "formal_charge_graphs": projected_charge_graphs,
            "stereochemistry_graphs": projected_chirality_graphs,
        },
    }


def _raw_placeholders(output_root: Path) -> list[dict[str, str]]:
    raw = output_root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    notice = (
        "GraphER-managed DiGress placeholder. Valid processed PyG artifacts "
        "already exist; this is not the original QM9 source file.\n"
    )
    records: list[dict[str, str]] = []
    for name in RAW_PLACEHOLDERS:
        path = raw / name
        path.write_text(notice, encoding="utf-8")
        records.append({"path": str(path), "sha256": _sha256(path)})
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare GraphER QM9 splits for isolated DiGress training."
    )
    parser.add_argument("--dataset", choices=("qm9",), required=True)
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
            output_root / "processed" / PROCESSED_FILES[split],
            model_view_destination=output_root / "model_view" / f"{split}.pkl",
            split=split,
        )

    placeholders = _raw_placeholders(output_root)
    import networkx as nx
    import torch
    import torch_geometric

    finished = datetime.now(timezone.utc)
    _atomic_json(
        args.manifest.expanduser().resolve(),
        {
            "format": FORMAT,
            "dataset": "qm9",
            "representation": "heavy_atom_categorical",
            "split_order_preserved": True,
            "graphs_dropped": 0,
            "splits": records,
            "vocabulary": {
                "atom_class_to_atomic_number": list(ATOM_VOCABULARY),
                "edge_class_zero": "no_edge",
                "present_edge_classes": list(EDGE_VOCABULARY),
            },
            "raw_placeholders": placeholders,
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
