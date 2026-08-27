#!/usr/bin/env python
"""Internal worker converting molecular splits into DeFoG PyG artifacts.

This program is intentionally executed with the isolated DeFoG interpreter.
It accepts only trusted GraphER pickle files containing ordered NetworkX graph
sequences and writes the exact processed files expected by the attached DeFoG
QM9 and ZINC data modules.  Conversion is all-or-nothing: a malformed graph
aborts the run, and no graph is filtered or reordered.

GraphER represents atoms by atomic number and bonds by the integer classes
``1=single, 2=double, 3=triple, 4=aromatic``.  DeFoG uses zero-based atom class
indices and reserves edge class zero for a non-edge.  QM9 supports all four
bond classes directly.  The attached DeFoG ZINC model has only the three
Kekule bond classes, so aromatic ZINC inputs are deterministically kekulized
from their recorded source SMILES and checked against the input graph before
they are serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAT = "grapher_to_defog_molecular_dataset_v1"
SPLITS = ("train", "val", "test")

ATOM_VOCABULARIES: dict[str, tuple[int, ...]] = {
    "qm9": (6, 7, 8, 9),
    "zinc": (6, 7, 8, 9, 15, 16, 17, 35, 53),
}
EDGE_VOCABULARIES: dict[str, tuple[int, ...]] = {
    "qm9": (1, 2, 3, 4),
    "zinc": (1, 2, 3),
}
BOND_ORDERS = {1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5}
QM9_PROJECTION_POLICY = "audit_and_project_from_categorical_graph_state_v1"

QM9_PROCESSED_FILES = {
    "train": "proc_tr_no_h.pt",
    "val": "proc_val_no_h.pt",
    "test": "proc_test_no_h.pt",
}
ZINC_PROCESSED_FILES = {split: f"{split}.pt" for split in SPLITS}

QM9_RAW_PLACEHOLDERS = (
    "gdb9.sdf",
    "gdb9.sdf.csv",
    "uncharacterized.txt",
)
ZINC_RAW_PLACEHOLDERS = (
    "train.pickle",
    "val.pickle",
    "test.pickle",
    "train.index",
    "val.index",
    "test.index",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_torch_save(path: Path, value: Any, torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _processed_path(output_root: Path, dataset: str, split: str) -> Path:
    if dataset == "qm9":
        return output_root / "processed" / QM9_PROCESSED_FILES[split]
    if dataset == "zinc":
        return output_root / "full" / "processed" / ZINC_PROCESSED_FILES[split]
    raise ValueError(f"Unsupported molecular DeFoG dataset: {dataset!r}.")


def _raw_placeholder_names(dataset: str) -> tuple[str, ...]:
    if dataset == "qm9":
        return QM9_RAW_PLACEHOLDERS
    if dataset == "zinc":
        return ZINC_RAW_PLACEHOLDERS
    raise ValueError(f"Unsupported molecular DeFoG dataset: {dataset!r}.")


def _atom_class(dataset: str, atomic_number: int) -> int:
    try:
        return ATOM_VOCABULARIES[dataset].index(int(atomic_number))
    except (KeyError, ValueError) as exc:
        expected = ATOM_VOCABULARIES.get(dataset)
        raise ValueError(
            f"Atomic number {atomic_number} is outside the DeFoG {dataset} "
            f"vocabulary {expected}."
        ) from exc


def _edge_class(dataset: str, bond_type: int) -> int:
    bond_type = int(bond_type)
    if bond_type not in EDGE_VOCABULARIES.get(dataset, ()):
        raise ValueError(
            f"Bond type {bond_type} is outside the DeFoG {dataset} edge "
            f"vocabulary {EDGE_VOCABULARIES.get(dataset)}."
        )
    # DeFoG reserves zero for a non-edge and uses GraphER's 1/2/3/4 IDs
    # unchanged for present bonds.
    return bond_type


def _load_graphs(path: Path, *, split: str) -> list[Any]:
    # Pickle loading is safe here only because the parent wrapper supplies
    # project-owned prepared-dataset artifacts, never arbitrary user files.
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, (list, tuple)) or not value:
        raise TypeError(f"{split}.pkl must contain a non-empty graph sequence.")
    return list(value)


def _atomic_number(node_data: Mapping[str, Any], *, graph_label: str) -> int:
    atomic_number = node_data.get("atomic_num", node_data.get("atom_type"))
    if atomic_number is None:
        raise ValueError(f"{graph_label} contains a node without atomic_num.")
    atomic_number = int(atomic_number)
    if "atomic_num" in node_data and "atom_type" in node_data:
        if int(node_data["atomic_num"]) != int(node_data["atom_type"]):
            raise ValueError(
                f"{graph_label} contains inconsistent atomic_num and atom_type."
            )
    return atomic_number


def _validate_supported_node_state(
    node_data: Mapping[str, Any], *, graph_label: str
) -> None:
    """Reject chemistry that DeFoG's QM9/ZINC node state cannot encode."""

    if "formal_charge" in node_data and int(node_data["formal_charge"]) != 0:
        raise ValueError(
            f"{graph_label} contains non-zero formal_charge, which the attached "
            "DeFoG categorical node state does not represent."
        )
    for key in ("chiral_tag", "chirality", "stereo"):
        value = node_data.get(key)
        if value not in (None, False, 0, "", "none", "unspecified"):
            raise ValueError(
                f"{graph_label} contains unsupported node attribute "
                f"{key}={value!r}."
            )


def _qm9_projection_metadata(
    graph: Any,
    *,
    graph_label: str,
) -> tuple[list[list[int]], list[int], list[int]]:
    """Require provenance produced by the canonical QM9 preparer."""

    policy = graph.graph.get("qm9_source_state_projection_policy")
    required = (
        "projected_formal_charge_atoms",
        "projected_chiral_atoms",
        "projected_stereo_bonds",
    )
    if policy != QM9_PROJECTION_POLICY or any(
        key not in graph.graph for key in required
    ):
        raise ValueError(
            f"{graph_label} lacks canonical QM9 source-state projection "
            "provenance. Regenerate qm9_attributed with the current "
            "scripts/prepare_qm9_dataset.py."
        )
    raw_charges = graph.graph["projected_formal_charge_atoms"]
    raw_chiral = graph.graph["projected_chiral_atoms"]
    raw_bond_stereo = graph.graph["projected_stereo_bonds"]
    if not isinstance(raw_charges, (list, tuple)) or not isinstance(
        raw_chiral, (list, tuple)
    ) or not isinstance(raw_bond_stereo, (list, tuple)):
        raise ValueError(f"{graph_label} has malformed QM9 projection metadata.")
    charges: list[list[int]] = []
    for item in raw_charges:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                f"{graph_label} has malformed projected formal-charge entry."
            )
        charges.append([int(item[0]), int(item[1])])
    return (
        charges,
        [int(value) for value in raw_chiral],
        [int(value) for value in raw_bond_stereo],
    )


def _bond_type(edge_data: Mapping[str, Any], *, graph_label: str) -> int:
    if "bond_type" not in edge_data:
        raise ValueError(f"{graph_label} contains an edge without bond_type.")
    bond_type = int(edge_data["bond_type"])
    if bond_type not in BOND_ORDERS:
        raise ValueError(f"{graph_label} contains unknown bond type {bond_type}.")
    if "bond_order" in edge_data:
        observed = float(edge_data["bond_order"])
        expected = BOND_ORDERS[bond_type]
        if abs(observed - expected) > 1e-6:
            raise ValueError(
                f"{graph_label} has bond_type={bond_type} but "
                f"bond_order={observed}; expected {expected}."
            )
    for key in ("bond_stereo", "stereo", "stereo_atoms"):
        value = edge_data.get(key)
        if value not in (None, False, 0, "", "none", "unspecified"):
            raise ValueError(
                f"{graph_label} contains unsupported edge attribute "
                f"{key}={value!r}."
            )
    return bond_type


def _validate_graph(graph: Any, *, graph_label: str) -> tuple[list[Any], dict[Any, int]]:
    import networkx as nx

    if not isinstance(graph, nx.Graph):
        raise TypeError(f"{graph_label} is not a NetworkX graph.")
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError(f"{graph_label} must be simple and undirected.")
    if graph.number_of_nodes() <= 0:
        raise ValueError(f"{graph_label} has no nodes.")
    if nx.number_of_selfloops(graph):
        raise ValueError(f"{graph_label} contains a self-loop.")
    if graph.number_of_nodes() > 1 and not nx.is_connected(graph):
        raise ValueError(f"{graph_label} is disconnected.")
    nodes = list(graph.nodes())
    if len(nodes) != len(set(nodes)):
        raise ValueError(f"{graph_label} contains duplicate node identifiers.")
    return nodes, {node: index for index, node in enumerate(nodes)}


def _rdkit_graph_bond_type(bond: Any, Chem: Any) -> int:
    if bond.GetIsAromatic() or bond.GetBondType() == Chem.BondType.AROMATIC:
        return 4
    mapping = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
    }
    value = mapping.get(bond.GetBondType())
    if value is None:
        raise ValueError(f"Unsupported RDKit bond type {bond.GetBondType()}.")
    return value


def _prepared_zinc_molecule(source_smiles: str, Chem: Any) -> Any:
    molecule = Chem.MolFromSmiles(source_smiles, sanitize=False)
    if molecule is None:
        raise ValueError("RDKit could not parse graph.source_smiles.")
    Chem.SanitizeMol(molecule)
    fragments = tuple(Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True))
    if not fragments:
        raise ValueError("graph.source_smiles contains no molecular fragment.")
    if len(fragments) > 1:
        molecule = max(
            enumerate(fragments),
            key=lambda item: (
                int(item[1].GetNumHeavyAtoms()),
                int(item[1].GetNumAtoms()),
                -item[0],
            ),
        )[1]
    molecule = Chem.RemoveHs(molecule, sanitize=True)
    charged_atoms = [
        (int(atom.GetIdx()), int(atom.GetFormalCharge()))
        for atom in molecule.GetAtoms()
        if int(atom.GetFormalCharge()) != 0
    ]
    if charged_atoms:
        raise ValueError(
            "DeFoG's ZINC atom state does not represent formal charge; "
            f"source_smiles contains charged atoms {charged_atoms}."
        )
    Chem.RemoveStereochemistry(molecule)
    Chem.SanitizeMol(molecule)
    return molecule


def _zinc_kekule_edges(
    graph: Any,
    *,
    nodes: Sequence[Any],
    node_positions: Mapping[Any, int],
    graph_label: str,
) -> tuple[dict[tuple[int, int], int], dict[str, Any]]:
    """Return DeFoG-compatible ZINC bonds without guessing aromatic orders."""

    input_edges: dict[tuple[int, int], int] = {}
    aromatic_count = 0
    for u, v, data in graph.edges(data=True):
        left, right = sorted((node_positions[u], node_positions[v]))
        value = _bond_type(data, graph_label=graph_label)
        input_edges[(left, right)] = value
        aromatic_count += int(value == 4)

    source_smiles = graph.graph.get("source_smiles")
    if not source_smiles:
        raise ValueError(
            f"{graph_label} has no source_smiles; verifying that formal charge "
            "and the project graph agree with DeFoG's ZINC state is impossible."
        )
    try:
        from rdkit import Chem
    except Exception as exc:
        raise ImportError(
            "RDKit is required to convert aromatic GraphER ZINC graphs to "
            "DeFoG's Kekule edge vocabulary."
        ) from exc

    molecule = _prepared_zinc_molecule(str(source_smiles), Chem)
    if molecule.GetNumAtoms() != len(nodes):
        raise ValueError(
            f"{graph_label} source_smiles has {molecule.GetNumAtoms()} heavy atoms, "
            f"but the serialized graph has {len(nodes)}."
        )
    graph_atoms = [
        _atomic_number(graph.nodes[node], graph_label=graph_label) for node in nodes
    ]
    molecule_atoms = [int(atom.GetAtomicNum()) for atom in molecule.GetAtoms()]
    if molecule_atoms != graph_atoms:
        raise ValueError(
            f"{graph_label} source_smiles atom order does not match the serialized "
            "GraphER node order."
        )

    molecule_edges: dict[tuple[int, int], int] = {}
    for bond in molecule.GetBonds():
        edge = tuple(sorted((int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx()))))
        molecule_edges[edge] = _rdkit_graph_bond_type(bond, Chem)
    if molecule_edges != input_edges:
        raise ValueError(
            f"{graph_label} source_smiles bonds do not exactly reproduce the "
            "serialized GraphER graph."
        )

    if aromatic_count == 0:
        return (
            {edge: _edge_class("zinc", value) for edge, value in input_edges.items()},
            {
                "aromatic_bonds_input": 0,
                "source_smiles_used": True,
                "kekulized": False,
            },
        )

    # Kekulize a private molecule copy.  clearAromaticFlags=True makes the
    # result independent of later aromaticity perception and leaves only
    # single/double/triple bonds, which is precisely DeFoG-ZINC's vocabulary.
    molecule = Chem.Mol(molecule)
    Chem.Kekulize(molecule, clearAromaticFlags=True)
    output_edges: dict[tuple[int, int], int] = {}
    for bond in molecule.GetBonds():
        edge = tuple(sorted((int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx()))))
        value = _rdkit_graph_bond_type(bond, Chem)
        output_edges[edge] = _edge_class("zinc", value)
    if set(output_edges) != set(input_edges):
        raise AssertionError("RDKit kekulization changed the ZINC graph topology.")
    return (
        output_edges,
        {
            "aromatic_bonds_input": aromatic_count,
            "source_smiles_used": True,
            "kekulized": True,
        },
    )


def _ordered_graph_digest_update(
    digest: Any,
    *,
    graph_index: int,
    atom_numbers: Sequence[int],
    edges: Mapping[tuple[int, int], int],
) -> None:
    record = {
        "index": int(graph_index),
        "atoms": [int(value) for value in atom_numbers],
        "edges": [
            [int(u), int(v), int(value)]
            for (u, v), value in sorted(edges.items())
        ],
    }
    digest.update(json.dumps(record, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\n")


def _convert_split(
    source: Path,
    destination: Path,
    *,
    model_view_destination: Path,
    dataset: str,
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
    input_bond_counts: Counter[int] = Counter()
    output_bond_counts: Counter[int] = Counter()
    source_smiles_used = 0
    kekulized_graphs = 0
    aromatic_bonds_input = 0
    projected_formal_charge_graphs = 0
    projected_formal_charge_atoms = 0
    projected_stereochemistry_graphs = 0
    projected_chiral_atoms = 0
    projected_stereo_bonds = 0
    input_digest = hashlib.sha256()
    output_digest = hashlib.sha256()

    atom_vocabulary = ATOM_VOCABULARIES[dataset]
    edge_vocabulary = EDGE_VOCABULARIES[dataset]
    edge_dimensions = len(edge_vocabulary) + 1

    for index, graph in enumerate(graphs):
        graph_label = f"{split}[{index}]"
        try:
            nodes, node_positions = _validate_graph(graph, graph_label=graph_label)
            atom_numbers = [
                _atomic_number(graph.nodes[node], graph_label=graph_label)
                for node in nodes
            ]
            for node in nodes:
                _validate_supported_node_state(
                    graph.nodes[node], graph_label=graph_label
                )
            atom_classes = [
                _atom_class(dataset, atomic_number)
                for atomic_number in atom_numbers
            ]

            input_edges: dict[tuple[int, int], int] = {}
            for u, v, data in graph.edges(data=True):
                edge = tuple(sorted((node_positions[u], node_positions[v])))
                if edge in input_edges:
                    raise ValueError(f"{graph_label} contains a duplicate edge.")
                input_edges[edge] = _bond_type(data, graph_label=graph_label)

            if dataset == "zinc":
                output_edges, zinc_record = _zinc_kekule_edges(
                    graph,
                    nodes=nodes,
                    node_positions=node_positions,
                    graph_label=graph_label,
                )
                source_smiles_used += int(zinc_record["source_smiles_used"])
                kekulized_graphs += int(zinc_record["kekulized"])
                aromatic_bonds_input += int(zinc_record["aromatic_bonds_input"])
            else:
                output_edges = {
                    edge: _edge_class(dataset, value)
                    for edge, value in input_edges.items()
                }
                (
                    charge_projection,
                    chiral_projection,
                    bond_stereo_projection,
                ) = _qm9_projection_metadata(
                    graph,
                    graph_label=graph_label,
                )
                projected_formal_charge_graphs += int(bool(charge_projection))
                projected_formal_charge_atoms += len(charge_projection)
                projected_stereochemistry_graphs += int(
                    bool(chiral_projection) or bool(bond_stereo_projection)
                )
                projected_chiral_atoms += len(chiral_projection)
                projected_stereo_bonds += len(bond_stereo_projection)

            row: list[int] = []
            col: list[int] = []
            edge_classes: list[int] = []
            for (u, v), edge_class in output_edges.items():
                row.extend((u, v))
                col.extend((v, u))
                edge_classes.extend((edge_class, edge_class))
            if row:
                order = sorted(
                    range(len(row)),
                    key=lambda edge_index: row[edge_index] * len(nodes)
                    + col[edge_index],
                )
                row = [row[item] for item in order]
                col = [col[item] for item in order]
                edge_classes = [edge_classes[item] for item in order]
                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_labels = torch.tensor(edge_classes, dtype=torch.long)
                edge_attr = F.one_hot(
                    edge_labels,
                    num_classes=edge_dimensions,
                ).to(torch.float)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, edge_dimensions), dtype=torch.float)

            x = F.one_hot(
                torch.tensor(atom_classes, dtype=torch.long),
                num_classes=len(atom_vocabulary),
            ).to(torch.float)
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.zeros((1, 0), dtype=torch.float),
                # A length-one tensor also keeps DeFoG's ``len(data.idx)``
                # diagnostics valid for a split containing exactly one graph.
                idx=torch.tensor([index], dtype=torch.long),
            )
            data_list.append(data)

            model_view = nx.Graph()
            for node_index, atomic_number in enumerate(atom_numbers):
                model_view.add_node(
                    node_index,
                    atomic_num=int(atomic_number),
                    atom_type=int(atomic_number),
                )
            for (u, v), bond_type in output_edges.items():
                model_view.add_edge(
                    int(u),
                    int(v),
                    bond_type=int(bond_type),
                    bond_order=float(BOND_ORDERS[int(bond_type)]),
                )
            model_view.graph.update(
                {
                    "source_graph_index": index,
                    "molecular_dataset": dataset,
                    "molecular_representation": "model",
                    "source_smiles": graph.graph.get("source_smiles"),
                    "canonical_smiles": graph.graph.get("canonical_smiles"),
                    "projected_formal_charge_atoms": graph.graph.get(
                        "projected_formal_charge_atoms", []
                    ),
                    "projected_chiral_atoms": graph.graph.get(
                        "projected_chiral_atoms", []
                    ),
                    "projected_stereo_bonds": graph.graph.get(
                        "projected_stereo_bonds", []
                    ),
                    "qm9_source_state_projection_policy": graph.graph.get(
                        "qm9_source_state_projection_policy"
                    ),
                }
            )
            model_view_graphs.append(model_view)

            node_counts.append(len(nodes))
            edge_counts.append(len(output_edges))
            atom_counts.update(atom_numbers)
            input_bond_counts.update(input_edges.values())
            output_bond_counts.update(output_edges.values())
            _ordered_graph_digest_update(
                input_digest,
                graph_index=index,
                atom_numbers=atom_numbers,
                edges=input_edges,
            )
            _ordered_graph_digest_update(
                output_digest,
                graph_index=index,
                atom_numbers=atom_numbers,
                edges=output_edges,
            )
        except Exception as exc:
            raise ValueError(
                f"{graph_label}: {type(exc).__name__}: {exc}"
            ) from exc

    collated = InMemoryDataset.collate(data_list)
    _atomic_torch_save(destination, collated, torch)
    _atomic_pickle(model_view_destination, model_view_graphs)
    return {
        "source": {"path": str(source), "sha256": _sha256(source)},
        "output": {"path": str(destination), "sha256": _sha256(destination)},
        "model_view": {
            "path": str(model_view_destination),
            "sha256": _sha256(model_view_destination),
            "graph_count": len(model_view_graphs),
            "representation": "model",
        },
        "graph_count": len(graphs),
        "source_to_processed_index": "identity",
        "ordered_input_graphs_sha256": input_digest.hexdigest(),
        "ordered_defog_graphs_sha256": output_digest.hexdigest(),
        "node_count": {"min": min(node_counts), "max": max(node_counts)},
        "edge_count": {"min": min(edge_counts), "max": max(edge_counts)},
        "atomic_number_counts": {
            str(key): int(value) for key, value in sorted(atom_counts.items())
        },
        "input_bond_type_counts": {
            str(key): int(value) for key, value in sorted(input_bond_counts.items())
        },
        "defog_bond_class_counts": {
            str(key): int(value) for key, value in sorted(output_bond_counts.items())
        },
        "zinc_kekulization": (
            {
                "policy": "rdkit_source_smiles_verified_v1",
                "source_smiles_used": source_smiles_used,
                "graphs_kekulized": kekulized_graphs,
                "aromatic_bonds_input": aromatic_bonds_input,
            }
            if dataset == "zinc"
            else None
        ),
        "qm9_source_state_projection": (
            {
                "policy": "audit_and_project_from_categorical_graph_state_v1",
                "formal_charge_graphs": projected_formal_charge_graphs,
                "formal_charge_atoms": projected_formal_charge_atoms,
                "stereochemistry_graphs": projected_stereochemistry_graphs,
                "chiral_atoms": projected_chiral_atoms,
                "stereo_bonds": projected_stereo_bonds,
            }
            if dataset == "qm9"
            else None
        ),
    }


def _create_raw_placeholders(output_root: Path, dataset: str) -> list[dict[str, str]]:
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    notice = (
        "GraphER-managed DeFoG placeholder. The corresponding processed PyG "
        "artifacts must be present; this file is not source data.\n"
    )
    for name in _raw_placeholder_names(dataset):
        path = raw_dir / name
        path.write_text(notice, encoding="utf-8")
        records.append({"path": str(path), "sha256": _sha256(path)})
    readme = output_root / "GRAPHER_DATASET_NOTICE.txt"
    readme.write_text(
        "This directory was generated from immutable GraphER NetworkX splits.\n"
        "Files under raw/ are sentinels that prevent the external DeFoG loader "
        "from downloading a different benchmark. Do not delete processed files "
        "or use the sentinels as source data. See the conversion manifest.\n",
        encoding="utf-8",
    )
    records.append({"path": str(readme), "sha256": _sha256(readme)})
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare GraphER QM9/ZINC splits for isolated DeFoG training."
    )
    parser.add_argument("--dataset", choices=sorted(ATOM_VOCABULARIES), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
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
            _processed_path(output_root, args.dataset, split),
            model_view_destination=output_root / "model_view" / f"{split}.pkl",
            dataset=args.dataset,
            split=split,
        )

    placeholders = _create_raw_placeholders(output_root, args.dataset)

    import networkx as nx
    import torch
    import torch_geometric

    rdkit_version = None
    if args.dataset == "zinc":
        try:
            import rdkit

            rdkit_version = str(rdkit.__version__)
        except Exception:
            # Conversion will already have failed if RDKit was unavailable;
            # retain this defensive branch for unusual import metadata issues.
            rdkit_version = None

    finished = datetime.now(timezone.utc)
    _atomic_json(
        args.manifest.expanduser().resolve(),
        {
            "format": FORMAT,
            "dataset": args.dataset,
            "representation": "molecular_attributed",
            "split_order_preserved": True,
            "graphs_dropped": 0,
            "splits": records,
            "vocabulary": {
                "node_attribute": "atomic_num",
                "atom_class_to_atomic_number": list(ATOM_VOCABULARIES[args.dataset]),
                "edge_attribute": "bond_type",
                "edge_class_zero": "no_edge",
                "present_edge_classes": list(EDGE_VOCABULARIES[args.dataset]),
                "bond_orders": {
                    str(key): value
                    for key, value in BOND_ORDERS.items()
                    if key in EDGE_VOCABULARIES[args.dataset]
                },
            },
            "zinc_kekulization_policy": (
                "rdkit_source_smiles_verified_v1"
                if args.dataset == "zinc"
                else None
            ),
            "unsupported_state_policy": {
                "formal_charge": (
                    "audit_and_project_at_qm9_preparation"
                    if args.dataset == "qm9"
                    else "reject_nonzero"
                ),
                "node_stereochemistry": (
                    "audit_and_project_at_qm9_preparation"
                    if args.dataset == "qm9"
                    else "reject_if_serialized"
                ),
                "bond_stereochemistry": (
                    "audit_and_project_at_qm9_preparation"
                    if args.dataset == "qm9"
                    else "reject_if_serialized"
                ),
                "graph_filtering_or_repair": "none",
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
                "rdkit": rdkit_version,
            },
        },
    )


if __name__ == "__main__":
    main()
