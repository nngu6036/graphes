from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx


BOND_TYPE_TO_ORDER: dict[int, float] = {1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5}


def molecular_graph_schema() -> dict[str, Any]:
    """Return the portable hard attributed-graph schema used by molecular GraphER."""

    return {
        "format": "networkx.Graph hard attributed molecular graph",
        "node_attributes": {
            "atomic_number": "required int atomic number",
            "z": "alias of atomic_number",
            "formal_charge": "optional int formal charge",
            "is_aromatic": "optional bool aromatic atom flag",
            "chiral_tag": "optional RDKit chiral tag string",
        },
        "edge_attributes": {
            "edge_type": "required int bond category: 1=single, 2=double, 3=triple, 4=aromatic",
            "bond_order": "optional float bond order; inferred from edge_type when missing",
            "is_aromatic": "optional bool aromatic bond flag",
            "edge_attr": "optional numeric edge feature vector",
        },
        "notes": [
            "Generated molecular samples are hard attributed graphs, not relaxed categorical tensors.",
            "Validity without correction converts this graph directly to RDKit and sanitizes without graph repair.",
        ],
    }


def _atomic_number(data: dict[str, Any]) -> int:
    for key in ("atomic_number", "z", "atom_type", "node_type"):
        value = data.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    raise KeyError("Molecular node is missing atomic_number/z.")


def _edge_type(data: dict[str, Any]) -> int:
    for key in ("edge_type", "bond_type", "bond"):
        value = data.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    raw = data.get("edge_attr")
    if isinstance(raw, (list, tuple)) and raw:
        return int(round(float(raw[0])))
    raise KeyError("Molecular edge is missing edge_type/bond_type.")


def canonicalize_molecular_graph(graph: nx.Graph) -> nx.Graph:
    """Return a compactly relabeled molecular graph with canonical attrs."""

    graph = nx.convert_node_labels_to_integers(nx.Graph(graph), ordering="sorted")
    out = nx.Graph()
    out.graph.update(graph.graph)
    for node, data in graph.nodes(data=True):
        attrs = dict(data)
        z = _atomic_number(attrs)
        attrs["atomic_number"] = int(z)
        attrs["z"] = int(z)
        attrs.setdefault("node_label", f"atomic_number={int(z)}")
        out.add_node(int(node), **attrs)
    for u, v, data in graph.edges(data=True):
        attrs = dict(data)
        edge_type = _edge_type(attrs)
        attrs["edge_type"] = int(edge_type)
        attrs.setdefault("edge_attr", [float(edge_type)])
        attrs["bond_order"] = float(attrs.get("bond_order", BOND_TYPE_TO_ORDER.get(int(edge_type), 1.0)))
        if int(edge_type) == 4:
            attrs.setdefault("is_aromatic", True)
        out.add_edge(int(u), int(v), **attrs)
    return out


@dataclass
class MolecularConversion:
    graph: nx.Graph
    valid: bool
    smiles: str
    mol: Any | None = None
    error: str | None = None


def _rdkit_bond_type(edge_type: int):
    from rdkit import Chem

    mapping = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    return mapping.get(int(edge_type), Chem.BondType.SINGLE)


def molecular_graph_to_rdkit(
    graph: nx.Graph,
    *,
    sanitize: bool = True,
    isomeric_smiles: bool = False,
) -> MolecularConversion:
    """Convert a hard attributed graph to RDKit without any graph repair."""

    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception as exc:  # pragma: no cover - optional dependency.
        return MolecularConversion(nx.Graph(graph), False, "", None, f"rdkit_missing:{exc}")

    try:
        canonical = canonicalize_molecular_graph(graph)
        rw = Chem.RWMol()
        node_to_atom: dict[int, int] = {}
        for node, data in canonical.nodes(data=True):
            atom = Chem.Atom(int(data["atomic_number"]))
            if "formal_charge" in data:
                try:
                    atom.SetFormalCharge(int(data["formal_charge"]))
                except Exception:
                    pass
            if bool(data.get("is_aromatic", False)):
                atom.SetIsAromatic(True)
            node_to_atom[int(node)] = int(rw.AddAtom(atom))
        for u, v, data in canonical.edges(data=True):
            edge_type = int(data["edge_type"])
            bond_type = _rdkit_bond_type(edge_type)
            rw.AddBond(node_to_atom[int(u)], node_to_atom[int(v)], bond_type)
            bond = rw.GetBondBetweenAtoms(node_to_atom[int(u)], node_to_atom[int(v)])
            if bond is not None and (edge_type == 4 or bool(data.get("is_aromatic", False))):
                bond.SetIsAromatic(True)
                rw.GetAtomWithIdx(node_to_atom[int(u)]).SetIsAromatic(True)
                rw.GetAtomWithIdx(node_to_atom[int(v)]).SetIsAromatic(True)
        mol = rw.GetMol()
        if sanitize:
            Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=bool(isomeric_smiles))
        return MolecularConversion(canonical, True, str(smiles), mol, None)
    except Exception as exc:
        return MolecularConversion(nx.Graph(graph), False, "", None, f"{type(exc).__name__}:{exc}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def graph_to_molecular_record(graph: nx.Graph) -> dict[str, Any]:
    graph = canonicalize_molecular_graph(graph)
    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "graph": {k: _jsonable(v) for k, v in graph.graph.items()},
        "nodes": [
            {"id": int(node), **{k: _jsonable(v) for k, v in data.items()}}
            for node, data in graph.nodes(data=True)
        ],
        "edges": [
            {"source": int(u), "target": int(v), **{k: _jsonable(value) for k, value in data.items()}}
            for u, v, data in graph.edges(data=True)
        ],
    }


def write_molecular_jsonl(
    graphs: Sequence[nx.Graph],
    path: str | Path,
    *,
    conversions: Sequence[MolecularConversion] | None = None,
    records: Sequence[dict[str, Any]] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conversions = list(conversions or [])
    records = list(records or [])
    with path.open("w", encoding="utf-8") as f:
        for index, graph in enumerate(graphs):
            payload = graph_to_molecular_record(graph)
            payload["index"] = int(index)
            if index < len(conversions):
                payload["rdkit"] = {
                    "valid": bool(conversions[index].valid),
                    "canonical_smiles": str(conversions[index].smiles or ""),
                    "error": conversions[index].error,
                }
            if index < len(records):
                payload["generation_record"] = records[index]
            f.write(json.dumps(payload, sort_keys=True) + "\n")
    return path


def write_smiles_file(smiles: Iterable[str | None], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for value in smiles:
            if value:
                f.write(str(value) + "\n")
    return path


def write_molecular_sdf(
    conversions: Sequence[MolecularConversion],
    path: str | Path,
) -> tuple[Path, int]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from rdkit import Chem
    except Exception as exc:  # pragma: no cover
        raise ModuleNotFoundError("Writing SDF requires RDKit.") from exc
    writer = Chem.SDWriter(str(path))
    count = 0
    try:
        for item in conversions:
            if item.valid and item.mol is not None:
                writer.write(item.mol)
                count += 1
    finally:
        writer.close()
    return path, count
