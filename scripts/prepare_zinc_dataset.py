#!/usr/bin/env python
"""Prepare the fixed ZINC heavy-atom graph benchmark from a local SMILES file."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np

from grapher.data.io import save_dataset_splits
from grapher.data.preparation_reporting import (
    common_preparation_report,
    print_preparation_summary,
)
from grapher.rewiring_mlp.molecular.graph_io import require_rdkit
from grapher.utils.io import ensure_dir, load_yaml, save_json

SPLIT_NAMES = ("train", "val", "test")
SUPPORTED_BOND_TYPES = (1, 2, 3, 4)


class ZincRecordError(ValueError):
    """A source record rejected for a stable, reportable reason."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = str(reason)


@dataclass(frozen=True)
class ZincProtocol:
    dataset_name: str
    expected_graphs: int
    selection: str
    seed: int
    split_counts: dict[str, int]
    remove_hydrogens: bool
    keep_largest_fragment: bool
    retain_formal_charge: bool
    retain_stereochemistry: bool
    max_nodes: int
    require_connected: bool
    neutral_only: bool
    uncharged_atoms_only: bool
    allowed_atomic_numbers: tuple[int, ...]
    allowed_bond_types: tuple[int, ...]
    bond_orders: dict[int, float]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ZincProtocol:
        source = config.get("source", {}) or {}
        subset = source.get("subset", {}) or {}
        preprocessing = config.get("preprocessing", {}) or {}
        filters = config.get("filters", {}) or {}
        categorical = config.get("categorical_state", {}) or {}

        selection = str(subset.get("selection", "first_valid_after_seeded_shuffle"))
        if selection != "first_valid_after_seeded_shuffle":
            raise NotImplementedError(
                "Only source.subset.selection="
                "'first_valid_after_seeded_shuffle' is implemented."
            )

        expected_graphs = int(subset.get("expected_graphs", 0))
        if expected_graphs <= 0:
            raise ValueError("source.subset.expected_graphs must be positive.")

        split_cfg = config.get("split", {}) or {}
        split_counts = {name: int(split_cfg.get(name, 0)) for name in SPLIT_NAMES}
        if any(count < 0 for count in split_counts.values()):
            raise ValueError("ZINC split counts must be non-negative.")
        if sum(split_counts.values()) != expected_graphs:
            raise ValueError(
                "ZINC split counts must sum to source.subset.expected_graphs."
            )

        if not bool(preprocessing.get("sanitize_with_rdkit", True)):
            raise ValueError("Strict RDKit sanitization cannot be disabled.")
        if not bool(preprocessing.get("undirected", True)):
            raise NotImplementedError("Directed molecular graphs are not supported.")
        if bool(preprocessing.get("kekulize", False)):
            raise ValueError(
                "ZINC preparation must keep kekulize=false to preserve aromatic bonds."
            )
        if not bool(preprocessing.get("retain_aromatic_bonds", True)):
            raise ValueError("ZINC preparation must retain aromatic bonds.")
        if bool(preprocessing.get("retain_stereochemistry", False)):
            raise NotImplementedError(
                "Stereochemical graph attributes are not implemented."
            )

        dataset_name = str(config.get("name", "zinc"))
        relative_name = Path(dataset_name)
        if (
            not dataset_name
            or relative_name.is_absolute()
            or len(relative_name.parts) != 1
            or dataset_name in {".", ".."}
        ):
            raise ValueError(
                f"Dataset name must be one directory name: {dataset_name!r}"
            )

        allowed_atoms = tuple(
            int(value)
            for value in filters.get(
                "allowed_atomic_numbers",
                categorical.get("node_categories", ()),
            )
        )
        if not allowed_atoms:
            raise ValueError("filters.allowed_atomic_numbers must not be empty.")

        allowed_bonds = tuple(
            int(value) for value in categorical.get("edge_categories", ())
        )
        if not allowed_bonds:
            allowed_bonds = SUPPORTED_BOND_TYPES
        if not set(allowed_bonds) <= set(SUPPORTED_BOND_TYPES):
            raise ValueError(
                "ZINC edge categories must be drawn from bond types 1, 2, 3, 4."
            )

        bond_orders = {
            int(key): float(value)
            for key, value in (config.get("bond_orders", {}) or {}).items()
        }
        for bond_type in allowed_bonds:
            bond_orders.setdefault(
                bond_type,
                1.5 if bond_type == 4 else float(bond_type),
            )

        max_nodes = int(preprocessing.get("max_nodes", 0))
        if max_nodes <= 0:
            raise ValueError("preprocessing.max_nodes must be positive.")

        return cls(
            dataset_name=dataset_name,
            expected_graphs=expected_graphs,
            selection=selection,
            seed=int(subset.get("seed", 0)),
            split_counts=split_counts,
            remove_hydrogens=bool(preprocessing.get("remove_hydrogens", True)),
            keep_largest_fragment=bool(
                preprocessing.get("keep_largest_fragment", True)
            ),
            retain_formal_charge=bool(preprocessing.get("retain_formal_charge", False)),
            retain_stereochemistry=False,
            max_nodes=max_nodes,
            require_connected=bool(filters.get("require_connected", True)),
            neutral_only=bool(filters.get("neutral_only", True)),
            uncharged_atoms_only=bool(
                filters.get("uncharged_atoms_only", False)
            ),
            allowed_atomic_numbers=allowed_atoms,
            allowed_bond_types=allowed_bonds,
            bond_orders=bond_orders,
        )


def download_bundled_zinc_source(
    destination: str | Path | None = None,
) -> Path:
    """Placeholder for a future licensed/bundled ZINC source downloader."""

    del destination
    raise NotImplementedError(
        "Bundled ZINC download is not available. Supply a local file with "
        "--smiles-file instead."
    )


def _resolve_csv_column(
    header: Sequence[str],
    smiles_column: str | int | None,
) -> int:
    if not header:
        raise ValueError("The delimited source file has no header.")
    if isinstance(smiles_column, int) or (
        isinstance(smiles_column, str) and smiles_column.isdigit()
    ):
        index = int(smiles_column)
        if index < 0 or index >= len(header):
            raise ValueError(f"SMILES column index {index} is outside the CSV header.")
        return index

    normalized = {name.strip().lower(): index for index, name in enumerate(header)}
    if smiles_column is not None:
        key = str(smiles_column).strip().lower()
        if key not in normalized:
            raise ValueError(
                f"SMILES column {smiles_column!r} is not in header {list(header)!r}."
            )
        return normalized[key]
    for candidate in ("smiles", "smile", "canonical_smiles"):
        if candidate in normalized:
            return normalized[candidate]
    if len(header) == 1:
        return 0
    raise ValueError(
        "Could not infer the SMILES column; pass --smiles-column explicitly."
    )


def read_zinc_smiles(
    path: str | Path,
    *,
    smiles_column: str | int | None = None,
) -> list[str]:
    """Read non-empty SMILES records from a local CSV, TSV, SMI, or text file."""

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"ZINC source file does not exist: {source_path}")

    if source_path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "\t" if source_path.suffix.lower() == ".tsv" else ","
        with source_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            try:
                header = next(reader)
            except StopIteration:
                return []
            index = _resolve_csv_column(header, smiles_column)
            records = []
            for row_number, row in enumerate(reader, start=2):
                if index >= len(row):
                    raise ValueError(
                        f"Row {row_number} does not contain SMILES column {index}."
                    )
                value = row[index].strip()
                if value:
                    records.append(value)
            return records

    if smiles_column not in {None, 0, "0"}:
        raise ValueError("Named SMILES columns are supported only for CSV/TSV sources.")
    records: list[str] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            token = stripped.split()[0]
            if token.lower() not in {"smiles", "smile", "canonical_smiles"}:
                records.append(token)
    return records


def _bond_type_from_rdkit(bond: Any) -> int:
    Chem = require_rdkit()
    if bond.GetIsAromatic() or bond.GetBondType() == Chem.BondType.AROMATIC:
        return 4
    mapping = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
    }
    bond_type = mapping.get(bond.GetBondType())
    if bond_type is None:
        raise ZincRecordError(
            "unsupported_bond_type",
            f"Unsupported RDKit bond type: {bond.GetBondType()}",
        )
    return bond_type


def smiles_to_zinc_graph(
    smiles: str,
    protocol: ZincProtocol,
    *,
    source_index: int | None = None,
) -> nx.Graph:
    """Strictly sanitize and convert one ZINC SMILES record."""

    Chem = require_rdkit()
    try:
        molecule = Chem.MolFromSmiles(str(smiles), sanitize=False)
    except Exception as exc:
        raise ZincRecordError("parse_failure", str(exc)) from exc
    if molecule is None:
        raise ZincRecordError(
            "parse_failure",
            "RDKit MolFromSmiles returned None.",
        )
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:
        raise ZincRecordError("sanitization_failure", str(exc)) from exc

    try:
        fragments = tuple(
            Chem.GetMolFrags(
                molecule,
                asMols=True,
                sanitizeFrags=True,
            )
        )
    except Exception as exc:
        raise ZincRecordError("sanitization_failure", str(exc)) from exc
    if len(fragments) > 1:
        if not protocol.keep_largest_fragment:
            raise ZincRecordError(
                "multiple_fragments",
                "The molecule contains multiple disconnected fragments.",
            )
        molecule = max(
            enumerate(fragments),
            key=lambda item: (
                int(item[1].GetNumHeavyAtoms()),
                int(item[1].GetNumAtoms()),
                -item[0],
            ),
        )[1]

    try:
        if protocol.remove_hydrogens:
            molecule = Chem.RemoveHs(molecule, sanitize=True)
        if not protocol.retain_stereochemistry:
            Chem.RemoveStereochemistry(molecule)
        Chem.SanitizeMol(molecule)
    except Exception as exc:
        raise ZincRecordError("sanitization_failure", str(exc)) from exc

    atoms = list(molecule.GetAtoms())
    if not atoms:
        raise ZincRecordError(
            "empty_heavy_atom_graph",
            "No atoms remain after heavy-atom preprocessing.",
        )
    if len(atoms) > protocol.max_nodes:
        raise ZincRecordError(
            "too_many_nodes",
            f"Molecule has {len(atoms)} nodes; maximum is {protocol.max_nodes}.",
        )

    formal_charges = [int(atom.GetFormalCharge()) for atom in atoms]
    if protocol.neutral_only and sum(formal_charges) != 0:
        raise ZincRecordError(
            "non_neutral",
            "The molecule has non-zero net formal charge.",
        )
    if protocol.uncharged_atoms_only and any(formal_charges):
        raise ZincRecordError(
            "charged_atom",
            "The molecule contains a formally charged atom, which cannot be "
            "represented by the configured categorical graph state.",
        )

    graph = nx.Graph()
    for atom in atoms:
        atomic_num = int(atom.GetAtomicNum())
        if atomic_num not in protocol.allowed_atomic_numbers:
            raise ZincRecordError(
                "disallowed_atom",
                f"Atomic number {atomic_num} is outside the configured vocabulary.",
            )
        attributes: dict[str, int] = {
            "atomic_num": atomic_num,
            "atom_type": atomic_num,
        }
        if protocol.retain_formal_charge:
            attributes["formal_charge"] = int(atom.GetFormalCharge())
        graph.add_node(int(atom.GetIdx()), **attributes)

    for bond in molecule.GetBonds():
        u = int(bond.GetBeginAtomIdx())
        v = int(bond.GetEndAtomIdx())
        if u == v:
            raise ZincRecordError(
                "self_loop",
                "The molecular graph contains a self-loop.",
            )
        bond_type = _bond_type_from_rdkit(bond)
        if bond_type not in protocol.allowed_bond_types:
            raise ZincRecordError(
                "disallowed_bond",
                f"Bond type {bond_type} is outside the configured vocabulary.",
            )
        graph.add_edge(
            min(u, v),
            max(u, v),
            bond_type=bond_type,
            bond_order=float(protocol.bond_orders[bond_type]),
        )

    graph = nx.convert_node_labels_to_integers(
        graph,
        first_label=0,
        ordering="sorted",
    )
    if (
        protocol.require_connected
        and graph.number_of_nodes() > 1
        and not nx.is_connected(graph)
    ):
        raise ZincRecordError(
            "disconnected_graph",
            "The heavy-atom graph is disconnected.",
        )

    try:
        canonical_smiles = str(
            Chem.MolToSmiles(
                molecule,
                canonical=True,
                isomericSmiles=False,
            )
        )
    except Exception as exc:
        raise ZincRecordError("canonicalization_failure", str(exc)) from exc
    graph.graph.update(
        {
            "source_dataset": protocol.dataset_name,
            "source_index": source_index,
            "source_smiles": str(smiles),
            "canonical_smiles": canonical_smiles,
        }
    )
    return graph


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_records_sha256(
    graphs: Sequence[nx.Graph],
) -> str:
    digest = hashlib.sha256()
    for graph in graphs:
        digest.update(str(graph.graph.get("source_index")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(graph.graph.get("source_smiles", "")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _split_selected_graphs(
    graphs: Sequence[nx.Graph],
    split_counts: dict[str, int],
) -> dict[str, list[nx.Graph]]:
    splits: dict[str, list[nx.Graph]] = {}
    offset = 0
    for name in SPLIT_NAMES:
        count = int(split_counts[name])
        splits[name] = list(graphs[offset : offset + count])
        offset += count
    if offset != len(graphs):
        raise AssertionError("Split counts do not consume all selected graphs.")
    return splits


def _graph_statistics(graphs: Sequence[nx.Graph]) -> dict[str, Any]:
    node_counts = [graph.number_of_nodes() for graph in graphs]
    edge_counts = [graph.number_of_edges() for graph in graphs]
    atom_counts: Counter[int] = Counter()
    bond_counts: Counter[int] = Counter()
    for graph in graphs:
        atom_counts.update(
            int(data["atomic_num"]) for _, data in graph.nodes(data=True)
        )
        bond_counts.update(
            int(data["bond_type"]) for _, _, data in graph.edges(data=True)
        )
    return {
        "min_nodes": min(node_counts),
        "max_nodes": max(node_counts),
        "mean_nodes": float(np.mean(node_counts)),
        "min_edges": min(edge_counts),
        "max_edges": max(edge_counts),
        "mean_edges": float(np.mean(edge_counts)),
        "all_connected": all(
            graph.number_of_nodes() > 0 and nx.is_connected(graph) for graph in graphs
        ),
        "atomic_number_counts": {
            str(key): value for key, value in sorted(atom_counts.items())
        },
        "bond_type_counts": {
            str(key): value for key, value in sorted(bond_counts.items())
        },
    }


def prepare_zinc_dataset(
    smiles_file: str | Path,
    config: dict[str, Any],
    *,
    root: str | Path | None = None,
    smiles_column: str | int | None = None,
) -> dict[str, Any]:
    """Prepare, persist, and report the configured deterministic ZINC subset."""

    protocol = ZincProtocol.from_config(config)
    source_path = Path(smiles_file)
    smiles_records = read_zinc_smiles(
        source_path,
        smiles_column=smiles_column,
    )
    if not smiles_records:
        raise RuntimeError(f"No SMILES records found in {source_path}.")

    rng = np.random.default_rng(protocol.seed)
    shuffled_indices = rng.permutation(len(smiles_records)).tolist()
    selected: list[nx.Graph] = []
    rejected: Counter[str] = Counter()
    attempted = 0
    for source_index in shuffled_indices:
        if len(selected) >= protocol.expected_graphs:
            break
        attempted += 1
        try:
            graph = smiles_to_zinc_graph(
                smiles_records[source_index],
                protocol,
                source_index=int(source_index),
            )
        except ZincRecordError as exc:
            rejected[exc.reason] += 1
            continue
        selected.append(graph)

    if len(selected) != protocol.expected_graphs:
        raise RuntimeError(
            f"Only {len(selected)} valid ZINC graphs were found; expected "
            f"{protocol.expected_graphs}. attempted={attempted}, "
            f"rejected={dict(sorted(rejected.items()))}"
        )

    splits = _split_selected_graphs(selected, protocol.split_counts)
    output_root = Path(
        root if root is not None else config.get("root", "outputs/datasets")
    )
    resolved_config = copy.deepcopy(config)
    resolved_source = resolved_config.setdefault("source", {})
    resolved_source["local_file"] = str(source_path.resolve())
    resolved_source["local_file_sha256"] = _sha256_file(source_path)
    resolved_source["smiles_column"] = smiles_column
    save_dataset_splits(
        protocol.dataset_name,
        splits,
        resolved_config,
        root=output_root,
    )

    rejection_reasons = dict(sorted(rejected.items()))
    common_report = common_preparation_report(
        input_records=len(smiles_records),
        processed_records=attempted,
        accepted_graphs=len(selected),
        rejection_reasons=rejection_reasons,
    )
    report = {
        "status": "pass",
        "dataset": protocol.dataset_name,
        "source": str(source_path.resolve()),
        "source_sha256": resolved_source["local_file_sha256"],
        "smiles_column": smiles_column,
        "selection": protocol.selection,
        "selection_seed": protocol.seed,
        **common_report,
        # Backward-compatible aliases retained for existing consumers.
        "num_attempted_records": attempted,
        "num_selected_graphs": len(selected),
        "selected_records_sha256": _selected_records_sha256(selected),
        "split_sizes": {name: len(splits[name]) for name in SPLIT_NAMES},
        "filter_diagnostics": {
            "num_rejected": int(sum(rejected.values())),
            "rejection_reasons": rejection_reasons,
        },
        "graph_statistics": _graph_statistics(selected),
        "schema": {
            "node_attributes": ["atomic_num", "atom_type"],
            "edge_attributes": ["bond_type", "bond_order"],
            "bond_types": list(protocol.allowed_bond_types),
        },
    }
    output_dir = ensure_dir(output_root / protocol.dataset_name)
    save_json(report, output_dir / "prep_report.json")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the deterministic ZINC heavy-atom benchmark from a local "
            "SMILES/CSV source."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/datasets/zinc.yaml",
        help="ZINC dataset protocol YAML.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--smiles-file",
        "--input-file",
        dest="smiles_file",
        help="Local .smi, .txt, .csv, or .tsv source.",
    )
    source.add_argument(
        "--download-bundled-source",
        action="store_true",
        help="Request the currently unavailable bundled-source downloader.",
    )
    parser.add_argument(
        "--smiles-column",
        default=None,
        help="CSV/TSV SMILES column name or zero-based index.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Output root override; defaults to config.root.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.download_bundled_source:
        download_bundled_zinc_source()
    if not args.smiles_file:
        raise ValueError("--smiles-file is required.")
    config = load_yaml(args.config)
    report = prepare_zinc_dataset(
        args.smiles_file,
        config,
        root=args.root,
        smiles_column=args.smiles_column,
    )
    output_root = Path(args.root or config.get("root", "outputs/datasets"))
    print_preparation_summary(
        dataset="ZINC250k-GraphER12k",
        source=report["source"],
        input_records=report["num_input_records"],
        processed_records=report["num_processed_records"],
        accepted_graphs=report["num_accepted_graphs"],
        rejection_reasons=report["rejection_reasons"],
        split_sizes=report["split_sizes"],
        outputs=(
            (
                "topology + attributes",
                output_root / report["dataset"],
                "node: atomic_num, atom_type; edge: bond_type, bond_order",
            ),
        ),
    )


if __name__ == "__main__":
    main()
