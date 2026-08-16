#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from grapher.data.io import save_dataset_splits
from grapher.data.preparation_reporting import (
    common_preparation_report,
    print_preparation_summary,
)
from grapher.rewiring_mlp.molecular.constants import QM9_ATOM_TYPES, QM9_BOND_TYPES
from grapher.rewiring_mlp.molecular.graph_io import (
    graphs_from_smiles,
    nx_to_topology,
    read_smiles_file,
    split_graphs,
)
from grapher.utils.io import ensure_dir, load_yaml, save_json

TOPOLOGY_SCHEMA = {
    "node_attributes": [],
    "edge_attributes": [],
}
ATTRIBUTED_SCHEMA = {
    "node_attributes": ["atomic_num", "atom_type"],
    "edge_attributes": ["bond_type", "bond_order"],
}

# DeFoG and PyG both pin the older ``molnet_publish/qm9.zip`` archive. Its
# ``gdb9.sdf`` is the report-facing source for this project. A newer DeepChem
# archive contains an OpenBabel-derived ``qm9.sdf`` with different connection
# tables; some characterized records become disconnected after hydrogen
# removal. Canonical runs therefore identify the source by content, not merely
# by filename or record count.
DEFOG_PYG_QM9_ARCHIVE_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/"
    "datasets/molnet_publish/qm9.zip"
)
DEFOG_PYG_QM9_SDF_SHA256 = (
    "98c4e97d50ac549b8c9f0b2114b348a9a944718e17e50d9a724b729f1deaa28e"
)
QM9_UNCHARACTERIZED_SHA256 = (
    "3aa5115d540b356de94791d4a74c3bf1ed91c469ecf52a4f5d7cc0506fe02e24"
)
KNOWN_INCOMPATIBLE_DEEPCHEM_SDF_SHA256 = (
    "d93a319831492355f44adfba9d73d358aa96c82216fe6754b9a83550e52cf718"
)


@dataclass(frozen=True)
class QM9Protocol:
    """Pinned preparation contract for the report-facing QM9 benchmark."""

    canonical: bool
    expected_source_records: int
    expected_excluded_records: int
    expected_graphs: int
    expected_sdf_sha256: str
    expected_uncharacterized_sha256: str
    split_seed: int
    split_counts: dict[str, int]
    project_formal_charge: bool
    project_stereochemistry: bool

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QM9Protocol":
        protocol = config.get("protocol", {}) or {}
        split = config.get("split", {}) or {}
        preprocessing = config.get("preprocessing", {}) or {}
        result = cls(
            canonical=bool(protocol.get("canonical", True)),
            expected_source_records=int(
                protocol.get("expected_source_records", 133885)
            ),
            expected_excluded_records=int(
                protocol.get("expected_excluded_records", 3054)
            ),
            expected_graphs=int(protocol.get("expected_graphs", 130831)),
            expected_sdf_sha256=str(
                protocol.get(
                    "expected_sdf_sha256",
                    DEFOG_PYG_QM9_SDF_SHA256,
                )
            ).strip().lower(),
            expected_uncharacterized_sha256=str(
                protocol.get(
                    "expected_uncharacterized_sha256",
                    QM9_UNCHARACTERIZED_SHA256,
                )
            ).strip().lower(),
            split_seed=int(split.get("seed", 42)),
            split_counts={
                name: int(split.get(name, 0)) for name in ("train", "val", "test")
            },
            project_formal_charge=bool(
                preprocessing.get("project_formal_charge", True)
            ),
            project_stereochemistry=bool(
                preprocessing.get("project_stereochemistry", True)
            ),
        )
        if min(
            result.expected_source_records,
            result.expected_excluded_records,
            result.expected_graphs,
        ) <= 0:
            raise ValueError("QM9 protocol counts must be positive.")
        if (
            result.expected_source_records - result.expected_excluded_records
            != result.expected_graphs
        ):
            raise ValueError(
                "QM9 source minus excluded count must equal expected_graphs."
            )
        if (
            len(result.expected_sdf_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in result.expected_sdf_sha256
            )
        ):
            raise ValueError(
                "protocol.expected_sdf_sha256 must be a 64-character "
                "hexadecimal SHA-256 digest."
            )
        if (
            result.canonical
            and result.expected_sdf_sha256 != DEFOG_PYG_QM9_SDF_SHA256
        ):
            raise ValueError(
                "Canonical QM9 must use the gdb9.sdf SHA-256 pinned by "
                "DeFoG/PyG. Use --allow-noncanonical for another source."
            )
        if (
            len(result.expected_uncharacterized_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in result.expected_uncharacterized_sha256
            )
        ):
            raise ValueError(
                "protocol.expected_uncharacterized_sha256 must be a "
                "64-character hexadecimal SHA-256 digest."
            )
        if (
            result.canonical
            and result.expected_uncharacterized_sha256
            != QM9_UNCHARACTERIZED_SHA256
        ):
            raise ValueError(
                "Canonical QM9 must use the official uncharacterized.txt "
                "SHA-256 pinned by DeFoG/PyG."
            )
        if any(value < 0 for value in result.split_counts.values()):
            raise ValueError("QM9 split counts must be non-negative.")
        if sum(result.split_counts.values()) != result.expected_graphs:
            raise ValueError("QM9 split counts must sum to expected_graphs.")
        if result.canonical and not (
            result.project_formal_charge and result.project_stereochemistry
        ):
            raise ValueError(
                "Canonical QM9 must explicitly declare projection of formal "
                "charge and stereochemistry from DeFoG's categorical state."
            )
        return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_canonical_sdf_source(
    path: Path,
    *,
    observed_sha256: str,
    expected_sha256: str,
) -> None:
    """Require the exact QM9 SDF source pinned by DeFoG and PyG."""

    if observed_sha256 == expected_sha256:
        return

    if observed_sha256 == KNOWN_INCOMPATIBLE_DEEPCHEM_SDF_SHA256:
        detail = (
            "This is the newer OpenBabel-derived DeepChem qm9.sdf. Its "
            "connection tables are not the source used by DeFoG/PyG and can "
            "produce disconnected characterized heavy-atom graphs."
        )
    else:
        detail = "The file is not the gdb9.sdf source pinned by DeFoG/PyG."

    raise RuntimeError(
        "Canonical QM9 source checksum mismatch. "
        f"{detail}\n"
        f"  path: {path}\n"
        f"  observed SHA-256: {observed_sha256}\n"
        f"  expected SHA-256: {expected_sha256}\n"
        "Download the DeFoG/PyG archive, extract gdb9.sdf, and pass that "
        f"file with --sdf-file. Archive: {DEFOG_PYG_QM9_ARCHIVE_URL}"
    )


def _ordered_indices_sha256(indices: list[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(str(int(index)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_uncharacterized_indices(
    path: str | Path,
    *,
    expected_count: int,
    expected_source_records: int,
) -> set[int]:
    """Parse QM9's official one-based uncharacterized-molecule list."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            "QM9 canonical preparation requires the official "
            f"uncharacterized.txt file: {source}"
        )
    lines = source.read_text(encoding="utf-8", errors="strict").splitlines()
    if len(lines) < 11:
        raise ValueError(f"Malformed QM9 uncharacterized file: {source}")
    indices: list[int] = []
    # This is the official QM9/Figshare layout also consumed by PyG/DeFoG:
    # nine header lines, one-based molecule rows, and one footer line. PyG's
    # ``text.split("\n")[9:-2]`` is equivalent because the file ends in a
    # newline; ``splitlines()`` removes that final empty element.
    for line_number, line in enumerate(lines[9:-1], start=10):
        fields = line.split()
        if not fields:
            continue
        try:
            index = int(fields[0]) - 1
        except ValueError as exc:
            raise ValueError(
                f"Invalid QM9 exclusion index on line {line_number}: {line!r}"
            ) from exc
        if index < 0 or index >= expected_source_records:
            raise ValueError(
                f"QM9 exclusion index {index} is outside the source range."
            )
        indices.append(index)
    if len(indices) != expected_count or len(set(indices)) != expected_count:
        raise ValueError(
            "QM9 uncharacterized index count mismatch: "
            f"found {len(indices)} rows/{len(set(indices))} unique, "
            f"expected {expected_count}."
        )
    return set(indices)


def _dataset_output_paths(
    root: str | Path, dataset_names: tuple[str, str]
) -> tuple[Path, Path]:
    """Resolve two distinct dataset directories directly beneath the output root."""
    if len(set(dataset_names)) != len(dataset_names):
        raise ValueError("Topology and attributed dataset names must be different.")

    resolved_root = Path(root).resolve()
    targets: list[Path] = []
    for name in dataset_names:
        relative = Path(name)
        if not name or relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError(
                f"Dataset name must be one directory name, not a path: {name!r}"
            )
        target = (resolved_root / relative).resolve()
        if target == resolved_root or target.parent != resolved_root:
            raise ValueError(f"Unsafe dataset output path: {target}")
        targets.append(target)
    return targets[0], targets[1]


def _clear_dataset_outputs(paths: tuple[Path, Path]) -> list[Path]:
    """Remove old dataset directories after new graph splits are ready to save."""
    removed: list[Path] = []
    for path in paths:
        if path.is_symlink() or path.is_file():
            path.unlink()
            removed.append(path)
        elif path.exists():
            shutil.rmtree(path)
            removed.append(path)
    return removed


def _pyg_bond_type(edge_attr) -> int:
    values = edge_attr.detach().cpu().numpy().reshape(-1).tolist()
    if len(values) >= len(QM9_BOND_TYPES):
        return int(
            QM9_BOND_TYPES[
                int(max(range(len(QM9_BOND_TYPES)), key=lambda i: values[i]))
            ]
        )
    if values:
        val = int(values[0])
        if val in QM9_BOND_TYPES:
            return val
        if 0 <= val < len(QM9_BOND_TYPES):
            return int(QM9_BOND_TYPES[val])
    return int(QM9_BOND_TYPES[0])


def _pyg_data_to_nx(data, *, remove_h: bool = True) -> nx.Graph:
    z = data.z.detach().cpu().numpy().astype(int).tolist()
    keep_old = []
    for idx, atomic_num in enumerate(z):
        if remove_h and int(atomic_num) == 1:
            continue
        if int(atomic_num) not in QM9_ATOM_TYPES:
            raise ValueError(
                f"Atom {atomic_num} is outside allowed atom set {QM9_ATOM_TYPES}."
            )
        keep_old.append(idx)

    node_map = {old: new for new, old in enumerate(keep_old)}
    graph = nx.Graph()
    for old, new in node_map.items():
        atomic_num = int(z[old])
        graph.add_node(new, atomic_num=atomic_num, atom_type=atomic_num)

    edge_index = data.edge_index.detach().cpu().numpy().astype(int)
    edge_attr = getattr(data, "edge_attr", None)
    seen: set[tuple[int, int]] = set()
    for col in range(edge_index.shape[1]):
        u_old = int(edge_index[0, col])
        v_old = int(edge_index[1, col])
        if u_old not in node_map or v_old not in node_map:
            continue
        u = int(node_map[u_old])
        v = int(node_map[v_old])
        if u == v:
            continue
        edge = (u, v) if u < v else (v, u)
        if edge in seen:
            continue
        seen.add(edge)
        bond_type = (
            _pyg_bond_type(edge_attr[col])
            if edge_attr is not None
            else int(QM9_BOND_TYPES[0])
        )
        graph.add_edge(
            edge[0],
            edge[1],
            bond_type=bond_type,
            bond_order=float(bond_type if bond_type != 4 else 1.5),
        )

    return nx.convert_node_labels_to_integers(graph, ordering="sorted")


def _graphs_from_pyg_qm9(
    root: str | Path, *, max_molecules: int | None = None, remove_h: bool = True
) -> tuple[list[nx.Graph], dict[str, int]]:
    try:
        from torch_geometric.datasets.qm9 import QM9  # type: ignore
    except ModuleNotFoundError as exc:
        raise ImportError(
            "PyTorch Geometric is required for --source pyg. Install torch-geometric "
            "or pass --source smiles --smiles-file PATH, or --source sdf --sdf-file PATH."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Could not import torch_geometric.datasets.qm9.QM9. This usually means "
            "the installed PyTorch/PyG versions are incompatible or PyG was installed "
            "without source files needed by TorchScript. Try reinstalling a PyG build "
            "matching your PyTorch/CUDA version, or use --source sdf --sdf-file PATH "
            "or --source smiles --smiles-file PATH."
        ) from exc

    try:
        dataset = QM9(str(root))
    except Exception as exc:
        downloaded_sdf = Path(root) / "raw" / "gdb9.sdf"
        raise RuntimeError(
            "Could not initialize torch_geometric.datasets.QM9. PyG preprocessing may "
            "stop when RDKit cannot parse an individual molecule. If the raw download "
            f"exists, retry with --source sdf --sdf-file {downloaded_sdf}; "
            "the direct SDF loader audits the pinned records. For import or "
            "TorchScript errors, reinstall "
            "a PyG build matching your PyTorch/CUDA version."
        ) from exc
    limit = (
        len(dataset) if max_molecules is None else min(int(max_molecules), len(dataset))
    )
    graphs: list[nx.Graph] = []
    errors: dict[str, int] = {}
    for idx in range(limit):
        try:
            datum = dataset[idx]
            graph = _pyg_data_to_nx(datum, remove_h=remove_h)
            if graph.number_of_nodes() == 0:
                raise ValueError("empty molecule after preprocessing")
            if graph.number_of_nodes() > 1 and not nx.is_connected(graph):
                raise ValueError("disconnected molecule")
            source_index = getattr(datum, "idx", idx)
            if hasattr(source_index, "item"):
                source_index = source_index.item()
            graph.graph["source_index"] = int(source_index)
            graphs.append(graph)
        except Exception as exc:
            name = type(exc).__name__
            errors[name] = errors.get(name, 0) + 1
    return graphs, errors


def _rdkit_bond_type(bond) -> int:
    """Map RDKit bond types to the integer convention used by QM9_BOND_TYPES."""
    try:
        from rdkit import Chem  # type: ignore
    except ModuleNotFoundError as exc:
        raise ImportError("RDKit is required for --source sdf.") from exc

    if bond.GetIsAromatic():
        value = 4
    else:
        value = {
            Chem.BondType.SINGLE: 1,
            Chem.BondType.DOUBLE: 2,
            Chem.BondType.TRIPLE: 3,
            Chem.BondType.AROMATIC: 4,
        }.get(bond.GetBondType())
    if value is None or int(value) not in QM9_BOND_TYPES:
        raise ValueError(f"Unsupported QM9 bond type: {bond.GetBondType()}")
    return int(value)


def _rdkit_mol_to_nx(mol, *, remove_h: bool = True, kekulize: bool = True) -> nx.Graph:
    try:
        from rdkit import Chem  # type: ignore
    except ModuleNotFoundError as exc:
        raise ImportError("RDKit is required for --source sdf.") from exc

    mol = Chem.Mol(mol)
    charged_atoms = [
        (int(atom.GetIdx()), int(atom.GetFormalCharge()))
        for atom in mol.GetAtoms()
        if int(atom.GetFormalCharge()) != 0
    ]
    chiral_atoms = [
        int(atom.GetIdx())
        for atom in mol.GetAtoms()
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    ]
    stereo_bonds = [
        int(bond.GetIdx())
        for bond in mol.GetBonds()
        if bond.GetStereo() != Chem.BondStereo.STEREONONE
    ]
    if kekulize:
        Chem.Kekulize(mol, clearAromaticFlags=True)

    keep_old = []
    for atom in mol.GetAtoms():
        old = int(atom.GetIdx())
        atomic_num = int(atom.GetAtomicNum())
        if remove_h and atomic_num == 1:
            continue
        if atomic_num not in QM9_ATOM_TYPES:
            raise ValueError(
                f"Atom {atomic_num} is outside allowed atom set {QM9_ATOM_TYPES}."
            )
        keep_old.append(old)

    node_map = {old: new for new, old in enumerate(keep_old)}
    graph = nx.Graph()
    for old, new in node_map.items():
        atom = mol.GetAtomWithIdx(int(old))
        atomic_num = int(atom.GetAtomicNum())
        graph.add_node(new, atomic_num=atomic_num, atom_type=atomic_num)

    for bond in mol.GetBonds():
        u_old = int(bond.GetBeginAtomIdx())
        v_old = int(bond.GetEndAtomIdx())
        if u_old not in node_map or v_old not in node_map:
            continue
        u = int(node_map[u_old])
        v = int(node_map[v_old])
        if u == v:
            continue
        bond_type = _rdkit_bond_type(bond)
        graph.add_edge(
            min(u, v),
            max(u, v),
            bond_type=bond_type,
            bond_order=float(1.5 if bond_type == 4 else bond_type),
        )

    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    # The standard DeFoG/DiGress-style QM9 state contains atom and bond
    # categories only. Preserve the canonical molecule rather than filtering
    # it, while making every projected source channel auditable.
    graph.graph.update(
        {
            "qm9_source_state_projection_policy": (
                "audit_and_project_from_categorical_graph_state_v1"
            ),
            "projected_formal_charge_atoms": [
                [int(index), int(charge)] for index, charge in charged_atoms
            ],
            "projected_chiral_atoms": [int(index) for index in chiral_atoms],
            "projected_stereo_bonds": [int(index) for index in stereo_bonds],
        }
    )
    return graph


def _graphs_from_sdf_qm9(
    sdf_file: str | Path,
    *,
    max_molecules: int | None = None,
    remove_h: bool = True,
    kekulize: bool = True,
    excluded_indices: set[int] | None = None,
) -> tuple[list[nx.Graph], dict[str, int], int]:
    try:
        from rdkit import Chem  # type: ignore
    except ModuleNotFoundError as exc:
        raise ImportError(
            "RDKit is required for --source sdf. Install rdkit or use --source smiles."
        ) from exc

    sdf_file = Path(sdf_file)
    if not sdf_file.exists():
        raise FileNotFoundError(f"SDF file does not exist: {sdf_file}")

    supplier = Chem.SDMolSupplier(
        str(sdf_file),
        removeHs=False,
        sanitize=False,
        strictParsing=True,
    )
    if supplier is None:
        raise RuntimeError(f"Could not open SDF file: {sdf_file}")

    graphs: list[nx.Graph] = []
    errors: dict[str, int] = {}
    num_seen = 0
    excluded = excluded_indices or set()
    for source_index, mol in enumerate(supplier):
        if max_molecules is not None and num_seen >= int(max_molecules):
            break
        num_seen += 1
        if source_index in excluded:
            continue
        try:
            if mol is None:
                raise ValueError("RDKit returned None for molecule")
            graph = _rdkit_mol_to_nx(mol, remove_h=remove_h, kekulize=kekulize)
            if graph.number_of_nodes() == 0:
                raise ValueError("empty molecule after preprocessing")
            if graph.number_of_nodes() > 1 and not nx.is_connected(graph):
                raise ValueError("disconnected molecule")
            graph.graph["source_index"] = int(source_index)
            graphs.append(graph)
        except Exception as exc:
            name = type(exc).__name__
            errors[name] = errors.get(name, 0) + 1
    return graphs, errors, num_seen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare QM9 heavy-atom topology and attributed graph splits."
    )
    parser.add_argument(
        "--config",
        default="configs/datasets/qm9.yaml",
        help="Pinned QM9 preparation protocol YAML.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "pyg", "sdf", "smiles"],
        default="auto",
        help=(
            "Data source. auto uses --smiles-file when provided, otherwise --sdf-file or "
            "<pyg-root>/raw/gdb9.sdf when present, otherwise PyG QM9."
        ),
    )
    parser.add_argument(
        "--smiles-file", default=None, help="Path to .smi/.txt/.csv containing SMILES."
    )
    parser.add_argument(
        "--smiles-column", default=None, help="Optional CSV/TSV column name for SMILES."
    )
    parser.add_argument(
        "--sdf-file",
        default=None,
        help="Path to QM9 gdb9.sdf. Defaults to <pyg-root>/raw/gdb9.sdf for --source sdf.",
    )
    parser.add_argument(
        "--uncharacterized-file",
        default=None,
        help=(
            "Official QM9 uncharacterized.txt. For SDF input, defaults to a "
            "file with this name beside the SDF."
        ),
    )
    parser.add_argument(
        "--pyg-root",
        default="data/pyg_qm9",
        help="Root directory for torch_geometric.datasets.QM9.",
    )
    parser.add_argument("--root", default="outputs/datasets")
    parser.add_argument(
        "--topology-name",
        default="qm9_topology",
        help="Output dataset name for the topology-only representation (always written).",
    )
    parser.add_argument(
        "--attributed-name",
        default="qm9_attributed",
        help=(
            "Output dataset name for topology with atom and bond attributes "
            "(always written)."
        ),
    )
    parser.add_argument("--max-molecules", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Split seed; defaults to the pinned value in the protocol config.",
    )
    parser.add_argument("--keep-hydrogens", action="store_true")
    parser.add_argument("--no-kekulize", action="store_true")
    parser.add_argument(
        "--allow-noncanonical",
        action="store_true",
        help=(
            "Permit development subsets or non-SDF inputs without enforcing "
            "the report-facing canonical counts. Such outputs are marked noncanonical."
        ),
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    protocol = QM9Protocol.from_config(config)
    split_seed = protocol.split_seed if args.seed is None else int(args.seed)
    canonical_run = bool(protocol.canonical and not args.allow_noncanonical)
    if canonical_run and args.max_molecules is not None:
        raise ValueError(
            "--max-molecules is a development option; combine it with "
            "--allow-noncanonical."
        )
    if canonical_run and split_seed != protocol.split_seed:
        raise ValueError(
            f"Canonical QM9 uses split seed {protocol.split_seed}; received "
            f"{split_seed}. Use --allow-noncanonical for a different split."
        )
    if canonical_run and args.keep_hydrogens:
        raise ValueError(
            "Canonical QM9 uses the heavy-atom representation. "
            "Use --allow-noncanonical to retain hydrogens."
        )

    # PyG and DeFoG preserve QM9's source SDF bond categories.
    effective_kekulize = False if canonical_run else not args.no_kekulize

    root = Path(args.root)
    output_paths = _dataset_output_paths(
        root,
        (args.topology_name, args.attributed_name),
    )

    source = args.source
    default_sdf = Path(args.pyg_root) / "raw" / "gdb9.sdf"
    if source == "auto":
        if args.smiles_file:
            source = "smiles"
        elif args.sdf_file or default_sdf.exists():
            source = "sdf"
        else:
            source = "pyg"

    smiles = None
    num_input_records = None
    source_sha256: str | None = None
    exclusions_path: Path | None = None
    exclusions_sha256: str | None = None
    excluded_indices: set[int] = set()
    if source == "smiles":
        if canonical_run:
            raise ValueError(
                "Canonical QM9 preparation requires the original SDF plus the "
                "official uncharacterized.txt; SMILES input is development-only."
            )
        if not args.smiles_file:
            raise ValueError("--smiles-file is required when --source smiles.")
        smiles = read_smiles_file(args.smiles_file, smiles_column=args.smiles_column)
        if args.max_molecules:
            smiles = smiles[: int(args.max_molecules)]
        num_input_records = len(smiles)
        graphs, errors = graphs_from_smiles(
            smiles,
            remove_h=not args.keep_hydrogens,
            kekulize=effective_kekulize,
        )
        source_path = str(args.smiles_file)
        source_sha256 = _sha256_file(Path(args.smiles_file).resolve())
    elif source == "sdf":
        sdf_file = Path(args.sdf_file) if args.sdf_file else default_sdf
        if not sdf_file.is_file():
            raise FileNotFoundError(f"QM9 SDF file does not exist: {sdf_file}")
        sdf_file = sdf_file.resolve()
        source_sha256 = _sha256_file(sdf_file)
        if canonical_run:
            _validate_canonical_sdf_source(
                sdf_file,
                observed_sha256=source_sha256,
                expected_sha256=protocol.expected_sdf_sha256,
            )
            exclusions_path = (
                Path(args.uncharacterized_file)
                if args.uncharacterized_file
                else sdf_file.parent / "uncharacterized.txt"
            )
            excluded_indices = _read_uncharacterized_indices(
                exclusions_path,
                expected_count=protocol.expected_excluded_records,
                expected_source_records=protocol.expected_source_records,
            )
            exclusions_path = exclusions_path.resolve()
            exclusions_sha256 = _sha256_file(exclusions_path)
            if exclusions_sha256 != protocol.expected_uncharacterized_sha256:
                raise RuntimeError(
                    "Canonical QM9 uncharacterized.txt checksum mismatch.\n"
                    f"  path: {exclusions_path}\n"
                    f"  observed SHA-256: {exclusions_sha256}\n"
                    "  expected SHA-256: "
                    f"{protocol.expected_uncharacterized_sha256}\n"
                    "Use the official file distributed with the DeFoG/PyG "
                    "QM9 source."
                )
        graphs, errors, num_input_records = _graphs_from_sdf_qm9(
            sdf_file,
            max_molecules=args.max_molecules,
            remove_h=not args.keep_hydrogens,
            kekulize=effective_kekulize,
            excluded_indices=excluded_indices,
        )
        source_path = str(sdf_file)
    else:
        if canonical_run:
            raise ValueError(
                "Canonical report preparation uses the original SDF so formal "
                "charge, stereochemistry, source count, and exclusion indices "
                "can be audited. Use --source sdf or --allow-noncanonical."
            )
        graphs, errors = _graphs_from_pyg_qm9(
            args.pyg_root,
            max_molecules=args.max_molecules,
            remove_h=not args.keep_hydrogens,
        )
        source_path = f"torch_geometric.datasets.QM9:{args.pyg_root}"

    if not graphs:
        raise RuntimeError(
            f"No valid molecules parsed from {source_path}; errors={errors}"
        )

    if canonical_run:
        if num_input_records != protocol.expected_source_records:
            raise RuntimeError(
                "QM9 source record count mismatch: "
                f"found {num_input_records}, expected "
                f"{protocol.expected_source_records}."
            )
        if errors:
            raise RuntimeError(
                "Characterized QM9 records must convert without additional "
                f"rejections; observed {errors}."
            )
        if len(graphs) != protocol.expected_graphs:
            raise RuntimeError(
                f"QM9 selected graph count is {len(graphs)}, expected "
                f"{protocol.expected_graphs}."
            )
        selected_source_indices = [
            int(graph.graph["source_index"]) for graph in graphs
        ]
        expected_selected_source_indices = [
            index
            for index in range(protocol.expected_source_records)
            if index not in excluded_indices
        ]
        if selected_source_indices != expected_selected_source_indices:
            mismatch = next(
                (
                    position,
                    expected,
                    observed,
                )
                for position, (expected, observed) in enumerate(
                    zip(
                        expected_selected_source_indices,
                        selected_source_indices,
                    )
                )
                if expected != observed
            )
            raise RuntimeError(
                "QM9 selected source indices are not the ordered complement "
                "of the official exclusions: "
                f"position={mismatch[0]}, expected={mismatch[1]}, "
                f"observed={mismatch[2]}."
            )

    attributed_splits = split_graphs(graphs, seed=split_seed)
    split_sizes = {name: len(values) for name, values in attributed_splits.items()}
    split_source_indices_sha256 = {
        name: _ordered_indices_sha256(
            [
                int(graph.graph.get("source_index", index))
                for index, graph in enumerate(values)
            ]
        )
        for name, values in attributed_splits.items()
    }
    projection_summary = {
        "formal_charge_graphs": sum(
            bool(graph.graph.get("projected_formal_charge_atoms"))
            for graph in graphs
        ),
        "formal_charge_atoms": sum(
            len(graph.graph.get("projected_formal_charge_atoms", ()))
            for graph in graphs
        ),
        "stereochemistry_graphs": sum(
            bool(graph.graph.get("projected_chiral_atoms"))
            or bool(graph.graph.get("projected_stereo_bonds"))
            for graph in graphs
        ),
        "chiral_atoms": sum(
            len(graph.graph.get("projected_chiral_atoms", ())) for graph in graphs
        ),
        "stereo_bonds": sum(
            len(graph.graph.get("projected_stereo_bonds", ())) for graph in graphs
        ),
        "policy": "audit_and_project_from_categorical_graph_state_v1",
    }
    if canonical_run and split_sizes != protocol.split_counts:
        raise RuntimeError(
            f"QM9 split sizes {split_sizes} do not match the pinned "
            f"protocol {protocol.split_counts}."
        )
    topology_splits = {
        k: [nx_to_topology(g) for g in v] for k, v in attributed_splits.items()
    }

    config_top = {
        "name": args.topology_name,
        "source": source_path,
        "source_type": source,
        "kind": "qm9_topology",
        "remove_h": not args.keep_hydrogens,
        "kekulize": effective_kekulize,
        "seed": split_seed,
        "canonical_protocol": canonical_run,
        "source_sha256": source_sha256,
        "expected_source_sha256": (
            protocol.expected_sdf_sha256 if canonical_run else None
        ),
        "source_sha256_verified": bool(
            canonical_run and source_sha256 == protocol.expected_sdf_sha256
        ),
        "uncharacterized_file": (
            str(exclusions_path) if exclusions_path is not None else None
        ),
        "uncharacterized_file_sha256": exclusions_sha256,
        "expected_uncharacterized_file_sha256": (
            protocol.expected_uncharacterized_sha256 if canonical_run else None
        ),
        "uncharacterized_file_sha256_verified": bool(
            canonical_run
            and exclusions_sha256 == protocol.expected_uncharacterized_sha256
        ),
        "selected_source_indices_sha256": _ordered_indices_sha256(
            [
                int(graph.graph.get("source_index", index))
                for index, graph in enumerate(graphs)
            ]
        ),
        "split_source_indices_sha256": split_source_indices_sha256,
        "source_state_projection": projection_summary,
    }
    config_attr = dict(config_top)
    config_attr["name"] = args.attributed_name
    config_attr["kind"] = "qm9_attributed"

    removed_paths = _clear_dataset_outputs(output_paths)
    for path in removed_paths:
        print(f"Removed old dataset artifacts: {path}")

    save_dataset_splits(args.topology_name, topology_splits, config_top, root=root)
    save_dataset_splits(args.attributed_name, attributed_splits, config_attr, root=root)
    input_records = (
        len(smiles)
        if smiles is not None
        else (
            int(num_input_records)
            if num_input_records is not None
            else len(graphs) + sum(int(value) for value in errors.values())
        )
    )
    report_rejections = dict(errors)
    if excluded_indices:
        report_rejections["official_uncharacterized"] = len(excluded_indices)
    common_report = common_preparation_report(
        input_records=input_records,
        processed_records=input_records,
        accepted_graphs=len(graphs),
        rejection_reasons=report_rejections,
    )
    prep_report = {
        "status": "pass",
        "dataset": "QM9",
        "source_type": source,
        "source": source_path,
        "source_sha256": source_sha256,
        "protocol": {
            "canonical": canonical_run,
            "expected_source_records": protocol.expected_source_records,
            "expected_excluded_records": protocol.expected_excluded_records,
            "expected_graphs": protocol.expected_graphs,
            "expected_sdf_sha256": protocol.expected_sdf_sha256,
            "source_sha256_verified": bool(
                canonical_run and source_sha256 == protocol.expected_sdf_sha256
            ),
            "expected_uncharacterized_sha256": (
                protocol.expected_uncharacterized_sha256
            ),
            "uncharacterized_sha256_verified": bool(
                canonical_run
                and exclusions_sha256
                == protocol.expected_uncharacterized_sha256
            ),
            "split_seed": split_seed,
            "uncharacterized_file": (
                str(exclusions_path) if exclusions_path is not None else None
            ),
            "uncharacterized_file_sha256": exclusions_sha256,
            "excluded_indices_sha256": (
                _ordered_indices_sha256(sorted(excluded_indices))
                if excluded_indices
                else None
            ),
            "selected_source_indices_sha256": config_top[
                "selected_source_indices_sha256"
            ],
            "split_source_indices_sha256": split_source_indices_sha256,
            "source_state_projection": projection_summary,
        },
        "num_input_smiles": len(smiles) if smiles is not None else None,
        **common_report,
        # Backward-compatible alias retained for existing consumers.
        "num_valid_graphs": len(graphs),
        "errors": report_rejections,
        "conversion_errors": errors,
        "topology_dataset": args.topology_name,
        "attributed_dataset": args.attributed_name,
        "split_sizes": {k: len(v) for k, v in topology_splits.items()},
        "representations": {
            "topology": TOPOLOGY_SCHEMA,
            "topology_and_attributes": ATTRIBUTED_SCHEMA,
        },
    }
    for dataset_name in (args.topology_name, args.attributed_name):
        save_json(
            prep_report,
            ensure_dir(root / dataset_name) / "qm9_prep_report.json",
        )
    print_preparation_summary(
        dataset="QM9",
        source=source_path,
        input_records=input_records,
        processed_records=input_records,
        accepted_graphs=len(graphs),
        rejection_reasons=report_rejections,
        split_sizes=prep_report["split_sizes"],
        outputs=(
            (
                "topology only",
                root / args.topology_name,
                "node attributes: none; edge attributes: none",
            ),
            (
                "topology + attributes",
                root / args.attributed_name,
                "node: atomic_num, atom_type; edge: bond_type, bond_order",
            ),
        ),
    )


if __name__ == "__main__":
    main()
