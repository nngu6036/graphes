from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grapher.evaluation.data_io import save_dataset_splits
from grapher.datasets.zinc_utils import assert_zinc_atomic_numbers
from grapher.utils.io import load_yaml


def _normalise_split(value: str | None) -> str | None:
    if value is None:
        return None
    key = value.strip().lower()
    aliases = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "dev": "val",
        "test": "test",
        "testing": "test",
    }
    return aliases.get(key)


def _bond_type_id(bond: Any) -> int | None:
    from rdkit import Chem

    mapping = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4,
    }
    return mapping.get(bond.GetBondType())


def _smiles_to_graph(
    smiles: str,
    *,
    keep_hs: bool = False,
    target: float | None = None,
    row_id: str | None = None,
) -> nx.Graph | None:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol) if keep_hs else Chem.RemoveHs(mol)

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    graph = nx.Graph()

    for atom in mol.GetAtoms():
        idx = int(atom.GetIdx())
        z = int(atom.GetAtomicNum())
        if z <= 0:
            return None

        graph.add_node(
            idx,
            node_label=f"atomic_number={z}",
            atomic_number=z,
            z=z,
            atom_symbol=str(atom.GetSymbol()),
            feats=[float(z)],
        )

    for bond in mol.GetBonds():
        edge_type = _bond_type_id(bond)
        if edge_type is None:
            return None

        graph.add_edge(
            int(bond.GetBeginAtomIdx()),
            int(bond.GetEndAtomIdx()),
            edge_type=edge_type,
            edge_attr=[float(edge_type)],
            bond_type_name=str(bond.GetBondType()),
        )

    graph.graph["smiles"] = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=False,
    )
    graph.graph["source_dataset"] = "zinc_smiles_rdkit"

    if row_id is not None:
        graph.graph["source_row_id"] = row_id
    if target is not None:
        graph.graph["molecular_target"] = [float(target)]
        graph.graph["molecular_target_value"] = float(target)

    return nx.convert_node_labels_to_integers(graph)


def _read_graphs(args: argparse.Namespace) -> list[tuple[str | None, nx.Graph]]:
    out: list[tuple[str | None, nx.Graph]] = []
    skipped = 0

    with open(args.csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if args.smiles_col not in (reader.fieldnames or []):
            raise KeyError(
                f"SMILES column {args.smiles_col!r} not found. "
                f"Available columns: {reader.fieldnames}"
            )

        for i, row in enumerate(reader):
            smiles = str(row.get(args.smiles_col, "")).strip()
            if not smiles:
                skipped += 1
                continue

            split = _normalise_split(row.get(args.split_col)) if args.split_col else None
            if args.split_col and split is None:
                skipped += 1
                continue

            target = None
            if args.target_col:
                raw_target = row.get(args.target_col)
                if raw_target not in (None, ""):
                    target = float(raw_target)

            row_id = row.get(args.id_col) if args.id_col else str(i)
            graph = _smiles_to_graph(
                smiles,
                keep_hs=args.keep_hs,
                target=target,
                row_id=row_id,
            )
            if graph is None or graph.number_of_nodes() == 0:
                skipped += 1
                continue

            out.append((split, graph))

    print(f"Loaded {len(out)} valid RDKit graphs; skipped {skipped} rows.")
    return out


def _make_splits(
    items: list[tuple[str | None, nx.Graph]],
    args: argparse.Namespace,
) -> dict[str, list[nx.Graph]]:
    splits = {"train": [], "val": [], "test": []}

    if args.split_col:
        for split, graph in items:
            assert split in splits
            splits[split].append(graph)
        return splits

    graphs = [graph for _, graph in items]
    rng = random.Random(args.seed)
    rng.shuffle(graphs)

    if args.train_count is not None:
        train_count = int(args.train_count)
        val_count = int(args.val_count or 0)
        test_count = int(args.test_count or 0)
        total = train_count + val_count + test_count
        graphs = graphs[:total]
        return {
            "train": graphs[:train_count],
            "val": graphs[train_count : train_count + val_count],
            "test": graphs[train_count + val_count : train_count + val_count + test_count],
        }

    if args.max_graphs is not None:
        graphs = graphs[: int(args.max_graphs)]

    n = len(graphs)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    return {
        "train": graphs[:n_train],
        "val": graphs[n_train : n_train + n_val],
        "test": graphs[n_train + n_val :],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare ZINC from a SMILES/RDKit source with explicit atomic-number node labels."
    )
    parser.add_argument("--csv", required=True, help="CSV containing at least one SMILES column.")
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--split-col", default=None, help="Optional train/val/test split column.")
    parser.add_argument("--target-col", default=None, help="Optional regression target column.")
    parser.add_argument("--id-col", default=None, help="Optional molecule id column.")
    parser.add_argument("--config", default="configs/datasets/zinc.yaml")
    parser.add_argument("--output-root", default="outputs/datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-hs", action="store_true", help="Keep explicit hydrogens. Leave off for heavy-atom ZINC.")
    parser.add_argument("--max-graphs", type=int, default=None)

    # Use these if no split column is available and you want a 12k-style split.
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--val-count", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=None)

    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg["seed"] = int(args.seed)
    cfg["source"] = "smiles_rdkit"
    cfg["smiles_path"] = str(Path(args.csv).resolve())
    cfg["smiles_column"] = args.smiles_col
    cfg["split_column"] = args.split_col
    cfg["target_column"] = args.target_col
    cfg["id_column"] = args.id_col
    cfg["preparation_script"] = "scripts/prepare_zinc_from_smiles.py"
    cfg["requires_atomic_numbers"] = True
    cfg["node_atomic_number_attr"] = "atomic_number"
    cfg["keep_hs"] = bool(args.keep_hs)
    cfg["max_graphs"] = args.max_graphs
    cfg["split"] = {"train": float(args.train_frac), "val": float(args.val_frac), "test": max(0.0, 1.0 - float(args.train_frac) - float(args.val_frac))}
    if args.train_count is not None:
        cfg["split_counts"] = {"train": int(args.train_count), "val": int(args.val_count or 0), "test": int(args.test_count or 0)}

    # Not needed when raw node labels are explicit strings like atomic_number=6.
    cfg["rdkit_atomic_number_mapping"] = None

    # This old diagnostic mapping is specific to PyG categorical ZINC and is stale
    # for the SMILES/RDKit-prepared dataset.
    cfg.pop("canonical_node_label_to_raw_atom_type", None)

    items = _read_graphs(args)
    splits = _make_splits(items, args)
    atomic_number_stats = assert_zinc_atomic_numbers(
        splits,
        context="scripts/prepare_zinc_from_smiles.py",
    )

    save_dataset_splits(
        "zinc",
        splits,
        cfg,
        output_root=args.output_root,
        force=args.force,
    )

    print("Saved ZINC splits:", {k: len(v) for k, v in splits.items()})
    print("Atomic-number coverage:", atomic_number_stats)


if __name__ == "__main__":
    main()