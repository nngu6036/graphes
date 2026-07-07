#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from grapher.data.io import save_dataset_splits
from grapher.molecular.graph_io import graphs_from_smiles, nx_to_topology, read_smiles_file, split_graphs
from grapher.utils.io import ensure_dir, save_json, save_pickle


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare QM9 heavy-atom topology and attributed graph splits from SMILES.")
    parser.add_argument("--smiles-file", required=True, help="Path to .smi/.txt/.csv containing SMILES.")
    parser.add_argument("--smiles-column", default=None, help="Optional CSV/TSV column name for SMILES.")
    parser.add_argument("--root", default="outputs/datasets")
    parser.add_argument("--topology-name", default="qm9_topology")
    parser.add_argument("--attributed-name", default="qm9_attributed")
    parser.add_argument("--max-molecules", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-hydrogens", action="store_true")
    parser.add_argument("--no-kekulize", action="store_true")
    args = parser.parse_args()

    smiles = read_smiles_file(args.smiles_file, smiles_column=args.smiles_column)
    if args.max_molecules:
        smiles = smiles[: int(args.max_molecules)]
    graphs, errors = graphs_from_smiles(
        smiles,
        remove_h=not args.keep_hydrogens,
        kekulize=not args.no_kekulize,
    )
    if not graphs:
        raise RuntimeError(f"No valid molecules parsed from {args.smiles_file}; errors={errors}")

    attributed_splits = split_graphs(graphs, seed=args.seed)
    topology_splits = {k: [nx_to_topology(g) for g in v] for k, v in attributed_splits.items()}

    root = Path(args.root)
    config_top = {
        "name": args.topology_name,
        "source": str(args.smiles_file),
        "kind": "qm9_topology",
        "remove_h": not args.keep_hydrogens,
        "kekulize": not args.no_kekulize,
        "seed": args.seed,
    }
    config_attr = dict(config_top)
    config_attr["name"] = args.attributed_name
    config_attr["kind"] = "qm9_attributed"

    save_dataset_splits(args.topology_name, topology_splits, config_top, root=root)
    save_dataset_splits(args.attributed_name, attributed_splits, config_attr, root=root)
    save_json(
        {
            "num_input_smiles": len(smiles),
            "num_valid_graphs": len(graphs),
            "errors": errors,
            "topology_dataset": args.topology_name,
            "attributed_dataset": args.attributed_name,
            "split_sizes": {k: len(v) for k, v in topology_splits.items()},
        },
        ensure_dir(root / args.topology_name) / "qm9_prep_report.json",
    )
    print("Prepared QM9 datasets")
    print(f"  valid molecules: {len(graphs)} / {len(smiles)}")
    print(f"  errors: {errors}")
    print(f"  topology:   {root / args.topology_name}")
    print(f"  attributed: {root / args.attributed_name}")


if __name__ == "__main__":
    main()
