#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx

from grapher.molecular.attribute_flow import load_attribute_flow_checkpoint
from grapher.molecular.graph_io import graph_to_smiles, is_valid_molecular_graph
from grapher.utils.io import ensure_dir, load_pickle, save_json, save_pickle


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign QM9 atom/bond labels to generated topologies using attribute flow.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--topology-graphs", required=True, help="Path to pkl containing generated topology graphs.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-graphs", type=int, default=None)
    args = parser.parse_args()

    topologies = list(load_pickle(args.topology_graphs))
    if args.max_graphs:
        topologies = topologies[: int(args.max_graphs)]
    model = load_attribute_flow_checkpoint(args.checkpoint, device=args.device)
    out_dir = ensure_dir(args.output_dir)
    mol_graphs = []
    smiles = []
    valid = 0
    for i, topo in enumerate(topologies):
        g = model.sample_attributes(topo, steps=args.steps, temperature=args.temperature, device=args.device, seed=args.seed + i)
        mol_graphs.append(g)
        smi = graph_to_smiles(g)
        if smi is not None:
            valid += 1
            smiles.append(smi)
        else:
            smiles.append("")
    save_pickle(mol_graphs, out_dir / "molecular_graphs.pkl")
    with (out_dir / "generated.smi").open("w", encoding="utf-8") as f:
        for smi in smiles:
            if smi:
                f.write(smi + "\n")
    save_json({"num_graphs": len(mol_graphs), "valid": valid, "validity": valid / max(len(mol_graphs), 1)}, out_dir / "attribute_flow_metrics.json")
    print(f"Generated attributed molecular graphs: {len(mol_graphs)}")
    print(f"Validity: {valid}/{len(mol_graphs)} = {valid / max(len(mol_graphs), 1):.4f}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
