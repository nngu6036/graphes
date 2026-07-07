#!/usr/bin/env python
from __future__ import annotations

import argparse

from grapher.molecular.dense_mixture_catflow import load_dense_mixture_catflow_checkpoint
from grapher.molecular.graph_io import graph_to_smiles
from grapher.utils.io import ensure_dir, save_json, save_pickle


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample full molecular graphs from joint dense mixture CatFlow.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample-categorical", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-nodes", type=int, default=None)
    args = parser.parse_args()

    model = load_dense_mixture_catflow_checkpoint(args.checkpoint, device=args.device)
    out_dir = ensure_dir(args.output_dir)
    mol_graphs = []
    smiles = []
    valid = 0
    unique_smiles = set()

    for i in range(int(args.num_samples)):
        g = model.sample_graph(
            num_nodes=args.num_nodes,
            steps=args.steps,
            temperature=args.temperature,
            device=args.device,
            seed=args.seed + i,
            sample_categorical=bool(args.sample_categorical),
        )
        mol_graphs.append(g)
        smi = graph_to_smiles(g)
        if smi is not None:
            valid += 1
            unique_smiles.add(smi)
            smiles.append(smi)
        else:
            smiles.append("")

    save_pickle(mol_graphs, out_dir / "molecular_graphs.pkl")
    with (out_dir / "generated.smi").open("w", encoding="utf-8") as f:
        for smi in smiles:
            if smi:
                f.write(smi + "\n")
    metrics = {
        "num_graphs": len(mol_graphs),
        "valid": valid,
        "validity": valid / max(len(mol_graphs), 1),
        "unique_valid": len(unique_smiles),
        "uniqueness_valid_only": len(unique_smiles) / max(valid, 1),
    }
    save_json(metrics, out_dir / "joint_mixture_catflow_metrics.json")
    print(f"Generated molecular graphs: {len(mol_graphs)}")
    print(f"Validity: {valid}/{len(mol_graphs)} = {metrics['validity']:.4f}")
    print(f"Unique valid: {len(unique_smiles)}/{max(valid, 1)} = {metrics['uniqueness_valid_only']:.4f}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
