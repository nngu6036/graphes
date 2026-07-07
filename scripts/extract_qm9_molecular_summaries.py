#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from grapher.data.io import load_dataset_splits
from grapher.molecular.topology_summary import molecular_attribute_topology_summary, molecular_topology_summary
from grapher.utils.io import ensure_dir, save_json


def aggregate(rows):
    if not rows:
        return {}
    out = {"num_graphs": len(rows)}
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float, bool))]
    for k in keys:
        vals = np.asarray([float(r[k]) for r in rows], dtype=float)
        out[k + "_mean"] = float(vals.mean())
        out[k + "_std"] = float(vals.std())
    for k, v in rows[0].items():
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            mat = np.asarray([r[k] for r in rows], dtype=float)
            out[k + "_mean"] = mat.mean(axis=0).tolist()
            out[k + "_std"] = mat.std(axis=0).tolist()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract topology and attribute-related molecular summaries for QM9.")
    parser.add_argument("--dataset", default="qm9_attributed")
    parser.add_argument("--root", default="outputs/datasets")
    parser.add_argument("--output-dir", default="outputs/molecular_summaries/qm9")
    parser.add_argument("--max-graphs", type=int, default=None)
    args = parser.parse_args()

    splits = load_dataset_splits(args.dataset, root=args.root, build_if_missing=False)
    out_dir = ensure_dir(args.output_dir)
    report = {}
    for split, graphs in splits.items():
        graphs = list(graphs)
        if args.max_graphs:
            graphs = graphs[: int(args.max_graphs)]
        topo_rows = [molecular_topology_summary(g) for g in graphs]
        attr_rows = [molecular_attribute_topology_summary(g) for g in graphs]
        with (out_dir / f"{split}_topology_summaries.jsonl").open("w", encoding="utf-8") as f:
            for row in topo_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        with (out_dir / f"{split}_attribute_topology_summaries.jsonl").open("w", encoding="utf-8") as f:
            for row in attr_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        report[split] = {
            "topology": aggregate(topo_rows),
            "attribute_topology": aggregate(attr_rows),
        }
    save_json(report, out_dir / "summary_report.json")
    print(f"Saved molecular summaries to {out_dir}")


if __name__ == "__main__":
    main()
