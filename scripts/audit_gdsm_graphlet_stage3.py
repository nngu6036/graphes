#!/usr/bin/env python3
"""Audit GDSM graphlet-refinement degree/connectivity invariants."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.generated_dir.expanduser().resolve()
    source_path = root / "pre_refinement_graphs.pkl"
    if not source_path.is_file():
        source_path = root / "vanilla_graphs.pkl"
    source = load_pickle(source_path)
    final = load_pickle(root / "base_graphs.pkl")
    frozen = load_pickle(root / "frozen_degree_sequences.pkl")
    if not (len(source) == len(final) == len(frozen)):
        raise SystemExit("Artifact lengths do not match")
    degree_ok = []
    connected_ok = []
    changed = []
    for index, (source_graph, graph, degree) in enumerate(zip(source, final, frozen)):
        expected = [int(v) for v in degree]
        source_degree = [int(source_graph.degree(v)) for v in sorted(source_graph.nodes())]
        final_degree = [int(graph.degree(v)) for v in sorted(graph.nodes())]
        if source_degree != expected:
            raise SystemExit(f"Frozen degree artifact disagrees with pre-refinement source {index}")
        degree_ok.append(final_degree == expected)
        source_connected = source_graph.number_of_nodes() <= 1 or nx.is_connected(source_graph)
        connected_ok.append(
            (not source_connected)
            or graph.number_of_nodes() <= 1
            or nx.is_connected(graph)
        )
        changed.append(set(source_graph.edges()) != set(graph.edges()))
    report = {
        "status": "passed" if all(degree_ok) and all(connected_ok) else "failed",
        "num_graphs": len(final),
        "source_artifact": source_path.name,
        "degree_preservation_rate": float(np.mean(degree_ok)) if degree_ok else 1.0,
        "connected_source_connectivity_preservation_rate": (
            float(np.mean(connected_ok)) if connected_ok else 1.0
        ),
        "changed_rate": float(np.mean(changed)) if changed else 0.0,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
