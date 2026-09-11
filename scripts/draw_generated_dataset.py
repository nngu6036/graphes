#!/usr/bin/env python3
"""Draw a random sample or all graphs from a generated NetworkX pickle.

Accepts the same --generated-graphs file as evaluate_generated_molecules.py,
including a list/tuple of graphs or a dictionary containing graphs,
molecular_graphs, or generated_graphs. Molecular input uses raw_valid RDKit
sanitization without inferred charges or bond correction. Duplicate generated
records remain separate samples; selection is without replacement by index.
Molecular records must be nonempty simple graphs with atomic numbers 1--118
and integer bond_type categories 1--4. Malformed records are excluded together
with molecules that fail sanitization, before either sampling or counting.

Uses draw_dataset's molecular/generic panels, typed induced-cycle drawings,
PDF pagination, and PNG export. Graphlet counts use every valid molecular graph
in the input file (every graph for generic input), independently of the sample.
No prepared reference splits, evaluation reports, or FCD installation are needed.

Example::

    PYTHONPATH=src python scripts/draw_generated_dataset.py \
      --generated-graphs "$GEN/molecular_graphs.pkl" --dataset qm9_attributed \
      --count 1024 --seed 42 --row 4 --col 4 --k-min 3 --k-max 5 \
      --output "$GEN/generated_1024.pdf"

Use --all to draw the entire eligible pool. If --output is omitted, a multipage
PDF is written beside the input. A *_drawing.json sidecar records the original
file indices and exclusions; *_graphlet_histogram.json records cycle counts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import networkx as nx

from grapher.utils.networkx_pickle import load_trusted_networkx_pickle

if __package__:
    from scripts import draw_dataset as draw
else:
    import draw_dataset as draw


def _load_generated_graphs(path: str | Path) -> list[nx.Graph]:
    path = Path(path).expanduser().resolve()
    payload = load_trusted_networkx_pickle(path)
    if isinstance(payload, dict):
        for key in ("graphs", "molecular_graphs", "generated_graphs"):
            if key in payload:
                payload = payload[key]
                break
    if isinstance(payload, (dict, nx.Graph, str, bytes)):
        raise TypeError(f"{path} must contain a collection of NetworkX graphs.")
    try:
        graphs = list(payload)
    except TypeError as exc:
        raise TypeError(f"{path} must contain a collection of NetworkX graphs.") from exc
    if not all(isinstance(graph, nx.Graph) for graph in graphs):
        raise TypeError(f"{path} contains non-NetworkX values.")
    if not graphs:
        raise ValueError(f"{path} contains no generated graphs.")
    return graphs


def build_parser() -> argparse.ArgumentParser:
    return draw.build_parser(generated=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    draw._validate_drawing_args(args)
    path = args.generated_graphs.expanduser().resolve()
    graphs = _load_generated_graphs(path)
    label = args.dataset or path.stem
    return draw._draw_graph_collection(
        args, graphs, path, label,
        [("generated", index) for index in range(len(graphs))], "generated",
        generated=True,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
