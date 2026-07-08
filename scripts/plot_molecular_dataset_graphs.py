#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from grapher.data.io import load_dataset_splits
from grapher.molecular.constants import (
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_SINGLE,
    BOND_TRIPLE,
    QM9_ATOM_SYMBOLS,
)


ATOM_COLORS = {
    6: "#4c78a8",  # C
    7: "#54a24b",  # N
    8: "#e45756",  # O
    9: "#f2cf5b",  # F
}

BOND_COLORS = {
    BOND_SINGLE: "#7f7f7f",
    BOND_DOUBLE: "#f58518",
    BOND_TRIPLE: "#b279a2",
    BOND_AROMATIC: "#72b7b2",
}
from grapher.utils.io import ensure_dir


def _node_label(graph: nx.Graph, node: int) -> str:
    data = graph.nodes[node]
    atomic_num = int(data.get("atomic_num", data.get("atom_type", 0)))
    return QM9_ATOM_SYMBOLS.get(atomic_num, str(atomic_num))


def _node_color(graph: nx.Graph, node: int) -> str:
    data = graph.nodes[node]
    atomic_num = int(data.get("atomic_num", data.get("atom_type", 0)))
    return ATOM_COLORS.get(atomic_num, "#bab0ac")


def _edge_color(graph: nx.Graph, u: int, v: int) -> str:
    data = graph.edges[u, v]
    bond_type = int(data.get("bond_type", data.get("bond_order", 1)))
    return BOND_COLORS.get(bond_type, "#7f7f7f")


def _edge_width(graph: nx.Graph, u: int, v: int) -> float:
    data = graph.edges[u, v]
    bond_type = int(data.get("bond_type", data.get("bond_order", 1)))
    if bond_type == BOND_DOUBLE:
        return 2.2
    if bond_type == BOND_TRIPLE:
        return 2.8
    return 1.6


def _draw_graph(ax, graph: nx.Graph, title: str) -> None:
    graph = nx.convert_node_labels_to_integers(nx.Graph(graph), ordering="sorted")
    pos = nx.spring_layout(graph, seed=0)
    labels = {node: _node_label(graph, node) for node in graph.nodes()}
    edge_colors = [_edge_color(graph, u, v) for u, v in graph.edges()]
    edge_widths = [_edge_width(graph, u, v) for u, v in graph.edges()]
    node_colors = [_node_color(graph, node) for node in graph.nodes()]

    nx.draw_networkx_edges(graph, pos, ax=ax, width=edge_widths, edge_color=edge_colors)
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=520,
        node_color=node_colors,
        edgecolors="#222222",
        linewidths=1.2,
    )
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=9, font_color="#ffffff")
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a sample of graphs from a molecular dataset split.")
    parser.add_argument("--dataset", default="qm9_attributed")
    parser.add_argument("--root", default="outputs/datasets")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--num-graphs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0, help="Seed used when --sample-random is set.")
    parser.add_argument("--sample-random", action="store_true", help="Sample random graphs instead of taking the first N.")
    parser.add_argument("--output-dir", default="outputs/plots/molecular_dataset")
    parser.add_argument("--filename", default=None, help="Defaults to <dataset>_<split>_<num>.png.")
    parser.add_argument("--cols", type=int, default=4)
    args = parser.parse_args()

    splits = load_dataset_splits(args.dataset, root=args.root, build_if_missing=False)
    graphs = list(splits[args.split])
    if not graphs:
        raise RuntimeError(f"No graphs found for {args.dataset}/{args.split}.")

    n = min(int(args.num_graphs), len(graphs))
    if args.sample_random:
        rng = np.random.default_rng(int(args.seed))
        indices = rng.choice(len(graphs), size=n, replace=False).tolist()
    else:
        indices = list(range(n))

    selected = [graphs[i] for i in indices]
    cols = max(1, int(args.cols))
    rows = int(math.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)

    for ax in axes.reshape(-1):
        ax.set_axis_off()
    for ax, graph, idx in zip(axes.reshape(-1), selected, indices):
        _draw_graph(
            ax,
            graph,
            f"{args.split}[{idx}] n={graph.number_of_nodes()} m={graph.number_of_edges()}",
        )

    fig.tight_layout()
    out_dir = ensure_dir(args.output_dir)
    filename = args.filename or f"{args.dataset}_{args.split}_{n}.png"
    out_path = Path(out_dir) / filename
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
