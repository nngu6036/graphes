#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from grapher.data.io import load_dataset_splits
from grapher.molecular.constants import BOND_TYPE_NAMES, QM9_ATOM_SYMBOLS
from grapher.utils.io import ensure_dir


def _node_label(graph: nx.Graph, node: int) -> str:
    data = graph.nodes[node]
    atomic_num = int(data.get("atomic_num", data.get("atom_type", 0)))
    return QM9_ATOM_SYMBOLS.get(atomic_num, str(atomic_num))


def _edge_label(graph: nx.Graph, u: int, v: int) -> str:
    data = graph.edges[u, v]
    bond_type = int(data.get("bond_type", data.get("bond_order", 1)))
    return BOND_TYPE_NAMES.get(bond_type, str(bond_type))


def _draw_graph(ax, graph: nx.Graph, title: str, *, show_edge_labels: bool) -> None:
    graph = nx.convert_node_labels_to_integers(nx.Graph(graph), ordering="sorted")
    pos = nx.spring_layout(graph, seed=0)
    labels = {node: _node_label(graph, node) for node in graph.nodes()}

    nx.draw_networkx_edges(graph, pos, ax=ax, width=1.4, edge_color="#777777")
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=520,
        node_color="#f4f4f4",
        edgecolors="#222222",
        linewidths=1.2,
    )
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=9)
    if show_edge_labels:
        edge_labels = {(u, v): _edge_label(graph, u, v) for u, v in graph.edges()}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax, font_size=7)
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
    parser.add_argument("--show-edge-labels", action="store_true")
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
            show_edge_labels=bool(args.show_edge_labels),
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
