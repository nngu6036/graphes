#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import itertools

from grapher.data.io import load_dataset_splits
from grapher.molecular.constants import (
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_SINGLE,
    BOND_TRIPLE,
    QM9_ATOM_SYMBOLS,
)
from grapher.utils.io import ensure_dir


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


def _graph_statistics(graphs: list[nx.Graph]) -> dict[str, object]:
    node_counts = np.asarray([g.number_of_nodes() for g in graphs], dtype=float)
    edge_counts = np.asarray([g.number_of_edges() for g in graphs], dtype=float)

    atom_counts: Counter[str] = Counter()
    bond_counts: Counter[str] = Counter()

    for graph in graphs:
        for node in graph.nodes():
            atom_counts[_node_label(graph, int(node))] += 1

        for u, v in graph.edges():
            data = graph.edges[u, v]
            bond_type = int(data.get("bond_type", data.get("bond_order", 1)))

            if bond_type == BOND_SINGLE:
                name = "single"
            elif bond_type == BOND_DOUBLE:
                name = "double"
            elif bond_type == BOND_TRIPLE:
                name = "triple"
            elif bond_type == BOND_AROMATIC:
                name = "aromatic"
            else:
                name = str(bond_type)

            bond_counts[name] += 1

    def summarize(values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {"min": 0.0, "mean": 0.0, "max": 0.0}

        return {
            "min": float(values.min()),
            "mean": float(values.mean()),
            "max": float(values.max()),
        }

    return {
        "num_graphs": len(graphs),
        "nodes": summarize(node_counts),
        "edges": summarize(edge_counts),
        "atom_counts": dict(sorted(atom_counts.items())),
        "bond_counts": dict(sorted(bond_counts.items())),
    }


def _print_statistics(name: str, graphs: list[nx.Graph]) -> None:
    stats = _graph_statistics(graphs)

    print(f"{name}:")
    print(f"  graphs: {stats['num_graphs']}")

    print(
        "  nodes: "
        f"min={stats['nodes']['min']:.0f} "
        f"mean={stats['nodes']['mean']:.2f} "
        f"max={stats['nodes']['max']:.0f}"
    )

    print(
        "  edges: "
        f"min={stats['edges']['min']:.0f} "
        f"mean={stats['edges']['mean']:.2f} "
        f"max={stats['edges']['max']:.0f}"
    )

    print(f"  atoms: {stats['atom_counts']}")
    print(f"  bonds: {stats['bond_counts']}")


def _draw_graph(
    ax,
    graph: nx.Graph,
    title: str,
    *,
    use_color: bool = True,
    show_node_labels: bool = True,
) -> None:
    graph = nx.convert_node_labels_to_integers(nx.Graph(graph), ordering="sorted")
    pos = nx.spring_layout(graph, seed=0)

    labels = {node: _node_label(graph, node) for node in graph.nodes()}
    edge_widths = [_edge_width(graph, u, v) for u, v in graph.edges()]

    if use_color:
        node_colors = [_node_color(graph, node) for node in graph.nodes()]
        edge_colors = [_edge_color(graph, u, v) for u, v in graph.edges()]
        font_color = "#ffffff"
    else:
        node_colors = "#f2f2f2"
        edge_colors = "#555555"
        font_color = "black"

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        width=edge_widths,
        edge_color=edge_colors,
    )

    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=520,
        node_color=node_colors,
        edgecolors="#222222",
        linewidths=1.2,
    )

    if show_node_labels:
        nx.draw_networkx_labels(
            graph,
            pos,
            labels=labels,
            ax=ax,
            font_size=9,
            font_color=font_color,
        )

    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def simplify_graph(graph: nx.Graph) -> nx.Graph:
    """
    Simplify a molecular graph by removing unnecessary attributes and
    keeping only atom type and bond type.
    """
    simple_graph = nx.Graph()

    for node, data in graph.nodes(data=True):
        atomic_num = int(data.get("atomic_num", data.get("atom_type", 0)))
        simple_graph.add_node(node, atomic_num=atomic_num)

    for u, v, data in graph.edges(data=True):
        bond_type = int(data.get("bond_type", data.get("bond_order", 1)))
        simple_graph.add_edge(u, v, bond_type=bond_type)

    return simple_graph


def _plot_grid(
    graphs: list[nx.Graph],
    indices: list[int],
    *,
    split: str,
    cols: int,
    output_path: Path,
    use_color: bool,
    show_node_labels: bool,
) -> None:
    rows = int(math.ceil(len(graphs) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.2 * cols, 3.0 * rows),
        squeeze=False,
    )

    for ax in axes.reshape(-1):
        ax.set_axis_off()

    for ax, graph, idx in zip(axes.reshape(-1), graphs, indices):
        _draw_graph(
            ax,
            graph,
            f"{split}[{idx}] n={graph.number_of_nodes()} m={graph.number_of_edges()}",
            use_color=use_color,
            show_node_labels=show_node_labels,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def canonical_key_bruteforce(H: nx.Graph) -> str:
    """
    Exact canonical key for a small simple undirected graph.

    Two isomorphic graphs will have the same key.
    This is suitable for small fixed k, e.g. k <= 6.
    """
    nodes = list(H.nodes())
    k = len(nodes)

    best_key = None

    for perm in itertools.permutations(nodes):
        bits = []

        for i in range(k):
            for j in range(i + 1, k):
                if H.has_edge(perm[i], perm[j]):
                    bits.append("1")
                else:
                    bits.append("0")

        key = "".join(bits)

        if best_key is None or key < best_key:
            best_key = key

    return best_key


def list_motifs(G: nx.Graph, k: int) -> list[nx.Graph]:
    """
    Return all unique connected induced motifs of size k from one graph.
    """

    if G.is_directed():
        raise ValueError("G must be undirected.")

    if any(u == v for u, v in G.edges()):
        raise ValueError("G must be simple: no self-loops.")

    if k <= 0:
        raise ValueError("k must be positive.")

    if k > G.number_of_nodes():
        return []

    motifs = []
    seen_keys = set()

    for node_subset in itertools.combinations(G.nodes(), k):
        H = G.subgraph(node_subset).copy()

        if not nx.is_connected(H):
            continue

        key = canonical_key_bruteforce(H)

        if key not in seen_keys:
            seen_keys.add(key)

            H = nx.convert_node_labels_to_integers(H, ordering="sorted")
            motifs.append(H)

    return motifs


def aggregate_unique_motifs(graphs: list[nx.Graph], k: int) -> list[nx.Graph]:
    """
    Given a list of graphs, return the unique connected induced k-node motifs
    appearing in any graph.

    Isomorphic motifs are merged, so only one representative is kept.
    """

    unique_motifs = []
    seen_keys = set()

    for G in graphs:
        motifs = list_motifs(G, k)

        for motif in motifs:
            key = canonical_key_bruteforce(motif)

            if key not in seen_keys:
                seen_keys.add(key)

                motif = nx.convert_node_labels_to_integers(
                    motif,
                    ordering="sorted",
                )

                unique_motifs.append(motif)

    return unique_motifs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a sample of graphs from a molecular dataset split."
    )

    parser.add_argument("--dataset", default="qm9_attributed")
    parser.add_argument("--root", default="outputs/datasets")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--num-graphs", type=int, default=16)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used when --sample-random is set.",
    )
    parser.add_argument(
        "--sample-random",
        action="store_true",
        help="Sample random graphs instead of taking the first N.",
    )
    parser.add_argument("--output-dir", default="outputs/plots/molecular_dataset")
    parser.add_argument(
        "--filename",
        default=None,
        help="Defaults to <dataset>_<split>_<num>.png.",
    )
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--motif-size", type=int, default=4, help="Node count k for unique connected induced motifs.")
    parser.add_argument("--max-motifs", type=int, default=64, help="Maximum number of unique motifs to plot.")
    parser.add_argument(
        "--stats-all-splits",
        action="store_true",
        help="Print statistics for train/val/test, not only the requested split.",
    )

    args = parser.parse_args()

    splits = load_dataset_splits(
        args.dataset,
        root=args.root,
        build_if_missing=False,
    )

    graphs = list(splits[args.split])

    if not graphs:
        raise RuntimeError(f"No graphs found for {args.dataset}/{args.split}.")

    print(f"Dataset: {args.dataset}")
    print(f"Root: {args.root}")
    print(f"Split sizes: { {split: len(items) for split, items in splits.items()} }")

    if args.stats_all_splits:
        for split, split_graphs in splits.items():
            _print_statistics(f"Split {split}", list(split_graphs))
    else:
        _print_statistics(f"Split {args.split}", graphs)

    n = min(int(args.num_graphs), len(graphs))

    if n <= 0:
        raise ValueError("--num-graphs must be positive.")

    if args.sample_random:
        rng = np.random.default_rng(int(args.seed))
        indices = rng.choice(len(graphs), size=n, replace=False).tolist()
    else:
        indices = list(range(n))

    selected = [graphs[i] for i in indices]

    _print_statistics("Selected graphs", selected)

    simplified = [simplify_graph(graph) for graph in selected]

    _print_statistics("Selected simplified graphs", simplified)
    motifs = aggregate_unique_motifs(simplified, int(args.motif_size))
    if args.max_motifs is not None and int(args.max_motifs) > 0:
        motifs = motifs[: int(args.max_motifs)]
    if motifs:
        _print_statistics(f"Unique {int(args.motif_size)}-node motifs", motifs)
    else:
        print(f"Unique {int(args.motif_size)}-node motifs: 0")

    cols = max(1, int(args.cols))
    out_dir = ensure_dir(args.output_dir)

    filename = args.filename or f"{args.dataset}_{args.split}_{n}.png"
    out_path = Path(out_dir) / filename

    simple_filename = f"{out_path.stem}_simple{out_path.suffix}"
    simple_out_path = out_path.with_name(simple_filename)
    motif_filename = f"{out_path.stem}_motifs_k{int(args.motif_size)}{out_path.suffix}"
    motif_out_path = out_path.with_name(motif_filename)

    # Original molecular graph plot: colored atoms and bonds.
    _plot_grid(
        selected,
        indices,
        split=args.split,
        cols=cols,
        output_path=out_path,
        use_color=True,
        show_node_labels=True,
    )

    # Simplified molecular graph plot: grayscale topology only.
    _plot_grid(
        simplified,
        indices,
        split=args.split,
        cols=cols,
        output_path=simple_out_path,
        use_color=False,
        show_node_labels=False,
    )
    if motifs:
        _plot_grid(
            motifs,
            list(range(len(motifs))),
            split="motif",
            cols=cols,
            output_path=motif_out_path,
            use_color=False,
            show_node_labels=False,
        )

    print(f"Saved plot to: {out_path}")
    print(f"Saved simplified plot to: {simple_out_path}")
    if motifs:
        print(f"Saved motif plot to: {motif_out_path}")


if __name__ == "__main__":
    main()
