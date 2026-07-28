#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from grapher.data.io import load_dataset_splits
from grapher.utils.motifs import aggregate_unique_attributed_k_motifs_with_counts_as_tuples


DATASET = "qm9_attributed"
NODE_LABEL_ATTR = "atomic_num"
EDGE_LABEL_ATTR = "bond_type"

ATOM_SYMBOLS = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
}
ATOM_COLORS = {
    1: "#f2f2f2",
    6: "#4d4d4d",
    7: "#4e79a7",
    8: "#e15759",
    9: "#59a14f",
}
BOND_COLORS = {
    1: "#8c8c8c",
    2: "#4e79a7",
    3: "#e15759",
    4: "#59a14f",
}


def log(message: str) -> None:
    print(f"[debug] {message}", flush=True)


def node_label(graph: nx.Graph, node: int) -> str:
    value = int(graph.nodes[node].get(NODE_LABEL_ATTR, graph.nodes[node].get("atom_type", 0)))
    return ATOM_SYMBOLS.get(value, str(value))


def node_color(graph: nx.Graph, node: int) -> str:
    value = int(graph.nodes[node].get(NODE_LABEL_ATTR, graph.nodes[node].get("atom_type", 0)))
    return ATOM_COLORS.get(value, "#bab0ac")


def edge_color(graph: nx.Graph, u: int, v: int) -> str:
    value = int(graph.edges[u, v].get(EDGE_LABEL_ATTR, graph.edges[u, v].get("bond_order", 1)))
    return BOND_COLORS.get(value, "#7f7f7f")


def edge_width(graph: nx.Graph, u: int, v: int) -> float:
    value = int(graph.edges[u, v].get(EDGE_LABEL_ATTR, graph.edges[u, v].get("bond_order", 1)))
    return {1: 1.4, 2: 2.2, 3: 3.0, 4: 1.8}.get(value, 1.4)


def plot_attributed_motifs(rows: list[tuple[nx.Graph, int, float]], *, output_path: Path, columns: int) -> None:
    if not rows:
        raise ValueError("No attributed motifs to plot.")

    columns = max(int(columns), 1)
    rows_count = int(math.ceil(len(rows) / columns))
    fig, axes = plt.subplots(rows_count, columns, figsize=(3.2 * columns, 3.0 * rows_count), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for idx, (ax, (motif, count, normalized_count)) in enumerate(zip(axes.ravel(), rows), start=1):
        motif = nx.convert_node_labels_to_integers(motif, ordering="sorted")
        log(
            f"plotting attributed motif {idx}/{len(rows)} "
            f"count={count} normalized_count={normalized_count:.4f} "
            f"n={motif.number_of_nodes()} m={motif.number_of_edges()}"
        )
        pos = nx.spring_layout(motif, seed=0)
        nx.draw_networkx_nodes(
            motif,
            pos,
            ax=ax,
            node_size=420,
            node_color=[node_color(motif, node) for node in motif.nodes()],
            edgecolors="#222222",
        )
        nx.draw_networkx_edges(
            motif,
            pos,
            ax=ax,
            edge_color=[edge_color(motif, u, v) for u, v in motif.edges()],
            width=[edge_width(motif, u, v) for u, v in motif.edges()],
        )
        nx.draw_networkx_labels(
            motif,
            pos,
            labels={node: node_label(motif, node) for node in motif.nodes()},
            ax=ax,
            font_size=8,
            font_color="#111111",
        )
        ax.set_title(f"count={count}\n{normalized_count:.4f}", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate unique connected attributed k-node motifs across QM9 attributed train graphs and plot them."
    )
    parser.add_argument("--root", default="outputs/datasets")
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument(
        "--max-subgraphs",
        type=int,
        default=48,
        help="Maximum number of top-occurrence unique attributed motifs to plot. Use 0 to plot all.",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=0,
        help="Maximum number of train graphs to aggregate. Use 0 for all train graphs.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="Print aggregation progress every N graphs within each k. Use 0 to disable.",
    )
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--output", default="outputs/plots/qm9_attributed_induced_subgraphs.png")
    args = parser.parse_args()

    if args.k_min <= 0 or args.k_max < args.k_min:
        raise ValueError("Require 1 <= --k-min <= --k-max.")

    config_path = Path("configs/datasets") / f"{DATASET}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing dataset config: {config_path}")

    log(f"loading dataset={DATASET} root={args.root} config={config_path}")
    splits = load_dataset_splits(DATASET, root=args.root, build_if_missing=True, config_path=config_path)
    graphs = list(splits["train"])
    if not graphs:
        raise RuntimeError(f"Dataset split {DATASET}/train is empty.")
    if int(args.max_graphs) > 0:
        graphs = graphs[: int(args.max_graphs)]
    log(f"loaded train graphs={len(graphs)} k_min={args.k_min} k_max={args.k_max}")

    output_path = Path(args.output)
    output_paths: list[Path] = []
    for k in range(int(args.k_min), int(args.k_max) + 1):
        log(f"processing attributed k={k}")
        total_induced = sum(math.comb(g.number_of_nodes(), k) for g in graphs if g.number_of_nodes() >= k)
        log(f"total possible induced subgraphs across graphs={total_induced}")
        log("aggregating unique attributed motifs with counts across all train graphs")
        motif_counts = aggregate_unique_attributed_k_motifs_with_counts_as_tuples(
            graphs,
            k,
            node_label_attr=NODE_LABEL_ATTR,
            edge_label_attr=EDGE_LABEL_ATTR,
            connected_only=True,
            missing_ok=False,
            preserve_original_attrs=True,
            progress_interval=max(int(args.progress_interval), 0),
            log_fn=log,
        )
        disconnected_count = sum(
            1 for motif, _ in motif_counts if not nx.is_connected(motif)
        )
        motif_counts = [
            (motif, count)
            for motif, count in motif_counts
            if nx.is_connected(motif)
        ]
        if disconnected_count:
            log(f"discarded disconnected attributed motifs={disconnected_count}")
        log(f"unique connected attributed motif count={len(motif_counts)}")
        total_count = sum(count for _, count in motif_counts)
        motif_counts = sorted(
            motif_counts,
            key=lambda item: (
                -int(item[1]),
                item[0].number_of_nodes(),
                item[0].number_of_edges(),
            ),
        )
        selected = [
            (motif, count, float(count) / max(float(len(graphs)), 1.0))
            for motif, count in motif_counts
        ]
        if int(args.max_subgraphs) > 0:
            selected = selected[: int(args.max_subgraphs)]
            log(f"capped plotted motifs to top max_subgraphs={args.max_subgraphs} by occurrence count")

        for idx, (motif, count, normalized_count) in enumerate(selected, start=1):
            log(
                f"attributed motif {idx}/{len(selected)} "
                f"count={count} normalized_count={normalized_count:.4f} "
                f"n={motif.number_of_nodes()} m={motif.number_of_edges()}"
            )

        print(
            f"graphs={len(graphs)} k={k} total_connected_occurrences={total_count} "
            f"unique_connected_attributed_motifs={len(motif_counts)} plotted={len(selected)}"
        )

        if args.k_min == args.k_max:
            k_output_path = output_path
        else:
            k_output_path = output_path.with_name(f"{output_path.stem}_k{k}{output_path.suffix}")
        plot_attributed_motifs(selected, output_path=k_output_path, columns=args.columns)
        output_paths.append(k_output_path)
        print(f"Saved plot to: {k_output_path}")

    print("Saved plots: " + ", ".join(str(path) for path in output_paths), flush=True)


if __name__ == "__main__":
    main()
