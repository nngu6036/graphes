#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from grapher.data.io import load_dataset_splits
from grapher.utils.motifs import NautyCanonicalizer, list_motifs


RANDOM_SEED = 0


def count_connected_induced_subgraphs(graph: nx.Graph, k: int) -> tuple[int, int]:
    total = 0
    connected = 0
    for nodes in itertools.combinations(graph.nodes(), int(k)):
        total += 1
        subgraph = graph.subgraph(nodes)
        if nx.is_connected(subgraph):
            connected += 1
    return total, connected


def plot_subgraphs(rows: list[tuple[nx.Graph, str]], *, output_path: Path, columns: int) -> None:
    if not rows:
        raise ValueError("No induced subgraphs to plot.")

    columns = max(int(columns), 1)
    rows_count = int(math.ceil(len(rows) / columns))
    fig, axes = plt.subplots(rows_count, columns, figsize=(3.2 * columns, 3.0 * rows_count), squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    for idx, (ax, (subgraph, key)) in enumerate(zip(axes.ravel(), rows), start=1):
        print(
            f"[debug] plotting motif {idx}/{len(rows)} "
            f"canonical={key} n={subgraph.number_of_nodes()} m={subgraph.number_of_edges()}",
            flush=True,
        )
        pos = nx.spring_layout(subgraph, seed=0)
        nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_size=360, node_color="#d9d9d9", edgecolors="#333333")
        nx.draw_networkx_edges(subgraph, pos, ax=ax, width=1.6, edge_color="#333333")
        nx.draw_networkx_labels(subgraph, pos, ax=ax, font_size=8)
        ax.set_title(key, fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List unique connected k-node motifs from one SBM graph and plot them with canonical keys."
    )
    parser.add_argument("--dataset", default="sbm")
    parser.add_argument("--root", default="outputs/datasets")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--max-subgraphs",
        type=int,
        default=48,
        help="Maximum number of unique motifs to list and plot. Use 0 to process all.",
    )
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--output", default="outputs/plots/sbm_induced_subgraphs.png")
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be positive.")

    config_path = Path("configs/datasets") / f"{args.dataset}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing dataset config: {config_path}")

    print(f"[debug] loading dataset={args.dataset} root={args.root} config={config_path}", flush=True)
    splits = load_dataset_splits(
        args.dataset,
        root=args.root,
        build_if_missing=True,
        config_path=config_path,
    )
    graphs = list(splits["train"])
    if not graphs:
        raise RuntimeError(f"Dataset split {args.dataset}/train is empty.")

    rng = np.random.default_rng(RANDOM_SEED)
    graph_index = int(rng.integers(0, len(graphs)))
    graph = graphs[graph_index]
    print(
        f"[debug] selected train graph index={graph_index} "
        f"n={graph.number_of_nodes()} m={graph.number_of_edges()} k={args.k}",
        flush=True,
    )

    print("[debug] counting induced subgraphs", flush=True)
    total_subgraphs, connected_subgraphs = count_connected_induced_subgraphs(graph, args.k)
    print(
        f"[debug] induced subgraph count total={total_subgraphs} connected={connected_subgraphs}",
        flush=True,
    )

    canonicalizer = NautyCanonicalizer()
    print("[debug] listing unique connected motifs", flush=True)
    motifs = list_motifs(graph, args.k, canonicalizer=canonicalizer, connected_only=True)
    print(f"[debug] unique connected motif count={len(motifs)}", flush=True)
    max_subgraphs = int(args.max_subgraphs)
    if max_subgraphs > 0:
        motifs = motifs[:max_subgraphs]
        print(f"[debug] capped plotted motifs to max_subgraphs={max_subgraphs}", flush=True)

    keys = canonicalizer.canonical_graph6_batch(motifs)
    selected = list(zip(motifs, keys))
    for idx, (motif, key) in enumerate(selected, start=1):
        print(
            f"[debug] motif {idx}/{len(selected)} "
            f"canonical={key} n={motif.number_of_nodes()} m={motif.number_of_edges()}",
            flush=True,
        )

    print(
        f"graph_index={graph_index} "
        f"n={graph.number_of_nodes()} m={graph.number_of_edges()} k={args.k} "
        f"unique_connected_motifs={len(keys)} plotted={len(selected)}"
    )

    plot_subgraphs(selected, output_path=Path(args.output), columns=args.columns)
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
