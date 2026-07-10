#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from grapher.data.io import load_dataset_splits
from grapher.utils.motifs import NautyCanonicalizer, k_induced_subgraph_canonical_strings


RANDOM_SEED = 0
MAX_EXACT_CONNECTED_COUNT = 200_000
MOTIF_SUBGRAPH_SAMPLES = 5_000


def count_connected_induced_subgraphs(graph: nx.Graph, k: int) -> tuple[int, int]:
    total = 0
    connected = 0
    for nodes in itertools.combinations(graph.nodes(), int(k)):
        total += 1
        subgraph = graph.subgraph(nodes)
        if nx.is_connected(subgraph):
            connected += 1
    return total, connected


def maybe_count_connected_induced_subgraphs(graph: nx.Graph, k: int) -> tuple[int, int | None]:
    total = math.comb(graph.number_of_nodes(), int(k))
    if total > MAX_EXACT_CONNECTED_COUNT:
        return total, None
    _, connected = count_connected_induced_subgraphs(graph, k)
    return total, connected


def plot_subgraphs(rows: list[tuple[nx.Graph, str, float]], *, output_path: Path, columns: int) -> None:
    if not rows:
        raise ValueError("No induced subgraphs to plot.")

    columns = max(int(columns), 1)
    rows_count = int(math.ceil(len(rows) / columns))
    fig, axes = plt.subplots(rows_count, columns, figsize=(3.2 * columns, 3.0 * rows_count), squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    for idx, (ax, (subgraph, key, frequency)) in enumerate(zip(axes.ravel(), rows), start=1):
        print(
            f"[debug] plotting motif {idx}/{len(rows)} "
            f"canonical={key} frequency={frequency:.6f} "
            f"n={subgraph.number_of_nodes()} m={subgraph.number_of_edges()}",
            flush=True,
        )
        pos = nx.spring_layout(subgraph, seed=0)
        nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_size=360, node_color="#d9d9d9", edgecolors="#333333")
        nx.draw_networkx_edges(subgraph, pos, ax=ax, width=1.6, edge_color="#333333")
        nx.draw_networkx_labels(subgraph, pos, ax=ax, font_size=8)
        ax.set_title(f"{key}\nf={frequency:.4f}", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List unique connected k-node motifs from one QM9 topology graph and plot them with canonical keys."
    )
    parser.add_argument("--dataset", default="qm9_topology")
    parser.add_argument("--root", default="outputs/datasets")
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument(
        "--max-subgraphs",
        type=int,
        default=48,
        help="Maximum number of unique motifs to list and plot. Use 0 to process all.",
    )
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--output", default="outputs/plots/qm9_topology_induced_subgraphs.png")
    args = parser.parse_args()

    if args.k_min <= 0 or args.k_max < args.k_min:
        raise ValueError("Require 1 <= --k-min <= --k-max.")

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
        f"n={graph.number_of_nodes()} m={graph.number_of_edges()} "
        f"k_min={args.k_min} k_max={args.k_max}",
        flush=True,
    )

    output_path = Path(args.output)
    output_paths: list[Path] = []
    for k in range(int(args.k_min), int(args.k_max) + 1):
        print(f"[debug] processing k={k}", flush=True)
        print("[debug] counting induced subgraphs", flush=True)
        total_subgraphs, connected_subgraphs = maybe_count_connected_induced_subgraphs(graph, k)
        if connected_subgraphs is None:
            print(
                f"[debug] induced subgraph count total={total_subgraphs}; "
                f"connected exact count skipped because total>{MAX_EXACT_CONNECTED_COUNT}",
                flush=True,
            )
        else:
            print(
                f"[debug] induced subgraph count total={total_subgraphs} connected={connected_subgraphs}",
                flush=True,
            )

        canonicalizer = NautyCanonicalizer()
        sample_count = min(MOTIF_SUBGRAPH_SAMPLES, total_subgraphs)
        print(
            f"[debug] listing unique connected motifs from sampled induced subgraphs "
            f"num_samples={sample_count}",
            flush=True,
        )
        sampled_keys = k_induced_subgraph_canonical_strings(
            graph,
            k,
            connected_only=True,
            unique=False,
            num_samples=sample_count,
            rng=np.random.default_rng(RANDOM_SEED + k),
            canonicalizer=canonicalizer,
        )
        key_counts = Counter(sampled_keys)
        total_key_count = sum(key_counts.values())
        keys = sorted(key_counts)
        motifs = [nx.from_graph6_bytes(key.encode("ascii")) for key in keys]
        print(f"[debug] sampled unique connected motif count={len(motifs)}", flush=True)
        max_subgraphs = int(args.max_subgraphs)
        if max_subgraphs > 0:
            keys = keys[:max_subgraphs]
            motifs = motifs[:max_subgraphs]
            print(f"[debug] capped plotted motifs to max_subgraphs={max_subgraphs}", flush=True)

        selected = [
            (motif, key, float(key_counts[key]) / max(float(total_key_count), 1.0))
            for motif, key in zip(motifs, keys)
        ]
        for idx, (motif, key, frequency) in enumerate(selected, start=1):
            print(
                f"[debug] motif {idx}/{len(selected)} "
                f"canonical={key} frequency={frequency:.6f} "
                f"count={key_counts[key]} n={motif.number_of_nodes()} m={motif.number_of_edges()}",
                flush=True,
            )

        print(
            f"graph_index={graph_index} "
            f"n={graph.number_of_nodes()} m={graph.number_of_edges()} k={k} "
            f"unique_connected_motifs={len(keys)} plotted={len(selected)}"
        )

        if args.k_min == args.k_max:
            k_output_path = output_path
        else:
            k_output_path = output_path.with_name(f"{output_path.stem}_k{k}{output_path.suffix}")
        plot_subgraphs(selected, output_path=k_output_path, columns=args.columns)
        output_paths.append(k_output_path)
        print(f"Saved plot to: {k_output_path}")

    print(
        "Saved plots: "
        + ", ".join(str(path) for path in output_paths),
        flush=True,
    )


if __name__ == "__main__":
    main()
