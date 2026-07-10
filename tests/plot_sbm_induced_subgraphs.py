#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from grapher.data.io import load_dataset_splits
from grapher.utils.motifs import NautyCanonicalizer, aggregate_unique_motifs_with_counts


def log(message: str) -> None:
    print(f"[debug] {message}", flush=True)


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
        description="Aggregate unique connected k-node motifs across QM9 topology train graphs and plot them with frequencies."
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
    if int(args.max_graphs) > 0:
        graphs = graphs[: int(args.max_graphs)]
    print(
        f"[debug] loaded train graphs={len(graphs)} "
        f"k_min={args.k_min} k_max={args.k_max}",
        flush=True,
    )

    output_path = Path(args.output)
    output_paths: list[Path] = []
    for k in range(int(args.k_min), int(args.k_max) + 1):
        print(f"[debug] processing k={k}", flush=True)
        total_induced = sum(math.comb(g.number_of_nodes(), k) for g in graphs if g.number_of_nodes() >= k)
        print(f"[debug] total possible induced subgraphs across graphs={total_induced}", flush=True)

        canonicalizer = NautyCanonicalizer()
        print("[debug] aggregating unique motifs with counts across all train graphs", flush=True)
        motif_counts = aggregate_unique_motifs_with_counts(
            graphs,
            k,
            connected_only=True,
            canonicalizer=canonicalizer,
            progress_interval=max(int(args.progress_interval), 0),
            log_fn=log,
        )
        print(f"[debug] unique connected motif count={len(motif_counts)}", flush=True)

        keys = canonicalizer.canonical_graph6_batch([motif for motif, _ in motif_counts])
        total_count = sum(count for _, count in motif_counts)
        selected = [
            (motif, key, float(count) / max(float(total_count), 1.0))
            for (motif, count), key in zip(motif_counts, keys)
        ]
        max_subgraphs = int(args.max_subgraphs)
        if max_subgraphs > 0:
            selected = selected[:max_subgraphs]
            print(f"[debug] capped plotted motifs to max_subgraphs={max_subgraphs}", flush=True)

        for idx, (motif, key, frequency) in enumerate(selected, start=1):
            count = int(round(frequency * max(float(total_count), 1.0)))
            print(
                f"[debug] motif {idx}/{len(selected)} "
                f"canonical={key} frequency={frequency:.6f} "
                f"count={count} n={motif.number_of_nodes()} m={motif.number_of_edges()}",
                flush=True,
            )

        print(
            f"graphs={len(graphs)} k={k} total_connected_occurrences={total_count} "
            f"unique_connected_motifs={len(motif_counts)} plotted={len(selected)}"
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
