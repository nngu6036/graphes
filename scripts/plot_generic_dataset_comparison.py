#!/usr/bin/env python3
"""Save separate training/generated graph grids for Community-small or Ego-small.

Run from the GraphES repository root, for example:
    python scripts/plot_generic_dataset_comparison.py --dataset community_small \
        --row 3 --column 3 --generated-path /path/to/generations/run/ablations/TR

Only trusted project NetworkX pickles should be supplied. Nodes are coloured by
communities detected independently in each graph, not by ground-truth labels.
Sampling is uniform without replacement by file index; disconnected and empty
graphs are retained. Neither graph topology nor the input files are modified.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Work on servers without a display.
import matplotlib.pyplot as plt
import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from grapher.utils.networkx_pickle import load_trusted_networkx_pickle

SERIALIZED = {"community_small": "sbm", "ego_small": "ego_small"}


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def generated_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "base_graphs.pkl"
    if not path.is_file():
        raise ValueError(f"Generated graph file not found: {path}. Pass a .pkl file or a directory containing base_graphs.pkl.")
    return path


def load_graphs(path: Path) -> list[nx.Graph]:
    if not path.is_file():
        raise ValueError(f"Training graph file not found: {path}. Set --dataset-root or --training-path.")
    data = load_trusted_networkx_pickle(path)
    if isinstance(data, dict):
        for key in ("graphs", "generated_graphs", "base_graphs", "train"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, (list, tuple)) or not data:
        raise ValueError(f"{path} must contain a nonempty list/tuple of NetworkX graphs (or a dictionary wrapping one).")
    for index, graph in enumerate(data):
        if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
            raise ValueError(f"{path}: record {index} is not an undirected, non-multigraph NetworkX graph.")
    return list(data)


def community_colours(graph: nx.Graph) -> list:
    """Greedy modularity on unweighted topology; deterministic for saved order."""
    nodes = list(graph)
    if not nodes:
        return []
    if graph.number_of_edges() == 0:
        return [plt.get_cmap("coolwarm")(0.15)] * len(nodes)
    # Relabel the drawing copy so arbitrary/mixed node labels are also supported.
    indexed = nx.convert_node_labels_to_integers(graph, ordering="default")
    communities = list(nx.community.greedy_modularity_communities(indexed, weight=None))
    communities.sort(key=lambda group: (-len(group), min(group)))
    assignment = {node: rank for rank, group in enumerate(communities) for node in group}
    cmap = plt.get_cmap("coolwarm")
    if len(communities) == 1:
        return [cmap(0.15)] * len(nodes)
    return [cmap(0.08 + 0.84 * assignment[index] / (len(communities) - 1)) for index in range(len(nodes))]


def plot_grid(graphs: list[nx.Graph], indices: list[int], path: Path, args) -> None:
    figure, axes = plt.subplots(args.rows, args.cols, squeeze=False,
                                figsize=(args.panel_size * args.cols, args.panel_size * args.rows),
                                facecolor="white")
    figure.subplots_adjust(left=0.025, right=0.975, bottom=0.025, top=0.975, wspace=0.04, hspace=0.04)
    try:
        for ax, index in zip(axes.flat, indices):
            # Integer labels give stable layout/colour behavior without touching input.
            graph = nx.convert_node_labels_to_integers(graphs[index], ordering="default")
            ax.set_axis_off()
            ax.set_aspect("equal", adjustable="box")
            if not graph:
                ax.text(0.5, 0.5, "Empty graph", ha="center", va="center", transform=ax.transAxes,
                        color="#777777", fontsize=9)
            else:
                layout_seed = (args.seed + index * 104729) % (2**32)
                positions = nx.spring_layout(graph, seed=layout_seed, iterations=200, weight=None,
                                             k=1.6 / math.sqrt(len(graph)))
                nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="#808080",
                                       width=args.edge_width, alpha=0.70)
                nx.draw_networkx_nodes(graph, positions, ax=ax, node_color=community_colours(graph),
                                       node_size=args.node_size, linewidths=0.25, edgecolors="white")
                ax.margins(0.13)
            if args.show_indices:
                ax.set_title(f"Graph {index}", fontsize=9, color="#555555", pad=1)
        figure.savefig(path, dpi=args.dpi, facecolor="white", transparent=False)
    finally:
        plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=tuple(SERIALIZED))
    parser.add_argument("--row", "--rows", dest="rows", required=True, type=positive_int)
    parser.add_argument("--column", "--col", "--cols", dest="cols", required=True, type=positive_int)
    parser.add_argument("--generated-path", "--generated-graphs", "--generated-dir", dest="generated_path",
                        required=True, type=Path, help="Graph pickle or directory containing base_graphs.pkl.")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "outputs/datasets",
                        help="Prepared dataset root (default: <repository>/outputs/datasets).")
    parser.add_argument("--training-path", type=Path, help="Override the automatically selected train.pkl.")
    parser.add_argument("--output-dir", type=Path, help="Default: plots/ beside the generated pickle.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling and layout seed (default: 42).")
    parser.add_argument("--dpi", type=positive_int, default=300)
    parser.add_argument("--panel-size", type=positive_float, default=2.8, help="Width/height in inches of each panel.")
    parser.add_argument("--node-size", type=positive_float, default=28, help="Node area in points squared.")
    parser.add_argument("--edge-width", type=positive_float, default=0.65)
    parser.add_argument("--show-indices", action="store_true", help="Show original file indices above panels.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        source = generated_file(args.generated_path)
        training = (args.training_path or args.dataset_root / SERIALIZED[args.dataset] / "train.pkl").expanduser().resolve()
        train_graphs = load_graphs(training)
        generated_graphs = load_graphs(source)
        count = args.rows * args.cols
        for label, graphs in (("training", train_graphs), ("generated", generated_graphs)):
            if len(graphs) < count:
                raise ValueError(f"Requested {count} panels but {label} contains only {len(graphs)} graphs. Reduce --row/--column.")
        # Separate RNG streams keep the selected training examples fixed when
        # comparing different generators with the same grid and seed.
        train_indices = random.Random(args.seed).sample(range(len(train_graphs)), count)
        generated_indices = random.Random(args.seed + 1).sample(range(len(generated_graphs)), count)
        output = (args.output_dir or source.parent / "plots").expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        suffix = f"{args.rows}x{args.cols}"
        train_png = output / f"{args.dataset}_training_{suffix}.png"
        generated_png = output / f"{args.dataset}_generated_{suffix}.png"
        plot_grid(train_graphs, train_indices, train_png, args)
        plot_grid(generated_graphs, generated_indices, generated_png, args)
        selection_path = output / f"{args.dataset}_selection_{suffix}.json"
        selection_path.write_text(json.dumps({
            "dataset": args.dataset, "rows": args.rows, "columns": args.cols, "seed": args.seed,
            "sampling": "uniform without replacement by file index; no graph filtering",
            "colour": "greedy modularity communities; local to each graph; coolwarm palette",
            "training_file": str(training), "generated_file": str(source),
            "training_indices": train_indices, "generated_indices": generated_indices,
            "training_png": str(train_png), "generated_png": str(generated_png),
            "dpi": args.dpi, "panel_size_inches": args.panel_size,
            "node_size": args.node_size, "edge_width": args.edge_width,
            "show_indices": args.show_indices,
            "networkx_version": nx.__version__, "matplotlib_version": matplotlib.__version__,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Training graphs: {training} ({len(train_graphs)} available)")
        print(f"Generated graphs: {source} ({len(generated_graphs)} available)")
        print(f"Saved {train_png}\nSaved {generated_png}\nSaved {selection_path}")
        return 0
    except (OSError, ValueError, TypeError, nx.NetworkXException) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
