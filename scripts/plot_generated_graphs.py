from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grapher.evaluation.run_utils import sample_path
from grapher.registry import available_datasets
from grapher.utils.io import load_pickle


def _default_output_path(dataset: str, run_id: int | None) -> Path:
    if run_id is None:
        return Path("outputs/plots") / dataset / "grapher.png"
    return Path("outputs/plots") / dataset / "grapher" / f"run_{run_id:03d}.png"


def _load_graphs(path: Path) -> list[nx.Graph]:
    payload = load_pickle(path)
    if not isinstance(payload, list):
        raise TypeError(f"Expected a list of generated graphs in {path}, got {type(payload).__name__}.")
    graphs: list[nx.Graph] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, nx.Graph):
            raise TypeError(f"Expected NetworkX graphs in {path}; item {idx} is {type(item).__name__}.")
        graphs.append(nx.convert_node_labels_to_integers(nx.Graph(item), ordering="sorted"))
    return graphs


def plot_generated_graphs(
    *,
    dataset: str,
    run_id: int | None,
    row: int,
    column: int,
    sample_file: str | None = None,
    output: str | None = None,
    start_index: int = 0,
    seed: int = 42,
    dpi: int = 180,
) -> Path:
    row = int(row)
    column = int(column)
    if row <= 0 or column <= 0:
        raise ValueError("--row and --column must be positive integers.")
    if start_index < 0:
        raise ValueError("--start-index must be non-negative.")

    path = Path(sample_file) if sample_file else sample_path(dataset, "grapher", run_id=run_id)
    if not path.exists():
        raise FileNotFoundError(f"Generated sample file not found: {path}")

    graphs = _load_graphs(path)
    total = row * column
    selected = graphs[start_index : start_index + total]
    if not selected:
        raise ValueError(f"No graphs available from start index {start_index}; sample file contains {len(graphs)} graphs.")

    fig_width = max(3.0 * column, 4.0)
    fig_height = max(3.0 * row, 3.0)
    fig, axes = plt.subplots(row, column, figsize=(fig_width, fig_height), squeeze=False)

    for slot, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if slot >= len(selected):
            continue
        graph_index = start_index + slot
        graph = selected[slot]
        if graph.number_of_nodes() == 0:
            ax.set_title(f"#{graph_index} | n=0 m=0", fontsize=9)
            continue
        layout_seed = int(seed) + graph_index
        k = 1.0 / math.sqrt(max(graph.number_of_nodes(), 1))
        pos = nx.spring_layout(graph, seed=layout_seed, k=k)
        node_size = max(60, min(260, int(1800 / max(graph.number_of_nodes(), 1))))
        nx.draw_networkx_edges(graph, pos, ax=ax, width=0.8, alpha=0.55, edge_color="#4b5563")
        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            node_size=node_size,
            node_color="#2563eb",
            edgecolors="#111827",
            linewidths=0.35,
        )
        ax.set_title(f"#{graph_index} | n={graph.number_of_nodes()} m={graph.number_of_edges()}", fontsize=9)

    out = Path(output) if output else _default_output_path(dataset, run_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.0)
    fig.savefig(out, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot generated GraphER samples in a grid.")
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--column", type=int, required=True)
    parser.add_argument("--sample-file", type=str, default=None, help="Optional explicit generated sample pickle path.")
    parser.add_argument("--output", type=str, default=None, help="Optional output image path. Defaults to outputs/plots/<dataset>/grapher/run_XXX.png.")
    parser.add_argument("--start-index", type=int, default=0, help="Index of the first graph to plot.")
    parser.add_argument("--seed", type=int, default=42, help="Layout seed.")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    output = plot_generated_graphs(
        dataset=args.dataset,
        run_id=args.run_id,
        row=args.row,
        column=args.column,
        sample_file=args.sample_file,
        output=args.output,
        start_index=args.start_index,
        seed=args.seed,
        dpi=args.dpi,
    )
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()
