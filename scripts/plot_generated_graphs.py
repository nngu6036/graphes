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

from grapher.evaluation.data_io import load_dataset_splits
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


def _select_graphs(graphs: list[nx.Graph], *, start_index: int, total: int, label: str) -> list[nx.Graph]:
    selected = graphs[start_index : start_index + total]
    if not selected:
        raise ValueError(f"No {label} graphs available from start index {start_index}; source contains {len(graphs)} graphs.")
    return selected


def _draw_graph_grid(
    *,
    axes,
    graphs: list[nx.Graph],
    start_index: int,
    seed: int,
) -> None:
    for slot, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if slot >= len(graphs):
            continue
        graph_index = start_index + slot
        graph = graphs[slot]
        if graph.number_of_nodes() == 0:
            ax.set_title(f"#{graph_index} | n=0 m=0", fontsize=9)
            continue
        layout_seed = int(seed) + graph_index
        k = 1.0 / math.sqrt(max(graph.number_of_nodes(), 1))
        pos = nx.spring_layout(graph, seed=layout_seed, k=k)
        nx.draw_networkx_edges(graph, pos, ax=ax, width=0.9, alpha=1.0, edge_color="black")
        ax.set_title(f"#{graph_index} | n={graph.number_of_nodes()} m={graph.number_of_edges()}", fontsize=9)


def plot_generated_graphs(
    *,
    dataset: str,
    run_id: int | None,
    row: int,
    column: int,
    dataset_root: str = "outputs/datasets",
    sample_file: str | None = None,
    output: str | None = None,
    train_start_index: int = 0,
    start_index: int = 0,
    seed: int = 42,
    dpi: int = 180,
) -> Path:
    row = int(row)
    column = int(column)
    if row <= 0 or column <= 0:
        raise ValueError("--row and --column must be positive integers.")
    if train_start_index < 0:
        raise ValueError("--train-start-index must be non-negative.")
    if start_index < 0:
        raise ValueError("--start-index must be non-negative.")

    path = Path(sample_file) if sample_file else sample_path(dataset, "grapher", run_id=run_id)
    if not path.exists():
        raise FileNotFoundError(f"Generated sample file not found: {path}")

    splits = load_dataset_splits(dataset, output_root=dataset_root, build_if_missing=True)
    if "train" not in splits:
        raise KeyError(f"Dataset {dataset!r} is missing split 'train'; available={sorted(splits)}")
    train_graphs = [
        nx.convert_node_labels_to_integers(nx.Graph(item), ordering="sorted")
        for item in splits["train"]
    ]
    generated_graphs = _load_graphs(path)
    total = row * column
    selected_train = _select_graphs(train_graphs, start_index=train_start_index, total=total, label="training")
    selected_generated = _select_graphs(generated_graphs, start_index=start_index, total=total, label="generated")

    fig_width = max(6.0 * column, 6.0)
    fig_height = max(3.0 * row, 3.0)
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    subfigures = fig.subfigures(1, 2, wspace=0.04)
    train_subfigure, generated_subfigure = subfigures
    train_subfigure.suptitle("Training graph", fontsize=11)
    generated_subfigure.suptitle("Generated graph", fontsize=11)
    train_axes = train_subfigure.subplots(row, column, squeeze=False)
    generated_axes = generated_subfigure.subplots(row, column, squeeze=False)
    _draw_graph_grid(axes=train_axes, graphs=selected_train, start_index=train_start_index, seed=seed)
    _draw_graph_grid(axes=generated_axes, graphs=selected_generated, start_index=start_index, seed=seed)

    out = Path(output) if output else _default_output_path(dataset, run_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training and generated GraphER samples side by side.")
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--row", type=int, default=1)
    parser.add_argument("--column", type=int, default=1)
    parser.add_argument("--dataset-root", type=str, default="outputs/datasets")
    parser.add_argument("--sample-file", type=str, default=None, help="Optional explicit generated sample pickle path.")
    parser.add_argument("--output", type=str, default=None, help="Optional output image path. Defaults to outputs/plots/<dataset>/grapher/run_XXX.png.")
    parser.add_argument("--train-start-index", type=int, default=0, help="Index of the first training graph to plot.")
    parser.add_argument("--start-index", type=int, default=0, help="Index of the first generated graph to plot.")
    parser.add_argument("--seed", type=int, default=42, help="Layout seed.")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    output = plot_generated_graphs(
        dataset=args.dataset,
        run_id=args.run_id,
        row=args.row,
        column=args.column,
        dataset_root=args.dataset_root,
        sample_file=args.sample_file,
        output=args.output,
        train_start_index=args.train_start_index,
        start_index=args.start_index,
        seed=args.seed,
        dpi=args.dpi,
    )
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()
