from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

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
from grapher.generation.molecular_rewiring import edge_type_value, node_type_value
from grapher.registry import available_datasets
from grapher.utils.io import load_pickle


MOLECULAR_DATASETS = {"qm9", "zinc"}
PLOT_MODELS = ("grapher", "grapher_molecular")
PLOT_WHAT = ("both", "train", "generated")
ATOM_COLORS = {
    1: "#f8fafc",   # H
    5: "#f59e0b",   # B
    6: "#cbd5e1",   # C
    7: "#60a5fa",   # N
    8: "#f87171",   # O
    9: "#34d399",   # F
    15: "#fbbf24",  # P
    16: "#fde047",  # S
    17: "#86efac",  # Cl
    35: "#d97706",  # Br
    53: "#a78bfa",  # I
}
BOND_COLORS = {
    1: "#475569",  # single
    2: "#2563eb",  # double
    3: "#7c3aed",  # triple
    4: "#ea580c",  # aromatic
}


def _default_model(dataset: str) -> str:
    return "grapher_molecular" if dataset.lower() in MOLECULAR_DATASETS else "grapher"


def _default_output_path(dataset: str, model: str, run_id: int | None) -> Path:
    if run_id is None:
        return Path("outputs/plots") / dataset / f"{model}.png"
    return Path("outputs/plots") / dataset / model / f"run_{run_id:03d}.png"


def _load_graphs(path: Path) -> list[nx.Graph]:
    payload = load_pickle(path)
    if isinstance(payload, dict) and "graphs" in payload:
        payload = payload["graphs"]
    if not isinstance(payload, list):
        raise TypeError(
            f"Expected a list of generated graphs or a bundle with graphs in {path}, "
            f"got {type(payload).__name__}."
        )
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


def _atom_label(graph: nx.Graph, node: int) -> str:
    try:
        atomic_number = node_type_value(graph, int(node))
    except Exception:
        return str(node)
    symbols = {
        1: "H",
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        15: "P",
        16: "S",
        17: "Cl",
        35: "Br",
        53: "I",
    }
    return symbols.get(int(atomic_number), str(int(atomic_number)))


def _bond_label(edge_type: int) -> str:
    return {1: "1", 2: "2", 3: "3", 4: "ar"}.get(int(edge_type), str(int(edge_type)))


def _atom_color(graph: nx.Graph, node: int) -> str:
    try:
        atomic_number = node_type_value(graph, int(node))
    except Exception:
        return "#e5e7eb"
    return ATOM_COLORS.get(int(atomic_number), "#e5e7eb")


def _flatten_axes(axes) -> list:
    """Flatten either a numpy-like axes array or a nested Python list of axes."""
    if hasattr(axes, "flat"):
        return list(axes.flat)
    flattened = []
    for item in axes:
        if isinstance(item, (list, tuple)):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def _style_sample_axis(ax, *, paper_style: bool) -> None:
    """Make each sample look like a small square cell as in GraphRNN-style figures."""
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.58, 0.58)
    ax.set_ylim(-0.58, 0.58)
    ax.set_facecolor("white")

    if paper_style:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.55)
            spine.set_color("black")
    else:
        ax.set_axis_off()


def _normalize_layout(pos: dict[int, tuple[float, float]], *, scale: float = 0.50) -> dict[int, tuple[float, float]]:
    """Center and scale layout coordinates so every graph fits inside the square cell."""
    if not pos:
        return pos
    xs = [float(p[0]) for p in pos.values()]
    ys = [float(p[1]) for p in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    span = max(max_x - min_x, max_y - min_y, 1e-9)
    return {node: ((float(p[0]) - cx) / span * 2.0 * scale, (float(p[1]) - cy) / span * 2.0 * scale) for node, p in pos.items()}


def _layout_graph(graph: nx.Graph, *, seed: int, layout: str) -> dict[int, tuple[float, float]]:
    if graph.number_of_nodes() == 1:
        return {next(iter(graph.nodes())): (0.0, 0.0)}
    if layout == "kamada_kawai":
        try:
            pos = nx.kamada_kawai_layout(graph)
        except Exception:
            pos = nx.spring_layout(graph, seed=seed, k=1.0 / math.sqrt(max(graph.number_of_nodes(), 1)), iterations=150)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    elif layout == "spectral":
        try:
            pos = nx.spectral_layout(graph)
        except Exception:
            pos = nx.spring_layout(graph, seed=seed, k=1.0 / math.sqrt(max(graph.number_of_nodes(), 1)), iterations=150)
    else:
        pos = nx.spring_layout(graph, seed=seed, k=1.0 / math.sqrt(max(graph.number_of_nodes(), 1)), iterations=150)
    return _normalize_layout(pos)


def _draw_topological_graph(
    ax,
    graph: nx.Graph,
    pos: dict[int, tuple[float, float]],
    *,
    node_size: float,
    edge_width: float,
    node_color: str,
) -> None:
    nx.draw_networkx_edges(graph, pos, ax=ax, width=edge_width, alpha=1.0, edge_color="black")
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=node_size,
        node_color=node_color,
        edgecolors=node_color,
        linewidths=0.0,
    )


def _draw_molecular_graph(ax, graph: nx.Graph, pos: dict[int, tuple[float, float]]) -> None:
    edge_widths = []
    edge_colors = []
    edge_labels = {}
    for u, v in graph.edges():
        try:
            edge_type = edge_type_value(graph, int(u), int(v))
        except Exception:
            edge_type = 1
        edge_widths.append({1: 0.9, 2: 1.4, 3: 1.9, 4: 1.2}.get(int(edge_type), 0.9))
        edge_colors.append(BOND_COLORS.get(int(edge_type), "#475569"))
        edge_labels[(u, v)] = _bond_label(edge_type)
    node_size = max(160, min(420, int(2600 / max(graph.number_of_nodes(), 1))))
    nx.draw_networkx_edges(graph, pos, ax=ax, width=edge_widths, alpha=0.95, edge_color=edge_colors)
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=node_size,
        node_color=[_atom_color(graph, int(node)) for node in graph.nodes()],
        edgecolors="#111827",
        linewidths=0.7,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={node: _atom_label(graph, int(node)) for node in graph.nodes()},
        ax=ax,
        font_size=8,
        font_color="black",
        font_weight="bold",
    )
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        ax=ax,
        font_size=6,
        font_color="black",
        rotate=False,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.2},
    )


def _draw_graph_grid(
    *,
    axes,
    graphs: list[nx.Graph],
    start_index: int,
    seed: int,
    molecular: bool,
    paper_style: bool,
    show_graph_titles: bool,
    node_size: float,
    edge_width: float,
    node_color: str,
    layout: str,
) -> None:
    for slot, ax in enumerate(_flatten_axes(axes)):
        _style_sample_axis(ax, paper_style=paper_style)
        if slot >= len(graphs):
            if not paper_style:
                ax.set_axis_off()
            continue

        graph_index = start_index + slot
        graph = graphs[slot]
        if graph.number_of_nodes() == 0:
            if show_graph_titles:
                ax.set_title(f"#{graph_index} | n=0 m=0", fontsize=9)
            continue

        layout_seed = int(seed) + graph_index
        pos = _layout_graph(graph, seed=layout_seed, layout=layout)
        if molecular:
            _draw_molecular_graph(ax, graph, pos)
        else:
            _draw_topological_graph(
                ax,
                graph,
                pos,
                node_size=node_size,
                edge_width=edge_width,
                node_color=node_color,
            )
        if show_graph_titles:
            ax.set_title(f"#{graph_index} | n={graph.number_of_nodes()} m={graph.number_of_edges()}", fontsize=9)


def _make_axes_grid(fig, outer_slot, row: int, column: int, *, wspace: float, hspace: float):
    inner = outer_slot.subgridspec(row, column, wspace=wspace, hspace=hspace)
    return [[fig.add_subplot(inner[r, c]) for c in range(column)] for r in range(row)]


def plot_generated_graphs(
    *,
    dataset: str,
    model: str,
    run_id: int | None,
    row: int,
    column: int,
    dataset_root: str = "outputs/datasets",
    sample_file: str | None = None,
    output: str | None = None,
    plot_what: str = "both",
    train_start_index: int = 0,
    start_index: int = 0,
    seed: int = 42,
    dpi: int = 180,
    paper_style: bool = False,
    train_caption: str = "(d) Training data",
    generated_caption: str = "(e) Generated samples",
    single_caption: str | None = None,
    show_graph_titles: bool | None = None,
    node_size: float | None = None,
    edge_width: float | None = None,
    node_color: str = "#1f77b4",
    layout: str = "spring",
) -> Path:
    row = int(row)
    column = int(column)
    if row <= 0 or column <= 0:
        raise ValueError("--row and --column must be positive integers.")
    if train_start_index < 0:
        raise ValueError("--train-start-index must be non-negative.")
    if start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    if model not in PLOT_MODELS:
        raise ValueError(f"--model must be one of {PLOT_MODELS}; got {model!r}.")
    if plot_what not in PLOT_WHAT:
        raise ValueError(f"--plot must be one of {PLOT_WHAT}; got {plot_what!r}.")

    path = Path(sample_file) if sample_file else sample_path(dataset, model, run_id=run_id)
    if plot_what in {"both", "generated"} and not path.exists():
        raise FileNotFoundError(f"Generated sample file not found: {path}")

    splits = load_dataset_splits(dataset, output_root=dataset_root, build_if_missing=True)
    if "train" not in splits:
        raise KeyError(f"Dataset {dataset!r} is missing split 'train'; available={sorted(splits)}")
    train_graphs = [nx.convert_node_labels_to_integers(nx.Graph(item), ordering="sorted") for item in splits["train"]]
    generated_graphs = _load_graphs(path) if plot_what in {"both", "generated"} else []
    molecular = model == "grapher_molecular"
    total = row * column
    selected_train = _select_graphs(train_graphs, start_index=train_start_index, total=total, label="training") if plot_what in {"both", "train"} else []
    selected_generated = _select_graphs(generated_graphs, start_index=start_index, total=total, label="generated") if plot_what in {"both", "generated"} else []

    if show_graph_titles is None:
        show_graph_titles = not paper_style
    if node_size is None:
        node_size = 8.0 if paper_style else 35.0
    if edge_width is None:
        edge_width = 0.65 if paper_style else 0.9

    num_panels = 2 if plot_what == "both" else 1
    if paper_style:
        cell_size = 0.63
        panel_gap = 0.55 if num_panels == 2 else 0.0
        caption_space = 0.36
        fig_width = max(num_panels * column * cell_size + panel_gap, 2.2)
        fig_height = max(row * cell_size + caption_space, 1.5)
        left, right, top, bottom = 0.02, 0.98, 0.98, 0.20
        outer_wspace = 0.26 if num_panels == 2 else 0.0
        inner_wspace, inner_hspace = 0.18, 0.18
        caption_fontsize = 13
    else:
        fig_width = max(3.0 * column * num_panels, 3.0 * num_panels)
        fig_height = max(3.0 * row, 3.0)
        left, right, top, bottom = 0.04, 0.98, 0.92, 0.08
        outer_wspace = 0.10 if num_panels == 2 else 0.0
        inner_wspace, inner_hspace = 0.15, 0.25
        caption_fontsize = 11

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
    outer = fig.add_gridspec(
        1,
        num_panels,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        wspace=outer_wspace,
    )

    if plot_what == "both":
        train_axes = _make_axes_grid(fig, outer[0, 0], row, column, wspace=inner_wspace, hspace=inner_hspace)
        generated_axes = _make_axes_grid(fig, outer[0, 1], row, column, wspace=inner_wspace, hspace=inner_hspace)
        _draw_graph_grid(
            axes=train_axes,
            graphs=selected_train,
            start_index=train_start_index,
            seed=seed,
            molecular=molecular,
            paper_style=paper_style,
            show_graph_titles=show_graph_titles,
            node_size=node_size,
            edge_width=edge_width,
            node_color=node_color,
            layout=layout,
        )
        _draw_graph_grid(
            axes=generated_axes,
            graphs=selected_generated,
            start_index=start_index,
            seed=seed,
            molecular=molecular,
            paper_style=paper_style,
            show_graph_titles=show_graph_titles,
            node_size=node_size,
            edge_width=edge_width,
            node_color=node_color,
            layout=layout,
        )
        if paper_style:
            fig.text(0.255, 0.055, train_caption, ha="center", va="center", fontsize=caption_fontsize, family="serif")
            fig.text(0.745, 0.055, generated_caption, ha="center", va="center", fontsize=caption_fontsize, family="serif")
        else:
            fig.text(0.255, 0.025, train_caption, ha="center", va="center", fontsize=caption_fontsize)
            fig.text(0.745, 0.025, generated_caption, ha="center", va="center", fontsize=caption_fontsize)
    else:
        axes = _make_axes_grid(fig, outer[0, 0], row, column, wspace=inner_wspace, hspace=inner_hspace)
        graphs = selected_train if plot_what == "train" else selected_generated
        graph_start_index = train_start_index if plot_what == "train" else start_index
        _draw_graph_grid(
            axes=axes,
            graphs=graphs,
            start_index=graph_start_index,
            seed=seed,
            molecular=molecular,
            paper_style=paper_style,
            show_graph_titles=show_graph_titles,
            node_size=node_size,
            edge_width=edge_width,
            node_color=node_color,
            layout=layout,
        )
        caption = single_caption
        if caption is None:
            caption = train_caption if plot_what == "train" else generated_caption
        fig.text(0.5, 0.055 if paper_style else 0.025, caption, ha="center", va="center", fontsize=caption_fontsize, family="serif" if paper_style else None)

    out = Path(output) if output else _default_output_path(dataset, model, run_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training and generated GraphER samples side by side.")
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--model", choices=PLOT_MODELS, default=None, help="Defaults to grapher_molecular for QM9/ZINC and grapher otherwise.")
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--row", type=int, default=1)
    parser.add_argument("--column", type=int, default=1)
    parser.add_argument("--dataset-root", type=str, default="outputs/datasets")
    parser.add_argument("--sample-file", type=str, default=None, help="Optional explicit generated sample pickle path.")
    parser.add_argument("--output", type=str, default=None, help="Optional output image path. Defaults to outputs/plots/<dataset>/<model>/run_XXX.png.")
    parser.add_argument("--plot", choices=PLOT_WHAT, default="both", help="Choose whether to plot both panels, training graphs only, or generated graphs only.")
    parser.add_argument("--train-start-index", type=int, default=0, help="Index of the first training graph to plot.")
    parser.add_argument("--start-index", type=int, default=0, help="Index of the first generated graph to plot.")
    parser.add_argument("--seed", type=int, default=42, help="Layout seed.")
    parser.add_argument("--dpi", type=int, default=180)

    # Paper-style controls: use these to reproduce GraphRNN-like qualitative panels.
    parser.add_argument("--paper-style", action="store_true", help="Use compact square cells, visible borders, no graph titles, and bottom panel captions.")
    parser.add_argument("--train-caption", type=str, default="(d) Training data")
    parser.add_argument("--generated-caption", type=str, default="(e) Generated samples")
    parser.add_argument("--single-caption", type=str, default=None)
    parser.add_argument("--show-graph-titles", action="store_true", help="Show per-sample graph index/title even in paper style.")
    parser.add_argument("--node-size", type=float, default=None)
    parser.add_argument("--edge-width", type=float, default=None)
    parser.add_argument("--node-color", type=str, default="#1f77b4")
    parser.add_argument("--layout", choices=("spring", "kamada_kawai", "circular", "spectral"), default="spring")
    args = parser.parse_args()

    output = plot_generated_graphs(
        dataset=args.dataset,
        model=args.model or _default_model(args.dataset),
        run_id=args.run_id,
        row=args.row,
        column=args.column,
        dataset_root=args.dataset_root,
        sample_file=args.sample_file,
        output=args.output,
        plot_what=args.plot,
        train_start_index=args.train_start_index,
        start_index=args.start_index,
        seed=args.seed,
        dpi=args.dpi,
        paper_style=args.paper_style,
        train_caption=args.train_caption,
        generated_caption=args.generated_caption,
        single_caption=args.single_caption,
        show_graph_titles=args.show_graph_titles if args.show_graph_titles else None,
        node_size=args.node_size,
        edge_width=args.edge_width,
        node_color=args.node_color,
        layout=args.layout,
    )
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()
