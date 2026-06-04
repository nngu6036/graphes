from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np

from grapher.datasets.base import graph_statistics


def _flatten_splits(splits: Mapping[str, Sequence[nx.Graph]], selected_splits: Sequence[str]) -> dict[str, list[nx.Graph]]:
    out: dict[str, list[nx.Graph]] = {}
    for split in selected_splits:
        if split == "all":
            out["all"] = [g for graphs in splits.values() for g in graphs]
        elif split in splits:
            out[split] = list(splits[split])
    return out


def compute_split_statistics(
    splits: Mapping[str, Sequence[nx.Graph]],
    *,
    selected_splits: Sequence[str] = ("all", "train", "val", "test"),
    schema: Mapping[str, Any] | None = None,
    full: bool = False,
    include_local_structure: bool = False,
    include_wl_hashes: bool = False,
    skip_planarity: bool = True,
    include_path_stats: bool = False,
    include_exact_isomorphism: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    for split, graphs in _flatten_splits(splits, selected_splits).items():
        graphs = list(graphs)
        stats = graph_statistics(graphs)
        node_counts = [g.number_of_nodes() for g in graphs]
        edge_counts = [g.number_of_edges() for g in graphs]
        avg_degrees = [(2.0 * g.number_of_edges() / g.number_of_nodes()) if g.number_of_nodes() else 0.0 for g in graphs]
        densities = [nx.density(g) if g.number_of_nodes() > 1 else 0.0 for g in graphs]
        component_counts = [nx.number_connected_components(g) if g.number_of_nodes() else 0 for g in graphs]
        isolate_counts = [nx.number_of_isolates(g) for g in graphs]
        degrees = [int(d) for g in graphs for _, d in g.degree()]
        row: dict[str, Any] = {
            "kind": "graphs",
            "split": split,
            "num_graphs": stats["num_graphs"],
            "node_count": _summary(node_counts),
            "edge_count": _summary(edge_counts),
            "avg_degree": _summary(avg_degrees),
            "density": _summary(densities),
            "component_count": _summary(component_counts),
            "isolate_count": _summary(isolate_counts),
            "degree_histogram": dict(sorted(Counter(degrees).items())),
            "node_min": stats["nodes"]["min"],
            "node_max": stats["nodes"]["max"],
            "node_mean": stats["nodes"]["mean"],
            "edge_min": stats["edges"]["min"],
            "edge_max": stats["edges"]["max"],
            "edge_mean": stats["edges"]["mean"],
            "max_degree": stats["max_degree"]["max"],
            "avg_degree_mean": stats["avg_degree"],
            "density_mean": stats["density"]["mean"],
            "connected_rate": stats["connected_rate"],
            "has_attributes": _has_attributes(graphs),
        }
        if schema is not None:
            row["schema"] = dict(schema)
        row.update(_attribute_summary(graphs))
        if not skip_planarity:
            vals = [nx.check_planarity(g)[0] for g in graphs]
            row["planarity_rate"] = float(np.mean(vals)) if vals else 0.0
        if full or include_local_structure:
            clustering = [nx.average_clustering(g) if g.number_of_nodes() else 0.0 for g in graphs]
            transitivity = [nx.transitivity(g) if g.number_of_edges() else 0.0 for g in graphs]
            triangles = [sum(nx.triangles(g).values()) / 3.0 for g in graphs]
            row["avg_clustering"] = _summary(clustering)
            row["transitivity"] = _summary(transitivity)
            row["triangle_count"] = _summary(triangles)
            row["triangles"] = row["triangle_count"]["mean"]
        if include_path_stats:
            connected_graphs = [g for g in graphs if g.number_of_nodes() and nx.is_connected(g)]
            row["diameter"] = _summary([nx.diameter(g) for g in connected_graphs])
            row["avg_shortest_path_length"] = _summary([nx.average_shortest_path_length(g) for g in connected_graphs])
        if include_wl_hashes:
            hashes = [nx.weisfeiler_lehman_graph_hash(g) for g in graphs]
            row["wl_unique"] = len(set(hashes))
            row["wl_unique_rate"] = len(set(hashes)) / len(hashes) if hashes else 0.0
            row["wl_duplicate_rate"] = 1.0 - row["wl_unique_rate"] if hashes else 0.0
        if include_exact_isomorphism:
            row["exact_isomorphism_unique"] = _exact_unique_count(graphs)
            row["exact_isomorphism_unique_rate"] = row["exact_isomorphism_unique"] / len(graphs) if graphs else 0.0
        rows.append(row)
    return rows


def compute_degree_sequence_collection_statistics(sequences: Sequence[Sequence[int]], *, split: str = "all") -> dict[str, Any]:
    seqs = [list(map(int, seq)) for seq in sequences]
    lengths = np.asarray([len(seq) for seq in seqs], dtype=float)
    values = [d for seq in seqs for d in seq]
    hist = Counter(values)
    edge_counts = [sum(seq) / 2.0 for seq in seqs]
    graphical = [nx.is_graphical(seq) for seq in seqs]
    connected_feasible = [nx.is_valid_degree_sequence_erdos_gallai(seq) and sum(seq) >= 2 * (len(seq) - 1) for seq in seqs]
    return {
        "kind": "degree_sequences",
        "split": split,
        "num_sequences": len(seqs),
        "length": _summary(lengths),
        "edge_count_implied": _summary(edge_counts),
        "length_min": float(lengths.min()) if lengths.size else 0.0,
        "length_max": float(lengths.max()) if lengths.size else 0.0,
        "length_mean": float(lengths.mean()) if lengths.size else 0.0,
        "degree_min": min(values) if values else 0,
        "degree_max": max(values) if values else 0,
        "degree_mean": float(np.mean(values)) if values else 0.0,
        "max_degree": max(values) if values else 0,
        "graphical_rate": float(np.mean(graphical)) if graphical else 0.0,
        "connected_feasible_rate": float(np.mean(connected_feasible)) if connected_feasible else 0.0,
        "degree_histogram": dict(sorted(hist.items())),
    }


def _summary(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
    }


def _has_attributes(graphs: Sequence[nx.Graph]) -> bool:
    return any(data for g in graphs for _, data in g.nodes(data=True)) or any(data for g in graphs for _, _, data in g.edges(data=True))


def _attribute_summary(graphs: Sequence[nx.Graph]) -> dict[str, Any]:
    node_labels = set()
    edge_labels = set()
    graph_labels = 0
    for graph in graphs:
        if "graph_label" in graph.graph:
            graph_labels += 1
        for _, data in graph.nodes(data=True):
            if "node_label" in data:
                node_labels.add(data["node_label"])
        for _, _, data in graph.edges(data=True):
            if "edge_type" in data:
                edge_labels.add(data["edge_type"])
    return {
        "node_label_vocab_size": len(node_labels),
        "edge_label_vocab_size": len(edge_labels),
        "graph_label_count": graph_labels,
    }


def _exact_unique_count(graphs: Sequence[nx.Graph]) -> int:
    unique: list[nx.Graph] = []
    for graph in graphs:
        if not any(nx.is_isomorphic(graph, other) for other in unique):
            unique.append(graph)
    return len(unique)


def format_graph_statistics_table(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = ["split", "num_graphs", "node_min", "node_max", "node_mean", "edge_min", "edge_max", "edge_mean", "max_degree", "avg_degree", "density_mean", "connected_rate"]
    return _format_table(rows, columns)


def format_degree_sequence_statistics_table(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = ["split", "num_sequences", "length_min", "length_max", "length_mean", "degree_min", "degree_max", "degree_mean"]
    return _format_table(rows, columns)


def _format_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, Mapping) and "mean" in value:
            value = value["mean"]
        return f"{value:.4g}" if isinstance(value, float) else str(value)

    table = [[fmt(row.get(col, "")) for col in columns] for row in rows]
    widths = [len(col) for col in columns]
    for row in table:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    header = "  ".join(col.ljust(width) for col, width in zip(columns, widths))
    sep = "  ".join("-" * width for width in widths)
    body = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)) for row in table]
    return "\n".join([header, sep, *body])
