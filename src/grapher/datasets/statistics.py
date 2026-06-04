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
    full: bool = False,
    include_wl_hashes: bool = False,
    skip_planarity: bool = True,
    include_path_stats: bool = False,
    include_exact_isomorphism: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    for split, graphs in _flatten_splits(splits, selected_splits).items():
        stats = graph_statistics(graphs)
        row: dict[str, Any] = {
            "split": split,
            "num_graphs": stats["num_graphs"],
            "node_min": stats["nodes"]["min"],
            "node_max": stats["nodes"]["max"],
            "node_mean": stats["nodes"]["mean"],
            "edge_min": stats["edges"]["min"],
            "edge_max": stats["edges"]["max"],
            "edge_mean": stats["edges"]["mean"],
            "max_degree": stats["max_degree"]["max"],
            "avg_degree": stats["avg_degree"],
            "density_mean": stats["density"]["mean"],
            "connected_rate": stats["connected_rate"],
        }
        if not skip_planarity:
            vals = [nx.check_planarity(g)[0] for g in graphs]
            row["planarity_rate"] = float(np.mean(vals)) if vals else 0.0
        if full:
            row["avg_clustering"] = float(np.mean([nx.average_clustering(g) if g.number_of_nodes() else 0.0 for g in graphs])) if graphs else 0.0
            row["transitivity"] = float(np.mean([nx.transitivity(g) if g.number_of_edges() else 0.0 for g in graphs])) if graphs else 0.0
            row["triangles"] = float(np.mean([sum(nx.triangles(g).values()) / 3.0 for g in graphs])) if graphs else 0.0
        if include_wl_hashes:
            hashes = [nx.weisfeiler_lehman_graph_hash(g) for g in graphs]
            row["wl_unique"] = len(set(hashes))
            row["wl_duplicate_rate"] = 1.0 - (len(set(hashes)) / len(hashes)) if hashes else 0.0
        rows.append(row)
    return rows


def compute_degree_sequence_collection_statistics(sequences: Sequence[Sequence[int]], *, split: str = "all") -> dict[str, Any]:
    seqs = [list(map(int, seq)) for seq in sequences]
    lengths = np.asarray([len(seq) for seq in seqs], dtype=float)
    values = [d for seq in seqs for d in seq]
    hist = Counter(values)
    return {
        "split": split,
        "num_sequences": len(seqs),
        "length_min": float(lengths.min()) if lengths.size else 0.0,
        "length_max": float(lengths.max()) if lengths.size else 0.0,
        "length_mean": float(lengths.mean()) if lengths.size else 0.0,
        "degree_min": min(values) if values else 0,
        "degree_max": max(values) if values else 0,
        "degree_mean": float(np.mean(values)) if values else 0.0,
        "degree_histogram": dict(sorted(hist.items())),
    }


def format_graph_statistics_table(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = ["split", "num_graphs", "node_min", "node_max", "node_mean", "edge_min", "edge_max", "edge_mean", "max_degree", "avg_degree", "density_mean", "connected_rate"]
    return _format_table(rows, columns)


def format_degree_sequence_statistics_table(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = ["split", "num_sequences", "length_min", "length_max", "length_mean", "degree_min", "degree_max", "degree_mean"]
    return _format_table(rows, columns)


def _format_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    def fmt(value: Any) -> str:
        return f"{value:.4g}" if isinstance(value, float) else str(value)

    table = [[fmt(row.get(col, "")) for col in columns] for row in rows]
    widths = [len(col) for col in columns]
    for row in table:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    header = "  ".join(col.ljust(width) for col, width in zip(columns, widths))
    sep = "  ".join("-" * width for width in widths)
    body = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)) for row in table]
    return "\n".join([header, sep, *body])
