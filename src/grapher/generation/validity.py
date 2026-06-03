from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import networkx as nx
import numpy as np


def graph_fingerprint(graph: nx.Graph) -> str:
    """Isomorphism-invariant fingerprint for simple graphs where possible."""
    try:
        return nx.weisfeiler_lehman_graph_hash(nx.Graph(graph))
    except Exception:
        edges = sorted(tuple(sorted(e)) for e in graph.edges())
        return repr((graph.number_of_nodes(), edges))


def _is_valid_graph(g) -> bool:
    return isinstance(g, nx.Graph) and g.number_of_nodes() > 0


def generic_validity_rate(graphs: Sequence[nx.Graph]) -> float:
    if not graphs:
        return 0.0
    return float(np.mean([_is_valid_graph(g) for g in graphs]))


def no_self_loop_rate(graphs: Sequence[nx.Graph]) -> float:
    if not graphs:
        return 0.0
    vals = []
    for g in graphs:
        try:
            vals.append(len(list(nx.selfloop_edges(g))) == 0)
        except Exception:
            vals.append(False)
    return float(np.mean(vals))


def connectedness_rate(graphs: Sequence[nx.Graph]) -> float:
    if not graphs:
        return 0.0
    vals = []
    for g in graphs:
        try:
            vals.append(g.number_of_nodes() > 0 and nx.is_connected(g))
        except Exception:
            vals.append(False)
    return float(np.mean(vals))


def planarity_rate(graphs: Sequence[nx.Graph]) -> float:
    if not graphs:
        return 0.0
    vals = []
    for g in graphs:
        try:
            vals.append(nx.check_planarity(g)[0])
        except Exception:
            vals.append(False)
    return float(np.mean(vals))


def uniqueness_rate(graphs: Sequence[nx.Graph]) -> float:
    if not graphs:
        return 0.0
    fp = [graph_fingerprint(g) for g in graphs]
    return float(len(set(fp)) / len(fp))


def novelty_rate(graphs: Sequence[nx.Graph], reference_graphs: Sequence[nx.Graph] | None) -> float | None:
    if reference_graphs is None:
        return None
    if not graphs:
        return 0.0
    ref_fp = {graph_fingerprint(g) for g in reference_graphs}
    gen_fp = [graph_fingerprint(g) for g in graphs]
    return float(np.mean([f not in ref_fp for f in gen_fp]))


def graph_size_summary(graphs: Sequence[nx.Graph]) -> dict[str, float | int]:
    if not graphs:
        return {"num_graphs": 0}
    nodes = np.asarray([g.number_of_nodes() for g in graphs], dtype=float)
    edges = np.asarray([g.number_of_edges() for g in graphs], dtype=float)
    density = np.asarray([nx.density(g) if g.number_of_nodes() > 1 else 0.0 for g in graphs], dtype=float)
    return {
        "num_graphs": int(len(graphs)),
        "num_nodes_mean": float(nodes.mean()),
        "num_nodes_std": float(nodes.std(ddof=0)),
        "num_edges_mean": float(edges.mean()),
        "num_edges_std": float(edges.std(ddof=0)),
        "density_mean": float(density.mean()),
        "density_std": float(density.std(ddof=0)),
        "empty_graph_rate": float(np.mean(nodes == 0)),
    }


def quality_metrics(
    graphs: Sequence[nx.Graph],
    *,
    reference_graphs: Sequence[nx.Graph] | None = None,
    dataset: str | None = None,
) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {}
    metrics.update(graph_size_summary(graphs))
    metrics.update(
        {
            "validity_rate": generic_validity_rate(graphs),
            "no_self_loop_rate": no_self_loop_rate(graphs),
            "connectedness_rate": connectedness_rate(graphs),
            "planarity_rate": planarity_rate(graphs),
            "uniqueness_rate": uniqueness_rate(graphs),
            "novelty_rate": novelty_rate(graphs, reference_graphs),
        }
    )

    # Dataset-specific aliases used in paper tables.
    if dataset == "planar":
        metrics["dataset_validity_rate"] = metrics["planarity_rate"]
    elif dataset in {"sbm"}:
        # There is no single hard validity constraint for SBM samples, so use a
        # basic simple non-empty/no-self-loop criterion.
        metrics["dataset_validity_rate"] = min(
            float(metrics["validity_rate"] or 0.0), float(metrics["no_self_loop_rate"] or 0.0)
        )
    else:
        metrics["dataset_validity_rate"] = metrics["validity_rate"]
    return metrics
