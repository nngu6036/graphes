from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np


ORCA_EXEC = os.environ.get("ORCA_EXEC") or shutil.which("orca")
ORBIT_MULTIPLICITY_4 = np.asarray([2, 2, 1, 6, 2, 1, 2, 4, 2, 1, 1, 2, 2, 4, 1], dtype=np.int64)


@dataclass(frozen=True)
class SummaryConfig:
    degree_hist_max_degree: int | None = None
    clustering_bins: int = 20
    spectral_bins: int = 20
    motif_proxy: bool = True
    orbit_backend: str = "python"

    @classmethod
    def from_dict(cls, data: dict[str, Any], graphs: list[nx.Graph] | None = None) -> "SummaryConfig":
        max_degree_raw = data.get("degree_hist_max_degree", "auto")
        if max_degree_raw in {None, "auto"}:
            max_degree = None
            if graphs:
                max_degree = max((max(dict(g.degree()).values()) if g.number_of_nodes() else 0) for g in graphs)
        else:
            max_degree = int(max_degree_raw)
        return cls(
            degree_hist_max_degree=max_degree,
            clustering_bins=int(data.get("clustering_bins", 20)),
            spectral_bins=int(data.get("spectral_bins", 20)),
            motif_proxy=bool(data.get("motif_proxy", True)),
            orbit_backend=str(data.get("orbit_backend", "python")),
        )


def _safe_normalize(x: np.ndarray) -> np.ndarray:
    total = float(np.sum(x))
    return x / total if total > 0 else x


def sorted_degree_sequence(graph: nx.Graph) -> list[int]:
    return sorted((int(d) for _, d in graph.degree()), reverse=True)


def degree_histogram(graph: nx.Graph, max_degree: int | None = None) -> np.ndarray:
    degrees = [int(d) for _, d in graph.degree()]
    if not degrees:
        return np.zeros(1, dtype=np.float64)
    if max_degree is None:
        max_degree = max(degrees)
    values = [d for d in degrees if 0 <= d <= max_degree]
    hist = np.bincount(values, minlength=max_degree + 1).astype(np.float64)
    return _safe_normalize(hist)


def clustering_histogram(graph: nx.Graph, bins: int = 20) -> np.ndarray:
    if graph.number_of_nodes() == 0:
        out = np.zeros(bins, dtype=np.float64)
        out[0] = 1.0
        return out
    values = list(nx.clustering(graph).values())
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0), density=False)
    return _safe_normalize(hist.astype(np.float64))


def spectral_histogram(graph: nx.Graph, bins: int = 20) -> np.ndarray:
    if graph.number_of_nodes() == 0:
        return np.zeros(bins, dtype=np.float64)
    adjacency = nx.to_numpy_array(graph, dtype=np.float64)
    degrees = adjacency.sum(axis=1)
    inv_sqrt = np.zeros_like(degrees)
    inv_sqrt[degrees > 0] = 1.0 / np.sqrt(degrees[degrees > 0])
    laplacian = np.eye(adjacency.shape[0]) - np.diag(inv_sqrt) @ adjacency @ np.diag(inv_sqrt)
    try:
        vals = np.linalg.eigvalsh(laplacian)
    except np.linalg.LinAlgError:
        vals = np.zeros(graph.number_of_nodes(), dtype=np.float64)
    hist, _ = np.histogram(vals, bins=bins, range=(0.0, 2.0), density=False)
    return _safe_normalize(hist.astype(np.float64))


def motif_proxy_vector(graph: nx.Graph) -> np.ndarray:
    n = max(graph.number_of_nodes(), 1)
    m = graph.number_of_edges()
    degrees = np.asarray([d for _, d in graph.degree()], dtype=np.float64)
    wedges = float(np.sum(degrees * np.maximum(degrees - 1.0, 0.0) / 2.0))
    triangles = float(sum(nx.triangles(graph).values()) / 3.0) if graph.number_of_nodes() else 0.0
    transitivity = float(nx.transitivity(graph)) if m > 0 else 0.0
    avg_clustering = float(nx.average_clustering(graph)) if graph.number_of_nodes() else 0.0
    return np.asarray(
        [
            m / n,
            wedges / n,
            triangles / n,
            transitivity,
            avg_clustering,
        ],
        dtype=np.float64,
    )


def python_orbit_count_vector(graph: nx.Graph) -> np.ndarray:
    """Mean connected graphlet orbit counts for graphlets with 2 to 4 nodes."""

    counts = np.zeros(15, dtype=np.float64)
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return counts

    adjacency = {u: set(graph.neighbors(u)) for u in nodes}

    def has_edge(u: Any, v: Any) -> bool:
        return v in adjacency[u]

    for a_idx in range(n):
        u = nodes[a_idx]
        for b_idx in range(a_idx + 1, n):
            v = nodes[b_idx]
            if has_edge(u, v):
                counts[0] += 2.0

    for a_idx in range(n):
        for b_idx in range(a_idx + 1, n):
            for c_idx in range(b_idx + 1, n):
                subset = [nodes[a_idx], nodes[b_idx], nodes[c_idx]]
                degrees = {u: sum(1 for v in subset if u != v and has_edge(u, v)) for u in subset}
                edge_count = sum(degrees.values()) // 2
                if edge_count == 2:
                    for degree in degrees.values():
                        counts[2 if degree == 2 else 1] += 1.0
                elif edge_count == 3:
                    counts[3] += 3.0

    for a_idx in range(n):
        for b_idx in range(a_idx + 1, n):
            for c_idx in range(b_idx + 1, n):
                for d_idx in range(c_idx + 1, n):
                    subset = [nodes[a_idx], nodes[b_idx], nodes[c_idx], nodes[d_idx]]
                    degrees = {u: sum(1 for v in subset if u != v and has_edge(u, v)) for u in subset}
                    edge_count = sum(degrees.values()) // 2
                    if edge_count == 3:
                        if sorted(degrees.values()) == [1, 1, 1, 3]:
                            for degree in degrees.values():
                                counts[5 if degree == 3 else 4] += 1.0
                        elif sorted(degrees.values()) == [1, 1, 2, 2]:
                            for degree in degrees.values():
                                counts[7 if degree == 2 else 6] += 1.0
                    elif edge_count == 4:
                        if all(degree == 2 for degree in degrees.values()):
                            counts[11] += 4.0
                        elif sorted(degrees.values()) == [1, 2, 2, 3]:
                            for degree in degrees.values():
                                if degree == 1:
                                    counts[8] += 1.0
                                elif degree == 3:
                                    counts[9] += 1.0
                                else:
                                    counts[10] += 1.0
                    elif edge_count == 5:
                        for degree in degrees.values():
                            counts[13 if degree == 3 else 12] += 1.0
                    elif edge_count == 6:
                        counts[14] += 4.0

    return counts / n


def orca_orbit_count_vector(graph: nx.Graph, orbit_size: int = 4) -> np.ndarray:
    """Connected graphlet orbit counts from ORCA.

    Set ORCA_EXEC to the ORCA executable path, or put an ``orca`` executable on PATH.
    """

    if not ORCA_EXEC:
        raise RuntimeError("ORCA module is not found. Set ORCA_EXEC or add orca to PATH.")
    if graph.number_of_nodes() == 0:
        return np.zeros(15, dtype=np.float64)

    temp1_path: str | None = None
    temp2_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp1, tempfile.NamedTemporaryFile(mode="r", delete=False) as temp2:
            temp1_path = temp1.name
            temp2_path = temp2.name

            nodes = sorted(graph.nodes())
            node_map = {node: idx for idx, node in enumerate(nodes)}
            temp1.write(f"{graph.number_of_nodes()} {graph.number_of_edges()}\n")
            for u, v in graph.edges():
                temp1.write(f"{node_map[u]} {node_map[v]}\n")
            temp1.flush()

            try:
                subprocess.run(
                    [ORCA_EXEC, "node", str(orbit_size), temp1_path, temp2_path],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
                raise RuntimeError(f"ORCA execution failed: {stderr}") from exc

        with open(temp2_path, "r", encoding="utf-8") as f:
            orbit_counts = [list(map(int, line.strip().split())) for line in f if line.strip()]
        if not orbit_counts:
            return np.zeros(15, dtype=np.float64)

        total_counts = np.asarray(orbit_counts, dtype=np.int64).sum(axis=0)
        multiplicity = ORBIT_MULTIPLICITY_4[: total_counts.size]
        return (total_counts // multiplicity).astype(np.float64)
    finally:
        for path in (temp1_path, temp2_path):
            if path and os.path.exists(path):
                os.remove(path)


def orbit_count_vector(graph: nx.Graph) -> np.ndarray:
    """Orbit-count descriptor, using ORCA when requested or available."""

    if ORCA_EXEC:
        return orca_orbit_count_vector(graph)
    return python_orbit_count_vector(graph)


def extract_summary(graph: nx.Graph, config: SummaryConfig | dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config if isinstance(config, SummaryConfig) else SummaryConfig.from_dict(config or {})
    n = int(graph.number_of_nodes())
    m = int(graph.number_of_edges())
    degree_seq = sorted_degree_sequence(graph)
    triangles = float(sum(nx.triangles(graph).values()) / 3.0) if n else 0.0
    return {
        "num_nodes": n,
        "num_edges": m,
        "degree_sequence": degree_seq,
        "density": float(nx.density(graph)) if n > 1 else 0.0,
        "triangle_count_norm": triangles / max(n, 1),
        "degree_hist": degree_histogram(graph, cfg.degree_hist_max_degree),
        "clustering_hist": clustering_histogram(graph, cfg.clustering_bins),
        "spectral_hist": spectral_histogram(graph, cfg.spectral_bins),
        "motif_proxy": motif_proxy_vector(graph) if cfg.motif_proxy else np.zeros(0, dtype=np.float64),
        "orbit_count": orbit_count_vector(graph),
    }


def _l2(a: Any, b: Any) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    width = max(av.size, bv.size)
    ap = np.zeros(width, dtype=np.float64)
    bp = np.zeros(width, dtype=np.float64)
    ap[: av.size] = av
    bp[: bv.size] = bv
    return float(np.linalg.norm(ap - bp))


def distance_to_summary(graph: nx.Graph, target: dict[str, Any], config: SummaryConfig | dict[str, Any] | None = None, weights: dict[str, float] | None = None) -> float:
    """Permutation-invariant energy between a graph and a target summary."""

    cfg = config if isinstance(config, SummaryConfig) else SummaryConfig.from_dict(config or {})
    current = extract_summary(graph, cfg)
    w = weights or {}
    energy = 0.0
    energy += float(w.get("degree_weight", 0.0)) * _l2(current["degree_hist"], target.get("degree_hist", []))
    energy += float(w.get("clustering_weight", 1.0)) * _l2(current["clustering_hist"], target.get("clustering_hist", []))
    energy += float(w.get("spectral_weight", 0.0)) * _l2(current["spectral_hist"], target.get("spectral_hist", []))
    energy += float(w.get("motif_weight", 0.0)) * _l2(current["motif_proxy"], target.get("motif_proxy", []))
    energy += float(w.get("orbit_weight", 0.0)) * _l2(current["orbit_count"], target.get("orbit_count", []))
    energy += float(w.get("density_weight", 0.0)) * abs(float(current["density"]) - float(target.get("density", 0.0)))
    energy += float(w.get("triangle_weight", 0.0)) * abs(float(current["triangle_count_norm"]) - float(target.get("triangle_count_norm", 0.0)))
    return float(energy)


def summary_to_jsonable(summary: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in summary.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            out[key] = value.item()
        else:
            out[key] = value
    return out
