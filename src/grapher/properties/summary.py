from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from math import comb
from typing import Any
from pathlib import Path
import networkx as nx
import numpy as np

from grapher.utils.motifs import (
    default_topology_canonicalizer,
    graphlet_count_dict,
    graphlet_history_l2_distance,
    normalize_count_dict,
    topology_graphlet_basis,
)

ORCA_EXEC = os.environ.get("ORCA_EXEC") or shutil.which("orca")


def configure_orca_executable(
    executable: str | os.PathLike[str] | None = None,
    *,
    required: bool = True,
) -> str | None:
    """Resolve and validate the ORCA executable used by summary metrics.

    ``executable`` may be an explicit file, a directory containing ``orca``,
    a command name on ``PATH``, or a string containing an environment
    variable such as ``${ORCA_EXEC}``.  Calling this function updates the
    module-level executable used by all later ORCA orbit and graphlet calls.
    """

    global ORCA_EXEC

    raw = str(executable).strip() if executable is not None else ""
    if raw:
        expanded = os.path.expandvars(os.path.expanduser(raw))
        if "$" in expanded:
            if required:
                raise RuntimeError(
                    "The configured ORCA executable contains an unresolved "
                    f"environment variable: {raw!r}."
                )
            return None
        candidate = expanded
    else:
        candidate = os.environ.get("ORCA_EXEC", "").strip() or "orca"

    path = Path(candidate)
    if path.is_dir():
        path = path / "orca"
        resolved = str(path)
    else:
        resolved = shutil.which(candidate) or str(path)

    resolved_path = Path(resolved)
    if not resolved_path.is_file() or not os.access(resolved_path, os.X_OK):
        if required:
            raise RuntimeError(
                "ORCA evaluation is enabled, but the executable could not be "
                f"resolved from {candidate!r}. Set evaluation.orca_exec, set "
                "ORCA_EXEC, or add orca to PATH."
            )
        return None

    ORCA_EXEC = str(resolved_path.resolve())
    return ORCA_EXEC


@dataclass(frozen=True)
class SummaryConfig:
    degree_hist_max_degree: int | None = None
    clustering_bins: int = 20
    spectral_bins: int = 20
    clustering_summary: bool = True
    spectral_summary: bool = True
    motif_proxy: bool = True
    orbit_count: bool = False

    # New proposal: graph summary = degree histogram + graphlet history.
    # graphlet_history=False keeps old configs/checkpoints backwards-compatible.
    graphlet_history: bool = False
    graphlet_k_min: int = 3
    graphlet_k_max: int = 5
    graphlet_connected_only: bool = True
    graphlet_num_samples: int | None = None
    graphlet_backend: str = "sampled"

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], graphs: list[nx.Graph] | None = None
    ) -> "SummaryConfig":
        data = data or {}
        max_degree_raw = data.get("degree_hist_max_degree", "auto")
        if max_degree_raw in {None, "auto"}:
            max_degree = None
            if graphs:
                max_degree = max(
                    (max(dict(g.degree()).values()) if g.number_of_nodes() else 0)
                    for g in graphs
                )
        else:
            max_degree = int(max_degree_raw)

        graphlet_enabled = bool(
            data.get(
                "graphlet_history",
                data.get("use_graphlet_history", data.get("graphlets", False)),
            )
        )
        k_min = int(data.get("graphlet_k_min", data.get("k_min", 3)))
        k_max = int(data.get("graphlet_k_max", data.get("k_max", 5)))
        num_samples_raw = data.get(
            "graphlet_num_samples", data.get("num_graphlet_samples", None)
        )
        num_samples = (
            None
            if num_samples_raw in {None, "", "none", "None"}
            else int(num_samples_raw)
        )

        return cls(
            degree_hist_max_degree=max_degree,
            clustering_bins=int(data.get("clustering_bins", 20)),
            spectral_bins=int(data.get("spectral_bins", 20)),
            clustering_summary=bool(
                data.get("clustering_summary", data.get("use_clustering", True))
            ),
            spectral_summary=bool(
                data.get("spectral_summary", data.get("use_spectral", True))
            ),
            motif_proxy=bool(data.get("motif_proxy", True)),
            orbit_count=bool(data.get("orbit_count", data.get("use_orbit", False))),
            graphlet_history=graphlet_enabled,
            graphlet_k_min=k_min,
            graphlet_k_max=k_max,
            graphlet_connected_only=bool(data.get("graphlet_connected_only", True)),
            graphlet_num_samples=num_samples,
            graphlet_backend=str(
                data.get("graphlet_backend", "sampled")
            ).lower(),
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
    laplacian = np.eye(adjacency.shape[0]) - np.diag(inv_sqrt) @ adjacency @ np.diag(
        inv_sqrt
    )
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
    triangles = (
        float(sum(nx.triangles(graph).values()) / 3.0)
        if graph.number_of_nodes()
        else 0.0
    )
    transitivity = float(nx.transitivity(graph)) if m > 0 else 0.0
    avg_clustering = (
        float(nx.average_clustering(graph)) if graph.number_of_nodes() else 0.0
    )
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
                degrees = {
                    u: sum(1 for v in subset if u != v and has_edge(u, v))
                    for u in subset
                }
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
                    degrees = {
                        u: sum(1 for v in subset if u != v and has_edge(u, v))
                        for u in subset
                    }
                    edge_count = sum(degrees.values()) // 2
                    if edge_count == 3:
                        if sorted(degrees.values()) == [1, 1, 1, 3]:
                            for degree in degrees.values():
                                counts[7 if degree == 3 else 6] += 1.0
                        elif sorted(degrees.values()) == [1, 1, 2, 2]:
                            for degree in degrees.values():
                                counts[5 if degree == 2 else 4] += 1.0
                    elif edge_count == 4:
                        if all(degree == 2 for degree in degrees.values()):
                            counts[8] += 4.0
                        elif sorted(degrees.values()) == [1, 2, 2, 3]:
                            for degree in degrees.values():
                                if degree == 1:
                                    counts[9] += 1.0
                                elif degree == 3:
                                    counts[11] += 1.0
                                else:
                                    counts[10] += 1.0
                    elif edge_count == 5:
                        for degree in degrees.values():
                            counts[13 if degree == 3 else 12] += 1.0
                    elif edge_count == 6:
                        counts[14] += 4.0

    return counts / n


def orca_node_orbit_matrix(graph: nx.Graph, orbit_size: int = 4) -> np.ndarray:
    """Return the raw per-node ORCA orbit-count matrix.

    Set ORCA_EXEC to the ORCA executable path, or put an ``orca`` executable on PATH.
    """

    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("ORCA requires a simple undirected graph.")
    if nx.number_of_selfloops(graph):
        raise ValueError("ORCA requires a graph without self-loops.")
    orbit_size = int(orbit_size)
    if orbit_size not in {4, 5}:
        raise ValueError("ORCA orbit_size must be 4 or 5.")
    if not ORCA_EXEC:
        raise RuntimeError(
            "ORCA module is not found. Set ORCA_EXEC or add orca to PATH."
        )
    if graph.number_of_nodes() == 0:
        width = 15 if orbit_size == 4 else 73
        return np.zeros((0, width), dtype=np.int64)

    temp1_path: str | None = None
    temp2_path: str | None = None
    try:
        with (
            tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp1,
            tempfile.NamedTemporaryFile(mode="r", delete=False) as temp2,
        ):
            temp1_path = temp1.name
            temp2_path = temp2.name

            # ORCA only needs a stable integer relabeling. Insertion order also
            # supports graphs whose original node labels are not comparable.
            nodes = list(graph.nodes())
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
            orbit_counts = [
                list(map(int, line.strip().split())) for line in f if line.strip()
            ]
        expected_width = 15 if orbit_size == 4 else 73
        matrix = np.asarray(orbit_counts, dtype=np.int64)
        expected_shape = (graph.number_of_nodes(), expected_width)
        if matrix.shape != expected_shape:
            raise RuntimeError(
                "ORCA returned an invalid orbit matrix shape: "
                f"expected {expected_shape}, got {matrix.shape}."
            )
        if np.any(matrix < 0):
            raise RuntimeError("ORCA returned a negative orbit count.")
        return matrix
    finally:
        for path in (temp1_path, temp2_path):
            if path and os.path.exists(path):
                os.remove(path)


def orca_orbit_count_vector(graph: nx.Graph, orbit_size: int = 4) -> np.ndarray:
    """Mean per-node connected graphlet-orbit descriptor."""

    orbit_counts = orca_node_orbit_matrix(graph, orbit_size=orbit_size)
    if orbit_counts.size == 0:
        width = 15 if int(orbit_size) == 4 else 73
        return np.zeros(width, dtype=np.float64)
    return orbit_counts.mean(axis=0, dtype=np.float64)


@lru_cache(maxsize=4)
def _orca_graphlet_orbit_mapping(
    k: int,
) -> tuple[tuple[str, tuple[tuple[int, int], ...]], ...]:
    """Map each connected topology to all of its standard ORCA roles."""

    k = int(k)
    roles_by_signature = {
        (2, 1, (1, 1)): ((0, 2),),
        (3, 2, (1, 1, 2)): ((1, 2), (2, 1)),
        (3, 3, (2, 2, 2)): ((3, 3),),
        (4, 3, (1, 1, 2, 2)): ((4, 2), (5, 2)),
        (4, 3, (1, 1, 1, 3)): ((6, 3), (7, 1)),
        (4, 4, (2, 2, 2, 2)): ((8, 4),),
        (4, 4, (1, 2, 2, 3)): ((9, 1), (10, 2), (11, 1)),
        (4, 5, (2, 2, 3, 3)): ((12, 2), (13, 2)),
        (4, 6, (3, 3, 3, 3)): ((14, 4),),
    }
    if k not in {2, 3, 4}:
        raise ValueError("Exact ORCA graphlet history supports k in {2, 3, 4}.")

    mapping: list[tuple[str, tuple[tuple[int, int], ...]]] = []
    for key, representative in topology_graphlet_basis(k, connected_only=True):
        signature = (
            k,
            representative.number_of_edges(),
            tuple(sorted(int(degree) for _, degree in representative.degree())),
        )
        roles = roles_by_signature.get(signature)
        if roles is None:
            raise RuntimeError(
                f"No standard ORCA role map for k={k} graphlet {key!r}."
            )
        mapping.append((str(key), roles))

    return tuple(sorted(mapping, key=lambda item: item[0]))


def orca_connected_graphlet_statistics(
    graph: nx.Graph,
    *,
    k_min: int = 3,
    k_max: int = 4,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Return exact connected graphlet laws and their induced-subset mass."""

    if int(k_min) < 2 or int(k_max) > 4 or int(k_max) < int(k_min):
        raise ValueError(
            "Exact ORCA graphlet history requires 2 <= k_min <= k_max <= 4."
        )
    orbit_totals = orca_node_orbit_matrix(graph, orbit_size=4).sum(axis=0)
    history: dict[str, dict[str, float]] = {}
    connected_mass: dict[str, float] = {}
    n = graph.number_of_nodes()
    for k in range(int(k_min), int(k_max) + 1):
        counts: dict[str, float] = {}
        for key, roles in _orca_graphlet_orbit_mapping(k):
            role_counts: list[int] = []
            for orbit, multiplicity in roles:
                total = int(orbit_totals[orbit])
                value, remainder = divmod(total, int(multiplicity))
                if remainder:
                    raise RuntimeError(
                        "ORCA orbit total is not divisible by its role "
                        f"multiplicity for k={k}, graphlet={key!r}, "
                        f"orbit={orbit}."
                    )
                role_counts.append(value)
            if len(set(role_counts)) != 1:
                raise RuntimeError(
                    "ORCA roles disagree on the graphlet count for "
                    f"k={k}, graphlet={key!r}: {role_counts}."
                )
            counts[key] = float(role_counts[0])
        total = float(sum(counts.values()))
        history[str(k)] = {
            key: (value / total if total > 0.0 else 0.0)
            for key, value in counts.items()
        }
        denominator = comb(n, k) if n >= k else 0
        connected_mass[str(k)] = (
            float(total / denominator) if denominator > 0 else 0.0
        )
    return history, connected_mass


def orca_connected_graphlet_history(
    graph: nx.Graph,
    *,
    k_min: int = 3,
    k_max: int = 4,
) -> dict[str, dict[str, float]]:
    """Count connected induced graphlets exactly using one ORCA invocation."""

    history, _ = orca_connected_graphlet_statistics(
        graph,
        k_min=k_min,
        k_max=k_max,
    )
    return history


def orbit_count_vector(graph: nx.Graph) -> np.ndarray:
    """Orbit-count descriptor, using ORCA when requested or available."""

    if ORCA_EXEC:
        return orca_orbit_count_vector(graph)
    return python_orbit_count_vector(graph)


def graphlet_statistics_summary(
    graph: nx.Graph,
    cfg: SummaryConfig,
    *,
    backend_override: str | None = None,
    num_samples_override: int | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    if not cfg.graphlet_history:
        return {}, {}
    backend = str(backend_override or cfg.graphlet_backend).lower()
    if backend in {"orca", "exact_orca", "exact"}:
        if not cfg.graphlet_connected_only:
            raise ValueError(
                "The ORCA backend counts connected induced graphlets only. "
                "Set graphlet_connected_only: true or use graphlet_backend: sampled."
            )
        return orca_connected_graphlet_statistics(
            graph,
            k_min=cfg.graphlet_k_min,
            k_max=cfg.graphlet_k_max,
        )
    if backend not in {"sampled", "sampling", "enumerate", "enumeration"}:
        raise ValueError(f"Unknown graphlet_backend: {backend!r}.")

    num_samples = (
        cfg.graphlet_num_samples
        if num_samples_override is None
        else int(num_samples_override)
    )
    canonicalizer = default_topology_canonicalizer()
    history: dict[str, dict[str, float]] = {}
    connected_mass: dict[str, float] = {}
    n = graph.number_of_nodes()
    for k in range(cfg.graphlet_k_min, cfg.graphlet_k_max + 1):
        counts = graphlet_count_dict(
            graph,
            k,
            connected_only=cfg.graphlet_connected_only,
            num_samples=num_samples,
            canonicalizer=canonicalizer,
        )
        history[str(k)] = normalize_count_dict(counts)
        total_subsets = comb(n, k) if n >= k else 0
        sampled_subsets = (
            min(total_subsets, int(num_samples))
            if num_samples is not None and int(num_samples) > 0
            else total_subsets
        )
        connected_mass[str(k)] = (
            float(sum(counts.values()) / sampled_subsets)
            if sampled_subsets > 0 and cfg.graphlet_connected_only
            else float(sampled_subsets > 0)
        )
    return history, connected_mass


def graphlet_history_summary(
    graph: nx.Graph, cfg: SummaryConfig
) -> dict[str, dict[str, float]]:
    history, _ = graphlet_statistics_summary(graph, cfg)
    return history


def extract_summary(
    graph: nx.Graph, config: SummaryConfig | dict[str, Any] | None = None
) -> dict[str, Any]:
    cfg = (
        config
        if isinstance(config, SummaryConfig)
        else SummaryConfig.from_dict(config or {})
    )
    n = int(graph.number_of_nodes())
    m = int(graph.number_of_edges())
    degree_seq = sorted_degree_sequence(graph)
    triangles = float(sum(nx.triangles(graph).values()) / 3.0) if n else 0.0
    graphlet_history, graphlet_connected_mass = graphlet_statistics_summary(
        graph,
        cfg,
    )
    return {
        "num_nodes": n,
        "num_edges": m,
        "degree_sequence": degree_seq,
        "density": float(nx.density(graph)) if n > 1 else 0.0,
        "triangle_count_norm": triangles / max(n, 1),
        "degree_hist": degree_histogram(graph, cfg.degree_hist_max_degree),
        "clustering_hist": (
            clustering_histogram(graph, cfg.clustering_bins)
            if cfg.clustering_summary
            else np.zeros(cfg.clustering_bins, dtype=np.float64)
        ),
        "spectral_hist": (
            spectral_histogram(graph, cfg.spectral_bins)
            if cfg.spectral_summary
            else np.zeros(cfg.spectral_bins, dtype=np.float64)
        ),
        "motif_proxy": motif_proxy_vector(graph)
        if cfg.motif_proxy
        else np.zeros(0, dtype=np.float64),
        "orbit_count": orbit_count_vector(graph)
        if cfg.orbit_count
        else np.zeros(0, dtype=np.float64),
        "graphlet_history": graphlet_history,
        "graphlet_connected_mass": graphlet_connected_mass,
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


def _weighted_vector_distance(current: Any, target: Any, *, normalize: bool) -> float:
    """L2 distance with optional dimension normalization."""

    av = np.asarray(current, dtype=np.float64).reshape(-1)
    bv = np.asarray(target, dtype=np.float64).reshape(-1)
    width = max(av.size, bv.size, 1)
    value = _l2(av, bv)
    return float(value / np.sqrt(width)) if normalize else float(value)


def _weight(weights: dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(weights.get(key, default) or 0.0)


def distance_to_summary(
    graph: nx.Graph,
    target: dict[str, Any],
    config: SummaryConfig | dict[str, Any] | None = None,
    weights: dict[str, Any] | None = None,
) -> float:
    """Permutation-invariant energy between a graph and a target summary.

    The new graphlet-history energy is enabled by setting either
    ``energy.graphlet_weight`` or ``energy.graphlet_history_weight`` to a
    non-zero value and ``summary.graphlet_history: true``.
    """

    cfg = (
        config
        if isinstance(config, SummaryConfig)
        else SummaryConfig.from_dict(config or {})
    )
    w = weights or {}
    normalize = bool(w.get("normalize_terms", False))
    energy = 0.0

    degree_w = _weight(w, "degree_weight", 0.0)
    if degree_w != 0.0:
        energy += degree_w * _weighted_vector_distance(
            degree_histogram(graph, cfg.degree_hist_max_degree),
            target.get("degree_hist", []),
            normalize=normalize,
        )

    clustering_w = _weight(w, "clustering_weight", 1.0)
    if clustering_w != 0.0:
        energy += clustering_w * _weighted_vector_distance(
            clustering_histogram(graph, cfg.clustering_bins),
            target.get("clustering_hist", []),
            normalize=normalize,
        )

    spectral_w = _weight(w, "spectral_weight", 0.0)
    if spectral_w != 0.0:
        energy += spectral_w * _weighted_vector_distance(
            spectral_histogram(graph, cfg.spectral_bins),
            target.get("spectral_hist", []),
            normalize=normalize,
        )

    motif_w = _weight(w, "motif_weight", 0.0)
    if motif_w != 0.0 and cfg.motif_proxy:
        energy += motif_w * _weighted_vector_distance(
            motif_proxy_vector(graph),
            target.get("motif_proxy", []),
            normalize=normalize,
        )

    orbit_w = _weight(w, "orbit_weight", 0.0)
    if orbit_w != 0.0:
        energy += orbit_w * _weighted_vector_distance(
            orbit_count_vector(graph),
            target.get("orbit_count", []),
            normalize=normalize,
        )

    graphlet_w = _weight(
        w, "graphlet_weight", _weight(w, "graphlet_history_weight", 0.0)
    )
    graphlet_mass_w = _weight(w, "graphlet_connected_mass_weight", 0.0)
    if (graphlet_w != 0.0 or graphlet_mass_w != 0.0) and cfg.graphlet_history:
        backend = str(w.get("graphlet_backend", cfg.graphlet_backend)).lower()
        num_samples_raw = w.get(
            "graphlet_num_samples",
            cfg.graphlet_num_samples,
        )
        num_samples = (
            None
            if num_samples_raw in {None, "", "none", "None"}
            else int(num_samples_raw)
        )
        current_history, current_connected_mass = graphlet_statistics_summary(
            graph,
            cfg,
            backend_override=backend,
            num_samples_override=num_samples,
        )
        if graphlet_w != 0.0:
            size_weights = w.get("graphlet_size_weights", {}) or {}
            energy += graphlet_w * graphlet_history_l2_distance(
                current_history,
                target.get("graphlet_history", {}) or {},
                size_weights={
                    str(k): float(v) for k, v in dict(size_weights).items()
                },
                normalize_terms=normalize,
            )
        if graphlet_mass_w != 0.0:
            target_mass = target.get("graphlet_connected_mass", {}) or {}
            keys = [
                str(k)
                for k in range(cfg.graphlet_k_min, cfg.graphlet_k_max + 1)
            ]
            energy += graphlet_mass_w * _weighted_vector_distance(
                [current_connected_mass.get(key, 0.0) for key in keys],
                [target_mass.get(key, 0.0) for key in keys],
                normalize=normalize,
            )

    density_w = _weight(w, "density_weight", 0.0)
    if density_w != 0.0:
        n = graph.number_of_nodes()
        density = float(nx.density(graph)) if n > 1 else 0.0
        energy += density_w * abs(density - float(target.get("density", 0.0)))

    triangle_w = _weight(w, "triangle_weight", 0.0)
    if triangle_w != 0.0:
        n = graph.number_of_nodes()
        triangles = float(sum(nx.triangles(graph).values()) / 3.0) if n else 0.0
        energy += triangle_w * abs(
            (triangles / max(n, 1)) - float(target.get("triangle_count_norm", 0.0))
        )

    return float(energy)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def summary_to_jsonable(summary: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(value) for key, value in summary.items()}
