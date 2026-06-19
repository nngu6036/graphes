from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel


ORBIT_4_MULTIPLICITY = np.asarray([2, 2, 1, 6, 2, 1, 2, 4, 2, 1, 1, 2, 2, 4, 1], dtype=np.int64)


def _safe_normalize(x: np.ndarray) -> np.ndarray:
    total = float(np.sum(x))
    if total > 0:
        return x / total
    return x


def degree_histogram(graph: nx.Graph, max_degree: int = 100) -> np.ndarray:
    degs = [int(d) for _, d in graph.degree() if 0 <= int(d) <= int(max_degree)]
    hist = np.bincount(degs, minlength=int(max_degree) + 1).astype(np.float64)
    return _safe_normalize(hist)


def degree_sequence_histogram(seq: Sequence[int], max_degree: int) -> np.ndarray:
    values = [int(d) for d in seq if 0 <= int(d) <= int(max_degree)]
    hist = np.bincount(values, minlength=int(max_degree) + 1).astype(np.float64)
    return _safe_normalize(hist)


def clustering_histogram(graph: nx.Graph, bins: int = 20) -> np.ndarray:
    if graph.number_of_nodes() == 0:
        out = np.zeros(int(bins), dtype=np.float64)
        out[0] = 1.0
        return out
    values = list(nx.clustering(graph).values())
    hist, _ = np.histogram(values, bins=int(bins), range=(0.0, 1.0), density=False)
    return _safe_normalize(hist.astype(np.float64))


def spectral_histogram(graph: nx.Graph, bins: int = 20) -> np.ndarray:
    if graph.number_of_nodes() == 0:
        return np.zeros(int(bins), dtype=np.float64)
    adjacency = nx.to_numpy_array(graph, dtype=np.float64)
    deg = adjacency.sum(axis=1)
    inv_sqrt = np.zeros_like(deg)
    inv_sqrt[deg > 0] = 1.0 / np.sqrt(deg[deg > 0])
    lap = np.eye(adjacency.shape[0]) - np.diag(inv_sqrt) @ adjacency @ np.diag(inv_sqrt)
    try:
        vals = np.linalg.eigvalsh(lap)
    except np.linalg.LinAlgError:
        vals = np.zeros(graph.number_of_nodes(), dtype=np.float64)
    hist, _ = np.histogram(vals, bins=int(bins), range=(0.0, 2.0), density=False)
    return _safe_normalize(hist.astype(np.float64))


def motif_proxy_vector(graph: nx.Graph) -> np.ndarray:
    """Lightweight higher-order motif proxy used when ORCA is unavailable.

    It is not a replacement for ORCA orbit counts, but it keeps evaluation
    scripts runnable and records a transparent proxy for diagnostics.
    """

    n = max(graph.number_of_nodes(), 1)
    m = graph.number_of_edges()
    triangles = sum(nx.triangles(graph).values()) / 3.0 if graph.number_of_nodes() else 0.0
    try:
        squares = sum(nx.square_clustering(graph).values()) * n / 4.0
    except Exception:
        squares = 0.0
    wedges = sum(d * (d - 1) / 2 for _, d in graph.degree())
    return np.asarray([n, m, wedges, triangles, squares, nx.transitivity(graph) if m else 0.0], dtype=np.float64)


def resolve_orca_executable(orca_exec: str | None = None) -> str | None:
    if orca_exec:
        return str(Path(orca_exec))
    env_path = os.environ.get("ORCA_EXEC")
    if env_path:
        return env_path
    return shutil.which("orca")


def graphlet_orbit_counts(graph: nx.Graph, *, orca_exec: str, orbit_size: int = 4) -> np.ndarray:
    if int(orbit_size) != 4:
        raise ValueError("Only orbit_size=4 is supported because ORCA node-4 output has 15 known orbit multiplicities.")
    if graph.number_of_nodes() == 0:
        return np.zeros(len(ORBIT_4_MULTIPLICITY), dtype=np.float64)

    graph = nx.convert_node_labels_to_integers(nx.Graph(graph), ordering="sorted")
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "graph.txt"
        output_path = Path(temp_dir) / "orbits.txt"
        with open(input_path, "w", encoding="utf-8") as f:
            f.write(f"{graph.number_of_nodes()} {graph.number_of_edges()}\n")
            for u, v in graph.edges():
                f.write(f"{int(u)} {int(v)}\n")
        try:
            subprocess.run(
                [orca_exec, "node", str(int(orbit_size)), str(input_path), str(output_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"ORCA executable not found: {orca_exec}") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else ""
            raise RuntimeError(f"ORCA execution failed for graph with n={graph.number_of_nodes()} m={graph.number_of_edges()}: {stderr}") from exc

        rows: list[list[int]] = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.strip().split()
                if values:
                    rows.append([int(value) for value in values])

    if not rows:
        return np.zeros(len(ORBIT_4_MULTIPLICITY), dtype=np.float64)
    counts = np.asarray(rows, dtype=np.int64).sum(axis=0)
    multiplicity = ORBIT_4_MULTIPLICITY[: counts.shape[0]]
    return (counts // multiplicity).astype(np.float64)


def orbit_histogram(graphs: Sequence[nx.Graph], *, orca_exec: str, orbit_size: int = 4, eps: float = 1e-8) -> np.ndarray:
    histograms = []
    for graph in graphs:
        counts = graphlet_orbit_counts(graph, orca_exec=orca_exec, orbit_size=orbit_size)
        histograms.append(counts / (float(counts.sum()) + float(eps)))
    if not histograms:
        return np.zeros((0, len(ORBIT_4_MULTIPLICITY)), dtype=np.float64)
    width = max(hist.size for hist in histograms)
    out = np.zeros((len(histograms), width), dtype=np.float64)
    for idx, hist in enumerate(histograms):
        out[idx, : hist.size] = hist
    return out


def orbit_count_mmd(reference: Sequence[nx.Graph], generated: Sequence[nx.Graph], *, orca_exec: str, orbit_size: int = 4, sigma: float = 1.0) -> float:
    ref_hist = orbit_histogram(reference, orca_exec=orca_exec, orbit_size=orbit_size)
    gen_hist = orbit_histogram(generated, orca_exec=orca_exec, orbit_size=orbit_size)
    return mmd_gaussian_emd(ref_hist, gen_hist, sigma=sigma)


def structural_summary(graph: nx.Graph) -> np.ndarray:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    density = nx.density(graph) if n > 1 else 0.0
    avg_clust = nx.average_clustering(graph) if n > 0 else 0.0
    trans = nx.transitivity(graph) if m > 0 else 0.0
    conn = float(nx.is_connected(graph)) if n > 0 else 0.0
    degs = np.asarray([d for _, d in graph.degree()], dtype=np.float64)
    return np.asarray(
        [
            float(n),
            float(m),
            float(density),
            float(avg_clust),
            float(trans),
            float(conn),
            float(degs.mean()) if degs.size else 0.0,
            float(degs.std(ddof=0)) if degs.size else 0.0,
        ],
        dtype=np.float64,
    )


def descriptor_matrix(graphs: Sequence[nx.Graph], fn) -> np.ndarray:
    rows = [np.asarray(fn(g), dtype=np.float64).reshape(-1) for g in graphs]
    if not rows:
        return np.zeros((0, 1), dtype=np.float64)
    width = max(row.size for row in rows)
    out = np.zeros((len(rows), width), dtype=np.float64)
    for i, row in enumerate(rows):
        out[i, : row.size] = row
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def gaussian_emd_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    out = np.zeros((x.shape[0], y.shape[0]), dtype=np.float64)
    support = np.arange(x.shape[1], dtype=np.float64)
    for i, a in enumerate(x):
        for j, b in enumerate(y):
            dist = wasserstein_distance(support, support, a, b)
            out[i, j] = math.exp(-(dist * dist) / (2.0 * sigma * sigma))
    return out


def mmd_from_kernel(kxx: np.ndarray, kyy: np.ndarray, kxy: np.ndarray) -> float:
    return float(np.mean(kxx) + np.mean(kyy) - 2.0 * np.mean(kxy))


def mmd_gaussian_emd(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    return mmd_from_kernel(gaussian_emd_kernel(x, x, sigma), gaussian_emd_kernel(y, y, sigma), gaussian_emd_kernel(x, y, sigma))


def _median_heuristic_sigma(x: np.ndarray, y: np.ndarray) -> float:
    z = np.vstack([x, y])
    if z.shape[0] < 2:
        return 1.0
    dists = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
    vals = dists[np.triu_indices_from(dists, k=1)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    return float(np.sqrt(0.5 * np.median(vals))) or 1.0


def mmd_rbf(x: np.ndarray, y: np.ndarray, sigma: float | None = None) -> float:
    sigma_value = _median_heuristic_sigma(x, y) if sigma is None else float(sigma)
    gamma = 1.0 / (2.0 * sigma_value * sigma_value)
    return mmd_from_kernel(rbf_kernel(x, x, gamma=gamma), rbf_kernel(y, y, gamma=gamma), rbf_kernel(x, y, gamma=gamma))


def degree_distribution_kl(reference: Sequence[Sequence[int]], generated: Sequence[Sequence[int]], max_degree: int, eps: float = 1e-8) -> float:
    ref = np.mean([degree_sequence_histogram(seq, max_degree) for seq in reference], axis=0)
    gen = np.mean([degree_sequence_histogram(seq, max_degree) for seq in generated], axis=0)
    ref = (ref + eps) / np.sum(ref + eps)
    gen = (gen + eps) / np.sum(gen + eps)
    return float(np.sum(ref * np.log(ref / gen)))
