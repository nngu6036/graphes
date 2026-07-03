from __future__ import annotations

from typing import Callable, Sequence

import networkx as nx
import numpy as np

from grapher.properties.summary import clustering_histogram, degree_histogram, motif_proxy_vector, orbit_count_vector, spectral_histogram


def descriptor_matrix(graphs: Sequence[nx.Graph], fn: Callable[[nx.Graph], np.ndarray]) -> np.ndarray:
    rows = [np.asarray(fn(g), dtype=np.float64).reshape(-1) for g in graphs]
    if not rows:
        return np.zeros((0, 1), dtype=np.float64)
    width = max(row.size for row in rows)
    out = np.zeros((len(rows), width), dtype=np.float64)
    for i, row in enumerate(rows):
        out[i, : row.size] = row
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _median_sigma(x: np.ndarray, y: np.ndarray) -> float:
    z = np.vstack([x, y])
    if z.shape[0] < 2:
        return 1.0
    d = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
    vals = d[np.triu_indices(z.shape[0], k=1)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    return float(np.sqrt(0.5 * np.median(vals))) or 1.0


def mmd_rbf(x: np.ndarray, y: np.ndarray, sigma: float | None = None) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")
    sigma = _median_sigma(x, y) if sigma is None else float(sigma)
    gamma = 1.0 / (2.0 * sigma * sigma)
    kxx = np.exp(-gamma * np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
    kyy = np.exp(-gamma * np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1))
    kxy = np.exp(-gamma * np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1))
    return float(np.mean(kxx) + np.mean(kyy) - 2.0 * np.mean(kxy))


def gaussian_emd_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    width = max(x.shape[1], y.shape[1])
    xp = np.zeros((x.shape[0], width), dtype=np.float64)
    yp = np.zeros((y.shape[0], width), dtype=np.float64)
    xp[:, : x.shape[1]] = x
    yp[:, : y.shape[1]] = y
    emd = np.sum(np.abs(np.cumsum(xp[:, None, :] - yp[None, :, :], axis=-1)), axis=-1)
    return np.exp(-(emd * emd) / (2.0 * float(sigma) * float(sigma)))


def orbit_histogram_matrix(graphs: Sequence[nx.Graph], backend: str = "auto") -> np.ndarray:
    def histogram(g: nx.Graph) -> np.ndarray:
        counts = np.asarray(orbit_count_vector(g, backend=backend), dtype=np.float64).reshape(-1)
        total = float(np.sum(counts))
        return counts / (total + 1e-8)

    return descriptor_matrix(graphs, histogram)


def mmd_orbit(reference: Sequence[nx.Graph], generated: Sequence[nx.Graph], sigma: float = 1.0, backend: str = "auto") -> float:
    h_ref = orbit_histogram_matrix(reference, backend=backend)
    h_gen = orbit_histogram_matrix(generated, backend=backend)
    if h_ref.size == 0 or h_gen.size == 0:
        return float("nan")
    k_xx = gaussian_emd_kernel(h_ref, h_ref, sigma)
    k_yy = gaussian_emd_kernel(h_gen, h_gen, sigma)
    k_xy = gaussian_emd_kernel(h_ref, h_gen, sigma)
    return float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())


def connectedness_rate(graphs: Sequence[nx.Graph]) -> float:
    if not graphs:
        return 0.0
    return float(np.mean([nx.is_connected(g) if g.number_of_nodes() > 0 else False for g in graphs]))


def simple_graph_validity_rate(graphs: Sequence[nx.Graph]) -> float:
    if not graphs:
        return 0.0
    vals = []
    for g in graphs:
        vals.append(nx.number_of_selfloops(g) == 0 and isinstance(g, nx.Graph))
    return float(np.mean(vals))


def wl_uniqueness_rate(graphs: Sequence[nx.Graph]) -> float:
    if not graphs:
        return 0.0
    hashes = [nx.weisfeiler_lehman_graph_hash(g) for g in graphs]
    return float(len(set(hashes)) / len(hashes))


def wl_novelty_rate(generated: Sequence[nx.Graph], train: Sequence[nx.Graph]) -> float:
    if not generated:
        return 0.0
    train_hashes = {nx.weisfeiler_lehman_graph_hash(g) for g in train}
    gen_hashes = [nx.weisfeiler_lehman_graph_hash(g) for g in generated]
    return float(np.mean([h not in train_hashes for h in gen_hashes]))


def degree_preservation_rate(before: Sequence[nx.Graph], after: Sequence[nx.Graph]) -> float:
    if not before:
        return 0.0
    vals = []
    for g0, g1 in zip(before, after):
        d0 = sorted([d for _, d in g0.degree()], reverse=True)
        d1 = sorted([d for _, d in g1.degree()], reverse=True)
        vals.append(d0 == d1)
    return float(np.mean(vals))


def evaluate_graph_sets(reference: Sequence[nx.Graph], generated: Sequence[nx.Graph], train: Sequence[nx.Graph] | None = None) -> dict[str, float]:
    max_degree = 0
    for g in list(reference) + list(generated):
        if g.number_of_nodes():
            max_degree = max(max_degree, max(dict(g.degree()).values()))
    deg_ref = descriptor_matrix(reference, lambda g: degree_histogram(g, max_degree))
    deg_gen = descriptor_matrix(generated, lambda g: degree_histogram(g, max_degree))
    clus_ref = descriptor_matrix(reference, lambda g: clustering_histogram(g, 20))
    clus_gen = descriptor_matrix(generated, lambda g: clustering_histogram(g, 20))
    spec_ref = descriptor_matrix(reference, lambda g: spectral_histogram(g, 20))
    spec_gen = descriptor_matrix(generated, lambda g: spectral_histogram(g, 20))
    motif_ref = descriptor_matrix(reference, motif_proxy_vector)
    motif_gen = descriptor_matrix(generated, motif_proxy_vector)
    metrics = {
        "num_graphs": float(len(generated)),
        "degree_mmd": mmd_rbf(deg_ref, deg_gen),
        "clustering_mmd": mmd_rbf(clus_ref, clus_gen),
        "spectral_mmd": mmd_rbf(spec_ref, spec_gen),
        "motif_proxy_mmd": mmd_rbf(motif_ref, motif_gen),
        "orbit_mmd": mmd_orbit(reference, generated),
        "connectedness_rate": connectedness_rate(generated),
        "validity_rate": simple_graph_validity_rate(generated),
        "uniqueness_rate": wl_uniqueness_rate(generated),
    }
    if train is not None:
        metrics["novelty_rate"] = wl_novelty_rate(generated, train)
    return metrics
