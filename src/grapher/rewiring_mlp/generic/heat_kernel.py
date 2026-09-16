from __future__ import annotations

from typing import Sequence

import networkx as nx
import numpy as np


def _laplacian_matrix(graph: nx.Graph) -> np.ndarray:
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("Heat-kernel GraphER requires a simple undirected graph.")
    if nx.number_of_selfloops(graph):
        raise ValueError("Heat-kernel GraphER does not support self-loops.")
    nodes = list(range(graph.number_of_nodes()))
    n = len(nodes)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, dtype=np.float64, weight=None)
    degree = adjacency.sum(axis=1)
    return np.diag(degree) - adjacency


def heat_kernel_scale(graph: nx.Graph, *, mode: str = "mean_degree") -> float:
    mode = str(mode).lower()
    n = graph.number_of_nodes()
    trace = 2.0 * float(graph.number_of_edges())
    if mode in {"none", "raw"}:
        return 1.0
    if mode in {"trace", "degree_sum"}:
        return max(trace, 1.0)
    if mode in {"mean_degree", "average_degree", "avg_degree"}:
        return max(trace / max(float(n), 1.0), 1.0e-8)
    raise ValueError("heat-kernel normalization must be mean_degree, trace, or none.")


def validate_heat_times(times: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in times)
    if not values:
        raise ValueError("heat_kernel_times must contain at least one diffusion scale.")
    if any((not np.isfinite(value)) or value <= 0.0 for value in values):
        raise ValueError("heat_kernel_times must contain finite positive values.")
    return values


def heat_kernel_stack(
    graph: nx.Graph,
    *,
    times: Sequence[float],
    normalization: str = "mean_degree",
) -> np.ndarray:
    """Return multiscale combinatorial-Laplacian heat kernels [n,n,T].

    We diffuse with ``exp(-tau * L / scale)``. Dividing the combinatorial
    Laplacian by a graph-level scale makes a fixed list of tau values usable
    across different graph sizes/densities. For graphs within one degree fibre
    the scale is identical, so this normalization cannot change candidate
    ordering merely through degree changes.

    The full matrix contains both eigenvalue decay and eigenvector/eigenspace
    information: ``H_tau = U exp(-tau Lambda/scale) U^T``. Unlike raw
    eigenvectors it is invariant to eigenvector sign flips and rotations within
    repeated eigenspaces.
    """

    taus = validate_heat_times(times)
    laplacian = _laplacian_matrix(graph)
    n = laplacian.shape[0]
    if n == 0:
        return np.zeros((0, 0, len(taus)), dtype=np.float64)
    values, vectors = np.linalg.eigh(laplacian)
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    scale = heat_kernel_scale(graph, mode=normalization)
    channels: list[np.ndarray] = []
    for tau in taus:
        decay = np.exp(-float(tau) * values / max(float(scale), 1.0e-12))
        kernel = (vectors * decay[None, :]) @ vectors.T
        kernel = 0.5 * (kernel + kernel.T)
        # Numerical eigensolver noise can create tiny negative entries.
        kernel[np.abs(kernel) < 1.0e-14] = 0.0
        channels.append(kernel)
    return np.stack(channels, axis=-1)


def heat_kernel_distance(
    kernel: np.ndarray,
    target: np.ndarray,
    *,
    metric: str = "rmse",
    node_mask: np.ndarray | None = None,
) -> float:
    left = np.asarray(kernel, dtype=np.float64)
    right = np.asarray(target, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"Heat-kernel shapes must match: {left.shape} != {right.shape}.")
    if left.ndim != 3 or left.shape[0] != left.shape[1]:
        raise ValueError("Heat kernels must have shape [n,n,num_scales].")
    if left.size == 0:
        return 0.0
    if node_mask is None:
        valid = np.ones(left.shape[:2], dtype=np.bool_)
    else:
        mask = np.asarray(node_mask, dtype=np.bool_).reshape(-1)
        if mask.size != left.shape[0]:
            raise ValueError("Heat-kernel node mask has the wrong width.")
        valid = mask[:, None] & mask[None, :]
    delta = (left - right)[valid]
    metric_name = str(metric).lower()
    if metric_name in {"mae", "l1"}:
        return float(np.mean(np.abs(delta)))
    if metric_name in {"mse", "l2_squared"}:
        return float(np.mean(np.square(delta)))
    if metric_name in {"rmse", "l2", "frobenius"}:
        return float(np.sqrt(np.mean(np.square(delta))))
    raise ValueError("heat-kernel distance must be rmse, mse, mae, or frobenius.")


def validate_heat_kernel_stack(
    values: np.ndarray,
    *,
    n: int | None = None,
    num_scales: int | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] != array.shape[1]:
        raise ValueError("Heat-kernel state must have shape [n,n,num_scales].")
    if n is not None and array.shape[:2] != (int(n), int(n)):
        raise ValueError("Heat-kernel state graph size mismatch.")
    if num_scales is not None and array.shape[2] != int(num_scales):
        raise ValueError("Heat-kernel state scale count mismatch.")
    if not np.isfinite(array).all():
        raise ValueError("Heat-kernel state contains nonfinite values.")
    return array
