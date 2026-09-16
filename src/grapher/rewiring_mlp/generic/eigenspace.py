from __future__ import annotations

"""Low-frequency Laplacian eigenspace utilities.

The eigenspace state is represented by an orthogonal projector

    P_k = U_k U_k^T,

where U_k contains the first ``k`` non-trivial eigenvectors of the
combinatorial Laplacian (the constant zero mode is excluded).  Projectors are
invariant to eigenvector sign flips and rotations within a repeated-eigenvalue
subspace, which makes them substantially better behaved than raw eigenvectors.
"""

from typing import Sequence

import networkx as nx
import numpy as np


def effective_projector_rank(num_nodes: int, rank: int) -> int:
    """Return the valid non-trivial Laplacian subspace rank for one graph."""

    n = int(num_nodes)
    k = int(rank)
    if n < 0:
        raise ValueError("num_nodes must be nonnegative.")
    if k < 0:
        raise ValueError("projector rank must be nonnegative.")
    return min(k, max(n - 1, 0))


def _laplacian_matrix(graph: nx.Graph) -> np.ndarray:
    nodes = list(range(graph.number_of_nodes()))
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight=None, dtype=np.float64)
    adjacency = (adjacency > 0).astype(np.float64)
    degree = adjacency.sum(axis=1)
    return np.diag(degree) - adjacency


def laplacian_eigenspace_projector(
    graph: nx.Graph,
    *,
    rank: int,
) -> np.ndarray:
    """Return the low-frequency non-trivial Laplacian projector ``P_k``.

    The constant eigenvector is omitted.  For graphs smaller than ``rank+1``
    the rank is reduced to ``n-1``.  The returned matrix is always ``[n,n]``.
    """

    n = graph.number_of_nodes()
    k = effective_projector_rank(n, rank)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if k == 0:
        return np.zeros((n, n), dtype=np.float64)
    laplacian = _laplacian_matrix(graph)
    _values, vectors = np.linalg.eigh(laplacian)
    # np.linalg.eigh returns eigenvectors in ascending eigenvalue order.  The
    # first vector is the constant zero mode for a connected graph.
    basis = np.asarray(vectors[:, 1 : 1 + k], dtype=np.float64)
    projector = basis @ basis.T
    projector = 0.5 * (projector + projector.T)
    return projector


def validate_eigenspace_projector(
    projector: Sequence[Sequence[float]] | np.ndarray,
    *,
    n: int | None = None,
    rank: int | None = None,
    atol: float = 1.0e-5,
) -> np.ndarray:
    values = np.asarray(projector, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Eigenspace projector must have shape [n,n].")
    if n is not None and values.shape != (int(n), int(n)):
        raise ValueError("Eigenspace projector size differs from the graph size.")
    if not np.isfinite(values).all():
        raise ValueError("Eigenspace projector contains non-finite values.")
    if not np.allclose(values, values.T, atol=atol, rtol=0.0):
        raise ValueError("Eigenspace projector must be symmetric.")
    if rank is not None:
        expected = effective_projector_rank(values.shape[0], int(rank))
        if not np.isclose(float(np.trace(values)), float(expected), atol=max(atol, 5.0e-4)):
            raise ValueError("Eigenspace projector trace does not match its expected rank.")
    return values


def _helmert_basis(num_nodes: int) -> np.ndarray:
    """Deterministic orthonormal basis for the subspace orthogonal to 1."""
    n = int(num_nodes)
    if n <= 1:
        return np.zeros((n, 0), dtype=np.float64)
    basis = np.zeros((n, n - 1), dtype=np.float64)
    for column in range(n - 1):
        j = column + 1
        denom = np.sqrt(float(j * (j + 1)))
        basis[:j, column] = 1.0 / denom
        basis[j, column] = -float(j) / denom
    return basis


def project_to_eigenspace_projector(
    matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    rank: int,
) -> np.ndarray:
    """Project a symmetric score matrix onto the rank-k Laplacian subspace manifold.

    The matrix is first restricted to the subspace orthogonal to the constant
    vector.  The eigenvectors with the largest eigenvalues then define the
    nearest rank-k orthogonal projector in that centered subspace.
    """

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Projector score matrix must have shape [n,n].")
    n = values.shape[0]
    k = effective_projector_rank(n, rank)
    if n == 0 or k == 0:
        return np.zeros((n, n), dtype=np.float64)
    values = 0.5 * (values + values.T)
    q = _helmert_basis(n)
    reduced = q.T @ values @ q
    eigvals, eigvecs = np.linalg.eigh(reduced)
    basis = q @ eigvecs[:, np.argsort(eigvals)[-k:]]
    projector = basis @ basis.T
    return 0.5 * (projector + projector.T)


def eigenspace_projector_distance(
    left: Sequence[Sequence[float]] | np.ndarray,
    right: Sequence[Sequence[float]] | np.ndarray,
    *,
    rank: int,
    metric: str = "chordal",
) -> float:
    """Distance between two low-frequency eigenspaces.

    ``chordal`` uses ``||P-Q||_F / sqrt(2k)`` and is in ``[0,1]`` for exact
    rank-k projectors. ``rmse`` is provided for diagnostics.
    """

    a = validate_eigenspace_projector(left)
    b = validate_eigenspace_projector(right, n=a.shape[0])
    delta = a - b
    name = str(metric).lower()
    if name == "rmse":
        return float(np.sqrt(np.mean(np.square(delta)))) if delta.size else 0.0
    if name in {"frobenius", "fro"}:
        return float(np.linalg.norm(delta, ord="fro"))
    if name not in {"chordal", "projector", "projector_chordal"}:
        raise ValueError("Projector distance must be chordal, rmse, or frobenius.")
    k = effective_projector_rank(a.shape[0], rank)
    if k == 0:
        return 0.0
    return float(np.linalg.norm(delta, ord="fro") / np.sqrt(2.0 * float(k)))


def projector_node_signatures(
    graph: nx.Graph,
    projector: np.ndarray,
) -> np.ndarray:
    """Permutation-invariant per-node signatures for degree-constrained matching.

    Each row uses local graph structure plus statistics of the corresponding
    projector row whose off-diagonal entries are sorted.  Sorting makes the row
    signature invariant to the *other* node labels while retaining node-wise
    low-frequency leverage information.
    """

    n = graph.number_of_nodes()
    p = validate_eigenspace_projector(projector, n=n)
    triangles = nx.triangles(graph)
    clustering = nx.clustering(graph)
    degrees = dict(graph.degree())
    rows: list[np.ndarray] = []
    scale = float(max(n - 1, 1))
    for node in range(n):
        neighbors = list(graph.neighbors(node))
        neighbor_degrees = sorted((degrees[v] / scale for v in neighbors), reverse=True)
        padded_neighbor_degrees = np.zeros(n - 1, dtype=np.float64)
        if neighbor_degrees:
            padded_neighbor_degrees[: len(neighbor_degrees)] = neighbor_degrees
        row_values = np.delete(p[node], node)
        row_values = np.sort(row_values)
        possible_triangles = max(degrees[node] * (degrees[node] - 1) / 2.0, 1.0)
        prefix = np.asarray(
            [
                float(clustering[node]),
                float(triangles[node]) / possible_triangles,
                float(p[node, node]),
            ],
            dtype=np.float64,
        )
        rows.append(np.concatenate([prefix, padded_neighbor_degrees, row_values]))
    return np.stack(rows, axis=0) if rows else np.zeros((0, 0), dtype=np.float64)
