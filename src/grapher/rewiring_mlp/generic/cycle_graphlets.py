from __future__ import annotations

from math import comb
from typing import Any

import networkx as nx
import numpy as np

CYCLE_GRAPHLET_K = 3
CYCLE_GRAPHLET_BINS = ("cycle", "other")


def validate_cycle_graphlet_k(value: Any) -> int:
    """This first implementation intentionally supports only induced C3."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value != 3:
        raise ValueError("cycle_graphlet_k / cycle_guidance.k must be the integer 3; only C3 is implemented.")
    return int(value)


def cycle_graphlet_k(config: dict[str, Any] | None = None) -> int | None:
    values = dict(config or {})
    if not bool(values.get("cycle_graphlet_histogram", False)):
        return None
    return validate_cycle_graphlet_k(values.get("cycle_graphlet_k", 3))


def count_cycle_graphlets(graph: nx.Graph, *, k: int = 3) -> int:
    """Count induced triangle occurrences, once per unordered node triple.

    We count ALL triangles, not just triangles in a cycle basis. Attributes and
    edge weights are ignored. The input must be simple, undirected and loop-free.
    """
    validate_cycle_graphlet_k(k)
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("Cycle graphlets require a simple undirected loop-free graph.")
    return sum(nx.triangles(graph).values()) // 3


def cycle_histogram_from_count(count: int, n: int, *, k: int = 3) -> np.ndarray:
    validate_cycle_graphlet_k(k)
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer.")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise ValueError("The cycle count must be an integer.")
    total = comb(int(n), k) if n >= k else 0
    if count < 0 or count > total:
        raise ValueError("Cycle count must be between zero and the number of node triples.")
    # No k-subsets exist when n<k. This placeholder is masked in learning and
    # has zero candidate discrepancy; it is NOT an observed empty-subset trial.
    p = float(count) / total if total else 0.0
    return np.array([p, 1.0 - p], dtype=np.float64)


def extract_cycle_graphlet_histogram(graph: nx.Graph, *, k: int = 3) -> np.ndarray:
    """[triangle density, all-other-triples density], normalized by choose(n,3).

    A histogram conditional on being a cycle would be identically [1] at k=3.
    The residual 'other' bin retains occurrence mass, including disconnected
    triples and connected non-cycle triples, without separate non-cycle heads.
    """
    count = count_cycle_graphlets(graph, k=k)
    return cycle_histogram_from_count(count, graph.number_of_nodes(), k=k)


def validate_cycle_graphlet_histogram(values: Any) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (2,):
        raise ValueError("Cycle graphlet histogram must have shape (2,): [cycle, other].")
    if not np.all(np.isfinite(vector)) or np.any(vector < -1e-8) or np.any(vector > 1.0 + 1e-8):
        raise ValueError("Cycle graphlet histogram must be finite and in [0,1].")
    if not np.isclose(vector.sum(), 1.0, atol=1e-6, rtol=0):
        raise ValueError("Cycle graphlet histogram must sum to one.")
    clipped = np.clip(vector, 0.0, 1.0)
    return clipped / clipped.sum()


def cycle_graphlet_histogram_distance(current: Any, target: Any) -> float:
    """Total variation on the two bins, equal to triangle-density error."""
    a = validate_cycle_graphlet_histogram(current)
    b = validate_cycle_graphlet_histogram(target)
    return float(0.5 * np.abs(a - b).sum())
