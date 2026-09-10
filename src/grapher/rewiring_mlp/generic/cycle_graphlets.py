from __future__ import annotations

from math import comb
from typing import Any

import networkx as nx
import numpy as np

from grapher.utils.motifs import induced_simple_cycle_node_sets

SUPPORTED_CYCLE_GRAPHLET_K = (3, 4, 5)
CYCLE_GRAPHLET_BINS = ("cycle", "other")


def validate_cycle_graphlet_k(value: Any) -> int:
    """Validate the exact induced cycle size used by the cycle summary.

    The minimal GraphER cycle head supports a single cycle order per model,
    chosen from C3, C4, or C5.  A k-node set counts only when its induced
    subgraph is exactly the chordless cycle C_k.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) not in SUPPORTED_CYCLE_GRAPHLET_K
    ):
        raise ValueError(
            "cycle_graphlet_k / cycle_guidance.k must be an integer in {3,4,5}."
        )
    return int(value)


def cycle_graphlet_k(config: dict[str, Any] | None = None) -> int | None:
    values = dict(config or {})
    if not bool(values.get("cycle_graphlet_histogram", False)):
        return None
    return validate_cycle_graphlet_k(values.get("cycle_graphlet_k", 3))


def _validate_graph(graph: nx.Graph) -> None:
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("Cycle graphlets require a simple undirected loop-free graph.")


def count_cycle_graphlets(graph: nx.Graph, *, k: int = 3) -> int:
    """Count induced chordless C_k occurrences for k in {3,4,5}.

    Each graphlet is counted once by its unordered node set.  For k=3 this is
    the ordinary triangle count.  For k=4 and k=5, cycles with a chord are not
    counted because their induced subgraph is not C_k.
    """
    k = validate_cycle_graphlet_k(k)
    _validate_graph(graph)
    if graph.number_of_nodes() < k:
        return 0
    if k == 3:
        # Fast path. Every triangle is automatically an induced C3 on its
        # three selected nodes.
        return int(sum(nx.triangles(graph).values()) // 3)
    return int(sum(1 for _ in induced_simple_cycle_node_sets(graph, k)))


def cycle_histogram_from_count(count: int, n: int, *, k: int = 3) -> np.ndarray:
    k = validate_cycle_graphlet_k(k)
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer.")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise ValueError("The cycle count must be an integer.")
    total = comb(int(n), k) if n >= k else 0
    if count < 0 or count > total:
        raise ValueError(
            f"Cycle count must be between zero and choose(n,{k})."
        )
    # No k-subsets exist when n<k. This placeholder is masked in learning and
    # has zero candidate discrepancy; it is not an observed empty-subset trial.
    p = float(count) / total if total else 0.0
    return np.array([p, 1.0 - p], dtype=np.float64)


def extract_cycle_graphlet_histogram(graph: nx.Graph, *, k: int = 3) -> np.ndarray:
    """Return [induced-C_k density, other-k-subset density].

    The cycle coordinate is count(C_k) / choose(n,k).  The residual coordinate
    aggregates every k-node subset whose induced subgraph is not exactly C_k.
    This retains occurrence mass and avoids the degenerate cycle-conditional
    histogram [1] when only one cycle isomorphism class exists at a fixed k.
    """
    k = validate_cycle_graphlet_k(k)
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
    """Total variation on [C_k, other], equal to C_k-density error."""
    a = validate_cycle_graphlet_histogram(current)
    b = validate_cycle_graphlet_histogram(target)
    return float(0.5 * np.abs(a - b).sum())
