from __future__ import annotations

import itertools
from collections import Counter
from math import comb
from typing import Any

import networkx as nx
import numpy as np


def canonical_key_bruteforce(H: nx.Graph) -> str:
    """
    Exact canonical key for a small simple undirected graph.

    Two isomorphic graphs will have the same key. This is suitable for small
    fixed k, for example k <= 6.
    """
    nodes = list(H.nodes())
    k = len(nodes)

    best_key = None
    for perm in itertools.permutations(nodes):
        bits = []
        for i in range(k):
            for j in range(i + 1, k):
                bits.append("1" if H.has_edge(perm[i], perm[j]) else "0")

        key = "".join(bits)
        if best_key is None or key < best_key:
            best_key = key

    return str(best_key)


def _validate_simple_undirected(G: nx.Graph) -> None:
    if G.is_directed():
        raise ValueError("G must be undirected.")
    if any(u == v for u, v in G.edges()):
        raise ValueError("G must be simple: no self-loops.")


def list_motifs(G: nx.Graph, k: int) -> list[nx.Graph]:
    """
    Return all unique connected induced motifs of size k from one graph.
    """
    _validate_simple_undirected(G)

    if k <= 0:
        raise ValueError("k must be positive.")

    if k > G.number_of_nodes():
        return []

    motifs: list[nx.Graph] = []
    seen_keys: set[str] = set()

    for node_subset in itertools.combinations(G.nodes(), k):
        H = G.subgraph(node_subset).copy()
        if not nx.is_connected(H):
            continue

        key = canonical_key_bruteforce(H)
        if key not in seen_keys:
            seen_keys.add(key)
            motifs.append(nx.convert_node_labels_to_integers(H, ordering="sorted"))

    return motifs


def aggregate_unique_motifs(graphs: list[nx.Graph], k: int) -> list[nx.Graph]:
    """
    Return unique connected induced k-node motifs appearing in any graph.

    Isomorphic motifs are merged, so only one representative is kept.
    """
    unique_motifs: list[nx.Graph] = []
    seen_keys: set[str] = set()

    for G in graphs:
        for motif in list_motifs(G, k):
            key = canonical_key_bruteforce(motif)
            if key not in seen_keys:
                seen_keys.add(key)
                unique_motifs.append(nx.convert_node_labels_to_integers(motif, ordering="sorted"))

    return unique_motifs


def aggregate_unique_motifs_with_counts(graphs: list[nx.Graph], k: int) -> list[tuple[nx.Graph, int]]:
    """
    Return unique connected induced k-node motifs and their total occurrence
    counts across all input graphs.
    """
    representatives: dict[str, nx.Graph] = {}
    counts: Counter[str] = Counter()

    for G in graphs:
        _validate_simple_undirected(G)

        if k <= 0:
            raise ValueError("k must be positive.")

        if k > G.number_of_nodes():
            continue

        for node_subset in itertools.combinations(G.nodes(), k):
            H = G.subgraph(node_subset).copy()
            if not nx.is_connected(H):
                continue

            key = canonical_key_bruteforce(H)
            counts[key] += 1
            if key not in representatives:
                representatives[key] = nx.convert_node_labels_to_integers(H, ordering="sorted")

    return [(representatives[key], int(counts[key])) for key in counts]


def _sample_node_subsets(
    nodes: list[Any],
    k: int,
    *,
    num_samples: int | None = None,
    rng: np.random.Generator | None = None,
):
    """Yield k-node subsets exactly, or sampled without replacement when requested."""

    n = len(nodes)
    if k > n:
        return

    total = comb(n, k)
    if num_samples is None or int(num_samples) <= 0 or total <= int(num_samples):
        yield from itertools.combinations(nodes, k)
        return

    generator = rng if rng is not None else np.random.default_rng(0)
    seen: set[tuple[Any, ...]] = set()
    max_attempts = int(num_samples) * 20
    attempts = 0
    while len(seen) < int(num_samples) and attempts < max_attempts:
        attempts += 1
        idx = generator.choice(n, size=k, replace=False)
        subset = tuple(sorted(nodes[int(i)] for i in idx))
        if subset in seen:
            continue
        seen.add(subset)
        yield subset


def graphlet_count_dict(
    G: nx.Graph,
    k: int,
    *,
    connected_only: bool = True,
    num_samples: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, int]:
    """Count induced graphlets of size ``k`` using exact canonical keys.

    The returned keys are canonical adjacency-bit strings produced by
    :func:`canonical_key_bruteforce`, so isomorphic induced subgraphs map to the
    same key. For large graphs, set ``num_samples`` to estimate frequencies from
    sampled node subsets.
    """

    _validate_simple_undirected(G)
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > G.number_of_nodes():
        return {}

    counts: Counter[str] = Counter()
    nodes = list(G.nodes())
    for subset in _sample_node_subsets(nodes, k, num_samples=num_samples, rng=rng):
        H = G.subgraph(subset).copy()
        if connected_only and (H.number_of_nodes() > 0 and not nx.is_connected(H)):
            continue
        key = canonical_key_bruteforce(H)
        counts[key] += 1
    return {str(key): int(value) for key, value in counts.items()}


def normalize_count_dict(counts: dict[str, int | float]) -> dict[str, float]:
    total = float(sum(float(v) for v in counts.values()))
    if total <= 0.0:
        return {}
    return {str(k): float(v) / total for k, v in counts.items() if float(v) > 0.0}


def graphlet_frequency_dict(
    G: nx.Graph,
    k: int,
    *,
    connected_only: bool = True,
    num_samples: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    return normalize_count_dict(
        graphlet_count_dict(G, k, connected_only=connected_only, num_samples=num_samples, rng=rng)
    )


def graphlet_history(
    G: nx.Graph,
    *,
    k_min: int = 3,
    k_max: int = 5,
    connected_only: bool = True,
    num_samples: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, dict[str, float]]:
    """Return graphlet-frequency history {"3": {...}, ..., "K": {...}}.

    Each size-specific dictionary is normalized independently, so h_k lies on a
    simplex. Sizes larger than the graph have an empty histogram.
    """

    if k_min <= 0 or k_max < k_min:
        raise ValueError("Require 1 <= k_min <= k_max.")
    return {
        str(k): graphlet_frequency_dict(
            G,
            k,
            connected_only=connected_only,
            num_samples=num_samples,
            rng=rng,
        )
        for k in range(int(k_min), int(k_max) + 1)
    }


def graphlet_keys_by_size(histories: list[dict[str, dict[str, float]]]) -> dict[str, list[str]]:
    """Collect a stable key basis for a list of graphlet histories."""

    keys: dict[str, set[str]] = {}
    for history in histories:
        for k, hist in (history or {}).items():
            keys.setdefault(str(k), set()).update(str(key) for key in hist.keys())
    return {str(k): sorted(vals) for k, vals in sorted(keys.items(), key=lambda item: int(item[0]))}


def flatten_graphlet_history(
    history: dict[str, dict[str, float]] | None,
    keys_by_k: dict[str, list[str]] | None,
) -> np.ndarray:
    """Flatten a graphlet history according to a fixed basis."""

    if not keys_by_k:
        return np.zeros(0, dtype=np.float64)
    history = history or {}
    parts = []
    for k in sorted(keys_by_k.keys(), key=lambda x: int(x)):
        hist = history.get(str(k), {}) or {}
        parts.append(np.asarray([float(hist.get(key, 0.0)) for key in keys_by_k[k]], dtype=np.float64))
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts, axis=0)


def unflatten_graphlet_history(
    vector: np.ndarray,
    keys_by_k: dict[str, list[str]] | None,
) -> dict[str, dict[str, float]]:
    """Turn a flat vector back into a size-indexed graphlet history."""

    if not keys_by_k:
        return {}
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    history: dict[str, dict[str, float]] = {}
    pos = 0
    for k in sorted(keys_by_k.keys(), key=lambda x: int(x)):
        keys = list(keys_by_k[k])
        width = len(keys)
        raw = np.maximum(vec[pos : pos + width], 0.0)
        pos += width
        total = float(raw.sum())
        if width == 0 or total <= 0.0:
            history[str(k)] = {}
        else:
            vals = raw / total
            history[str(k)] = {key: float(vals[i]) for i, key in enumerate(keys) if float(vals[i]) > 0.0}
    return history


def graphlet_history_l2_distance(
    current: dict[str, dict[str, float]] | None,
    target: dict[str, dict[str, float]] | None,
    *,
    size_weights: dict[str, float] | None = None,
    normalize_terms: bool = True,
) -> float:
    """L2 distance between two graphlet histories using dynamic key unions."""

    current = current or {}
    target = target or {}
    size_weights = size_weights or {}
    sizes = sorted(set(current.keys()) | set(target.keys()), key=lambda x: int(x))
    total = 0.0
    for k in sizes:
        c = current.get(str(k), {}) or {}
        t = target.get(str(k), {}) or {}
        keys = sorted(set(c.keys()) | set(t.keys()))
        if not keys:
            continue
        cv = np.asarray([float(c.get(key, 0.0)) for key in keys], dtype=np.float64)
        tv = np.asarray([float(t.get(key, 0.0)) for key in keys], dtype=np.float64)
        dist = float(np.linalg.norm(cv - tv))
        if normalize_terms:
            dist /= float(np.sqrt(max(len(keys), 1)))
        total += float(size_weights.get(str(k), 1.0)) * dist
    return float(total)
