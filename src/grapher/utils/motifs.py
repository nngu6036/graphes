from __future__ import annotations

import itertools
import os
import subprocess
from collections import Counter
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any, Iterable, Iterator

import networkx as nx
import numpy as np


NAUTY_EXEC = os.environ.get("NAUTY_EXEC")


# ============================================================
# Validation
# ============================================================

def _validate_simple_undirected(G: nx.Graph) -> None:
    if G.is_directed():
        raise ValueError("G must be undirected.")

    if G.is_multigraph():
        raise ValueError("G must be simple: multigraphs are not supported.")

    if nx.number_of_selfloops(G) > 0:
        raise ValueError("G must be simple: no self-loops.")

# ============================================================
# Nauty canonicalization utility
# ============================================================

@dataclass(frozen=True)
class NautyCanonicalizer:
    """
    Canonicalize simple undirected NetworkX graphs using nauty labelg.

    The canonical key is a canonical graph6 string.

    Parameters
    ----------
    nauty_exec:
        Either path to labelg or path to the nauty build directory.

    use_traces:
        If True, call labelg with -t.
        Do not mix True and False in the same cache/evaluation.

    use_sparse_internal:
        If True, call labelg with -S.
        This may help on sparse larger graphs, but it changes the canonical labelling.
        For small k-motifs, the default False is usually fine.
    """

    nauty_exec: str | os.PathLike[str] = NAUTY_EXEC
    use_traces: bool = False
    use_sparse_internal: bool = False

    def _labelg_path(self) -> str:
        path = Path(self.nauty_exec)

        if path.is_dir():
            path = path / "labelg"

        if not path.exists():
            raise FileNotFoundError(
                f"Cannot find labelg at {path}. "
                "Set NAUTY_EXEC to either the labelg executable or the nauty directory."
            )

        if not os.access(path, os.X_OK):
            raise PermissionError(f"{path} exists but is not executable.")

        return str(path)

    @staticmethod
    def _to_graph6_line(G: nx.Graph) -> bytes:
        """
        Convert a simple undirected graph to one graph6 line.

        This only prepares the input for labelg. It is not canonicalization.
        """
        _validate_simple_undirected(G)

        H = nx.convert_node_labels_to_integers(G, ordering="default")
        return nx.to_graph6_bytes(H, header=False).strip()

    def canonical_graph6(self, G: nx.Graph) -> str:
        """
        Return the canonical graph6 string for one graph.
        """
        return self.canonical_graph6_batch([G])[0]

    def canonical_graph(self, G: nx.Graph) -> tuple[str, nx.Graph]:
        """
        Return both canonical graph6 string and the canonical NetworkX graph.
        """
        key = self.canonical_graph6(G)
        return key, nx.from_graph6_bytes(key.encode("ascii"))

    def canonical_graph6_batch(self, graphs: Iterable[nx.Graph]) -> list[str]:
        """
        Canonicalize many graphs in one labelg subprocess call.

        This is much faster than launching labelg once per subgraph.
        """
        graph6_lines = [self._to_graph6_line(G) for G in graphs]

        if not graph6_lines:
            return []

        input_bytes = b"\n".join(graph6_lines) + b"\n"

        cmd = [self._labelg_path(), "-q", "-g"]

        if self.use_sparse_internal:
            cmd.append("-S")

        if self.use_traces:
            cmd.append("-t")

        result = subprocess.run(
            cmd,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "nauty labelg failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr:\n{result.stderr.decode(errors='replace')}"
            )

        keys = [
            line.strip()
            for line in result.stdout.decode("ascii").splitlines()
            if line.strip()
        ]

        if len(keys) != len(graph6_lines):
            raise RuntimeError(
                "labelg returned a different number of graphs than expected: "
                f"expected {len(graph6_lines)}, got {len(keys)}"
            )

        return keys


# ============================================================
# Subset sampling
# ============================================================

def _sample_node_subsets(
    nodes: list[Any],
    k: int,
    *,
    num_samples: int | None = None,
    rng: np.random.Generator | None = None,
) -> Iterator[tuple[Any, ...]]:
    """
    Yield k-node subsets.

    If num_samples is None, enumerate all subsets exactly.
    If num_samples is positive, sample that many distinct k-subsets approximately
    uniformly without replacement.

    This version avoids sorted(node_labels), which can fail when node labels
    are mixed or not orderable.
    """
    n = len(nodes)

    if k > n:
        return

    total = comb(n, k)

    if num_samples is None or int(num_samples) <= 0 or total <= int(num_samples):
        yield from itertools.combinations(nodes, k)
        return

    wanted = int(num_samples)
    generator = rng if rng is not None else np.random.default_rng(0)

    seen_index_subsets: set[tuple[int, ...]] = set()
    max_attempts = max(100, wanted * 50)
    attempts = 0

    while len(seen_index_subsets) < wanted and attempts < max_attempts:
        attempts += 1

        idx = generator.choice(n, size=k, replace=False)
        idx_tuple = tuple(sorted(int(i) for i in idx))

        if idx_tuple in seen_index_subsets:
            continue

        seen_index_subsets.add(idx_tuple)
        yield tuple(nodes[i] for i in idx_tuple)

    if len(seen_index_subsets) < wanted:
        raise RuntimeError(
            f"Only sampled {len(seen_index_subsets)} unique subsets out of requested "
            f"{wanted}. Try reducing num_samples or use exact enumeration."
        )


def _batched(items: Iterable[Any], batch_size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []

    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def _iter_k_induced_subgraphs(
    G: nx.Graph,
    k: int,
    *,
    connected_only: bool,
    num_samples: int | None,
    rng: np.random.Generator | None,
) -> Iterator[nx.Graph]:
    _validate_simple_undirected(G)

    if k <= 0:
        raise ValueError("k must be positive.")

    if k > G.number_of_nodes():
        return

    nodes = list(G.nodes())

    for subset in _sample_node_subsets(
        nodes,
        k,
        num_samples=num_samples,
        rng=rng,
    ):
        H = G.subgraph(subset)

        if connected_only and not nx.is_connected(H):
            continue

        yield H


# ============================================================
# Improved motif / graphlet functions
# ============================================================

def k_induced_subgraph_canonical_strings(
    G: nx.Graph,
    k: int,
    *,
    connected_only: bool = True,
    unique: bool = False,
    num_samples: int | None = None,
    rng: np.random.Generator | None = None,
    canonicalizer: NautyCanonicalizer | None = None,
    batch_size: int = 4096,
) -> list[str]:
    """
    Return canonical graph6 strings of k-node induced subgraphs.

    If unique=False, returns one key per accepted k-subset.
    If unique=True, returns unique canonical keys only.
    """
    canonicalizer = canonicalizer or NautyCanonicalizer()

    keys: list[str] = []

    subgraphs = _iter_k_induced_subgraphs(
        G,
        k,
        connected_only=connected_only,
        num_samples=num_samples,
        rng=rng,
    )

    for batch in _batched(subgraphs, batch_size=batch_size):
        keys.extend(canonicalizer.canonical_graph6_batch(batch))

    if unique:
        return sorted(set(keys))

    return keys


def graphlet_count_dict(
    G: nx.Graph,
    k: int,
    *,
    connected_only: bool = True,
    num_samples: int | None = None,
    rng: np.random.Generator | None = None,
    canonicalizer: NautyCanonicalizer | None = None,
    batch_size: int = 4096,
) -> dict[str, int]:
    """
    Count induced graphlets of size k using nauty canonical graph6 keys.
    """
    canonicalizer = canonicalizer or NautyCanonicalizer()

    counts: Counter[str] = Counter()

    subgraphs = _iter_k_induced_subgraphs(
        G,
        k,
        connected_only=connected_only,
        num_samples=num_samples,
        rng=rng,
    )

    for batch in _batched(subgraphs, batch_size=batch_size):
        keys = canonicalizer.canonical_graph6_batch(batch)
        counts.update(keys)

    return {str(key): int(value) for key, value in counts.items()}


def list_motifs(
    G: nx.Graph,
    k: int,
    *,
    canonicalizer: NautyCanonicalizer | None = None,
    connected_only: bool = True,
    batch_size: int = 4096,
) -> list[nx.Graph]:
    """
    Return all unique induced k-node motif types appearing in G.

    Representatives are returned as canonical NetworkX graphs with labels 0..k-1.
    """
    keys = k_induced_subgraph_canonical_strings(
        G,
        k,
        connected_only=connected_only,
        unique=True,
        canonicalizer=canonicalizer,
        batch_size=batch_size,
    )

    return [nx.from_graph6_bytes(key.encode("ascii")) for key in keys]


def aggregate_unique_motifs(
    graphs: list[nx.Graph],
    k: int,
    *,
    canonicalizer: NautyCanonicalizer | None = None,
    connected_only: bool = True,
    batch_size: int = 4096,
) -> list[nx.Graph]:
    """
    Return unique induced k-node motif types appearing in any graph.
    """
    canonicalizer = canonicalizer or NautyCanonicalizer()

    seen_keys: set[str] = set()

    for G in graphs:
        keys = k_induced_subgraph_canonical_strings(
            G,
            k,
            connected_only=connected_only,
            unique=True,
            canonicalizer=canonicalizer,
            batch_size=batch_size,
        )
        seen_keys.update(keys)

    return [nx.from_graph6_bytes(key.encode("ascii")) for key in sorted(seen_keys)]


def aggregate_unique_motifs_with_counts(
    graphs: list[nx.Graph],
    k: int,
    *,
    canonicalizer: NautyCanonicalizer | None = None,
    connected_only: bool = True,
    batch_size: int = 4096,
    progress_interval: int = 0,
    log_fn=None,
) -> list[tuple[nx.Graph, int]]:
    """
    Return unique induced k-node motif types and their total occurrence counts
    across all input graphs.
    """
    canonicalizer = canonicalizer or NautyCanonicalizer()

    counts: Counter[str] = Counter()

    total_graphs = len(graphs)
    for graph_idx, G in enumerate(graphs, start=1):
        counts.update(
            graphlet_count_dict(
                G,
                k,
                connected_only=connected_only,
                canonicalizer=canonicalizer,
                batch_size=batch_size,
            )
        )
        if progress_interval and (
            graph_idx == 1
            or graph_idx % int(progress_interval) == 0
            or graph_idx == total_graphs
        ):
            message = (
                f"k={k} graph={graph_idx}/{total_graphs} "
                f"unique_motifs={len(counts)} total_occurrences={sum(counts.values())}"
            )
            if log_fn is not None:
                log_fn(message)
            else:
                print(message, flush=True)

    return [
        (nx.from_graph6_bytes(key.encode("ascii")), int(counts[key]))
        for key in sorted(counts.keys())
    ]


# ============================================================
# Frequency / history utilities
# ============================================================

def normalize_count_dict(counts: dict[str, int | float]) -> dict[str, float]:
    total = float(sum(float(v) for v in counts.values()))
    if total <= 0.0:
        return {}
    return {
        str(k): float(v) / total
        for k, v in counts.items()
        if float(v) > 0.0
    }


def graphlet_frequency_dict(
    G: nx.Graph,
    k: int,
    *,
    connected_only: bool = True,
    num_samples: int | None = None,
    rng: np.random.Generator | None = None,
    canonicalizer: NautyCanonicalizer | None = None,
    batch_size: int = 4096,
) -> dict[str, float]:
    return normalize_count_dict(
        graphlet_count_dict(
            G,
            k,
            connected_only=connected_only,
            num_samples=num_samples,
            rng=rng,
            canonicalizer=canonicalizer,
            batch_size=batch_size,
        )
    )


def graphlet_history(
    G: nx.Graph,
    *,
    k_min: int = 3,
    k_max: int = 5,
    connected_only: bool = True,
    num_samples: int | None = None,
    rng: np.random.Generator | None = None,
    canonicalizer: NautyCanonicalizer | None = None,
    batch_size: int = 4096,
) -> dict[str, dict[str, float]]:
    """
    Return graphlet-frequency history {"3": {...}, ..., "K": {...}}.
    Each size-specific dictionary is normalized independently.
    """
    if k_min <= 0 or k_max < k_min:
        raise ValueError("Require 1 <= k_min <= k_max.")

    canonicalizer = canonicalizer or NautyCanonicalizer()

    return {
        str(k): graphlet_frequency_dict(
            G,
            k,
            connected_only=connected_only,
            num_samples=num_samples,
            rng=rng,
            canonicalizer=canonicalizer,
            batch_size=batch_size,
        )
        for k in range(int(k_min), int(k_max) + 1)
    }


def graphlet_keys_by_size(
    histories: list[dict[str, dict[str, float]]],
) -> dict[str, list[str]]:
    """
    Collect a stable key basis for a list of graphlet histories.
    """
    keys: dict[str, set[str]] = {}

    for history in histories:
        for k, hist in (history or {}).items():
            keys.setdefault(str(k), set()).update(str(key) for key in hist.keys())

    return {
        str(k): sorted(vals)
        for k, vals in sorted(keys.items(), key=lambda item: int(item[0]))
    }


def flatten_graphlet_history(
    history: dict[str, dict[str, float]] | None,
    keys_by_k: dict[str, list[str]] | None,
) -> np.ndarray:
    """
    Flatten a graphlet history according to a fixed basis.
    """
    if not keys_by_k:
        return np.zeros(0, dtype=np.float64)

    history = history or {}
    parts: list[np.ndarray] = []

    for k in sorted(keys_by_k.keys(), key=lambda x: int(x)):
        hist = history.get(str(k), {}) or {}
        parts.append(
            np.asarray(
                [float(hist.get(key, 0.0)) for key in keys_by_k[k]],
                dtype=np.float64,
            )
        )

    if not parts:
        return np.zeros(0, dtype=np.float64)

    return np.concatenate(parts, axis=0)


def unflatten_graphlet_history(
    vector: np.ndarray,
    keys_by_k: dict[str, list[str]] | None,
) -> dict[str, dict[str, float]]:
    """
    Turn a flat vector back into a size-indexed graphlet history.
    """
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
            history[str(k)] = {
                key: float(vals[i])
                for i, key in enumerate(keys)
                if float(vals[i]) > 0.0
            }

    return history


def graphlet_history_l2_distance(
    current: dict[str, dict[str, float]] | None,
    target: dict[str, dict[str, float]] | None,
    *,
    size_weights: dict[str, float] | None = None,
    normalize_terms: bool = True,
) -> float:
    """
    L2 distance between two graphlet histories using dynamic key unions.
    """
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
