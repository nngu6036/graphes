from __future__ import annotations

import itertools
import json
import os
import shutil
import string
import subprocess
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from math import comb
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

NAUTY_EXEC = os.environ.get("NAUTY_EXEC") or shutil.which("labelg")


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
        if not self.nauty_exec:
            raise FileNotFoundError(
                "Cannot find nauty labelg. Set NAUTY_EXEC or add labelg to PATH."
            )
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


@dataclass(frozen=True)
class PythonCanonicalizer:
    """Exact dependency-free canonicalizer for small topology graphlets.

    It enumerates node orders and selects the lexicographically smallest
    graph6 representation.  This is deliberately a fallback for tests and
    small k; nauty remains strongly recommended for dataset-scale extraction.
    """

    max_nodes: int = 8

    def canonical_graph6(self, graph: nx.Graph) -> str:
        return self.canonical_graph6_batch([graph])[0]

    def canonical_graph6_batch(
        self,
        graphs: Iterable[nx.Graph],
    ) -> list[str]:
        return [self._canonical_graph6(graph) for graph in graphs]

    def _canonical_graph6(self, graph: nx.Graph) -> str:
        _validate_simple_undirected(graph)
        normalized = nx.convert_node_labels_to_integers(graph, ordering="default")
        raw = nx.to_graph6_bytes(normalized, header=False).strip().decode("ascii")
        return self._canonical_graph6_from_raw(raw, self.max_nodes)

    @staticmethod
    @lru_cache(maxsize=65536)
    def _canonical_graph6_from_raw(raw: str, max_nodes: int) -> str:
        graph = nx.from_graph6_bytes(raw.encode("ascii"))
        nodes = list(graph.nodes())
        n = len(nodes)
        if n > max_nodes:
            raise RuntimeError(
                "The Python graphlet canonicalizer supports at most "
                f"{max_nodes} nodes; install nauty labelg for k={n}."
            )
        best: bytes | None = None
        for order in itertools.permutations(nodes):
            position = {node: idx for idx, node in enumerate(order)}
            # Rebuild the graph so its insertion order is exactly 0..n-1.
            # ``nx.relabel_nodes`` preserves the old insertion order, while
            # NetworkX's graph6 writer follows insertion order even when a
            # ``nodes`` iterable is supplied.  Rebuilding is therefore
            # necessary for a genuinely label-invariant fallback.
            relabeled = nx.Graph()
            relabeled.add_nodes_from(range(n))
            relabeled.add_edges_from(
                (position[u], position[v]) for u, v in graph.edges()
            )
            encoded = nx.to_graph6_bytes(
                relabeled,
                nodes=range(n),
                header=False,
            ).strip()
            if best is None or encoded < best:
                best = encoded
        if best is None:
            best = nx.to_graph6_bytes(nx.Graph(), header=False).strip()
        return best.decode("ascii")


TOPOLOGY_CANONICALIZER_CONVENTION = "python_lexicographic_graph6_v1"

GRAPHLET_TOPOLOGY_FILTERS = frozenset({"all", "cyclic", "simple_cycle"})


def normalize_graphlet_topology_filter(value: Any = "all") -> str:
    """Return the canonical graphlet-topology filter name.

    ``all`` retains the historical connected-induced graphlet basis.
    ``cyclic`` retains every selected induced graphlet containing at least one
    cycle. ``simple_cycle`` retains only chordless cycles :math:`C_k` (one
    topology per order before attributes are considered).

    Several aliases are accepted so old experiment scripts can expose a
    concise ``graphlet_cycle_only`` boolean without changing checkpoint
    semantics.
    """

    if value is None:
        return "all"
    if isinstance(value, bool):
        return "simple_cycle" if value else "all"
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "": "all",
        "none": "all",
        "any": "all",
        "connected": "all",
        "cycle_containing": "cyclic",
        "contains_cycle": "cyclic",
        "has_cycle": "cyclic",
        "with_cycle": "cyclic",
        "cycle": "simple_cycle",
        "cycle_only": "simple_cycle",
        "simple_cycles": "simple_cycle",
        "induced_cycle": "simple_cycle",
        "induced_cycles": "simple_cycle",
        "chordless_cycle": "simple_cycle",
        "chordless_cycles": "simple_cycle",
        "ring": "simple_cycle",
        "rings": "simple_cycle",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in GRAPHLET_TOPOLOGY_FILTERS:
        choices = ", ".join(sorted(GRAPHLET_TOPOLOGY_FILTERS))
        raise ValueError(
            f"Unknown graphlet topology filter {value!r}; expected one of {choices}."
        )
    return normalized


def graphlet_topology_matches(
    graph: nx.Graph,
    topology_filter: str = "all",
) -> bool:
    """Return whether an induced graphlet belongs to the selected topology set."""

    selected = normalize_graphlet_topology_filter(topology_filter)
    if selected == "all":
        return True

    _validate_simple_undirected(graph)
    n = int(graph.number_of_nodes())
    m = int(graph.number_of_edges())
    if n < 3:
        return False

    if selected == "simple_cycle":
        # In a simple graph, connected + every degree equal to two is exactly C_k.
        return (
            m == n
            and all(int(degree) == 2 for _, degree in graph.degree())
            and nx.is_connected(graph)
        )

    # A simple undirected graph has a cycle iff its cyclomatic number is
    # positive: m - n + c > 0, where c is the number of components.
    components = int(nx.number_connected_components(graph)) if n else 0
    return m - n + components > 0


def default_topology_canonicalizer() -> PythonCanonicalizer:
    """Return the portable topology canonicalizer used by checkpoints.

    Topology graphlets are at most seven nodes, so the cached exact Python
    implementation is practical and, unlike environment-dependent dispatch to
    ``labelg``, gives training and generation identical coordinate keys.
    Nauty remains available explicitly for attributed workflows.
    """

    return PythonCanonicalizer()


@lru_cache(maxsize=32)
def _topology_graphlet_basis_cached(
    k: int,
    connected_only: bool,
    topology_filter: str,
) -> tuple[tuple[str, bytes], ...]:
    """Return every unlabeled topology graphlet of size ``k``.

    NetworkX's graph atlas contains one representative of every unlabeled
    simple graph with at most seven vertices.  Canonicalizing those
    representatives gives a stable, complete basis instead of learning the
    basis only from graphlet types observed in the training split.
    """

    k = int(k)
    if k <= 0 or k > 7:
        raise ValueError(
            "The built-in complete topology graphlet basis supports 1 <= k <= 7."
        )

    selected_filter = normalize_graphlet_topology_filter(topology_filter)
    representatives: list[nx.Graph] = []
    for graph in nx.graph_atlas_g():
        if graph.number_of_nodes() != k:
            continue
        graph = nx.convert_node_labels_to_integers(
            nx.Graph(graph),
            first_label=0,
            ordering="sorted",
        )
        if connected_only and k > 1 and not nx.is_connected(graph):
            continue
        if not graphlet_topology_matches(graph, selected_filter):
            continue
        representatives.append(graph)

    canonicalizer = default_topology_canonicalizer()
    keys = canonicalizer.canonical_graph6_batch(representatives)
    encoded = [
        (
            str(key),
            nx.to_graph6_bytes(graph, header=False).strip(),
        )
        for key, graph in zip(keys, representatives)
    ]
    return tuple(sorted(encoded, key=lambda item: item[0]))


def topology_graphlet_basis(
    k: int,
    *,
    connected_only: bool = True,
    topology_filter: str = "all",
) -> list[tuple[str, nx.Graph]]:
    """Return a complete canonical-key/representative basis for one size."""

    return [
        (key, nx.from_graph6_bytes(raw))
        for key, raw in _topology_graphlet_basis_cached(
            int(k),
            bool(connected_only),
            normalize_graphlet_topology_filter(topology_filter),
        )
    ]


def topology_graphlet_keys_by_size(
    k_min: int,
    k_max: int,
    *,
    connected_only: bool = True,
    topology_filter: str = "all",
) -> dict[str, list[str]]:
    """Return the complete canonical topology basis for every requested size."""

    if int(k_min) <= 0 or int(k_max) < int(k_min):
        raise ValueError("Require 1 <= k_min <= k_max.")
    return {
        str(k): [
            key
            for key, _ in topology_graphlet_basis(
                k,
                connected_only=connected_only,
                topology_filter=topology_filter,
            )
        ]
        for k in range(int(k_min), int(k_max) + 1)
    }


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


def induced_simple_cycle_node_sets(
    G: nx.Graph,
    k: int,
) -> Iterator[tuple[Any, ...]]:
    """Enumerate the node sets whose induced subgraph is a chordless ``C_k``.

    The search grows induced paths from the smallest-labelled node of each
    candidate ring.  Requiring the first endpoint to precede the last endpoint
    removes the two traversal-direction duplicates.  For bounded maximum
    degree this avoids scanning all ``binom(n, k)`` node subsets and visits at
    most ``O(n * Delta * (Delta - 1) ** (k - 2))`` path prefixes.
    """

    _validate_simple_undirected(G)
    k = int(k)
    if k < 3 or k > G.number_of_nodes():
        return

    nodes = sorted(G.nodes(), key=lambda node: (type(node).__name__, repr(node)))
    rank = {node: index for index, node in enumerate(nodes)}
    rings: set[frozenset[Any]] = set()

    for start in nodes:
        start_rank = rank[start]
        first_neighbors = sorted(
            (
                node
                for node in G.neighbors(start)
                if rank[node] > start_rank
            ),
            key=rank.__getitem__,
        )
        for first in first_neighbors:
            path: list[Any] = [start, first]
            visited: set[Any] = {start, first}

            def visit(current: Any) -> None:
                if len(path) == k:
                    if not G.has_edge(current, start):
                        return
                    # Each undirected ring is reached in both directions.
                    if rank[path[1]] >= rank[path[-1]]:
                        return
                    subset = frozenset(path)
                    induced = G.subgraph(subset)
                    if (
                        induced.number_of_edges() == k
                        and all(int(degree) == 2 for _, degree in induced.degree())
                    ):
                        rings.add(subset)
                    return

                next_length = len(path) + 1
                for neighbor in G.neighbors(current):
                    if neighbor == start or neighbor in visited:
                        continue
                    # ``start`` is the unique minimum-ranked node for this
                    # traversal, preventing rotations of the same ring.
                    if rank[neighbor] <= start_rank:
                        continue

                    # Preserve an induced path while growing it.  The only
                    # permitted earlier adjacency is the closing edge to
                    # ``start`` when adding the final node.
                    chord = False
                    for previous in path[:-1]:
                        if not G.has_edge(neighbor, previous):
                            continue
                        if next_length == k and previous == start:
                            continue
                        chord = True
                        break
                    if chord:
                        continue

                    path.append(neighbor)
                    visited.add(neighbor)
                    visit(neighbor)
                    visited.remove(neighbor)
                    path.pop()

            visit(first)

    for subset in sorted(
        rings,
        key=lambda values: tuple(sorted((rank[node] for node in values))),
    ):
        yield tuple(sorted(subset, key=rank.__getitem__))


def _iter_k_induced_subgraphs(
    G: nx.Graph,
    k: int,
    *,
    connected_only: bool,
    topology_filter: str,
    num_samples: int | None,
    rng: np.random.Generator | None,
) -> Iterator[nx.Graph]:
    _validate_simple_undirected(G)

    if k <= 0:
        raise ValueError("k must be positive.")

    if k > G.number_of_nodes():
        return

    selected_filter = normalize_graphlet_topology_filter(topology_filter)
    nodes = sorted(G.nodes(), key=lambda node: (type(node).__name__, repr(node)))

    total_subsets = comb(len(nodes), int(k))
    exact = (
        num_samples is None
        or int(num_samples) <= 0
        or total_subsets <= int(num_samples)
    )
    if selected_filter == "simple_cycle" and exact:
        for subset in induced_simple_cycle_node_sets(G, int(k)):
            yield G.subgraph(subset)
        return

    for subset in _sample_node_subsets(
        nodes,
        k,
        num_samples=num_samples,
        rng=rng,
    ):
        H = G.subgraph(subset)

        # Cycle filters provide a cheaper degree/edge-count rejection before
        # generic connectivity testing.  This matters during molecular basis
        # fitting, where almost every induced subset is acyclic.
        if selected_filter != "all":
            if not graphlet_topology_matches(H, selected_filter):
                continue
            if (
                connected_only
                and selected_filter == "cyclic"
                and not nx.is_connected(H)
            ):
                continue
        elif connected_only and not nx.is_connected(H):
            continue

        yield H


def _iter_connected_k_induced_subgraphs_exact(
    G: nx.Graph,
    k: int,
    *,
    topology_filter: str = "all",
) -> Iterator[nx.Graph]:
    """Enumerate connected k-sets by local frontier expansion."""

    _validate_simple_undirected(G)
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > G.number_of_nodes():
        return
    selected_filter = normalize_graphlet_topology_filter(topology_filter)
    if selected_filter == "simple_cycle":
        for subset in induced_simple_cycle_node_sets(G, int(k)):
            yield G.subgraph(subset)
        return
    nodes = sorted(G.nodes(), key=lambda node: (type(node).__name__, repr(node)))
    connected_sets: set[frozenset[Any]] = {
        frozenset((node,)) for node in nodes
    }
    for _size in range(2, int(k) + 1):
        expanded: set[frozenset[Any]] = set()
        for subset in connected_sets:
            frontier: set[Any] = set()
            for node in subset:
                frontier.update(G.neighbors(node))
            frontier.difference_update(subset)
            for node in frontier:
                expanded.add(subset | {node})
        connected_sets = expanded
        if not connected_sets:
            return
    ordered_sets = sorted(
        connected_sets,
        key=lambda subset: tuple(
            (type(node).__name__, repr(node))
            for node in sorted(
                subset,
                key=lambda value: (type(value).__name__, repr(value)),
            )
        ),
    )
    for subset in ordered_sets:
        subgraph = G.subgraph(subset)
        if graphlet_topology_matches(subgraph, selected_filter):
            yield subgraph


def graphlet_count_dict(
    G: nx.Graph,
    k: int,
    *,
    connected_only: bool = True,
    topology_filter: str = "all",
    num_samples: int | None = None,
    rng: np.random.Generator | None = None,
    canonicalizer: NautyCanonicalizer | PythonCanonicalizer | None = None,
    batch_size: int = 4096,
) -> dict[str, int]:
    """
    Count induced graphlets of size k using nauty canonical graph6 keys.
    """
    canonicalizer = canonicalizer or default_topology_canonicalizer()

    counts: Counter[str] = Counter()

    subgraphs = _iter_k_induced_subgraphs(
        G,
        k,
        connected_only=connected_only,
        topology_filter=topology_filter,
        num_samples=num_samples,
        rng=rng,
    )

    for batch in _batched(subgraphs, batch_size=batch_size):
        keys = canonicalizer.canonical_graph6_batch(batch)
        counts.update(keys)

    return {str(key): int(value) for key, value in counts.items()}


def connected_graphlet_count_dict_exact(
    G: nx.Graph,
    k: int,
    *,
    canonicalizer: NautyCanonicalizer | PythonCanonicalizer | None = None,
    batch_size: int = 4096,
    topology_filter: str = "all",
) -> dict[str, int]:
    """Exactly count connected induced topology graphlets of size ``k``."""

    canonicalizer = canonicalizer or default_topology_canonicalizer()
    counts: Counter[str] = Counter()
    subgraphs = _iter_connected_k_induced_subgraphs_exact(
        G,
        int(k),
        topology_filter=topology_filter,
    )
    for batch in _batched(subgraphs, batch_size=batch_size):
        counts.update(canonicalizer.canonical_graph6_batch(batch))
    return {str(key): int(value) for key, value in counts.items()}


# ============================================================
# Frequency / history utilities
# ============================================================


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
    topology_filter: str = "all",
    num_samples: int | None = None,
    rng: np.random.Generator | None = None,
    canonicalizer: NautyCanonicalizer | PythonCanonicalizer | None = None,
    batch_size: int = 4096,
) -> dict[str, float]:
    return normalize_count_dict(
        graphlet_count_dict(
            G,
            k,
            connected_only=connected_only,
            topology_filter=topology_filter,
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
    topology_filter: str = "all",
    num_samples: int | None = None,
    rng: np.random.Generator | None = None,
    canonicalizer: NautyCanonicalizer | PythonCanonicalizer | None = None,
    batch_size: int = 4096,
) -> dict[str, dict[str, float]]:
    """
    Return graphlet-frequency history {"3": {...}, ..., "K": {...}}.
    Each size-specific dictionary is normalized independently.
    """
    if k_min <= 0 or k_max < k_min:
        raise ValueError("Require 1 <= k_min <= k_max.")

    canonicalizer = canonicalizer or default_topology_canonicalizer()

    return {
        str(k): graphlet_frequency_dict(
            G,
            k,
            connected_only=connected_only,
            topology_filter=topology_filter,
            num_samples=num_samples,
            rng=rng,
            canonicalizer=canonicalizer,
            batch_size=batch_size,
        )
        for k in range(int(k_min), int(k_max) + 1)
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


# ============================================================
# Attributed graphlet data classes
# ============================================================


@dataclass(frozen=True)
class ColoredIncidenceTransform:
    colored_graph: nx.Graph
    original_to_colored_node: dict[Any, int]
    edge_to_aux_node: dict[frozenset[Any], int]


@dataclass
class AttributedMotifOccurrence:
    motif: nx.Graph
    count: int
    canonical_key: str


# ============================================================
# Basic helpers
# ============================================================


def _resolve_labelg(nauty_exec: str | os.PathLike[str]) -> str:
    """
    Resolve NAUTY_EXEC into the labelg executable.

    Accepts either:
      1. /path/to/labelg
      2. /path/to/nauty_directory
    """
    path = Path(nauty_exec)

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


def _stable_label_token(label: Any) -> str:
    """
    Convert a Python label value into a stable string token.

    Examples:
        6       -> "builtins.int:6"
        "C"     -> "builtins.str:'C'"
        ("C",1) -> "builtins.tuple:('C', 1)"
    """
    return f"{type(label).__module__}.{type(label).__qualname__}:{label!r}"


def canonicalize_attributed_graph_python(
    graph: nx.Graph,
    *,
    node_label_attr: str = "node_label",
    edge_label_attr: str = "edge_label",
    missing_ok: bool = False,
    max_nodes: int = 7,
) -> str:
    """Return an exact node/edge-label-preserving canonical key.

    The implementation enumerates node orders and is therefore intended for
    graphlets, not whole molecular graphs.  It provides a dependency-free
    counterpart to the nauty coloured-incidence implementation and keeps
    attributed graphlet training usable when ``labelg`` is unavailable.
    """

    _validate_simple_undirected(graph)
    nodes = list(graph.nodes())
    if len(nodes) > int(max_nodes):
        raise RuntimeError(
            "The Python attributed canonicalizer supports at most "
            f"{max_nodes} nodes; install nauty labelg for larger motifs."
        )

    node_tokens: dict[Any, str] = {}
    for node, data in graph.nodes(data=True):
        if node_label_attr not in data and not missing_ok:
            raise KeyError(f"Node {node!r} is missing {node_label_attr!r}")
        node_tokens[node] = _stable_label_token(
            data.get(node_label_attr, "__MISSING__")
        )

    edge_tokens: dict[frozenset[Any], str] = {}
    for u, v, data in graph.edges(data=True):
        if edge_label_attr not in data and not missing_ok:
            raise KeyError(f"Edge {(u, v)!r} is missing {edge_label_attr!r}")
        edge_tokens[frozenset((u, v))] = _stable_label_token(
            data.get(edge_label_attr, "__MISSING__")
        )

    best: str | None = None
    for order in itertools.permutations(nodes):
        encoded_nodes = [node_tokens[node] for node in order]
        encoded_edges: list[str | None] = []
        for left in range(len(order)):
            for right in range(left + 1, len(order)):
                encoded_edges.append(
                    edge_tokens.get(frozenset((order[left], order[right])))
                )
        candidate = json.dumps(
            [encoded_nodes, encoded_edges],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        if best is None or candidate < best:
            best = candidate
    if best is None:
        best = "[[],[]]"
    return f"ATTR_PY_V1|{best}"


def canonicalize_attributed_simple_cycle(
    graph: nx.Graph,
    *,
    node_label_attr: str = "node_label",
    edge_label_attr: str = "edge_label",
    missing_ok: bool = False,
) -> str:
    """Return an exact attributed key for a chordless cycle in ``O(k^2)``.

    A cycle :math:`C_k` has only ``2k`` topology automorphisms (the dihedral
    group), whereas the generic attributed fallback checks all ``k!`` node
    orders.  Encoding every rotation in both directions is therefore exact
    and much cheaper for ring-only graphlet guidance.
    """

    _validate_simple_undirected(graph)
    if not graphlet_topology_matches(graph, "simple_cycle"):
        raise ValueError("canonicalize_attributed_simple_cycle requires C_k.")

    def node_sort_key(node: Any) -> tuple[str, str]:
        return type(node).__name__, repr(node)

    node_tokens: dict[Any, str] = {}
    for node, data in graph.nodes(data=True):
        if node_label_attr not in data and not missing_ok:
            raise KeyError(f"Node {node!r} is missing {node_label_attr!r}")
        node_tokens[node] = _stable_label_token(
            data.get(node_label_attr, "__MISSING__")
        )

    edge_tokens: dict[frozenset[Any], str] = {}
    for u, v, data in graph.edges(data=True):
        if edge_label_attr not in data and not missing_ok:
            raise KeyError(f"Edge {(u, v)!r} is missing {edge_label_attr!r}")
        edge_tokens[frozenset((u, v))] = _stable_label_token(
            data.get(edge_label_attr, "__MISSING__")
        )

    start = min(graph.nodes(), key=node_sort_key)
    first_neighbor = min(graph.neighbors(start), key=node_sort_key)
    cycle_order: list[Any] = [start]
    previous = start
    current = first_neighbor
    while current != start:
        cycle_order.append(current)
        next_nodes = [node for node in graph.neighbors(current) if node != previous]
        if len(next_nodes) != 1:
            raise ValueError("Input graph is not a simple cycle.")
        previous, current = current, next_nodes[0]
        if len(cycle_order) > graph.number_of_nodes():
            raise ValueError("Input graph is not a simple cycle.")
    if len(cycle_order) != graph.number_of_nodes():
        raise ValueError("Input graph is not a simple cycle.")

    best: str | None = None
    forward = tuple(cycle_order)
    reverse = tuple(reversed(cycle_order))
    for orientation in (forward, reverse):
        for offset in range(len(orientation)):
            order = orientation[offset:] + orientation[:offset]
            encoded = [
                [
                    node_tokens[order[index]],
                    edge_tokens[
                        frozenset(
                            (order[index], order[(index + 1) % len(order)])
                        )
                    ],
                ]
                for index in range(len(order))
            ]
            candidate = json.dumps(
                encoded,
                ensure_ascii=True,
                separators=(",", ":"),
            )
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("A simple cycle must contain at least three nodes.")
    return f"ATTR_CYCLE_V1|{best}"


def attributed_graphlet_count_dict(
    graph: nx.Graph,
    k: int,
    *,
    node_label_attr: str = "node_label",
    edge_label_attr: str = "edge_label",
    connected_only: bool = True,
    topology_filter: str = "all",
    num_samples: int | None = None,
    rng: np.random.Generator | None = None,
    missing_ok: bool = False,
    backend: str = "auto",
    nauty_exec: str | os.PathLike[str] | None = NAUTY_EXEC,
) -> dict[str, int]:
    """Count selected induced attributed graphlets of one size.

    ``topology_filter`` selects all graphlets, every cycle-containing graphlet,
    or chordless cycles only. ``backend='auto'`` uses nauty when available and
    otherwise uses exact Python canonicalization. In ``simple_cycle`` mode the
    Python backend compares only the ``2k`` dihedral cycle symmetries. Node and
    edge labels are part of every key, so graphlets with the same topology but
    different chemistry remain distinct classes.
    """

    selected_filter = normalize_graphlet_topology_filter(topology_filter)
    backend = str(backend).lower()
    if backend not in {"auto", "python", "nauty"}:
        raise ValueError("backend must be 'auto', 'python', or 'nauty'.")
    use_nauty = backend == "nauty" or (backend == "auto" and bool(nauty_exec))
    if use_nauty and not nauty_exec:
        raise FileNotFoundError(
            "Attributed graphlet backend 'nauty' requires NAUTY_EXEC or labelg."
        )

    counts: Counter[str] = Counter()
    for subgraph in _iter_k_induced_subgraphs(
        graph,
        int(k),
        connected_only=bool(connected_only),
        topology_filter=selected_filter,
        num_samples=num_samples,
        rng=rng,
    ):
        if use_nauty:
            transformed = attributed_to_colored_incidence_graph(
                subgraph,
                node_label_attr=node_label_attr,
                edge_label_attr=edge_label_attr,
                missing_ok=missing_ok,
            )
            key = canonicalize_colored_graph_nauty(
                transformed.colored_graph,
                nauty_exec=nauty_exec,
            )
        elif selected_filter == "simple_cycle":
            key = canonicalize_attributed_simple_cycle(
                subgraph,
                node_label_attr=node_label_attr,
                edge_label_attr=edge_label_attr,
                missing_ok=missing_ok,
            )
        else:
            key = canonicalize_attributed_graph_python(
                subgraph,
                node_label_attr=node_label_attr,
                edge_label_attr=edge_label_attr,
                missing_ok=missing_ok,
            )
        counts[str(key)] += 1
    return dict(counts)


def _safe_colour_alphabet() -> list[str]:
    """
    Single-character colour alphabet for labelg -f.

    Avoid:
      - '-' because -f-xxx has special meaning.
      - '^' because x^N has repetition meaning in labelg -f.
      - whitespace.
    """
    chars = string.ascii_letters + string.digits + "!#$%&()*+,./:;<=>?@[]_{}~"
    return list(dict.fromkeys(chars))


def _make_colour_map(colour_tokens: list[str]) -> dict[str, str]:
    """
    Map semantic colour tokens to single ASCII characters for labelg -f.
    """
    unique_tokens = sorted(set(colour_tokens))
    alphabet = _safe_colour_alphabet()

    if len(unique_tokens) > len(alphabet):
        raise ValueError(
            f"Too many distinct colours: {len(unique_tokens)}. "
            f"This wrapper supports at most {len(alphabet)} colours. "
            "For many colours, use the nauty C API with lab/ptn instead."
        )

    return {token: alphabet[i] for i, token in enumerate(unique_tokens)}


# ============================================================
# Attributed graph -> coloured incidence graph
# ============================================================


def attributed_to_colored_incidence_graph(
    G: nx.Graph,
    *,
    node_label_attr: str = "node_label",
    edge_label_attr: str = "edge_label",
    label_normalizer: Callable[[Any], str] = _stable_label_token,
    missing_ok: bool = False,
) -> ColoredIncidenceTransform:
    """
    Convert an attributed graph into a vertex-coloured incidence graph.

    Original node:
        v with node label X[v]
        -> coloured vertex with colour token "node:<label>"

    Original labelled edge:
        (u, v) with edge label Y[u, v]
        -> auxiliary coloured vertex with colour token "edge:<label>"
        -> edges u -- aux -- v

    This converts node/edge-labelled graph isomorphism into
    vertex-coloured graph isomorphism.
    """
    _validate_simple_undirected(G)

    H = nx.Graph()
    original_to_colored_node: dict[Any, int] = {}
    edge_to_aux_node: dict[frozenset[Any], int] = {}

    # Original nodes become coloured vertices.
    for v, data in G.nodes(data=True):
        if node_label_attr not in data:
            if not missing_ok:
                raise KeyError(f"Node {v!r} is missing {node_label_attr!r}")
            raw_label = "__MISSING__"
        else:
            raw_label = data[node_label_attr]

        new_id = H.number_of_nodes()
        original_to_colored_node[v] = new_id

        node_attrs = dict(data)
        H.add_node(
            new_id,
            kind="node",
            color_token="node:" + label_normalizer(raw_label),
            label=raw_label,
            original_node=v,
            original_attrs=node_attrs,
        )

    # Original edges become auxiliary coloured vertices.
    for u, v, data in G.edges(data=True):
        if edge_label_attr not in data:
            if not missing_ok:
                raise KeyError(f"Edge {(u, v)!r} is missing {edge_label_attr!r}")
            raw_label = "__MISSING__"
        else:
            raw_label = data[edge_label_attr]

        aux_id = H.number_of_nodes()
        edge_key = frozenset((u, v))
        edge_to_aux_node[edge_key] = aux_id

        edge_attrs = dict(data)
        H.add_node(
            aux_id,
            kind="edge",
            color_token="edge:" + label_normalizer(raw_label),
            label=raw_label,
            original_edge=(u, v),
            original_attrs=edge_attrs,
        )

        H.add_edge(original_to_colored_node[u], aux_id)
        H.add_edge(aux_id, original_to_colored_node[v])

    return ColoredIncidenceTransform(
        colored_graph=H,
        original_to_colored_node=original_to_colored_node,
        edge_to_aux_node=edge_to_aux_node,
    )


# ============================================================
# Coloured graph canonicalization using nauty labelg -f
# ============================================================


def canonicalize_colored_graph_nauty(
    H: nx.Graph,
    *,
    color_attr: str = "color_token",
    nauty_exec: str | os.PathLike[str] = NAUTY_EXEC,
    use_traces: bool = False,
) -> str:
    """
    Canonicalize a vertex-coloured simple graph using nauty labelg -f.

    Returns a canonical string that includes:
      - semantic colour legend
      - canonical colour sequence
      - canonical graph6 topology

    Important:
        graph6 stores only topology, not colours.
        Therefore the returned key includes colour information as well.
    """
    _validate_simple_undirected(H)

    # Relabel to 0..n-1 for graph6 and for the colour string.
    mapping = {v: i for i, v in enumerate(H.nodes())}
    J = nx.relabel_nodes(H, mapping, copy=True)

    colour_tokens: list[str] = []
    for i in range(J.number_of_nodes()):
        if color_attr not in J.nodes[i]:
            raise KeyError(f"Node {i!r} is missing colour attribute {color_attr!r}")
        colour_tokens.append(str(J.nodes[i][color_attr]))

    colour_map = _make_colour_map(colour_tokens)
    colour_string = "".join(colour_map[token] for token in colour_tokens)

    graph6_input = nx.to_graph6_bytes(J, header=False)

    cmd = [
        _resolve_labelg(nauty_exec),
        "-q",
        "-g",
        f"-f{colour_string}",
    ]

    if use_traces:
        cmd.append("-t")

    result = subprocess.run(
        cmd,
        input=graph6_input,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "labelg failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr:\n{result.stderr.decode(errors='replace')}"
        )

    canonical_graph6 = result.stdout.decode("ascii").strip()

    # labelg orders colour classes by ASCII order.
    canonical_colour_string = "".join(sorted(colour_string))

    # Include semantic legend so local character mappings cannot collide.
    legend = sorted(
        [(char, token) for token, char in colour_map.items()],
        key=lambda x: x[0],
    )
    legend_json = json.dumps(legend, separators=(",", ":"))

    return f"ATTR_NAUTY_V1|{legend_json}|{canonical_colour_string}|{canonical_graph6}"


# ============================================================
# Coloured incidence subgraph -> attributed nx.Graph
# ============================================================


def colored_incidence_subgraph_to_attributed_graph(
    H_sub: nx.Graph,
    *,
    node_label_attr: str = "node_label",
    edge_label_attr: str = "edge_label",
    preserve_original_attrs: bool = True,
) -> nx.Graph:
    """
    Convert a valid coloured incidence subgraph back to an attributed nx.Graph.

    The input must contain:
      - original vertices with kind="node"
      - auxiliary edge vertices with kind="edge"

    Each edge-auxiliary vertex must be adjacent to exactly two original-node vertices.
    """
    G_out = nx.Graph()

    node_vertices = [
        v for v, data in H_sub.nodes(data=True) if data.get("kind") == "node"
    ]

    # Relabel output motif nodes to 0..k-1.
    node_map = {v: i for i, v in enumerate(node_vertices)}

    for v in node_vertices:
        data = H_sub.nodes[v]

        attrs = {}
        if preserve_original_attrs:
            attrs.update(dict(data.get("original_attrs", {})))

        attrs[node_label_attr] = data.get("label")
        G_out.add_node(node_map[v], **attrs)

    for aux, data in H_sub.nodes(data=True):
        if data.get("kind") != "edge":
            continue

        nbrs = [
            nbr
            for nbr in H_sub.neighbors(aux)
            if H_sub.nodes[nbr].get("kind") == "node"
        ]

        if len(nbrs) != 2:
            raise ValueError(
                f"Invalid incidence subgraph: edge-auxiliary vertex {aux!r} "
                f"has {len(nbrs)} node-neighbours, expected 2."
            )

        u, v = nbrs

        attrs = {}
        if preserve_original_attrs:
            attrs.update(dict(data.get("original_attrs", {})))

        attrs[edge_label_attr] = data.get("label")
        G_out.add_edge(node_map[u], node_map[v], **attrs)

    return G_out


# ============================================================
# Graph list: aggregate unique attributed k-induced motifs + counts
# ============================================================


def aggregate_unique_attributed_k_motifs_with_counts_nauty(
    graphs: list[nx.Graph],
    k: int,
    *,
    node_label_attr: str = "node_label",
    edge_label_attr: str = "edge_label",
    connected_only: bool = True,
    nauty_exec: str | os.PathLike[str] = NAUTY_EXEC,
    use_traces: bool = False,
    label_normalizer: Callable[[Any], str] = _stable_label_token,
    missing_ok: bool = False,
    preserve_original_attrs: bool = True,
    progress_interval: int = 0,
    log_fn=None,
) -> list[AttributedMotifOccurrence]:
    """
    Aggregate unique attributed k-induced motifs across a list of attributed graphs.

    Here k means k original graph nodes, not k vertices in the coloured incidence graph.

    For each input graph:
      1. Transform the attributed graph into a coloured incidence graph.
      2. Enumerate all k-node induced subgraphs in the original graph.
      3. For each induced subgraph, include:
           - selected original-node vertices
           - auxiliary edge vertices for edges among selected nodes
      4. Canonicalize the coloured incidence subgraph using nauty/Traces.
      5. Count occurrences by canonical key.
      6. Keep one attributed nx.Graph representative for each unique key.

    Returns
    -------
    list[AttributedMotifOccurrence]
        Each item contains:
          - motif: representative attributed nx.Graph
          - count: total number of occurrences across all graphs
          - canonical_key: nauty-based attributed canonical key
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    counts: Counter[str] = Counter()
    representatives: dict[str, nx.Graph] = {}

    total_graphs = len(graphs)
    for graph_idx, G in enumerate(graphs, start=1):
        _validate_simple_undirected(G)

        if k > G.number_of_nodes():
            continue

        transform = attributed_to_colored_incidence_graph(
            G,
            node_label_attr=node_label_attr,
            edge_label_attr=edge_label_attr,
            label_normalizer=label_normalizer,
            missing_ok=missing_ok,
        )

        H_full = transform.colored_graph
        original_nodes = list(G.nodes())

        for subset in itertools.combinations(original_nodes, k):
            original_subgraph = G.subgraph(subset)

            if connected_only and not nx.is_connected(original_subgraph):
                continue

            # Include selected original-node vertices in the coloured graph.
            coloured_nodes = [transform.original_to_colored_node[v] for v in subset]

            # Include edge-label auxiliary vertices for edges induced by the subset.
            for u, v in original_subgraph.edges():
                aux = transform.edge_to_aux_node[frozenset((u, v))]
                coloured_nodes.append(aux)

            H_sub = H_full.subgraph(coloured_nodes).copy()

            key = canonicalize_colored_graph_nauty(
                H_sub,
                color_attr="color_token",
                nauty_exec=nauty_exec,
                use_traces=use_traces,
            )

            counts[key] += 1

            if key not in representatives:
                representatives[key] = colored_incidence_subgraph_to_attributed_graph(
                    H_sub,
                    node_label_attr=node_label_attr,
                    edge_label_attr=edge_label_attr,
                    preserve_original_attrs=preserve_original_attrs,
                )
        if progress_interval and (
            graph_idx == 1
            or graph_idx % int(progress_interval) == 0
            or graph_idx == total_graphs
        ):
            message = (
                f"attributed k={k} graph={graph_idx}/{total_graphs} "
                f"unique_motifs={len(counts)} total_occurrences={sum(counts.values())}"
            )
            if log_fn is not None:
                log_fn(message)
            else:
                print(message, flush=True)

    return [
        AttributedMotifOccurrence(
            motif=representatives[key],
            count=int(counts[key]),
            canonical_key=key,
        )
        for key in sorted(counts.keys())
    ]
