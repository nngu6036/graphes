from __future__ import annotations

import itertools
import os
import subprocess
from collections import Counter
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any, Iterable, Iterator

import itertools
import json
import os
import string
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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


def attributed_to_colored_incidence_graph(
    G: nx.Graph,
    *,
    node_label_attr: str = "node_label",
    edge_label_attr: str = "edge_label",
) -> nx.Graph:
    """
    Convert a node/edge-attributed graph into a vertex-coloured incidence graph.

    Original nodes become coloured vertices.
    Original labelled edges become auxiliary coloured vertices.

    The output graph is simple, undirected, and suitable for nauty-style
    vertex-colour canonicalization.
    """
    H = nx.Graph()

    # Add original nodes.
    for v, data in G.nodes(data=True):
        node_label = data.get(node_label_attr, "__missing_node_label__")
        H.add_node(
            ("node", v),
            color=("node", node_label),
            original_node=v,
        )

    # Add one auxiliary vertex per edge.
    for edge_id, (u, v, data) in enumerate(G.edges(data=True)):
        edge_label = data.get(edge_label_attr, "__missing_edge_label__")

        aux = ("edge", edge_id, u, v)
        H.add_node(
            aux,
            color=("edge", edge_label),
            original_edge=(u, v),
        )

        H.add_edge(("node", u), aux)
        H.add_edge(aux, ("node", v))

    return H

# ============================================================
# Data classes
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


def _validate_simple_undirected(G: nx.Graph) -> None:
    """
    Validate that G is a simple undirected NetworkX graph.
    """
    if G.is_directed():
        raise ValueError("Expected an undirected nx.Graph.")

    if G.is_multigraph():
        raise ValueError("Expected a simple nx.Graph, not a MultiGraph.")

    if nx.number_of_selfloops(G) > 0:
        raise ValueError("Self-loops are not supported in this graph6 workflow.")


def _stable_label_token(label: Any) -> str:
    """
    Convert a Python label value into a stable string token.

    Examples:
        6       -> "builtins.int:6"
        "C"     -> "builtins.str:'C'"
        ("C",1) -> "builtins.tuple:('C', 1)"
    """
    return f"{type(label).__module__}.{type(label).__qualname__}:{repr(label)}"


def _safe_colour_alphabet() -> list[str]:
    """
    Single-character colour alphabet for labelg -f.

    Avoid:
      - '-' because -f-xxx has special meaning.
      - '^' because x^N has repetition meaning in labelg -f.
      - whitespace.
    """
    chars = (
        string.ascii_letters
        + string.digits
        + "!#$%&()*+,./:;<=>?@[]_{}~"
    )
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
        v for v, data in H_sub.nodes(data=True)
        if data.get("kind") == "node"
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
            nbr for nbr in H_sub.neighbors(aux)
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
# Single graph: list unique attributed k-induced motifs
# ============================================================

def list_unique_attributed_k_motifs_nauty(
    G: nx.Graph,
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
) -> list[nx.Graph]:
    """
    Return unique attributed k-induced subgraphs of one attributed graph.

    Here k means k original graph nodes, not k vertices in the transformed
    coloured incidence graph.

    Steps:
      1. Transform G into a coloured incidence graph.
      2. For each k-subset of original nodes:
           - include those k original-node vertices
           - include auxiliary edge vertices for edges among them
      3. Canonicalize the coloured incidence subgraph with nauty labelg -f.
      4. Keep one representative per canonical string.
      5. Convert representatives back to attributed nx.Graph objects.

    Returns
    -------
    list[nx.Graph]
        Unique attributed motif representatives.
        Nodes are relabelled to 0..k-1.
    """
    _validate_simple_undirected(G)

    if k <= 0:
        raise ValueError("k must be positive.")

    if k > G.number_of_nodes():
        return []

    transform = attributed_to_colored_incidence_graph(
        G,
        node_label_attr=node_label_attr,
        edge_label_attr=edge_label_attr,
        label_normalizer=label_normalizer,
        missing_ok=missing_ok,
    )

    H_full = transform.colored_graph
    seen: dict[str, nx.Graph] = {}

    original_nodes = list(G.nodes())

    for subset in itertools.combinations(original_nodes, k):
        original_subgraph = G.subgraph(subset)

        if connected_only and not nx.is_connected(original_subgraph):
            continue

        # Include selected original-node vertices in the coloured graph.
        coloured_nodes = [
            transform.original_to_colored_node[v]
            for v in subset
        ]

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

        if key not in seen:
            seen[key] = colored_incidence_subgraph_to_attributed_graph(
                H_sub,
                node_label_attr=node_label_attr,
                edge_label_attr=edge_label_attr,
                preserve_original_attrs=preserve_original_attrs,
            )

    return list(seen.values())


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

    for graph_idx, G in enumerate(graphs):
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
            coloured_nodes = [
                transform.original_to_colored_node[v]
                for v in subset
            ]

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

    return [
        AttributedMotifOccurrence(
            motif=representatives[key],
            count=int(counts[key]),
            canonical_key=key,
        )
        for key in sorted(counts.keys())
    ]


def aggregate_unique_attributed_k_motifs_with_counts_as_tuples(
    graphs: list[nx.Graph],
    k: int,
    **kwargs,
) -> list[tuple[nx.Graph, int]]:
    """
    Convenience wrapper.

    Returns exactly:
        list[(motif_graph, occurrence_count)]
    """
    results = aggregate_unique_attributed_k_motifs_with_counts_nauty(
        graphs,
        k,
        **kwargs,
    )
    return [(item.motif, item.count) for item in results]
