from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Optional, Sequence

import networkx as nx
import numpy as np
import torch


@dataclass
class SimpleData:
    """Small PyG-like data container used when torch_geometric is unavailable."""

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_pairs: torch.Tensor | None = None
    num_nodes: int | None = None
    num_edges: int | None = None
    pe: torch.Tensor | None = None

    def to(self, device: torch.device | str):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        if self.edge_pairs is not None:
            self.edge_pairs = self.edge_pairs.to(device)
        if self.pe is not None:
            self.pe = self.pe.to(device)
        return self


try:  # pragma: no cover - optional dependency.
    from torch_geometric.data import Data as PyGData  # type: ignore
except Exception:  # pragma: no cover
    PyGData = None


@dataclass(frozen=True)
class RewireAction:
    """A complete Graph-ER double-edge-swap action a=(e1,e2,r).

    ``e1`` and ``e2`` are the two removed edges. ``orientation`` is the
    reconnection pattern: 0 creates (u,x),(v,y), and 1 creates (u,y),(v,x), for
    e1=(u,v), e2=(x,y).  Endpoints are sorted only within each edge so that
    equality/hashing are stable for simple undirected graphs.  The order of the
    two removed edges is kept because the orientation convention is defined with
    respect to that order.
    """

    e1: tuple[int, int]
    e2: tuple[int, int]
    orientation: int

    def __post_init__(self) -> None:
        e1 = tuple(sorted((int(self.e1[0]), int(self.e1[1]))))
        e2 = tuple(sorted((int(self.e2[0]), int(self.e2[1]))))
        object.__setattr__(self, "e1", e1)
        object.__setattr__(self, "e2", e2)
        object.__setattr__(self, "orientation", int(self.orientation))
        if self.orientation not in (0, 1):
            raise ValueError("RewireAction.orientation must be 0 or 1.")

    def as_tuple(self) -> tuple[tuple[int, int], tuple[int, int], int]:
        return self.e1, self.e2, self.orientation


def action_new_edges(action: RewireAction) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the two edges created by a complete rewiring action."""

    (u, v), (x, y) = action.e1, action.e2
    if int(action.orientation) == 0:
        return tuple(sorted((u, x))), tuple(sorted((v, y)))
    return tuple(sorted((u, y))), tuple(sorted((v, x)))


def action_removed_edges(action: RewireAction) -> tuple[tuple[int, int], tuple[int, int]]:
    return tuple(sorted(action.e1)), tuple(sorted(action.e2))


def action_signature(action: RewireAction) -> tuple[tuple[int, int], tuple[int, int], int]:
    return action.e1, action.e2, int(action.orientation)


def degree_sequence(graph: nx.Graph) -> list[int]:
    return sorted((int(d) for _, d in graph.degree()), reverse=True)


def check_sequence_validity(seq: Sequence[int]) -> tuple[bool, int]:
    """Erdos-Gallai graphicality check.

    Returns (is_valid, code).  code=0 means valid; non-zero codes are used by
    scripts for diagnostics.
    """

    if len(seq) == 0:
        return False, 1
    if any(int(d) < 0 for d in seq):
        return False, 2
    if sum(int(d) for d in seq) % 2 != 0:
        return False, 3
    sorted_seq = sorted((int(d) for d in seq), reverse=True)
    n = len(sorted_seq)
    if sorted_seq and sorted_seq[0] >= n:
        return False, 4
    prefix = np.cumsum([0] + sorted_seq)
    for k in range(1, n + 1):
        lhs = int(prefix[k])
        rhs = k * (k - 1) + sum(min(d, k) for d in sorted_seq[k:])
        if lhs > rhs:
            return False, 5
    return True, 0


def connected_sequence_feasible(seq: Sequence[int]) -> tuple[bool, str]:
    seq = [int(d) for d in seq]
    ok, code = check_sequence_validity(seq)
    if not ok:
        return False, f"not_graphical:{code}"
    n = len(seq)
    if n <= 1:
        return True, "ok"
    if any(d <= 0 for d in seq):
        return False, "connected_graph_with_more_than_one_node_cannot_have_zero_degree"
    if sum(seq) < 2 * (n - 1):
        return False, "too_few_edges_for_connected_graph"
    return True, "ok"


def nx_to_undirected_edge_index(graph: nx.Graph) -> torch.Tensor:
    edges = [(int(u), int(v)) for u, v in graph.edges()]
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def _stack_node_attr(graph: nx.Graph, key: str, node_order: list[int]) -> torch.Tensor | None:
    rows = []
    for node in node_order:
        if key not in graph.nodes[node]:
            return None
        value = graph.nodes[node][key]
        if torch.is_tensor(value):
            rows.append(value.detach().cpu().reshape(-1).float())
        else:
            rows.append(torch.tensor(np.asarray(value, dtype=np.float32).reshape(-1), dtype=torch.float32))
    if not rows:
        return None
    width = max(int(row.numel()) for row in rows)
    padded = []
    for row in rows:
        if row.numel() < width:
            row = torch.cat([row, row.new_zeros(width - row.numel())])
        padded.append(row)
    return torch.stack(padded, dim=0)


def laplacian_positional_encoding(graph: nx.Graph, node_order: list[int], k: int) -> torch.Tensor:
    n = len(node_order)
    if k <= 0 or n == 0:
        return torch.empty((n, 0), dtype=torch.float32)
    adjacency = nx.to_numpy_array(graph, nodelist=node_order, dtype=np.float64)
    deg = adjacency.sum(axis=1)
    inv_sqrt = np.zeros_like(deg)
    inv_sqrt[deg > 0] = 1.0 / np.sqrt(deg[deg > 0])
    lap = np.eye(n) - np.diag(inv_sqrt) @ adjacency @ np.diag(inv_sqrt)
    try:
        _, eigvecs = np.linalg.eigh(lap)
    except np.linalg.LinAlgError:
        eigvecs = np.eye(n)
    cols = min(k, eigvecs.shape[1])
    pe = torch.tensor(eigvecs[:, :cols], dtype=torch.float32)
    if cols < k:
        pe = torch.cat([pe, pe.new_zeros((n, k - cols))], dim=1)
    return pe


def graph_to_data(graph: nx.Graph, k_eigen: int = 4, *, include_edge_pairs: bool = True):
    """Convert a NetworkX graph to the data object consumed by GraphER.

    The model's original input is kept: degree feature plus optional Laplacian
    positional encodings.  When PyG is installed, a torch_geometric Data object
    is returned; otherwise a lightweight container is used.
    """

    g = nx.convert_node_labels_to_integers(nx.Graph(graph), ordering="sorted")
    node_order = list(g.nodes())
    n = len(node_order)
    x_rows = _stack_node_attr(g, "x", node_order)
    if x_rows is None:
        degrees = torch.tensor([float(g.degree(u)) for u in node_order], dtype=torch.float32).view(-1, 1)
        x = degrees
    else:
        x = x_rows.float()
    pe_rows = _stack_node_attr(g, "pe", node_order)
    if pe_rows is None:
        pe = laplacian_positional_encoding(g, node_order, int(k_eigen))
    else:
        pe = pe_rows[:, : int(k_eigen)].float()
        if pe.size(1) < int(k_eigen):
            pe = torch.cat([pe, pe.new_zeros((n, int(k_eigen) - pe.size(1)))], dim=1)
    if pe.numel() > 0:
        x = torch.cat([x, pe], dim=-1)

    if include_edge_pairs:
        edges = list(g.edges())
        valid_pairs: list[list[int]] = []
        for i, (u, v) in enumerate(edges):
            for j in range(i + 1, len(edges)):
                a, b = edges[j]
                if len({u, v, a, b}) == 4:
                    valid_pairs.append([i, j])
        edge_pairs = torch.tensor(valid_pairs, dtype=torch.long) if valid_pairs else torch.empty((0, 2), dtype=torch.long)
    else:
        edge_pairs = torch.empty((0, 2), dtype=torch.long)
    edge_index = nx_to_undirected_edge_index(g)
    cls = PyGData or SimpleData
    data = cls(x=x, edge_index=edge_index, edge_pairs=edge_pairs, num_nodes=g.number_of_nodes(), num_edges=g.number_of_edges())
    data.pe = pe
    return data


def _simple_valid(a: int, b: int, c: int, d: int) -> bool:
    return len({a, b, c, d}) == 4 and a != b and c != d


def rewire(
    graph: nx.Graph,
    e1: tuple[int, int],
    e2: tuple[int, int],
    orientation: int,
    ensure_connected: bool = True,
):
    """Apply a valid degree-preserving double-edge swap, or return None."""

    (u, v), (x, y) = tuple(e1), tuple(e2)
    if len({u, v, x, y}) != 4:
        return None
    if orientation == 0:
        a, b, c, d = u, x, v, y
    else:
        a, b, c, d = u, y, v, x
    if not _simple_valid(a, b, c, d):
        return None
    if not graph.has_edge(u, v) or not graph.has_edge(x, y):
        return None
    # The new edges cannot equal the removed edges when all four endpoints are
    # distinct, so duplicate-edge checks can happen before copying the graph.
    # This avoids many expensive graph copies during candidate enumeration.
    if graph.has_edge(a, b) or graph.has_edge(c, d):
        return None
    g = graph.copy()
    g.remove_edge(u, v)
    g.remove_edge(x, y)
    g.add_edge(a, b)
    g.add_edge(c, d)
    if ensure_connected and g.number_of_nodes() > 0 and not nx.is_connected(g):
        return None
    return g, ((a, b), (c, d)), ((u, v), (x, y))


def _within_k_hop(graph: nx.Graph, e1: tuple[int, int], e2: tuple[int, int], k_hop: int | None) -> bool:
    if k_hop is None:
        return True
    u, v = e1
    x, y = e2
    dist_u = nx.single_source_shortest_path_length(graph, u, cutoff=int(k_hop))
    dist_v = nx.single_source_shortest_path_length(graph, v, cutoff=int(k_hop))
    return x in dist_u or y in dist_u or x in dist_v or y in dist_v

def _k_hop_reachability_cache(graph: nx.Graph, k_hop: int | None) -> dict[int, set[int]] | None:
    if k_hop is None:
        return None
    cutoff = int(k_hop)
    return {int(node): {int(dst) for dst in nx.single_source_shortest_path_length(graph, node, cutoff=cutoff)} for node in graph.nodes()}


def _within_k_hop_cached(
    reachability: dict[int, set[int]] | None,
    e1: tuple[int, int],
    e2: tuple[int, int],
) -> bool:
    if reachability is None:
        return True
    u, v = e1
    x, y = e2
    return x in reachability.get(u, set()) or y in reachability.get(u, set()) or x in reachability.get(v, set()) or y in reachability.get(v, set())


def rewire_action(
    graph: nx.Graph,
    action: RewireAction,
    *,
    ensure_connected: bool = True,
):
    """Apply a complete double-edge-swap action, or return None if invalid."""

    return rewire(graph, action.e1, action.e2, action.orientation, ensure_connected=ensure_connected)


def enumerate_rewire_actions(
    graph: nx.Graph,
    *,
    ensure_connected: bool = True,
    k_hop: int | None = 2,
    max_candidates: int | None = None,
    anchor_edges: Sequence[tuple[int, int]] | None = None,
    rng: random.Random | None = None,
    shuffle: bool = False,
) -> list[RewireAction]:
    """Enumerate or sample valid complete Graph-ER actions.

    This is the target-free candidate constructor used for neural training and
    generation.  It returns actions a=(e1,e2,r), not only partner edges, so the
    model can score the reconnection orientation r as required by the paper's
    categorical action field.
    """

    g_edges = [tuple(sorted((int(u), int(v)))) for u, v in graph.edges()]
    g_edges = sorted(set(g_edges))
    if len(g_edges) < 2:
        return []

    reachability = _k_hop_reachability_cache(graph, k_hop)
    edge_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if anchor_edges is not None:
        seen_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        anchors = [tuple(sorted((int(u), int(v)))) for u, v in anchor_edges]
        for e1 in anchors:
            if e1 not in g_edges:
                continue
            for e2 in g_edges:
                if e1 == e2:
                    continue
                if len(set(e1 + e2)) != 4:
                    continue
                if not _within_k_hop_cached(reachability, e1, e2):
                    continue
                key = (e1, e2) if e1 <= e2 else (e2, e1)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                # Keep the requested anchor as e1 for deterministic teacher
                # construction; orientation is defined relative to this order.
                edge_pairs.append((e1, e2))
    else:
        for i, e1 in enumerate(g_edges):
            for e2 in g_edges[i + 1 :]:
                if len(set(e1 + e2)) != 4:
                    continue
                if not _within_k_hop_cached(reachability, e1, e2):
                    continue
                edge_pairs.append((e1, e2))

    if shuffle:
        (rng or random).shuffle(edge_pairs)

    actions: list[RewireAction] = []
    seen_actions: set[tuple[tuple[int, int], tuple[int, int], int]] = set()
    for e1, e2 in edge_pairs:
        for orient in (0, 1):
            action = RewireAction(e1=e1, e2=e2, orientation=orient)
            sig = action_signature(action)
            if sig in seen_actions:
                continue
            if rewire_action(graph, action, ensure_connected=ensure_connected) is None:
                continue
            seen_actions.add(sig)
            actions.append(action)
            if max_candidates is not None and len(actions) >= int(max_candidates):
                return actions
    return actions


def merge_action_sets(*action_sets: Sequence[RewireAction], max_candidates: int | None = None) -> list[RewireAction]:
    """Merge action lists while preserving order and removing duplicates."""

    merged: list[RewireAction] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()
    for actions in action_sets:
        for action in actions:
            sig = action_signature(action)
            if sig in seen:
                continue
            seen.add(sig)
            merged.append(action)
            if max_candidates is not None and len(merged) >= int(max_candidates):
                return merged
    return merged


def build_candidates(
    graph: nx.Graph,
    anchor_edge: tuple[int, int],
    *,
    ensure_connected: bool = True,
    k_hop: int | None = 2,
    max_candidates: int | None = None,
) -> list[tuple[int, int]]:
    """Target-free feasible partner-edge candidates for one anchor edge."""

    candidates: list[tuple[int, int]] = []
    edges = [tuple(e) for e in graph.edges()]
    reachability = _k_hop_reachability_cache(graph, k_hop)
    for e2 in edges:
        if e2 == anchor_edge or set(e2) == set(anchor_edge):
            continue
        if len(set(anchor_edge + e2)) != 4:
            continue
        if not _within_k_hop_cached(reachability, anchor_edge, e2):
            continue
        if any(rewire(graph, anchor_edge, e2, orient, ensure_connected=ensure_connected) is not None for orient in (0, 1)):
            candidates.append(e2)
            if max_candidates is not None and len(candidates) >= int(max_candidates):
                break
    return candidates


def havel_hakimi_construction(graph: nx.Graph) -> nx.Graph:
    pairs = sorted(((int(d), int(u)) for u, d in graph.degree()), key=lambda x: (-x[0], x[1]))
    seq = [d for d, _ in pairs]
    h_int = nx.havel_hakimi_graph(seq)
    mapping = {idx: pairs[idx][1] for idx in range(len(seq))}
    return nx.relabel_nodes(h_int, mapping, copy=True)


def _merge_two_components_deterministically(graph: nx.Graph, comp1_nodes: Iterable[int], comp2_nodes: Iterable[int]) -> None:
    comp1_nodes = sorted(comp1_nodes)
    comp2_nodes = sorted(comp2_nodes)
    g1 = graph.subgraph(comp1_nodes).copy()
    g2 = graph.subgraph(comp2_nodes).copy()
    edges1 = sorted(tuple(sorted(e)) for e in g1.edges())
    edges2 = sorted(tuple(sorted(e)) for e in g2.edges())
    if not edges1 or not edges2:
        raise RuntimeError("Cannot connect components with no internal edges; degree sequence may not admit a connected realization.")
    bridges1 = set(tuple(sorted(e)) for e in nx.bridges(g1))
    bridges2 = set(tuple(sorted(e)) for e in nx.bridges(g2))
    edges1 = [e for e in edges1 if e not in bridges1] + [e for e in edges1 if e in bridges1]
    edges2 = [e for e in edges2 if e not in bridges2] + [e for e in edges2 if e in bridges2]
    base_components = nx.number_connected_components(graph)
    for a, b in edges1:
        for c, d in edges2:
            for orient in (0, 1):
                out = rewire(graph, (a, b), (c, d), orient, ensure_connected=False)
                if out is None:
                    continue
                candidate = out[0]
                if nx.number_connected_components(candidate) < base_components:
                    graph.clear()
                    graph.add_nodes_from(candidate.nodes(data=True))
                    graph.add_edges_from(candidate.edges(data=True))
                    graph.graph.update(candidate.graph)
                    return
    raise RuntimeError("Could not find a deterministic component-merging double-edge swap.")


def deterministic_connected_havel_hakimi(G: nx.Graph | None = None, seq: Sequence[int] | None = None) -> nx.Graph:
    """Deterministic connected Havel-Hakimi-style source graph.

    This is the canonical source builder used by the scripts.  It preserves the
    model implementation while centralizing the graph-theoretic utility outside
    ad hoc script-level helpers.
    """

    if G is None and seq is None:
        raise ValueError("Provide either G or seq.")
    if seq is None and G is not None:
        seq = degree_sequence(G)
    assert seq is not None
    feasible, reason = connected_sequence_feasible(seq)
    if not feasible:
        raise ValueError(f"Degree sequence cannot produce a connected simple graph: {reason}; seq={list(seq)}")
    if G is not None:
        h = havel_hakimi_construction(nx.convert_node_labels_to_integers(nx.Graph(G), ordering="sorted"))
    else:
        h = nx.havel_hakimi_graph([int(d) for d in seq])
    if h.number_of_nodes() <= 1 or nx.is_connected(h):
        return nx.convert_node_labels_to_integers(h, ordering="sorted")
    while not nx.is_connected(h):
        components = [sorted(c) for c in nx.connected_components(h)]
        components.sort(key=lambda comp: (len(comp), comp[0]))
        _merge_two_components_deterministically(h, components[0], components[-1])
    return nx.convert_node_labels_to_integers(h, ordering="sorted")


def configuration_model_from_multiset(degrees: Sequence[int]) -> nx.Graph:
    g = nx.configuration_model([int(d) for d in degrees])
    g = nx.Graph(g)
    g.remove_edges_from(nx.selfloop_edges(g))
    return nx.convert_node_labels_to_integers(g, ordering="sorted")


def edge_symmetric_difference_size(a: nx.Graph, b: nx.Graph) -> int:
    ea = {tuple(sorted(e)) for e in a.edges()}
    eb = {tuple(sorted(e)) for e in b.edges()}
    return len(ea ^ eb)
