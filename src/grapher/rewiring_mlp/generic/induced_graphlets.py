"""Exact untyped induced-graphlet distributions for a fixed k in {3, 4, 5}.

Bins are isomorphism classes, not node labels. ``all`` separates every connected
and disconnected topology; ``connected`` retains every connected class plus one
aggregate disconnected bin. Both divide by choose(n,k), never by connected mass.
Catalogue ordering is versioned and fingerprinted; it is not ORCA/atlas ordering.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations, islice
from math import comb
import hashlib
import json
from typing import Any, Mapping

import networkx as nx
import numpy as np

VERSION = "induced_topology_minmask_v1"
SUPPORTED_K = (3, 4, 5)


def validate_k(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) not in SUPPORTED_K:
        raise ValueError("induced_graphlet_k must be an integer in {3,4,5}; larger k is not implemented.")
    return int(value)


def validate_scope(value: str) -> str:
    if value not in ("all", "connected"):
        raise ValueError("induced_graphlet_scope must be 'all' or 'connected' (connected + disconnected bin).")
    return value


@dataclass(frozen=True)
class InducedGraphletSpec:
    k: int = 5
    scope: str = "all"

    def __post_init__(self):
        object.__setattr__(self, "k", validate_k(self.k))
        object.__setattr__(self, "scope", validate_scope(self.scope))

    @property
    def width(self) -> int:
        return len(catalogue(self.k, self.scope)["bins"])

    def metadata(self) -> dict:
        cat = catalogue(self.k, self.scope)
        return {"k": self.k, "scope": self.scope, "width": self.width,
                "version": VERSION, "fingerprint": cat["fingerprint"],
                "normalization": f"choose(n,{self.k})", "attributed": False,
                "bin_ids": [x["id"] for x in cat["bins"]]}

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None):
        values = config or {}
        if not values.get("induced_graphlet_histogram", False):
            return None
        return cls(values.get("induced_graphlet_k", 5), values.get("induced_graphlet_scope", "all"))


def _graph(k, code):
    g = nx.empty_graph(k)
    g.add_edges_from(pair for bit, pair in enumerate(combinations(range(k), 2)) if code & (1 << bit))
    return g


@lru_cache(maxsize=3)
def _canonical_codes(k: int):
    """Lookup every labeled adjacency bitmask -> minimum permuted bitmask."""
    k = validate_k(k)
    pairs = tuple(combinations(range(k), 2))
    pair_index = {p: i for i, p in enumerate(pairs)}
    codes = np.arange(1 << len(pairs), dtype=np.int64)
    result = codes.copy()
    for perm in permutations(range(k)):
        transformed = np.zeros_like(codes)
        for old, (u, v) in enumerate(pairs):
            new = pair_index[tuple(sorted((perm[u], perm[v])))]
            transformed |= ((codes >> old) & 1) << new
        np.minimum(result, transformed, out=result)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=6)
def catalogue(k: int = 5, scope: str = "all") -> dict:
    k, scope = validate_k(k), validate_scope(scope)
    codes = sorted(set(_canonical_codes(k).tolist()))
    bins = []
    for code in codes:
        g = _graph(k, code)
        connected = nx.is_connected(g)
        if scope == "connected" and not connected:
            continue
        bins.append({"id": f"k{k}_mask{code}", "canonical_mask": code,
                     "connected": connected, "num_edges": g.number_of_edges(),
                     "degrees": sorted(dict(g.degree()).values()),
                     "edges": [list(e) for e in g.edges()]})
    if scope == "connected":
        bins.append({"id": "disconnected", "canonical_mask": None, "connected": False,
                     "num_edges": None, "degrees": None, "edges": None})
    payload = {"version": VERSION, "k": k, "scope": scope, "attributed": False, "bins": bins}
    payload["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


@lru_cache(maxsize=6)
def _bin_lookup(k, scope):
    cat = catalogue(k, scope)
    indices = {b["canonical_mask"]: i for i, b in enumerate(cat["bins"]) if b["canonical_mask"] is not None}
    lookup = np.array([indices.get(int(c), len(cat["bins"]) - 1) for c in _canonical_codes(k)], dtype=np.int64)
    lookup.setflags(write=False)
    return lookup


def validate_histogram(values, spec: InducedGraphletSpec | None = None):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or (spec is not None and x.shape != (spec.width,)):
        raise ValueError("Induced graphlet histogram width/catalogue mismatch.")
    if not np.isfinite(x).all() or np.any(x < -1e-7) or np.any(x > 1 + 1e-7) or not np.isclose(x.sum(), 1, atol=1e-6, rtol=0):
        raise ValueError("Induced graphlet histogram must be a finite normalized probability vector.")
    x = np.clip(x, 0, 1)
    return x / x.sum()


def histogram_distance(left, right, spec=None) -> float:
    a, b = validate_histogram(left, spec), validate_histogram(right, spec)
    if a.shape != b.shape:
        raise ValueError("Induced histogram shapes differ.")
    return float(0.5 * np.abs(a-b).sum())


def histogram_from_counts(counts, n, spec):
    counts = np.asarray(counts)
    total = comb(int(n), spec.k) if n >= spec.k else 0
    if counts.shape != (spec.width,) or np.any(counts < 0) or not np.isfinite(counts).all() or not np.all(counts == np.floor(counts)) or int(counts.sum()) != total:
        raise ValueError("Counts must partition all k-node subsets exactly.")
    if total:
        return counts.astype(np.float64) / total
    # No observations: deterministic placeholder, always masked in loss/scoring.
    out = np.zeros(spec.width, dtype=np.float64)
    out[0 if spec.scope == "all" else -1] = 1.0
    return out


def _validate_graph(g):
    if g.is_directed() or g.is_multigraph() or nx.number_of_selfloops(g):
        raise ValueError("Induced graphlets require a simple undirected graph without self-loops.")


def _subset_chunks(n, k, chunk_size=65536):
    it = combinations(range(n), k)
    while True:
        block = list(islice(it, chunk_size))
        if not block:
            return
        yield np.asarray(block, dtype=np.intp)


def _classify(adjacency, subsets, spec):
    if len(subsets) == 0:
        return np.zeros(0, dtype=np.int64)
    codes = np.zeros(len(subsets), dtype=np.int64)
    for bit, (i, j) in enumerate(combinations(range(spec.k), 2)):
        codes |= adjacency[subsets[:, i], subsets[:, j]].astype(np.int64) << bit
    return _bin_lookup(spec.k, spec.scope)[codes]


def extract_counts(graph, spec: InducedGraphletSpec):
    _validate_graph(graph)
    adjacency = nx.to_numpy_array(graph, nodelist=list(graph), weight=None, dtype=bool)
    out = np.zeros(spec.width, dtype=np.int64)
    for subsets in _subset_chunks(len(graph), spec.k):
        out += np.bincount(_classify(adjacency, subsets, spec), minlength=spec.width)
    return out


def extract_histogram(graph, spec: InducedGraphletSpec):
    return histogram_from_counts(extract_counts(graph, spec), len(graph), spec)


class InducedGraphletCounter:
    """Exact current-state counts and local candidate deltas for changed pairs.

    Only subsets containing the endpoints of at least one changed adjacency
    entry can change their induced class. No per-subset isomorphism calls are
    made. Atom and bond labels deliberately do not affect this topology summary.
    """
    def __init__(self, graph, spec: InducedGraphletSpec):
        _validate_graph(graph)
        self.spec = spec
        self.nodes = list(graph)
        self.adjacency = nx.to_numpy_array(graph, nodelist=self.nodes, weight=None, dtype=bool)
        self.counts = np.zeros(spec.width, dtype=np.int64)
        for subsets in _subset_chunks(len(graph), spec.k):
            self.counts += np.bincount(_classify(self.adjacency, subsets, spec), minlength=spec.width)

    def histogram(self):
        return histogram_from_counts(self.counts, len(self.nodes), self.spec)

    def candidate_counts(self, graph):
        _validate_graph(graph)
        if set(graph) != set(self.nodes):
            raise ValueError("A graphlet delta requires the same node set.")
        n, k = len(self.nodes), self.spec.k
        candidate = nx.to_numpy_array(graph, nodelist=self.nodes, weight=None, dtype=bool)
        changed = np.argwhere(np.triu(candidate != self.adjacency, 1))
        if not len(changed) or n < k:
            return self.counts.copy()
        # For many edits, a complete recount is cheaper and uses bounded memory.
        if len(changed) * comb(n-2, k-2) >= 2 * comb(n, k):
            return extract_counts(graph, self.spec)
        affected = set()
        for u, v in changed:
            remaining = [i for i in range(n) if i != u and i != v]
            affected.update(tuple(sorted((int(u), int(v), *rest))) for rest in combinations(remaining, k-2))
        subsets = np.asarray(sorted(affected), dtype=np.intp).reshape(-1, k)
        before = np.bincount(_classify(self.adjacency, subsets, self.spec), minlength=self.spec.width)
        after = np.bincount(_classify(candidate, subsets, self.spec), minlength=self.spec.width)
        result = self.counts + after - before
        if np.any(result < 0) or result.sum() != self.counts.sum():
            raise AssertionError("Invalid incremental induced graphlet counts.")
        return result

    def candidate_histogram(self, graph):
        return histogram_from_counts(self.candidate_counts(graph), len(self.nodes), self.spec)


def prediction_and_loss(logits, target, sizes, spec):
    """Shared PyTorch loss: unordered-bin Brier loss, optional CE, TV/count metrics.

    Losses average over graphs with n>=k. No CDF/W1 on arbitrarily ordered bins.
    Returns differentiable losses (brier, ce) and tensor-valued diagnostics.
    """
    import torch
    import torch.nn.functional as F
    if logits.ndim != 2 or logits.shape[-1] != spec.width:
        raise ValueError("Wrong induced graphlet output shape.")
    if target.shape != logits.shape:
        raise ValueError("Induced graphlet target/prediction shape mismatch.")
    target = target.to(logits)
    if not torch.isfinite(target).all() or (target < -1e-6).any() or not torch.allclose(target.sum(-1), torch.ones_like(sizes, dtype=target.dtype), atol=1e-5):
        raise ValueError("Induced graphlet targets must be probability distributions.")
    valid = (sizes >= spec.k).to(logits.dtype)
    denominator = valid.sum().clamp_min(1)
    p = logits.softmax(-1)
    delta = p-target
    brier = (delta.square().sum(-1)*valid).sum()/denominator
    ce = (-(target*F.log_softmax(logits,-1)).sum(-1)*valid).sum()/denominator
    totals = torch.as_tensor([comb(int(n), spec.k) if n >= spec.k else 0 for n in sizes.detach().cpu().tolist()], dtype=logits.dtype, device=logits.device)
    metrics = {"induced_graphlet_histogram_loss": brier, "induced_graphlet_histogram_ce": ce,
               "induced_graphlet_histogram_tv": (0.5*delta.abs().sum(-1)*valid).sum()/denominator,
               "induced_graphlet_count_mae": (delta.abs().mean(-1)*totals*valid).sum()/denominator,
               "induced_graphlet_valid_fraction": valid.mean()}
    return brier, ce, metrics


def mask_prediction(probabilities, sizes, spec):
    import torch
    placeholder = torch.zeros_like(probabilities)
    placeholder[:, 0 if spec.scope == "all" else -1] = 1
    return torch.where((sizes >= spec.k).unsqueeze(-1), probabilities, placeholder)
