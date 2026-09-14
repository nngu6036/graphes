"""Exact untyped induced-graphlet distributions for k in {3,4,5}.

A single-order ``InducedGraphletSpec`` remains checkpoint-compatible.  New
configs may request ``induced_graphlet_k_min``/``induced_graphlet_k_max`` and
receive an ``InducedGraphletCollectionSpec``.  Each order is represented by a
separately normalized probability block; blocks are concatenated only for the
network head.  ``all`` separates every connected/disconnected isomorphism
class. ``connected`` retains connected classes plus one pooled disconnected bin.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations, islice
from math import comb
import hashlib
import json
from typing import Any, Mapping, Sequence, Union

import networkx as nx
import numpy as np

VERSION = "induced_topology_minmask_multik_v2"
LEGACY_VERSION = "induced_topology_minmask_v1"
SUPPORTED_K = (3, 4, 5)


def validate_k(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) not in SUPPORTED_K:
        raise ValueError("induced graphlet order must be an integer in {3,4,5}; larger k is not implemented.")
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
    def sizes(self) -> tuple[int, ...]:
        return (self.k,)

    @property
    def width(self) -> int:
        return len(catalogue(self.k, self.scope)["bins"])

    @property
    def slices(self) -> tuple[tuple[int, int], ...]:
        return ((0, self.width),)

    @property
    def block_widths(self) -> tuple[int, ...]:
        return (self.width,)

    @property
    def min_k(self) -> int:
        return self.k

    @property
    def max_k(self) -> int:
        return self.k

    def metadata(self) -> dict:
        cat = catalogue(self.k, self.scope)
        # Keep legacy keys/version for old single-k checkpoint equality tests.
        payload = {
            "k": self.k,
            "scope": self.scope,
            "width": self.width,
            "version": LEGACY_VERSION,
            "normalization": f"choose(n,{self.k})",
            "attributed": False,
            "bin_ids": [x["id"] for x in cat["bins"]],
        }
        payload["fingerprint"] = cat["fingerprint"]
        return payload

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None):
        values = config or {}
        if not values.get("induced_graphlet_histogram", False):
            return None
        scope = validate_scope(str(values.get("induced_graphlet_scope", "all")))
        if "induced_graphlet_k_min" in values or "induced_graphlet_k_max" in values:
            k_min = validate_k(values.get("induced_graphlet_k_min", values.get("induced_graphlet_k", 3)))
            k_max = validate_k(values.get("induced_graphlet_k_max", values.get("induced_graphlet_k", 5)))
            if k_min > k_max:
                raise ValueError("induced_graphlet_k_min must be <= induced_graphlet_k_max.")
            sizes = tuple(range(k_min, k_max + 1))
            if len(sizes) == 1:
                return cls(sizes[0], scope)
            return InducedGraphletCollectionSpec(sizes=sizes, scope=scope)
        return cls(values.get("induced_graphlet_k", 5), scope)


@dataclass(frozen=True)
class InducedGraphletCollectionSpec:
    sizes: tuple[int, ...] = (3, 4, 5)
    scope: str = "all"

    def __post_init__(self):
        sizes = tuple(validate_k(k) for k in self.sizes)
        if not sizes or tuple(sorted(set(sizes))) != sizes:
            raise ValueError("induced graphlet sizes must be a nonempty increasing subset of {3,4,5}.")
        object.__setattr__(self, "sizes", sizes)
        object.__setattr__(self, "scope", validate_scope(self.scope))

    # Backward-friendly convenience: code that only needs a size threshold can
    # use max k, while block-aware code should use sizes/slices.
    @property
    def k(self) -> int:
        return max(self.sizes)

    @property
    def min_k(self) -> int:
        return min(self.sizes)

    @property
    def max_k(self) -> int:
        return max(self.sizes)

    @property
    def block_widths(self) -> tuple[int, ...]:
        return tuple(len(catalogue(k, self.scope)["bins"]) for k in self.sizes)

    @property
    def slices(self) -> tuple[tuple[int, int], ...]:
        out, start = [], 0
        for width in self.block_widths:
            out.append((start, start + width))
            start += width
        return tuple(out)

    @property
    def width(self) -> int:
        return sum(self.block_widths)

    def metadata(self) -> dict:
        payload = {
            "version": VERSION,
            "sizes": list(self.sizes),
            "k_min": self.min_k,
            "k_max": self.max_k,
            "scope": self.scope,
            "width": self.width,
            "block_widths": list(self.block_widths),
            "block_slices": [list(x) for x in self.slices],
            "normalization": {str(k): f"choose(n,{k})" for k in self.sizes},
            "attributed": False,
            "bin_ids_by_k": {
                str(k): [x["id"] for x in catalogue(k, self.scope)["bins"]]
                for k in self.sizes
            },
        }
        payload["fingerprint"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return payload


# Type aliases are evaluated at import time, even with postponed annotations.
InducedSpec = Union[InducedGraphletSpec, InducedGraphletCollectionSpec]


def _graph(k, code):
    g = nx.empty_graph(k)
    g.add_edges_from(pair for bit, pair in enumerate(combinations(range(k), 2)) if code & (1 << bit))
    return g


@lru_cache(maxsize=3)
def _canonical_codes(k: int):
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
    payload = {"version": LEGACY_VERSION, "k": k, "scope": scope, "attributed": False, "bins": bins}
    payload["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


@lru_cache(maxsize=6)
def _bin_lookup(k, scope):
    cat = catalogue(k, scope)
    indices = {b["canonical_mask"]: i for i, b in enumerate(cat["bins"]) if b["canonical_mask"] is not None}
    lookup = np.array([indices.get(int(c), len(cat["bins"]) - 1) for c in _canonical_codes(k)], dtype=np.int64)
    lookup.setflags(write=False)
    return lookup


def _block_spec(spec: InducedSpec, k: int) -> InducedGraphletSpec:
    return InducedGraphletSpec(int(k), spec.scope)


def validate_histogram(values, spec: InducedSpec | None = None):
    x = np.asarray(values, dtype=np.float64)
    if spec is None:
        if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or np.any(x < -1e-7) or np.any(x > 1 + 1e-7) or not np.isclose(x.sum(), 1, atol=1e-6, rtol=0):
            raise ValueError("Induced graphlet histogram must be a finite normalized probability vector.")
        x = np.clip(x, 0, 1)
        return x / x.sum()
    if x.ndim != 1 or x.shape != (spec.width,):
        raise ValueError("Induced graphlet histogram width/catalogue mismatch.")
    if not np.isfinite(x).all() or np.any(x < -1e-7) or np.any(x > 1 + 1e-7):
        raise ValueError("Induced graphlet histogram must be finite and nonnegative.")
    out = np.clip(x, 0, 1)
    for k, (start, stop) in zip(spec.sizes, spec.slices):
        total = float(out[start:stop].sum())
        if not np.isclose(total, 1.0, atol=1e-6, rtol=0):
            raise ValueError(f"Induced graphlet block k={k} must sum to one.")
        out[start:stop] /= max(total, 1e-12)
    return out


def histogram_distance(left, right, spec: InducedSpec | None = None) -> float:
    a, b = validate_histogram(left, spec), validate_histogram(right, spec)
    if a.shape != b.shape:
        raise ValueError("Induced histogram shapes differ.")
    if spec is None or len(spec.sizes) == 1:
        return float(0.5 * np.abs(a-b).sum())
    return float(np.mean([0.5*np.abs(a[s:e]-b[s:e]).sum() for s,e in spec.slices]))


def histogram_from_counts(counts, n, spec: InducedSpec):
    if len(spec.sizes) > 1:
        if not isinstance(counts, Mapping):
            raise ValueError("Multi-order induced counts must be a mapping from k to count arrays.")
        return np.concatenate([
            histogram_from_counts(counts[int(k)] if int(k) in counts else counts[str(k)], n, _block_spec(spec, k))
            for k in spec.sizes
        ])
    k = spec.sizes[0]
    block = _block_spec(spec, k)
    counts = np.asarray(counts)
    total = comb(int(n), k) if n >= k else 0
    if counts.shape != (block.width,) or np.any(counts < 0) or not np.isfinite(counts).all() or not np.all(counts == np.floor(counts)) or int(counts.sum()) != total:
        raise ValueError("Counts must partition all k-node subsets exactly.")
    if total:
        return counts.astype(np.float64) / total
    out = np.zeros(block.width, dtype=np.float64)
    out[0 if block.scope == "all" else -1] = 1.0
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


def _classify(adjacency, subsets, spec: InducedGraphletSpec):
    if len(subsets) == 0:
        return np.zeros(0, dtype=np.int64)
    codes = np.zeros(len(subsets), dtype=np.int64)
    for bit, (i, j) in enumerate(combinations(range(spec.k), 2)):
        codes |= adjacency[subsets[:, i], subsets[:, j]].astype(np.int64) << bit
    return _bin_lookup(spec.k, spec.scope)[codes]


def _extract_counts_single(graph, spec: InducedGraphletSpec):
    adjacency = nx.to_numpy_array(graph, nodelist=list(graph), weight=None, dtype=bool)
    out = np.zeros(spec.width, dtype=np.int64)
    for subsets in _subset_chunks(len(graph), spec.k):
        out += np.bincount(_classify(adjacency, subsets, spec), minlength=spec.width)
    return out


def extract_counts(graph, spec: InducedSpec):
    _validate_graph(graph)
    if len(spec.sizes) == 1:
        return _extract_counts_single(graph, _block_spec(spec, spec.sizes[0]))
    return {int(k): _extract_counts_single(graph, _block_spec(spec, k)) for k in spec.sizes}


def extract_histogram(graph, spec: InducedSpec):
    return histogram_from_counts(extract_counts(graph, spec), len(graph), spec)


class InducedGraphletCounter:
    """Exact current-state counts and local candidate deltas for one/many k."""
    def __init__(self, graph, spec: InducedSpec):
        _validate_graph(graph)
        self.spec = spec
        self.nodes = list(graph)
        self.adjacency = nx.to_numpy_array(graph, nodelist=self.nodes, weight=None, dtype=bool)
        self.counts = extract_counts(graph, spec)

    def histogram(self):
        return histogram_from_counts(self.counts, len(self.nodes), self.spec)

    def _candidate_counts_single(self, candidate, current_counts, block_spec):
        n, k = len(self.nodes), block_spec.k
        changed = np.argwhere(np.triu(candidate != self.adjacency, 1))
        if not len(changed) or n < k:
            return np.asarray(current_counts).copy()
        if len(changed) * comb(n-2, k-2) >= 2 * comb(n, k):
            g = nx.from_numpy_array(candidate.astype(np.uint8))
            return _extract_counts_single(g, block_spec)
        affected = set()
        for u, v in changed:
            remaining = [i for i in range(n) if i != u and i != v]
            affected.update(tuple(sorted((int(u), int(v), *rest))) for rest in combinations(remaining, k-2))
        subsets = np.asarray(sorted(affected), dtype=np.intp).reshape(-1, k)
        before = np.bincount(_classify(self.adjacency, subsets, block_spec), minlength=block_spec.width)
        after = np.bincount(_classify(candidate, subsets, block_spec), minlength=block_spec.width)
        result = np.asarray(current_counts) + after - before
        if np.any(result < 0) or result.sum() != np.asarray(current_counts).sum():
            raise AssertionError("Invalid incremental induced graphlet counts.")
        return result

    def candidate_counts(self, graph):
        _validate_graph(graph)
        if set(graph) != set(self.nodes):
            raise ValueError("A graphlet delta requires the same node set.")
        candidate = nx.to_numpy_array(graph, nodelist=self.nodes, weight=None, dtype=bool)
        if len(self.spec.sizes) == 1:
            k = self.spec.sizes[0]
            return self._candidate_counts_single(candidate, self.counts, _block_spec(self.spec, k))
        return {
            int(k): self._candidate_counts_single(candidate, self.counts[int(k)], _block_spec(self.spec, k))
            for k in self.spec.sizes
        }

    def candidate_histogram(self, graph):
        return histogram_from_counts(self.candidate_counts(graph), len(self.nodes), self.spec)

    def accept(self, graph):
        self.counts = self.candidate_counts(graph)
        self.adjacency = nx.to_numpy_array(graph, nodelist=self.nodes, weight=None, dtype=bool)


def block_softmax(logits, spec: InducedSpec):
    import torch
    if len(spec.sizes) == 1:
        return logits.softmax(-1)
    return torch.cat([logits[:,s:e].softmax(-1) for s,e in spec.slices], dim=-1)


def prediction_and_loss(logits, target, sizes, spec: InducedSpec):
    import torch
    import torch.nn.functional as F
    if logits.ndim != 2 or logits.shape[-1] != spec.width or target.shape != logits.shape:
        raise ValueError("Wrong induced graphlet output/target shape.")
    target = target.to(logits)
    briers=[]; ces=[]; tvs=[]; maes=[]; valid_fracs=[]
    for k,(start,stop) in zip(spec.sizes,spec.slices):
        t = target[:,start:stop]
        if not torch.isfinite(t).all() or (t < -1e-6).any() or not torch.allclose(t.sum(-1), torch.ones_like(sizes,dtype=t.dtype), atol=1e-5):
            raise ValueError(f"Induced graphlet target block k={k} must be a probability distribution.")
        valid=(sizes>=k).to(logits.dtype); denom=valid.sum().clamp_min(1)
        z=logits[:,start:stop]; p=z.softmax(-1); delta=p-t
        briers.append((delta.square().sum(-1)*valid).sum()/denom)
        ces.append((-(t*F.log_softmax(z,-1)).sum(-1)*valid).sum()/denom)
        tvs.append((0.5*delta.abs().sum(-1)*valid).sum()/denom)
        totals=torch.as_tensor([comb(int(n),k) if n>=k else 0 for n in sizes.detach().cpu().tolist()],dtype=logits.dtype,device=logits.device)
        maes.append((delta.abs().mean(-1)*totals*valid).sum()/denom)
        valid_fracs.append(valid.mean())
    brier=torch.stack(briers).mean(); ce=torch.stack(ces).mean()
    metrics={"induced_graphlet_histogram_loss":brier,"induced_graphlet_histogram_ce":ce,
             "induced_graphlet_histogram_tv":torch.stack(tvs).mean(),
             "induced_graphlet_count_mae":torch.stack(maes).mean(),
             "induced_graphlet_valid_fraction":torch.stack(valid_fracs).mean()}
    for k,tv in zip(spec.sizes,tvs): metrics[f"induced_graphlet_histogram_tv_k{k}"]=tv
    return brier,ce,metrics


def mask_prediction(probabilities, sizes, spec: InducedSpec):
    import torch
    blocks=[]
    for k,(start,stop) in zip(spec.sizes,spec.slices):
        block=probabilities[:,start:stop]
        placeholder=torch.zeros_like(block)
        placeholder[:,0 if spec.scope=="all" else -1]=1
        blocks.append(torch.where((sizes>=k).unsqueeze(-1),block,placeholder))
    return torch.cat(blocks,dim=-1)


__all__ = [
    "InducedGraphletSpec","InducedGraphletCollectionSpec","InducedGraphletCounter",
    "block_softmax","catalogue","extract_counts","extract_histogram","histogram_distance",
    "histogram_from_counts","mask_prediction","prediction_and_loss","validate_histogram",
    "validate_k","validate_scope",
]
