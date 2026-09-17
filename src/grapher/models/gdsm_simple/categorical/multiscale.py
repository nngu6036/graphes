"""Exact connected induced categorical graphlets of orders 3--5.

Training-only vocabularies; a separate overflow and connected-subset mass per
order. No Monte Carlo counting and no silent truncation. Candidate swap deltas
visit only connected subsets of the union graph containing a changed pair.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations, product
from math import comb
from typing import Mapping

import numpy as np

from .data import canonical_triple


@lru_cache(maxsize=512)
def _pairs(k):
    return tuple(combinations(range(k), 2))


@lru_cache(maxsize=4096)
def _label_permutations(labels):
    # Node labels are the prefix of the canonical key. Only permutations that
    # put that prefix in sorted order can minimize the full key.
    groups = [[i for i, x in enumerate(labels) if x == value]
              for value in sorted(set(labels))]
    return tuple(tuple(i for group in groups_perm for i in group)
                 for groups_perm in product(*(permutations(g) for g in groups)))


@lru_cache(maxsize=200000)
def canonical_pattern(k: int, raw: tuple[int, ...]) -> tuple[int, ...]:
    if k not in (3, 4, 5) or len(raw) != k + k * (k - 1) // 2:
        raise ValueError('Canonicalization requires an order-3, 4 or 5 pattern')
    if k == 3:
        return canonical_triple(raw)
    labels = raw[:k]
    lookup = {p: raw[k+i] for i, p in enumerate(_pairs(k))}
    edges = min(tuple(lookup[(min(p[i], p[j]), max(p[i], p[j]))]
                      for i, j in _pairs(k))
                for p in _label_permutations(labels))
    return tuple(sorted(labels)) + edges


def _neighbors(e):
    return [sum(1 << int(j) for j in np.flatnonzero(row)) for row in e]


def _nodes(bits):
    out = []
    while bits:
        bit = bits & -bits
        out.append(bit.bit_length() - 1)
        bits ^= bit
    return out


def _expand(level, neighbors, limit):
    result = set()
    for bits in level:
        adjacent = 0
        for v in _nodes(bits):
            adjacent |= neighbors[v]
        available = adjacent & ~bits
        while available:
            bit = available & -available
            result.add(bits | bit)
            available ^= bit
            if limit is not None and len(result) > limit:
                raise RuntimeError(
                    f'Exact graphlet enumeration exceeded max_connected_subsets={limit}. '
                    'No approximate or truncated counts were returned. Increase this '
                    'explicit resource limit or reduce graph sizes; document any change.')
    return result


def connected_subsets(e, sizes=(3, 4, 5), *, limit=None, seed_pairs=None):
    """Yield each selected connected vertex subset once at each requested order.

    A connected k-subset contains a connected (k-1)-subset. Expanding all edges
    and deduplicating vertex bitsets is therefore exact, including non-cycles.
    With seed_pairs, only subsets containing at least one seed pair are yielded.
    """
    sizes = tuple(sorted(set(int(k) for k in sizes)))
    if not sizes or any(k not in (3, 4, 5) for k in sizes):
        raise ValueError('Supported graphlet orders are 3, 4 and 5')
    neighbors = _neighbors(e)
    edges = zip(*np.nonzero(np.triu(e, 1))) if seed_pairs is None else seed_pairs
    level = {(1 << int(i)) | (1 << int(j)) for i, j in edges}
    for k in range(3, min(max(sizes), len(e)) + 1):
        level = _expand(level, neighbors, limit)
        if k in sizes:
            yield k, level
        if not level:
            break


def _connected(bits, neighbors):
    reached = bits & -bits
    pending = reached
    while pending:
        bit = pending & -pending
        pending ^= bit
        new = neighbors[bit.bit_length()-1] & bits & ~reached
        reached |= new
        pending |= new
    return reached == bits


def _key(x, e, nodes):
    k = len(nodes)
    raw = tuple(int(x[v]) for v in nodes) + tuple(int(e[nodes[i], nodes[j]]) for i, j in _pairs(k))
    return canonical_pattern(k, raw)


def count_multi(x, e, sizes=(3, 4, 5), *, limit=None):
    result = {int(k): Counter() for k in sizes}
    for k, subsets in connected_subsets(e, sizes, limit=limit):
        for bits in subsets:
            result[k][_key(x, e, _nodes(bits))] += 1
    return result


def update_counts(x, before, after, counts, sizes=(3, 4, 5), *, limit=None):
    """Exact counts after an edge edit, with unchanged indexed node categories.

    Any changed induced subgraph contains a changed unordered pair. Any subset
    connected before OR after is connected in the union adjacency. Expanding
    changed pairs in that union thus covers every affected selected subset.
    """
    result = {int(k): Counter(counts[int(k)]) for k in sizes}
    changed = list(zip(*np.nonzero(np.triu(before != after, 1))))
    if not changed:
        return result
    union = (before > 0) | (after > 0)
    old_n, new_n = _neighbors(before), _neighbors(after)
    for k, subsets in connected_subsets(union, sizes, limit=limit, seed_pairs=changed):
        block = result[k]
        for bits in subsets:
            nodes = _nodes(bits)
            if _connected(bits, old_n):
                block[_key(x, before, nodes)] -= 1
            if _connected(bits, new_n):
                block[_key(x, after, nodes)] += 1
        if any(v < 0 for v in block.values()):
            raise AssertionError('Negative exact graphlet count after local delta')
        result[k] = +block  # Drop zero entries, preserving integer counts.
    return result



@dataclass
class IndexedCounts:
    """Compact training targets: per-order integer indices and occurrence counts."""
    blocks: dict


def pack_training_counts(counts, codebooks):
    blocks={}
    for k, block in counts.items():
        index=codebooks[k];ids=[];values=[]
        for key,value in block.items():
            if key not in index: index[key]=len(index)
            ids.append(index[key]);values.append(value)
        blocks[k]=(np.asarray(ids,np.int32),np.asarray(values,np.int32))
    return IndexedCounts(blocks)


def remap_training_counts(packed, lookup):
    blocks={}
    for k,(ids,values) in packed.blocks.items():
        mapped=lookup[k][ids]
        unique,inverse=np.unique(mapped,return_inverse=True)
        sums=np.zeros(len(unique),np.int32)
        np.add.at(sums,inverse,values)
        blocks[k]=(unique.astype(np.int32),sums)
    return IndexedCounts(blocks)


def pack_basis_counts(counts,basis):
    blocks={}
    for k in basis.orders:
        ids=[];values=[]
        for key,value in counts[k].items():
            ids.append(basis.index[k].get(key,len(basis.keys_by_order[k])))
            values.append(value)
        # Coalesce all out-of-vocabulary patterns into the order's overflow.
        unique,inverse=np.unique(ids,return_inverse=True)
        sums=np.zeros(len(unique),np.int32)
        np.add.at(sums,inverse,np.asarray(values,np.int32))
        blocks[k]=(unique.astype(np.int32),sums)
    return IndexedCounts(blocks)


class TypedGraphletsMulti:
    """Concatenated separately normalized histograms, not a pooled cross-order simplex."""
    multiscale = True

    def __init__(self, keys_by_order: Mapping, *, size_weights=None, limit=None):
        self.orders = tuple(sorted(int(k) for k in keys_by_order))
        if not self.orders or any(k not in (3, 4, 5) for k in self.orders):
            raise ValueError('Supported graphlet orders are 3, 4 and 5')
        keys_by_order = {int(k): v for k, v in keys_by_order.items()}
        self.keys_by_order = {k: tuple(tuple(int(a) for a in key) for key in keys_by_order[k]) for k in self.orders}
        for k, keys in self.keys_by_order.items():
            if len(set(keys)) != len(keys) or any(len(key) != k+k*(k-1)//2 for key in keys):
                raise ValueError(f'Invalid or duplicate vocabulary at order {k}')
        self.index = {k: {key: i for i, key in enumerate(keys)} for k, keys in self.keys_by_order.items()}
        self.block_sizes = tuple(len(self.keys_by_order[k])+1 for k in self.orders)
        offsets = np.r_[0, np.cumsum(self.block_sizes)]
        self.slices = {k: slice(int(offsets[i]), int(offsets[i+1])) for i, k in enumerate(self.orders)}
        self.dimension = int(offsets[-1])
        self.limit = limit
        weights = np.ones(len(self.orders)) if size_weights is None else np.asarray(size_weights, float)
        if weights.shape != (len(self.orders),) or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError('Provide one finite positive size weight per order')
        self.size_weights = weights / weights.sum()

    def count(self, x, e):
        return count_multi(x, e, self.orders, limit=self.limit)

    def encode_counts(self, counts, n):
        hist = np.zeros(self.dimension, np.float32)
        mass = np.zeros(len(self.orders), np.float32)
        for i, k in enumerate(self.orders):
            block = hist[self.slices[k]]
            if isinstance(counts,IndexedCounts):
                ids,values=counts.blocks[k]
                np.add.at(block,ids,values)
            else:
                for key, count in counts.get(k, {}).items():
                    block[self.index[k].get(key, len(block)-1)] += count
            total = float(block.sum())
            if n >= k:
                mass[i] = total / comb(n, k)
            elif total:
                raise ValueError('Nonzero graphlet counts for an unavailable order')
            if total:
                block /= total
        return hist, mass

    def summary(self, x, e):
        return self.encode_counts(self.count(x, e), len(x))

    def overflow_by_order(self, histogram):
        return {str(k): float(histogram[self.slices[k]][-1]) for k in self.orders}

    def overflow_mean(self, histogram):
        return float(sum(w*histogram[self.slices[k]][-1] for k, w in zip(self.orders, self.size_weights)))

    def schema(self):
        return {'graphlet_schema_version': 2, 'graphlet_orders': list(self.orders),
                'graphlet_keys_by_order': {str(k): [list(v) for v in self.keys_by_order[k]] for k in self.orders},
                'graphlet_size_weights': self.size_weights.tolist(),
                'graphlet_counting': 'exact_connected_induced',
                'graphlet_max_connected_subsets': self.limit,
                'graphlets': 'separate_connected_induced_typed_histograms_plus_overflow_and_mass_per_order'}

    @classmethod
    def from_schema(cls, schema):
        return cls(schema['graphlet_keys_by_order'], size_weights=schema['graphlet_size_weights'],
                   limit=schema.get('graphlet_max_connected_subsets'))


def fit_basis(counts_by_order, graphlet_config):
    """Counts accumulated only over training graphs. Stable frequency/lexical tie-breaking."""
    sizes = tuple(graphlet_config['sizes'])
    cap = graphlet_config['max_vocab_per_size']
    minimum = int(graphlet_config['min_train_count'])
    keys = {}
    coverage = {}
    for k in sizes:
        counts = counts_by_order[k]
        eligible = sorted((key for key in counts if counts[key] >= minimum), key=lambda key: (-counts[key], key))
        kept = eligible if cap is None else eligible[:int(cap)]
        keys[k] = sorted(kept)
        total = sum(counts.values())
        coverage[str(k)] = {'observed_classes': len(counts), 'retained_classes': len(kept),
                            'training_occurrences': int(total),
                            'retained_occurrence_fraction': sum(counts[a] for a in kept)/max(total, 1)}
    return TypedGraphletsMulti(keys, size_weights=graphlet_config['size_weights'],
                              limit=graphlet_config['max_connected_subsets']), coverage
