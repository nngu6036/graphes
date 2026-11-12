"""Explicit, audited donor schedules for generation-only basis controls.

The checkpoint bank is never modified.  Schedules refer to original bank
indices, retain the parent node-count sequence, and have their own RNG stream.
Changing a donor therefore changes the reverse trajectory; it is not an
ablation of an already sampled continuous endpoint.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping

import numpy as np


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def resolve_schedule(sample_cfg: Mapping, basis_sizes, count: int):
    raw = sample_cfg.get("basis_ablation", {}) or {}
    if not raw.get("enabled", False):
        return None
    allowed = {"enabled", "indices", "expected_node_counts", "schedule_sha256", "design"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown basis_ablation settings: {sorted(unknown)}")
    indices = raw.get("indices")
    sizes = np.asarray(basis_sizes, dtype=np.int64)
    if not isinstance(indices, list) or len(indices) != count:
        raise ValueError("basis_ablation.indices must contain one original bank index per sample")
    if any(type(i) is not int or not 0 <= i < len(sizes) for i in indices):
        raise ValueError("basis_ablation contains invalid original donor bank indices")
    expected = raw.get("expected_node_counts")
    if not isinstance(expected, list) or len(expected) != count:
        raise ValueError("basis_ablation.expected_node_counts must match the sample count")
    observed = sizes[np.asarray(indices)].tolist()
    if observed != expected:
        raise ValueError("Basis donor schedule changes the matched node-count sequence")
    if raw.get("schedule_sha256") != digest({"indices": indices, "node_counts": expected}):
        raise ValueError("Basis donor schedule digest differs")
    return np.asarray(indices, dtype=np.int64)


def select_indices(rng, bank_size: int, batch_size: int, start: int, schedule):
    # Consume the original donor RNG draw even under a schedule.  This preserves
    # any later use of that stream and leaves the diffusion RNG untouched.
    original = rng.integers(0, bank_size, size=batch_size)
    if schedule is None:
        return original
    selected = schedule[start:start + batch_size]
    if len(selected) != batch_size:
        raise ValueError("Basis donor schedule ended before the requested population")
    return selected


def make_schedule(parent_indices, basis_sizes, *, seed: int, cap_per_size=None,
                  donor_offset: int = 0):
    """Make nested, training-only reservoirs stratified by exact node count.

    ``cap_per_size=None, donor_offset=0`` exactly replays the parent's donors.
    Finite caps select the first K entries of one fixed permutation per stratum;
    inverse-CDF common random draws couple arms without changing node counts.
    """
    sizes = np.asarray(basis_sizes, dtype=np.int64)
    parents = [int(i) for i in parent_indices]
    if any(i < 0 or i >= len(sizes) for i in parents):
        raise ValueError("Parent donor index is outside the checkpoint bank")
    if cap_per_size is not None and (type(cap_per_size) is not int or cap_per_size < 1):
        raise ValueError("Reservoir cap must be a positive number of donors per size")
    pools = {}
    for n in sorted(set(int(x) for x in sizes)):
        candidates = np.flatnonzero(sizes == n)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 920003, n]))
        pool = rng.permutation(candidates)
        pools[n] = pool if cap_per_size is None else pool[:cap_per_size]
    expected = [int(sizes[i]) for i in parents]
    if cap_per_size is None and donor_offset == 0:
        chosen = parents
    else:
        chosen = []
        for i, n in enumerate(expected):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 920009, donor_offset, i]))
            pool = pools[n]
            chosen.append(int(pool[min(int(rng.random() * len(pool)), len(pool) - 1)]))
    bank_counts = Counter(int(x) for x in sizes)
    design = {
        "cap_per_node_count": cap_per_size,
        "donor_resampling_offset": donor_offset,
        "full_bank_size": len(sizes),
        "reservoir_size": sum(len(v) for v in pools.values()),
        "bank_counts_by_node_count": dict(sorted(bank_counts.items())),
        "reservoir_counts_by_node_count": {n: len(v) for n, v in pools.items()},
        "reservoir_original_indices": {n: v.tolist() for n, v in pools.items()},
        "matched_node_counts": True,
        "reverse_diffusion_rerun": True,
        "pairing": "shared checkpoint, node counts and diffusion RNG; endpoints change",
    }
    return {"enabled": True, "indices": chosen, "expected_node_counts": expected,
            "schedule_sha256": digest({"indices": chosen, "node_counts": expected}),
            "design": design}
