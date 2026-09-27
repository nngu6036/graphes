"""Optional final-only same-type graphlet refinement after edge thresholding.

This uses the existing exact graphlet delta and swap candidate helpers. The
legacy categorical NLL is replaced by weighted-adjacency MSE, and the spectral
term is the WEIGHTED adjacency spectrum. No invented edge probabilities/head.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from grapher.models.gdsm_simple.categorical.data import topology_summary
from grapher.models.gdsm_simple.categorical.multiscale import update_counts
from grapher.models.gdsm_simple.categorical.refiner import candidates


def energy(x, e, target, basis, codec, bins, weights, *, counts):
    hist, mass = basis.encode_counts(counts, len(x))
    active = sum(w for k, w in zip(basis.orders, basis.size_weights) if len(x) >= k)
    raw = {"graphlet": 0., "mass": 0.}
    for i, (k, w) in enumerate(zip(basis.orders, basis.size_weights)):
        if len(x) < k:
            continue
        weight = float(w / active)
        raw["graphlet"] += weight * float(np.abs(hist[basis.slices[k]]-target["histogram"][basis.slices[k]]).sum()/2)
        raw["mass"] += weight * float((mass[i]-target["mass"][i])**2)
    clustering, orbit = topology_summary(e, bins)
    raw["clustering"] = float(np.abs(clustering.cumsum()-target["clustering"].cumsum()).mean())
    raw["orbit"] = float(np.mean((orbit-target["orbit"])**2))
    structure = sum(weights[key]*raw[key] for key in raw)
    matrix = codec.encode(e).astype(np.float64)
    raw["spectral"] = (float(np.mean((np.linalg.eigvalsh(matrix)/len(x)**.5-target["spectrum"])**2))
                       if weights["spectral"] else 0.)
    ij = np.triu_indices(len(x), 1)
    raw["adjacency"] = (float(np.mean((matrix[ij]-target["weighted_adjacency"][ij])**2)) if len(ij[0]) else 0.)
    return {**raw, "structure": float(structure),
            "total": float(structure + weights["spectral"]*raw["spectral"] + weights["adjacency"]*raw["adjacency"])}


def refine(x, edges, target, basis, codec, bins, cfg, rng):
    before, e = edges.copy(), edges.copy()
    counts = basis.count(x, e)
    target = dict(target)
    if cfg["weights"]["spectral"]:
        target["spectrum"] = np.linalg.eigvalsh(target["weighted_adjacency"].astype(np.float64))/len(x)**.5
    old = energy(x, e, target, basis, codec, bins, cfg["weights"], counts=counts)
    initial = dict(old)
    keep_connected = bool(cfg["preserve_connectivity_if_connected"]) and nx.is_connected(nx.from_numpy_array(e > 0))
    seen = {e.tobytes()}
    accepted, tested = [], 0
    for _ in range(cfg["max_steps"]):
        best, best_counts, best_energy = None, None, old
        for candidate in candidates(e, cfg, rng):
            if candidate.tobytes() in seen:
                continue
            if keep_connected and not nx.is_connected(nx.from_numpy_array(candidate > 0)):
                continue
            tested += 1
            cc = update_counts(x, e, candidate, counts, basis.orders, limit=basis.limit)
            score = energy(x, candidate, target, basis, codec, bins, cfg["weights"], counts=cc)
            if cfg["require_structure_improvement"] and score["structure"] >= old["structure"]-cfg["min_improvement"]:
                continue
            if score["total"] < best_energy["total"]-cfg["min_improvement"]:
                best, best_counts, best_energy = candidate, cc, score
        if best is None:
            break
        accepted.append({"before": old, "after": best_energy})
        e, counts, old = best, best_counts, best_energy
        seen.add(e.tobytes())
    preserved = all(np.array_equal((e == k).sum(1), (before == k).sum(1)) for k in range(1, len(codec.values)))
    if not preserved:
        raise AssertionError("Same-type refinement changed typed degrees")
    return e, {"initial": initial, "final": old, "accepted_steps": len(accepted),
               "tested_candidates": tested, "typed_degrees_preserved": preserved,
               "degree_preserved": np.array_equal((e > 0).sum(1), (before > 0).sum(1)),
               "connectivity_preserved_if_initially_connected": keep_connected,
               "accepted": accepted, "changed": not np.array_equal(e, before)}
