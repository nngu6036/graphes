"""Optional final-only same-type graphlet refinement after edge thresholding.

Schema v1 keeps its historical weighted-adjacency spectral energy. Schema v2
uses normalized-Laplacian spectral consistency: the fixed continuous target is
soft-thresholded into edge presence, while each discrete swap candidate is
scored by the normalized-Laplacian spectrum of its actual topology. The
adjacency energy remains weighted-adjacency MSE. No categorical edge head is
introduced.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from grapher.models.gdsm_simple.categorical.data import topology_summary
from grapher.models.gdsm_simple.categorical.multiscale import update_counts
from grapher.models.gdsm_simple.categorical.refiner import candidates


def _normalized_laplacian_spectrum(adjacency: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    matrix = np.asarray(adjacency, dtype=np.float64)
    degrees = matrix.sum(axis=1)
    positive = degrees > float(eps)
    inv = np.zeros_like(degrees)
    inv[positive] = 1.0 / np.sqrt(degrees[positive])
    lap = np.diag(positive.astype(np.float64)) - inv[:, None] * matrix * inv[None, :]
    return np.linalg.eigvalsh(lap)


def _soft_presence_numpy(scaled_weighted: np.ndarray, consistency: dict, edge_scale: float) -> np.ndarray:
    threshold = float(consistency["soft_threshold_physical"]) / float(edge_scale)
    temperature = float(consistency["temperature_physical"]) / float(edge_scale)
    z = np.clip((np.asarray(scaled_weighted, dtype=np.float64)-threshold)/temperature, -60., 60.)
    soft = 1.0 / (1.0 + np.exp(-z))
    soft = .5 * (soft + soft.T)
    np.fill_diagonal(soft, 0.)
    return soft


def energy(x, e, target, basis, codec, bins, weights, *, counts,
           spectral_mode="sqrt_n", consistency=None, edge_scale=1.0):
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
    if weights["spectral"]:
        if spectral_mode == "normalized_laplacian":
            candidate_spectrum = _normalized_laplacian_spectrum((e > 0).astype(np.float64),
                                                                 float(consistency["normalized_laplacian_epsilon"]))
        else:
            candidate_spectrum = np.linalg.eigvalsh(matrix)/len(x)**.5
        raw["spectral"] = float(np.mean((candidate_spectrum-target["spectrum"])**2))
    else:
        raw["spectral"] = 0.
    ij = np.triu_indices(len(x), 1)
    raw["adjacency"] = (float(np.mean((matrix[ij]-target["weighted_adjacency"][ij])**2)) if len(ij[0]) else 0.)
    return {**raw, "structure": float(structure),
            "total": float(structure + weights["spectral"]*raw["spectral"] + weights["adjacency"]*raw["adjacency"])}


def refine(x, edges, target, basis, codec, bins, cfg, rng, *,
           spectral_mode="sqrt_n", consistency=None, edge_scale=1.0):
    before, e = edges.copy(), edges.copy()
    counts = basis.count(x, e)
    target = dict(target)
    if cfg["weights"]["spectral"]:
        if spectral_mode == "normalized_laplacian":
            if consistency is None:
                raise ValueError("normalized-Laplacian refinement requires consistency configuration")
            soft = _soft_presence_numpy(target["weighted_adjacency"], consistency, edge_scale)
            target["spectrum"] = _normalized_laplacian_spectrum(
                soft, float(consistency["normalized_laplacian_epsilon"]))
        else:
            target["spectrum"] = np.linalg.eigvalsh(target["weighted_adjacency"].astype(np.float64))/len(x)**.5
    kwargs = dict(spectral_mode=spectral_mode, consistency=consistency, edge_scale=edge_scale)
    old = energy(x, e, target, basis, codec, bins, cfg["weights"], counts=counts, **kwargs)
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
            score = energy(x, candidate, target, basis, codec, bins, cfg["weights"], counts=cc, **kwargs)
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
