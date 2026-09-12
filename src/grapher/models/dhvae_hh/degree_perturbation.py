"""Training-only empirical degree priors with auditable, constrained perturbations.

All kernels preserve n and m. ``moment_preserving`` also preserves sum(d**2).
The kernels operate on degree MULTISETS, not node identities. No validation/test
invariants or original training adjacencies are retained by this sampler.

A failed perturbation is either a labelled identity transition (keep_original)
or an error. We never silently redraw the parent, repair its degrees, or switch
methods. This matters when a degree sequence admits no permitted perturbation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

from grapher.models.dhvae_hh.havel_hakimi import construct_coarse_graph

METHODS = ("unit_transfer", "moment_preserving", "edge_relocation", "interpolation")


class DegreePerturbationError(ValueError):
    """A requested perturbation failed; do not hide this by resampling parents."""


@dataclass(frozen=True)
class DegreePerturbationConfig:
    method: str = "unit_transfer"
    probability: float = 0.25
    steps: int = 1
    max_attempts: int = 256  # candidate checks PER accepted-step target
    failure_policy: str = "error"
    max_degree: int | None = None
    max_distance: float | None = 4.0  # half sorted L1 from the original parent
    require_novel: bool = False
    block_size: int = 4
    max_block_patterns: int = 20000
    max_block_alternatives: int = 20000
    interpolation_neighbors: int = 8
    interpolation_alpha: float = 0.5
    interpolation_max_parent_distance: float | None = 4.0

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Unknown degree perturbation method {self.method!r}; choose {METHODS}.")
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("degree_perturbation.probability must be in [0, 1].")
        for key in ("steps", "max_attempts", "max_block_patterns", "max_block_alternatives", "interpolation_neighbors"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"degree_perturbation.{key} must be a positive integer.")
        if self.failure_policy not in {"error", "keep_original"}:
            raise ValueError("failure_policy must be error or keep_original (explicit identity transition).")
        if not isinstance(self.require_novel, bool):
            raise ValueError("require_novel must be a boolean.")
        if self.max_degree is not None and (
            isinstance(self.max_degree, bool) or not isinstance(self.max_degree, (int, np.integer)) or self.max_degree < 1
        ):
            raise ValueError("max_degree must be a positive integer or null.")
        if isinstance(self.block_size, bool) or not isinstance(self.block_size, (int, np.integer)) or not 3 <= self.block_size <= 6:
            raise ValueError("block_size must be an integer between 3 and 6.")
        if not math.isfinite(self.interpolation_alpha) or not 0 < self.interpolation_alpha < 1:
            raise ValueError("interpolation_alpha must be strictly between 0 and 1.")
        for key in ("max_distance", "interpolation_max_parent_distance"):
            value = getattr(self, key)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be positive or null.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None) -> "DegreePerturbationConfig":
        values = dict(data or {})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown degree_perturbation options: {sorted(unknown)}")
        return cls(**values)


def canonical_degrees(degrees: Sequence[int]) -> tuple[int, ...]:
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in degrees):
        raise ValueError("Degrees must be integers, not rounded floats or booleans.")
    return tuple(sorted((int(x) for x in degrees), reverse=True))


def degree_distance(a: Sequence[int], b: Sequence[int]) -> float:
    if len(a) != len(b):
        raise ValueError("Degree distance requires equal node counts.")
    return sum(abs(x - y) for x, y in zip(sorted(a), sorted(b))) / 2.0


def sequence_fingerprint(sequences: Iterable[Sequence[int]]) -> str:
    payload = json.dumps([list(canonical_degrees(s)) for s in sequences], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def connected_feasible(degrees: Sequence[int], max_degree: int | None = None) -> bool:
    """Check simple graphicality plus existence of a connected realization.

    For n>1 use positive degrees and sum(d)>=2(n-1), in addition to EG.
    This certifies existence, NOT that ordinary HH itself is connected.
    """
    n = len(degrees)
    if n == 1:
        return list(degrees) == [0]
    if n < 1:
        return False
    ceiling = min(n - 1, max_degree) if max_degree is not None else n - 1
    return (
        min(degrees) >= 1 and max(degrees) <= ceiling
        and sum(degrees) % 2 == 0 and sum(degrees) >= 2 * (n - 1)
        and nx.is_graphical(list(degrees), method="eg")
    )


def degree_summary(degrees: Sequence[int]) -> dict[str, Any]:
    seq = canonical_degrees(degrees)
    n, m = len(seq), sum(seq) // 2
    hist = np.bincount(seq, minlength=max(max(seq, default=0), 1) + 1).astype(np.float64)
    hist /= max(float(hist.sum()), 1.0)
    return {"num_nodes": n, "num_edges": m, "degree_sequence": list(seq), "degree_hist": hist,
            "density": 2.0 * m / (n * (n - 1)) if n > 1 else 0.0}


@lru_cache(maxsize=4096)
def _moment_blocks(size: int, total: int, squares: int, ceiling: int, limit: int) -> tuple[tuple[int, ...], ...]:
    """Enumerate positive nondecreasing integer blocks with two exact moments.

    Fixed-moment pruning avoids enumerating the full degree^block_size grid.
    A hard catalogue guard raises instead of silently truncating the kernel.
    """
    result: list[tuple[int, ...]] = []

    def visit(prefix: tuple[int, ...], left: int, s: int, q: int, low: int) -> None:
        if left == 0:
            if s == 0 and q == 0:
                result.append(prefix)
                if len(result) > limit:
                    raise DegreePerturbationError("Moment-block catalogue exceeds max_block_alternatives.")
            return
        if s < left * low or s > left * ceiling or q < 0 or q * left < s * s:
            return
        if q < left * low * low or q > left * ceiling * ceiling:
            return
        if left == 1:
            if low <= s <= ceiling and s * s == q:
                visit(prefix + (s,), 0, 0, 0, s)
            return
        for value in range(low, min(ceiling, s // left, math.isqrt(q)) + 1):
            visit(prefix + (value,), left - 1, s - value, q - value * value, value)

    visit((), size, total, squares, 1)
    return tuple(result)


def _available_blocks(seq: Sequence[int], size: int, limit: int) -> list[tuple[int, ...]]:
    """Distinct block multisets present in seq (multiplicity-aware)."""
    counts = Counter(seq)
    values = sorted(counts)
    result: list[tuple[int, ...]] = []

    def visit(start: int, left: int, prefix: tuple[int, ...]) -> None:
        if left == 0:
            result.append(prefix)
            if len(result) > limit:
                raise DegreePerturbationError("Too many source blocks; lower block_size or raise max_block_patterns.")
            return
        for pos in range(start, len(values)):
            value = values[pos]
            if counts[value]:
                counts[value] -= 1
                visit(pos, left - 1, prefix + (value,))
                counts[value] += 1

    visit(0, size, ())
    return result


def sum_preserving_round(
    x: Sequence[float], total: int, rng: np.random.Generator, *, sort_result: bool = True
) -> tuple[int, ...]:
    """Dependent randomized rounding: exact sum and coordinate expectations.

    Each pairwise update preserves the fractional sum and fixes at least one
    coordinate. Sorting is performed only AFTER rounding unless sort_result=False. Typed
    callers must disable sorting to keep node/category correspondence intact.
    No iid rounding or degree repair is used.
    """
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Rounding requires finite nonnegative one-dimensional values.")
    if not np.isclose(values.sum(), total, rtol=0, atol=1e-8):
        raise ValueError("Interpolation sum does not match the requested degree sum.")
    base = np.floor(values).astype(np.int64)
    frac = values - base
    while True:
        active = np.flatnonzero((frac > 1e-10) & (frac < 1 - 1e-10))
        if len(active) < 2:
            break
        i, j = rng.choice(active, size=2, replace=False)
        up = min(1 - frac[i], frac[j])
        down = min(frac[i], 1 - frac[j])
        if rng.random() < down / (up + down):
            frac[i] += up
            frac[j] -= up
        else:
            frac[i] -= down
            frac[j] += down
        frac[np.abs(frac) < 1e-10] = 0.0
        frac[np.abs(frac - 1) < 1e-10] = 1.0
    rounded = base + np.rint(frac).astype(np.int64)
    if int(rounded.sum()) != total:
        raise ArithmeticError("Dependent rounding failed the exact degree-sum check.")
    return canonical_degrees(rounded.tolist()) if sort_result else tuple(int(v) for v in rounded)


class PerturbedEmpiricalDegreeSampler:
    """One parent draw per sample; only kernel randomness uses a separate stream.

    ``sample(rng)`` consumes exactly one integer from rng to pick the parent.
    The mixture coin and perturbation kernel each have per-sample RNG streams.
    Thus differing rejection counts do not change later parent/mixture draws.
    """

    def __init__(self, degree_sequences: Sequence[Sequence[int]],
                 config: DegreePerturbationConfig | Mapping[str, Any] | None = None,
                 *, seed: int = 0, support_max_degree: int | None = None):
        self.config = config if isinstance(config, DegreePerturbationConfig) else DegreePerturbationConfig.from_dict(config)
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("seed must be nonnegative.")
        self.degree_sequences = tuple(canonical_degrees(s) for s in degree_sequences)
        if not self.degree_sequences:
            raise ValueError("Perturbation prior requires nonempty training degree sequences.")
        limits = [v for v in (self.config.max_degree, support_max_degree) if v is not None]
        self.max_degree = min(limits) if limits else None
        for i, seq in enumerate(self.degree_sequences):
            if not connected_feasible(seq, self.max_degree):
                raise ValueError(f"Training parent {i} is not connected-feasible within the degree support; parents are never clipped.")
        self.training_set = set(self.degree_sequences)
        grouped: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
        for seq in self.degree_sequences:
            grouped[(len(seq), sum(seq))].add(seq)
        self.groups = {k: sorted(v) for k, v in grouped.items()}
        self.records: list[dict[str, Any]] = []
        self._parent_rng = np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(3,)))
        self._neighbor_cache: dict[tuple[int, ...], list[tuple[int, ...]]] = {}

    @classmethod
    def fit_from_graphs(cls, graphs: Sequence[nx.Graph], config=None, *, seed=0, support_max_degree=None):
        # Extract invariants and then discard adjacency. In particular option 3
        # reconstructs its temporary witness from degrees, not training edges.
        seqs = []
        for graph in graphs:
            if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
                raise ValueError("Degree perturbation supports simple undirected generic graphs only.")
            if graph.number_of_nodes() == 0 or not nx.is_connected(graph):
                raise ValueError("Degree perturbation expects nonempty connected training graphs.")
            seqs.append([int(d) for _, d in graph.degree()])
        return cls(seqs, config, seed=seed, support_max_degree=support_max_degree)

    def _check(self, proposal, current, parent, visited, rejections, *, partner=None) -> tuple[int, ...] | None:
        seq = canonical_degrees(proposal)
        if seq == current or seq == partner:
            rejections["unchanged_multiset_or_copied_partner"] += 1
            return None
        if seq in visited:
            rejections["revisited_degree_multiset"] += 1
            return None
        if len(seq) != len(parent) or sum(seq) != sum(parent):
            raise AssertionError("Perturbation changed n or m.")
        if self.config.max_distance is not None and degree_distance(parent, seq) > self.config.max_distance:
            rejections["distance_limit"] += 1
            return None
        if self.config.require_novel and seq in self.training_set:
            rejections["already_in_training_degree_support"] += 1
            return None
        if not connected_feasible(seq, self.max_degree):
            rejections["non_graphical_or_outside_connected_support"] += 1
            return None
        return seq

    def _neighbors(self, current: tuple[int, ...]) -> list[tuple[int, ...]]:
        if current not in self._neighbor_cache:
            entries = []
            for seq in self.groups.get((len(current), sum(current)), []):
                distance = degree_distance(current, seq)
                if distance > 0 and (self.config.interpolation_max_parent_distance is None or distance <= self.config.interpolation_max_parent_distance):
                    entries.append((distance, seq))
            entries.sort()
            # Include all equally-close ties at the k-th boundary.
            if len(entries) > self.config.interpolation_neighbors:
                cutoff = entries[self.config.interpolation_neighbors - 1][0]
                entries = [e for e in entries if e[0] <= cutoff]
            self._neighbor_cache[current] = [e[1] for e in entries]
        return self._neighbor_cache[current]

    def _one_step(self, current, parent, visited, rng, rejections, witness):
        cfg, n = self.config, len(current)
        checks = 0
        if n <= 1:
            return None, witness, checks, "singleton_has_no_degree_perturbation", {}
        if cfg.method == "unit_transfer":
            pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
            for pos in rng.permutation(len(pairs))[:cfg.max_attempts]:
                i, j = pairs[int(pos)]
                proposal = list(current)
                proposal[i] -= 1
                proposal[j] += 1
                checks += 1
                accepted = self._check(proposal, current, parent, visited, rejections)
                if accepted is not None:
                    return accepted, witness, checks, None, {"donor_degree": current[i], "recipient_degree": current[j]}
            reason = "no_valid_unit_transfer" if len(pairs) <= cfg.max_attempts else "candidate_budget_exhausted"
            return None, witness, checks, reason, {}

        if cfg.method == "moment_preserving":
            if n < cfg.block_size:
                return None, witness, 0, "too_few_nodes_for_block", {}
            blocks = _available_blocks(current, cfg.block_size, cfg.max_block_patterns)
            ceiling = min(n - 1, self.max_degree) if self.max_degree is not None else n - 1
            for pos in rng.permutation(len(blocks)):
                old = blocks[int(pos)]
                replacements = [b for b in _moment_blocks(len(old), sum(old), sum(d*d for d in old), ceiling, cfg.max_block_alternatives) if b != old]
                for alt_pos in rng.permutation(len(replacements)):
                    if checks >= cfg.max_attempts:
                        return None, witness, checks, "candidate_budget_exhausted", {}
                    new = replacements[int(alt_pos)]
                    proposal = list(current)
                    for value in old:
                        proposal.remove(value)
                    proposal.extend(new)
                    checks += 1
                    accepted = self._check(proposal, current, parent, visited, rejections)
                    if accepted is not None:
                        assert sum(d*d for d in accepted) == sum(d*d for d in current)
                        return accepted, witness, checks, None, {"old_block": list(old), "new_block": list(new)}
            return None, witness, checks, "no_valid_moment_preserving_block", {}

        if cfg.method == "edge_relocation":
            if witness is None:
                try:
                    witness = construct_coarse_graph(degree_summary(current),
                        {"ensure_connected": True, "random_relabel": False}, rng)
                except (ValueError, RuntimeError, AssertionError) as exc:
                    return None, None, 0, "temporary_witness_constructor_failed", {"error": str(exc)}
            actions = []
            for a, b in sorted(witness.edges()):
                for u, v in ((a, b), (b, a)):
                    if witness.degree(u) <= 1:
                        continue
                    for w in sorted(witness.nodes()):
                        if w not in (u, v) and not witness.has_edge(w, v):
                            actions.append((u, v, w))
            for pos in rng.permutation(len(actions))[:cfg.max_attempts]:
                u, v, w = actions[int(pos)]
                candidate = witness.copy()
                candidate.remove_edge(u, v)
                candidate.add_edge(w, v)
                checks += 1
                if not nx.is_connected(candidate):
                    rejections["edge_relocation_disconnects_witness"] += 1
                    continue
                accepted = self._check([int(d) for _, d in candidate.degree()], current, parent, visited, rejections)
                if accepted is not None:
                    return accepted, candidate, checks, None, {"removed_edge": [u, v], "inserted_edge": [w, v], "witness_connected": True}
            reason = "no_valid_edge_relocation" if len(actions) <= cfg.max_attempts else "candidate_budget_exhausted"
            return None, witness, checks, reason, {}

        neighbors = self._neighbors(current)
        if not neighbors:
            return None, witness, 0, "no_distinct_training_partner_same_n_m", {}
        for _ in range(cfg.max_attempts):
            partner = neighbors[int(rng.integers(len(neighbors)))]
            x = (1 - cfg.interpolation_alpha) * np.asarray(current) + cfg.interpolation_alpha * np.asarray(partner)
            proposal = sum_preserving_round(x, sum(current), rng)
            checks += 1
            accepted = self._check(proposal, current, parent, visited, rejections, partner=partner)
            if accepted is not None:
                return accepted, witness, checks, None, {"partner_degree_sequence": list(partner),
                    "alpha": cfg.interpolation_alpha, "parent_partner_distance": degree_distance(current, partner)}
        return None, witness, checks, "no_valid_nonparent_interpolation_within_budget", {}

    def sample(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        parent_rng = self._parent_rng if rng is None else rng
        parent_index = int(parent_rng.integers(len(self.degree_sequences)))
        return self.perturb_parent(parent_index)

    def perturb_parent(self, parent_index: int, *, requested: bool | None = None) -> dict[str, Any]:
        """Perturb an explicit training parent; useful for paired prior diagnostics."""
        if parent_index < 0 or parent_index >= len(self.degree_sequences):
            raise IndexError("Training parent index out of range.")
        sample_index = len(self.records)
        coin_rng = np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(4, sample_index, 0)))
        kernel_rng = np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(4, sample_index, 1)))
        selected = bool(coin_rng.random() < self.config.probability) if requested is None else bool(requested)
        parent = self.degree_sequences[parent_index]
        current, visited, witness = parent, {parent}, None
        checks, operations = 0, []
        rejections: Counter[str] = Counter()
        failure = None
        if selected:
            for _ in range(self.config.steps):
                result, witness, count, failure, info = self._one_step(current, parent, visited, kernel_rng, rejections, witness)
                checks += count
                if result is None:
                    break
                current = result
                visited.add(current)
                operations.append(info)
        failed = selected and failure is not None
        # Transactional semantics: incomplete multi-step proposals roll back.
        # Partial changes are not mislabeled as the requested perturbation.
        if failed:
            current = parent
        changed = current != parent
        record = {
            "sample_index": sample_index, "method": self.config.method, "parent_train_index": int(parent_index),
            "parent_degree_sequence": list(parent), "degree_sequence": list(current),
            "requested": selected, "changed": changed, "failure_reason": failure,
            "failure_policy": self.config.failure_policy,
            "fallback_used": bool(failed and self.config.failure_policy == "keep_original"),
            "candidate_checks": checks, "proposal_rejections": dict(sorted(rejections.items())),
            "requested_steps": self.config.steps if selected else 0,
            "accepted_steps": len(operations) if not failed else 0,
            "rolled_back_steps": len(operations) if failed else 0,
            "distance_half_l1": degree_distance(parent, current),
            "degree_l1_distance": int(2 * degree_distance(parent, current)),
            "novel_vs_training": current not in self.training_set,
            "num_nodes": len(current), "num_edges": sum(current) // 2,
            "preserved_n_m": len(current) == len(parent) and sum(current) == sum(parent),
            "parent_second_moment": sum(d*d for d in parent), "second_moment": sum(d*d for d in current),
            "preserved_second_moment": sum(d*d for d in parent) == sum(d*d for d in current),
            "connected_feasible": connected_feasible(current, self.max_degree),
            "operations": operations, "raw_graphical": True, "raw_connected_feasible": True,
            "repair_used": False, "attempts_used": 1,
        }
        self.records.append(record)
        if failed and self.config.failure_policy == "error":
            raise DegreePerturbationError(
                f"{self.config.method} failed for training parent {parent_index}: {failure}. "
                "No parent redraw or degree repair was performed. Explicitly use "
                "failure_policy=keep_original to record an identity transition."
            )
        summary = degree_summary(current)
        summary["sampling_diagnostics"] = record
        return summary

    def report(self) -> dict[str, Any]:
        rows = self.records
        requested = sum(r["requested"] for r in rows)
        changed = sum(r["changed"] for r in rows)
        failures = Counter(r["failure_reason"] for r in rows if r["failure_reason"])
        rejections: Counter[str] = Counter()
        for row in rows:
            rejections.update(row["proposal_rejections"])
        mean = lambda key: float(np.mean([r[key] for r in rows])) if rows else 0.0
        return {
            "format": "empirical_degree_perturbation_v1", "method": self.config.method,
            "training_only": True, "config": asdict(self.config), "effective_max_degree": self.max_degree,
            "num_training_graphs": len(self.degree_sequences), "num_training_degree_multisets": len(self.training_set),
            "training_degree_fingerprint": sequence_fingerprint(self.degree_sequences),
            "parent_degree_fingerprint": sequence_fingerprint(r["parent_degree_sequence"] for r in rows),
            "sampled_degree_fingerprint": sequence_fingerprint(r["degree_sequence"] for r in rows),
            "num_samples": len(rows), "num_requested": requested, "num_changed": changed,
            "requested_fraction": requested / len(rows) if rows else 0.0,
            "changed_fraction": changed / len(rows) if rows else 0.0,
            "success_given_requested": changed / requested if requested else None,
            "num_failed_requests": sum(failures.values()),
            "num_identity_fallbacks": sum(bool(r["fallback_used"]) for r in rows),
            "failure_reasons": dict(sorted(failures.items())),
            "proposal_rejections": dict(sorted(rejections.items())),
            "mean_distance_half_l1": mean("distance_half_l1"),
            "novel_degree_fraction": mean("novel_vs_training"),
            "num_unique_degree_multisets": len({tuple(r["degree_sequence"]) for r in rows}),
            "all_preserve_n_m": all(r["preserved_n_m"] for r in rows),
            "all_connected_feasible": all(r["connected_feasible"] for r in rows),
            "all_preserve_second_moment": all(r["preserved_second_moment"] for r in rows),
            "rng": "parent=caller; mixture=(seed,4,sample,0); kernel=(seed,4,sample,1)",
            "records": rows,
        }
