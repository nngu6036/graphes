"""Stochastic same-bond-type diversification for attributed GraphER samples.

The input graph is already a complete attributed sample.  A move chooses two
edges with the same configured edge category and applies a valid double-edge
swap while assigning that same edge category to the two inserted edges.  The
strict variant additionally requires every node to keep the same multiset of
``(neighbor atom type, bond type)`` incidences.  Hence node attributes, indexed
ordinary degrees, indexed typed degrees, local atom-pair/bond environments, the
global edge-category histogram, and per-node weighted bond valence are hard
invariants.  Simplicity/connectivity are checked before a move is scored.
Optional RDKit filtering provides an additional molecular-validity guard.

Candidate selection is deliberately stochastic.  Predicted topology and typed
structure summaries are frozen at the pre-rewiring endpoint and define a trust
region; they guide exploration rather than forcing greedy repair.
"""
from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np

from grapher.properties.summary import SummaryConfig
from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary, GraphletBasis
from grapher.rewiring_mlp.core.rewiring import candidate_actions_from_edge_pair, is_valid_action
from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.graphlets import (
    candidate_topology_graphlet_counts,
    extract_topology_graphlet_counts,
    topology_structural_discrepancy_from_counts,
)
from grapher.rewiring_mlp.molecular.constraints import (
    QM9_PROJECTED_MAX_VALENCE,
    bond_order,
)
from grapher.rewiring_mlp.molecular.typed_invariants import (
    extract_typed_invariant,
    typed_invariant_matches_graph,
)


DEFAULT_ATTRIBUTED_DIVERSIFICATION: dict[str, Any] = {
    "enabled": False,
    "mode": "stochastic_same_bond_diversification",
    "random_steps": {
        "distribution": "uniform_integer",
        "min_steps": 0,
        "max_steps": 4,
        "mean_steps": 2.0,
    },
    "proposal_budget": 128,
    "valid_candidate_budget": 48,
    "preserve_connectivity": True,
    "strict_same_bond": True,
    # Stronger molecular locality invariant: for every node preserve the
    # number of neighbours of each atom type under each bond type.
    "preserve_neighbor_atom_type_by_bond_type": True,
    # Same-bond swaps already imply exact weighted bond-valence preservation;
    # keep an explicit check/diagnostic so this contract is audited at runtime.
    "verify_bond_valence_preservation": True,
    "rdkit_candidate_filter": True,
    "reject_revisited_states": True,
    "normalization": "initial",
    "epsilon": 1.0e-6,
    "guidance": {
        "topology_graphlet_weight": 0.50,
        "topology_graphlet_mass_weight": 0.05,
        "topology_orbit_weight": 0.25,
        "typed_graphlet_weight": 1.00,
        "typed_graphlet_mass_weight": 0.10,
    },
    "soft_selection": {
        "temperature_start": 0.75,
        "temperature_end": 0.25,
        "temperature_schedule": "linear",
        "score_normalization": "std",
        "max_target_worsening": 0.02,
        "max_relative_target_worsening": 0.25,
    },
    "target_trust_region": {
        "enabled": True,
        "absolute_worsening": 0.02,
        "relative_worsening": 0.25,
    },
}


def _deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_update(dict(out[key]), value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_options(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = _deep_update(DEFAULT_ATTRIBUTED_DIVERSIFICATION, dict(raw or {}))
    validate_options(cfg)
    return cfg


def validate_options(cfg: Mapping[str, Any]) -> None:
    allowed = {
        "enabled", "mode", "random_steps", "proposal_budget",
        "valid_candidate_budget", "preserve_connectivity", "strict_same_bond",
        "preserve_neighbor_atom_type_by_bond_type",
        "verify_bond_valence_preservation",
        "rdkit_candidate_filter", "reject_revisited_states", "normalization",
        "epsilon", "guidance", "soft_selection", "target_trust_region",
    }
    unknown = sorted(set(cfg) - allowed)
    if unknown:
        raise ValueError(f"Unknown attributed_diversification settings: {unknown}")
    if not bool(cfg.get("enabled", False)):
        return
    if str(cfg.get("mode", "")).lower() != "stochastic_same_bond_diversification":
        raise ValueError(
            "attributed_diversification.mode must be stochastic_same_bond_diversification"
        )
    if not bool(cfg.get("strict_same_bond", True)):
        raise ValueError("Attributed diversification requires strict_same_bond=true")
    if not bool(cfg.get("preserve_neighbor_atom_type_by_bond_type", True)):
        raise ValueError(
            "Strict molecular diversification requires "
            "preserve_neighbor_atom_type_by_bond_type=true"
        )
    if not bool(cfg.get("verify_bond_valence_preservation", True)):
        raise ValueError(
            "Strict molecular diversification requires "
            "verify_bond_valence_preservation=true"
        )
    if not bool(cfg.get("preserve_connectivity", True)):
        raise ValueError("Attributed diversification requires preserve_connectivity=true")
    for key in ("proposal_budget", "valid_candidate_budget"):
        if int(cfg.get(key, 0)) <= 0:
            raise ValueError(f"attributed_diversification.{key} must be positive")
    eps = float(cfg.get("epsilon", 0.0))
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("attributed_diversification.epsilon must be finite and positive")
    if str(cfg.get("normalization", "initial")).lower() not in {"initial", "none"}:
        raise ValueError("attributed_diversification.normalization must be initial or none")

    random_steps = dict(cfg.get("random_steps", {}) or {})
    distribution = str(random_steps.get("distribution", "uniform_integer")).lower()
    if distribution not in {"uniform_integer", "poisson"}:
        raise ValueError("random_steps.distribution must be uniform_integer or poisson")
    minimum = int(random_steps.get("min_steps", 0))
    maximum = int(random_steps.get("max_steps", 0))
    if minimum < 0 or maximum < minimum:
        raise ValueError("random_steps requires 0 <= min_steps <= max_steps")
    if distribution == "poisson" and float(random_steps.get("mean_steps", -1.0)) < 0.0:
        raise ValueError("random_steps.mean_steps must be nonnegative")

    guidance = dict(cfg.get("guidance", {}) or {})
    allowed_guidance = {
        "topology_graphlet_weight", "topology_graphlet_mass_weight",
        "topology_orbit_weight", "typed_graphlet_weight",
        "typed_graphlet_mass_weight",
    }
    unknown_guidance = sorted(set(guidance) - allowed_guidance)
    if unknown_guidance:
        raise ValueError(f"Unknown attributed diversification guidance weights: {unknown_guidance}")
    weights = [float(guidance.get(key, 0.0)) for key in allowed_guidance]
    if any((not math.isfinite(value) or value < 0.0) for value in weights):
        raise ValueError("Attributed diversification guidance weights must be finite and nonnegative")
    if sum(weights) <= 0.0:
        raise ValueError("Attributed diversification requires at least one positive guidance weight")

    soft = dict(cfg.get("soft_selection", {}) or {})
    if str(soft.get("temperature_schedule", "linear")).lower() not in {"constant", "linear", "cosine"}:
        raise ValueError("soft_selection.temperature_schedule must be constant, linear, or cosine")
    if str(soft.get("score_normalization", "std")).lower() not in {"none", "std"}:
        raise ValueError("soft_selection.score_normalization must be none or std")
    for key in ("temperature_start", "temperature_end"):
        value = float(soft.get(key, 0.0))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"soft_selection.{key} must be finite and positive")


def sample_step_budget(rng: np.random.Generator, cfg: Mapping[str, Any]) -> int:
    values = dict(cfg.get("random_steps", {}) or {})
    minimum = int(values.get("min_steps", 0))
    maximum = int(values.get("max_steps", minimum))
    distribution = str(values.get("distribution", "uniform_integer")).lower()
    if distribution == "uniform_integer":
        return int(rng.integers(minimum, maximum + 1))
    mean = float(values.get("mean_steps", 0.5 * (minimum + maximum)))
    return int(np.clip(rng.poisson(mean), minimum, maximum))


def _temperature(cfg: Mapping[str, Any], progress: float) -> float:
    soft = dict(cfg.get("soft_selection", {}) or {})
    start = float(soft.get("temperature_start", 0.75))
    end = float(soft.get("temperature_end", start))
    p = float(np.clip(progress, 0.0, 1.0))
    schedule = str(soft.get("temperature_schedule", "linear")).lower()
    if schedule == "constant":
        return start
    if schedule == "linear":
        return start + (end - start) * p
    cooling = 0.5 * (1.0 + np.cos(np.pi * p))
    return end + (start - end) * cooling


def _state_key(graph: nx.Graph, edge_attribute: str) -> tuple[Any, ...]:
    edges = tuple(
        sorted(
            (min(int(u), int(v)), max(int(u), int(v)), repr(data.get(edge_attribute)))
            for u, v, data in graph.edges(data=True)
        )
    )
    return edges


def _neighbor_atom_bond_signature(
    graph: nx.Graph,
    *,
    node_attribute: str,
    edge_attribute: str,
) -> tuple[Any, ...]:
    """Return an indexed radius-one chemical-environment signature.

    For each node this records its own atom type and a multiset of
    ``(neighbor_atom_type, bond_type)`` incidences.  Equality therefore means
    that a rewiring cannot replace, for example, a C--C single-bond neighbour
    by an N neighbour even though ordinary/typed bond degrees are unchanged.
    """

    rows: list[Any] = []
    for node in sorted(graph.nodes()):
        data = graph.nodes[node]
        if node_attribute not in data:
            raise KeyError(f"Node {node!r} is missing {node_attribute!r}")
        counts: Counter = Counter()
        for neighbor, edge_data in graph[node].items():
            if node_attribute not in graph.nodes[neighbor]:
                raise KeyError(f"Node {neighbor!r} is missing {node_attribute!r}")
            if edge_attribute not in edge_data:
                raise KeyError(f"Edge {(node, neighbor)!r} is missing {edge_attribute!r}")
            counts[(graph.nodes[neighbor][node_attribute], edge_data[edge_attribute])] += 1
        environment = tuple(
            sorted(
                ((atom_type, bond_type, int(count))
                 for (atom_type, bond_type), count in counts.items()),
                key=lambda value: (repr(value[0]), repr(value[1]), value[2]),
            )
        )
        rows.append((data[node_attribute], environment))
    return tuple(rows)


def _weighted_bond_valence_vector(
    graph: nx.Graph,
    *,
    edge_attribute: str,
) -> tuple[float, ...]:
    """Per-node bond-order sums used to audit exact bond-valence invariance."""

    values: list[float] = []
    for node in sorted(graph.nodes()):
        total = 0.0
        for _, _, edge_data in graph.edges(node, data=True):
            if edge_attribute not in edge_data:
                raise KeyError(f"Incident edge at node {node!r} is missing {edge_attribute!r}")
            total += float(bond_order(int(edge_data[edge_attribute])))
        values.append(float(total))
    return tuple(values)


def _bond_valence_vectors_equal(
    left: tuple[float, ...],
    right: tuple[float, ...],
    *,
    atol: float = 1.0e-10,
) -> bool:
    if len(left) != len(right):
        return False
    return bool(np.allclose(np.asarray(left), np.asarray(right), rtol=0.0, atol=atol))


def _qm9_projected_bond_valence_valid(
    graph: nx.Graph,
    *,
    node_attribute: str,
    edge_attribute: str,
) -> bool | None:
    """Check the projected QM9 valence envelope when all atoms are supported.

    Returns ``None`` for non-QM9 atom vocabularies (for example ZINC atoms P,
    S, Cl, Br, I), because applying the QM9 envelope there would be invalid.
    """

    valence = _weighted_bond_valence_vector(graph, edge_attribute=edge_attribute)
    for index, node in enumerate(sorted(graph.nodes())):
        atom = graph.nodes[node].get(node_attribute)
        if atom not in QM9_PROJECTED_MAX_VALENCE:
            return None
        if valence[index] > float(QM9_PROJECTED_MAX_VALENCE[atom]) + 1.0e-10:
            return False
    return True


def _candidate_graphs(
    current: nx.Graph,
    *,
    edge_attribute: str,
    node_attribute: str,
    neighbor_atom_bond_invariant: tuple[Any, ...],
    bond_valence_invariant: tuple[float, ...],
    cfg: Mapping[str, Any],
    rng: np.random.Generator,
    seen: set[tuple[Any, ...]],
    stats: Counter,
):
    groups: dict[Any, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    for u, v, data in current.edges(data=True):
        if edge_attribute not in data:
            raise KeyError(f"Edge {(u, v)!r} is missing {edge_attribute!r}")
        groups[data[edge_attribute]].append(
            (min(int(u), int(v)), max(int(u), int(v)), copy.deepcopy(dict(data)))
        )
    pairs = [
        (edge_type, left, right)
        for edge_type, edges in groups.items()
        for left, right in combinations(edges, 2)
    ]
    if not pairs:
        return
    order = rng.permutation(len(pairs))
    proposed = 0
    valid = 0
    proposal_budget = int(cfg.get("proposal_budget", 128))
    valid_budget = int(cfg.get("valid_candidate_budget", 48))
    preserve = bool(cfg.get("preserve_connectivity", True))

    for pair_index in order:
        edge_type, left, right = pairs[int(pair_index)]
        e1 = (left[0], left[1])
        e2 = (right[0], right[1])
        actions = list(candidate_actions_from_edge_pair(e1, e2))
        rng.shuffle(actions)
        for action in actions:
            if proposed >= proposal_budget or valid >= valid_budget:
                return
            proposed += 1
            stats["proposals"] += 1
            if not is_valid_action(current, action, preserve_connectivity=preserve):
                stats["topology_rejections"] += 1
                continue
            candidate = current.copy()
            removed, added = action
            candidate.remove_edges_from(removed)
            # Both removed edges share the same primary edge category. Preserve
            # the complete edge-attribute multiset by carrying one removed edge
            # record to each inserted edge.
            payloads = [copy.deepcopy(left[2]), copy.deepcopy(right[2])]
            for (u, v), payload in zip(added, payloads):
                payload[edge_attribute] = edge_type
                candidate.add_edge(int(u), int(v), **payload)
            key = _state_key(candidate, edge_attribute)
            if bool(cfg.get("reject_revisited_states", True)) and key in seen:
                stats["visited_rejections"] += 1
                continue
            if bool(cfg.get("preserve_neighbor_atom_type_by_bond_type", True)):
                observed_local = _neighbor_atom_bond_signature(
                    candidate,
                    node_attribute=node_attribute,
                    edge_attribute=edge_attribute,
                )
                if observed_local != neighbor_atom_bond_invariant:
                    stats["neighbor_atom_bond_signature_rejections"] += 1
                    continue
            if bool(cfg.get("verify_bond_valence_preservation", True)):
                observed_valence = _weighted_bond_valence_vector(
                    candidate, edge_attribute=edge_attribute
                )
                if not _bond_valence_vectors_equal(
                    observed_valence, bond_valence_invariant
                ):
                    stats["bond_valence_rejections"] += 1
                    continue
            if bool(cfg.get("rdkit_candidate_filter", True)):
                from grapher.rewiring_mlp.molecular.graph_io import is_valid_molecular_graph

                if not is_valid_molecular_graph(candidate):
                    stats["rdkit_rejections"] += 1
                    continue
            valid += 1
            stats["valid_candidates_scored"] += 1
            yield action, candidate, key


def _typed_graphlet_summary(
    graph: nx.Graph,
    basis: GraphletBasis,
    summary_cfg: SummaryConfig,
) -> tuple[np.ndarray, np.ndarray]:
    history, mass = basis.statistics_for_graph(
        graph, summary_cfg, rng=np.random.default_rng(0)
    )
    return (
        basis.flatten_history(history).astype(np.float64),
        basis.flatten_mass(mass).astype(np.float64),
    )


def _block_tv(left: np.ndarray, right: np.ndarray, slices) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"Graphlet target width mismatch: {a.shape} != {b.shape}")
    values = [0.5 * float(np.abs(a[s:e] - b[s:e]).sum()) for s, e in slices]
    return float(np.mean(values)) if values else 0.0


def _typed_discrepancy(
    graph: nx.Graph,
    *,
    basis: GraphletBasis,
    summary_cfg: SummaryConfig,
    target_hist: np.ndarray,
    target_mass: np.ndarray,
) -> tuple[float, float]:
    hist, mass = _typed_graphlet_summary(graph, basis, summary_cfg)
    hist_distance = _block_tv(hist, target_hist, basis.slices)
    expected_mass = np.asarray(target_mass, dtype=np.float64).reshape(-1)
    if mass.shape != expected_mass.shape:
        raise ValueError("Typed graphlet mass target width mismatch")
    mass_distance = float(np.mean(np.abs(mass - expected_mass))) if mass.size else 0.0
    return hist_distance, mass_distance


def _energy_from_parts(parts: Mapping[str, float], scales: Mapping[str, float], weights: Mapping[str, float]) -> float:
    return float(
        sum(float(weights.get(name, 0.0)) * float(value) / float(scales.get(name, 1.0))
            for name, value in parts.items())
    )


def _softmax_choice(
    rng: np.random.Generator,
    energies: np.ndarray,
    *,
    temperature: float,
    normalization: str,
) -> int:
    values = np.asarray(energies, dtype=np.float64)
    if normalization == "std" and values.size > 1:
        scale = float(values.std())
        if scale > 1.0e-12:
            values = (values - float(values.mean())) / scale
    logits = -values / max(float(temperature), 1.0e-12)
    logits -= float(logits.max())
    probabilities = np.exp(logits)
    probabilities /= float(probabilities.sum())
    return int(rng.choice(len(values), p=probabilities))


def diversify_same_bond_graph(
    source: nx.Graph,
    *,
    topology_prediction: Mapping[str, Any],
    attribute_prediction: Mapping[str, Any],
    topology_basis: TopologyGraphletBasis,
    typed_basis: GraphletBasis,
    topology_summary_config: SummaryConfig,
    typed_summary_config: SummaryConfig,
    vocabulary: GraphCategoryVocabulary,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[nx.Graph, dict[str, Any]]:
    """Sample a same-bond rewiring trajectory within a frozen summary region."""

    cfg = resolve_options(config)
    if not bool(cfg.get("enabled", False)):
        return source.copy(), {
            "requested_steps": 0,
            "accepted_steps": 0,
            "stop_reason": "disabled",
            "changed": False,
        }
    if vocabulary.node_attribute is None or vocabulary.edge_attribute is None:
        raise ValueError("Attributed diversification requires node and edge attributes")
    current = nx.convert_node_labels_to_integers(
        nx.Graph(source), first_label=0, ordering="sorted"
    )
    if current.number_of_nodes() > 1 and not nx.is_connected(current):
        return current, {
            "requested_steps": 0,
            "accepted_steps": 0,
            "stop_reason": "disconnected_source_skipped",
            "changed": False,
            "typed_degree_preserved": True,
            "degree_preserved": True,
            "neighbor_atom_bond_signature_preserved": True,
            "bond_valence_preserved": True,
            "connected": False,
        }

    invariant = extract_typed_invariant(
        current,
        edge_types=vocabulary.edge_values,
        node_attribute=str(vocabulary.node_attribute),
        edge_attribute=str(vocabulary.edge_attribute),
    )
    neighbor_atom_bond_invariant = _neighbor_atom_bond_signature(
        current,
        node_attribute=str(vocabulary.node_attribute),
        edge_attribute=str(vocabulary.edge_attribute),
    )
    bond_valence_invariant = _weighted_bond_valence_vector(
        current, edge_attribute=str(vocabulary.edge_attribute)
    )
    source_qm9_bond_valence_valid = _qm9_projected_bond_valence_valid(
        current,
        node_attribute=str(vocabulary.node_attribute),
        edge_attribute=str(vocabulary.edge_attribute),
    )
    source_rdkit_valid: bool | None = None
    if bool(cfg.get("rdkit_candidate_filter", True)):
        from grapher.rewiring_mlp.molecular.graph_io import is_valid_molecular_graph

        source_rdkit_valid = bool(is_valid_molecular_graph(current))
    requested_steps = sample_step_budget(rng, cfg)
    if requested_steps <= 0:
        return current, {
            "requested_steps": requested_steps,
            "accepted_steps": 0,
            "stop_reason": "zero_budget",
            "changed": False,
            "typed_degree_preserved": True,
            "degree_preserved": True,
            "neighbor_atom_bond_signature_preserved": True,
            "bond_valence_preserved": True,
            "source_qm9_bond_valence_valid": source_qm9_bond_valence_valid,
            "final_qm9_bond_valence_valid": source_qm9_bond_valence_valid,
            "source_rdkit_valid": source_rdkit_valid,
            "final_rdkit_valid": source_rdkit_valid,
            "rdkit_validity_preserved_for_valid_source": (
                True if source_rdkit_valid is True else None
            ),
            "connected": True,
        }

    top_target = np.asarray(topology_prediction["graphlet_histogram"], dtype=np.float64)
    top_mass_target = np.asarray(topology_prediction["graphlet_mass"], dtype=np.float64)
    orbit_target = np.asarray(
        topology_prediction.get("orbit_mean_counts", np.zeros(0)), dtype=np.float64
    )
    typed_target = np.asarray(
        attribute_prediction["typed_graphlet_histogram"], dtype=np.float64
    )
    typed_mass_target = np.asarray(
        attribute_prediction["typed_graphlet_mass"], dtype=np.float64
    )

    guidance = dict(cfg.get("guidance", {}) or {})
    weights = {
        "topology_graphlet": float(guidance.get("topology_graphlet_weight", 0.0)),
        "topology_mass": float(guidance.get("topology_graphlet_mass_weight", 0.0)),
        "orbit": float(guidance.get("topology_orbit_weight", 0.0)),
        "typed_graphlet": float(guidance.get("typed_graphlet_weight", 0.0)),
        "typed_mass": float(guidance.get("typed_graphlet_mass_weight", 0.0)),
    }

    top_counts = extract_topology_graphlet_counts(current, graphlet_basis=topology_basis)

    def score(graph: nx.Graph, counts) -> dict[str, float]:
        top = topology_structural_discrepancy_from_counts(
            graph,
            counts,
            graphlet_target=top_target,
            graphlet_mass_target=top_mass_target,
            clustering_target=np.zeros(0, dtype=np.float64),
            orbit_target=orbit_target,
            graphlet_basis=topology_basis,
            graphlet_weight=1.0,
            graphlet_mass_weight=0.0,
            clustering_weight=0.0,
            orbit_weight=1.0 if weights["orbit"] > 0.0 and orbit_target.size else 0.0,
        )
        typed_hist, typed_mass = _typed_discrepancy(
            graph,
            basis=typed_basis,
            summary_cfg=typed_summary_config,
            target_hist=typed_target,
            target_mass=typed_mass_target,
        )
        return {
            "topology_graphlet": float(top["graphlet_histogram"]),
            "topology_mass": float(top["graphlet_mass"]),
            "orbit": float(top["orbit"]),
            "typed_graphlet": float(typed_hist),
            "typed_mass": float(typed_mass),
        }

    initial_parts = score(current, top_counts)
    eps = float(cfg.get("epsilon", 1.0e-6))
    if str(cfg.get("normalization", "initial")).lower() == "initial":
        scales = {name: max(float(value), eps) for name, value in initial_parts.items()}
    else:
        scales = {name: 1.0 for name in initial_parts}
    initial_energy = _energy_from_parts(initial_parts, scales, weights)
    current_parts = dict(initial_parts)
    current_energy = float(initial_energy)
    seen = {_state_key(current, str(vocabulary.edge_attribute))}
    trace: list[dict[str, Any]] = []
    search_totals: Counter = Counter()
    stop_reason = "step_limit"
    trust = dict(cfg.get("target_trust_region", {}) or {})
    soft = dict(cfg.get("soft_selection", {}) or {})

    for step in range(requested_steps):
        candidate_rows = []
        step_stats: Counter = Counter()
        for action, candidate, key in _candidate_graphs(
            current,
            edge_attribute=str(vocabulary.edge_attribute),
            node_attribute=str(vocabulary.node_attribute),
            neighbor_atom_bond_invariant=neighbor_atom_bond_invariant,
            bond_valence_invariant=bond_valence_invariant,
            cfg=cfg,
            rng=rng,
            seen=seen,
            stats=step_stats,
        ) or ():
            candidate_counts = candidate_topology_graphlet_counts(
                current,
                candidate,
                action,
                current_counts=top_counts,
                graphlet_basis=topology_basis,
            )
            parts = score(candidate, candidate_counts)
            energy = _energy_from_parts(parts, scales, weights)

            max_abs = float(soft.get("max_target_worsening", 0.02))
            max_rel = float(soft.get("max_relative_target_worsening", 0.25))
            step_limit = current_energy + max_abs
            if max_rel >= 0.0:
                step_limit += max_rel * max(abs(current_energy), eps)
            if energy > step_limit + 1.0e-12:
                step_stats["step_worsening_rejections"] += 1
                continue

            if bool(trust.get("enabled", True)):
                bound = initial_energy + float(trust.get("absolute_worsening", 0.02))
                relative = float(trust.get("relative_worsening", 0.25))
                if relative >= 0.0:
                    bound += relative * max(abs(initial_energy), eps)
                if energy > bound + 1.0e-12:
                    step_stats["trust_region_rejections"] += 1
                    continue
            candidate_rows.append((energy, candidate, key, action, parts, candidate_counts))

        search_totals.update(step_stats)
        if not candidate_rows:
            stop_reason = "no_candidate_inside_structural_trust_region"
            break
        temperature = _temperature(cfg, step / max(requested_steps - 1, 1))
        choice = _softmax_choice(
            rng,
            np.asarray([row[0] for row in candidate_rows], dtype=np.float64),
            temperature=temperature,
            normalization=str(soft.get("score_normalization", "std")).lower(),
        )
        energy, candidate, key, action, parts, candidate_counts = candidate_rows[choice]
        if not typed_invariant_matches_graph(candidate, invariant):
            raise AssertionError("Same-bond swap changed indexed node/typed-degree invariant")
        if _neighbor_atom_bond_signature(
            candidate,
            node_attribute=str(vocabulary.node_attribute),
            edge_attribute=str(vocabulary.edge_attribute),
        ) != neighbor_atom_bond_invariant:
            raise AssertionError(
                "Strict same-bond swap changed neighbor-atom-type-by-bond-type signature"
            )
        if not _bond_valence_vectors_equal(
            _weighted_bond_valence_vector(
                candidate, edge_attribute=str(vocabulary.edge_attribute)
            ),
            bond_valence_invariant,
        ):
            raise AssertionError("Strict same-bond swap changed per-node bond valence")
        if candidate.number_of_nodes() > 1 and not nx.is_connected(candidate):
            raise AssertionError("Same-bond diversification disconnected the graph")
        trace.append(
            {
                "step": step + 1,
                "temperature": float(temperature),
                "energy_before": float(current_energy),
                "energy_after": float(energy),
                "discrepancies_before": dict(current_parts),
                "discrepancies_after": dict(parts),
                "removed": [list(edge) for edge in action[0]],
                "added": [list(edge) for edge in action[1]],
                "candidate_search": dict(step_stats),
                "candidates_inside_trust_region": len(candidate_rows),
            }
        )
        current = candidate
        current_parts = dict(parts)
        current_energy = float(energy)
        top_counts = candidate_counts
        seen.add(key)

    preserved = typed_invariant_matches_graph(current, invariant)
    if not preserved:
        raise AssertionError("Attributed diversification violated typed-degree preservation")
    neighbor_preserved = (
        _neighbor_atom_bond_signature(
            current,
            node_attribute=str(vocabulary.node_attribute),
            edge_attribute=str(vocabulary.edge_attribute),
        )
        == neighbor_atom_bond_invariant
    )
    if not neighbor_preserved:
        raise AssertionError(
            "Attributed diversification violated neighbor-atom-type-by-bond-type preservation"
        )
    final_bond_valence = _weighted_bond_valence_vector(
        current, edge_attribute=str(vocabulary.edge_attribute)
    )
    bond_valence_preserved = _bond_valence_vectors_equal(
        final_bond_valence, bond_valence_invariant
    )
    if not bond_valence_preserved:
        raise AssertionError("Attributed diversification violated bond-valence preservation")
    final_qm9_bond_valence_valid = _qm9_projected_bond_valence_valid(
        current,
        node_attribute=str(vocabulary.node_attribute),
        edge_attribute=str(vocabulary.edge_attribute),
    )
    final_rdkit_valid: bool | None = None
    if bool(cfg.get("rdkit_candidate_filter", True)):
        from grapher.rewiring_mlp.molecular.graph_io import is_valid_molecular_graph

        final_rdkit_valid = bool(is_valid_molecular_graph(current))
        if source_rdkit_valid is True and not final_rdkit_valid:
            raise AssertionError(
                "RDKit-valid source became invalid after strict molecular rewiring"
            )
    changed = _state_key(current, str(vocabulary.edge_attribute)) != _state_key(source, str(vocabulary.edge_attribute))
    return current, {
        "requested_steps": int(requested_steps),
        "accepted_steps": len(trace),
        "stop_reason": stop_reason,
        "changed": bool(changed),
        "typed_degree_preserved": True,
        "degree_preserved": True,
        "neighbor_atom_bond_signature_preserved": bool(neighbor_preserved),
        "bond_valence_preserved": bool(bond_valence_preserved),
        "source_bond_valence_vector": [float(value) for value in bond_valence_invariant],
        "final_bond_valence_vector": [float(value) for value in final_bond_valence],
        "source_qm9_bond_valence_valid": source_qm9_bond_valence_valid,
        "final_qm9_bond_valence_valid": final_qm9_bond_valence_valid,
        "qm9_bond_valence_validity_preserved": (
            None
            if source_qm9_bond_valence_valid is None
            else (not source_qm9_bond_valence_valid or bool(final_qm9_bond_valence_valid))
        ),
        "source_rdkit_valid": source_rdkit_valid,
        "final_rdkit_valid": final_rdkit_valid,
        "rdkit_validity_preserved_for_valid_source": (
            None
            if source_rdkit_valid is not True
            else bool(final_rdkit_valid)
        ),
        "node_types_preserved": True,
        "bond_type_counts_preserved": True,
        "connected": bool(current.number_of_nodes() <= 1 or nx.is_connected(current)),
        "initial_discrepancies": initial_parts,
        "final_discrepancies": current_parts,
        "initial_energy": float(initial_energy),
        "final_energy": float(current_energy),
        "normalization_scales": scales,
        "weights": weights,
        "trace": trace,
        "candidate_search_totals": dict(search_totals),
        "target_scope": "frozen_topology_and_typed_summary_heads",
        "strict_same_bond": True,
        "preserve_neighbor_atom_type_by_bond_type": True,
        "verify_bond_valence_preservation": True,
    }
