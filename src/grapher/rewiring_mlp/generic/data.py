from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import networkx as nx
import numpy as np
import torch

from grapher.properties.summary import SummaryConfig
from grapher.rewiring_mlp.core.rewiring import Action, make_action
from grapher.rewiring_mlp.generic.graphlets import (
    candidate_topology_graphlet_counts,
    extract_topology_graphlet_counts,
    extract_topology_structural_target,
    topology_structural_discrepancy_from_counts,
)
from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.rewiring import propose_valid_topology_swaps


def normalize_topology_graph(graph: nx.Graph) -> nx.Graph:
    """Return a simple integer-labelled generic graph."""

    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("Topology GraphER requires a simple undirected graph.")
    out = nx.convert_node_labels_to_integers(
        nx.Graph(graph),
        first_label=0,
        ordering="sorted",
    )
    if nx.number_of_selfloops(out):
        raise ValueError("Topology GraphER does not support self-loops.")
    return out


@dataclass
class TopologyTrainingPair:
    """One completed base output explicitly coupled to one summary target."""

    source_graph: nx.Graph
    target_graph: nx.Graph
    base_generator: str
    source_index: int = -1
    target_index: int = -1
    split: str = "train"
    matching_method: str = "unspecified"
    matching_cost: float = 0.0
    source_graph_path: str | None = None
    source_manifest_path: str | None = None


@dataclass
class TopologyGraphletExample:
    current_graph: nx.Graph
    time: float
    graphlet_target: np.ndarray
    graphlet_mass_target: np.ndarray
    clustering_target: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    orbit_target: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    base_generator: str = "legacy_havel_hakimi"
    source_index: int = -1
    target_index: int = -1
    matching_cost: float = 0.0
    trajectory_id: int = -1
    step: int = -1
    teacher_actions: tuple[Action, ...] = ()
    teacher_distribution: np.ndarray | None = None
    teacher_selected_index: int = -1


@dataclass
class TopologyGraphletBatch:
    adjacency: torch.Tensor
    node_mask: torch.Tensor
    pair_mask: torch.Tensor
    degrees: torch.Tensor
    graph_size: torch.Tensor
    time: torch.Tensor
    graphlet_target: torch.Tensor
    graphlet_mass_target: torch.Tensor
    clustering_target: torch.Tensor
    orbit_target: torch.Tensor

    def to(self, device: torch.device | str) -> "TopologyGraphletBatch":
        return TopologyGraphletBatch(
            **{key: value.to(device) for key, value in self.__dict__.items()}
        )


def collate_topology_examples(
    examples: Sequence[TopologyGraphletExample],
) -> TopologyGraphletBatch:
    """Collate current topology states without terminal pair labels."""

    if not examples:
        raise ValueError("Cannot collate an empty topology batch.")
    max_nodes = max(example.current_graph.number_of_nodes() for example in examples)
    batch_size = len(examples)
    graphlet_width = int(np.asarray(examples[0].graphlet_target).size)
    mass_width = int(np.asarray(examples[0].graphlet_mass_target).size)
    clustering_width = int(np.asarray(examples[0].clustering_target).size)
    orbit_width = int(np.asarray(examples[0].orbit_target).size)

    adjacency = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.bool_)
    node_mask = np.zeros((batch_size, max_nodes), dtype=np.bool_)
    pair_mask = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.bool_)
    degrees = np.zeros((batch_size, max_nodes), dtype=np.float32)
    graph_sizes = np.zeros(batch_size, dtype=np.float32)
    times = np.zeros(batch_size, dtype=np.float32)
    graphlets = np.zeros((batch_size, graphlet_width), dtype=np.float32)
    masses = np.zeros((batch_size, mass_width), dtype=np.float32)
    clustering = np.zeros((batch_size, clustering_width), dtype=np.float32)
    orbit = np.zeros((batch_size, orbit_width), dtype=np.float32)

    for index, example in enumerate(examples):
        graph = normalize_topology_graph(example.current_graph)
        n = graph.number_of_nodes()
        matrix = nx.to_numpy_array(
            graph,
            nodelist=list(range(n)),
            dtype=np.float32,
        )
        adjacency[index, :n, :n] = matrix > 0.0
        node_mask[index, :n] = True
        pair_mask[index, :n, :n] = True
        np.fill_diagonal(pair_mask[index], False)
        degrees[index, :n] = np.asarray(
            [float(graph.degree(node)) / max(n - 1, 1) for node in range(n)],
            dtype=np.float32,
        )
        graph_sizes[index] = float(n)
        times[index] = float(np.clip(example.time, 0.0, 1.0))
        graphlet = np.asarray(example.graphlet_target, dtype=np.float32).reshape(-1)
        mass = np.asarray(example.graphlet_mass_target, dtype=np.float32).reshape(-1)
        clustering_target = np.asarray(
            example.clustering_target, dtype=np.float32
        ).reshape(-1)
        orbit_target = np.asarray(example.orbit_target, dtype=np.float32).reshape(-1)
        if (
            graphlet.size != graphlet_width
            or mass.size != mass_width
            or clustering_target.size != clustering_width
            or orbit_target.size != orbit_width
        ):
            raise ValueError("Topology examples use inconsistent structural targets.")
        graphlets[index] = graphlet
        masses[index] = mass
        clustering[index] = clustering_target
        orbit[index] = orbit_target

    return TopologyGraphletBatch(
        adjacency=torch.from_numpy(adjacency),
        node_mask=torch.from_numpy(node_mask),
        pair_mask=torch.from_numpy(pair_mask),
        degrees=torch.from_numpy(degrees),
        graph_size=torch.from_numpy(graph_sizes),
        time=torch.from_numpy(times),
        graphlet_target=torch.from_numpy(graphlets),
        graphlet_mass_target=torch.from_numpy(masses),
        clustering_target=torch.from_numpy(clustering),
        orbit_target=torch.from_numpy(orbit),
    )


def _construct_source_from_degree_sequence(
    degree_sequence: Sequence[int],
    *,
    ensure_connected: bool,
    max_repair_trials: int,
    random_relabel: bool,
    source_randomization_steps: int,
    rng: np.random.Generator,
) -> nx.Graph:
    # Lazy legacy bridge: ordinary GraphER data/model imports do not load the
    # optional DH-VAE+HH baseline. New post-correction training should provide
    # completed base outputs instead of relying on this source constructor.
    from grapher.models.dhvae_hh.havel_hakimi import (
        assert_constructor_validity,
        construct_coarse_graph,
    )

    sequence = sorted((int(value) for value in degree_sequence), reverse=True)
    summary = {
        "num_nodes": len(sequence),
        "num_edges": int(sum(sequence) // 2),
        "degree_sequence": sequence,
    }
    source = construct_coarse_graph(
        summary,
        {
            "type": "havel_hakimi",
            "ensure_connected": bool(ensure_connected),
            "random_relabel": bool(random_relabel),
            "max_repair_trials": int(max_repair_trials),
        },
        rng,
    )
    assert_constructor_validity(
        source,
        summary,
        require_connected=bool(ensure_connected),
    )
    for _ in range(max(int(source_randomization_steps), 0)):
        actions, candidate_graphs, _ = propose_valid_topology_swaps(
            source,
            proposal_budget=64,
            valid_candidate_budget=1,
            preserve_connectivity=bool(ensure_connected),
            rng=rng,
        )
        if not actions:
            break
        source = candidate_graphs[actions[0]]
    return source


def _randomly_relabel_topology_graph(
    graph: nx.Graph,
    *,
    rng: np.random.Generator,
) -> nx.Graph:
    nodes = sorted(graph.nodes())
    permutation = rng.permutation(len(nodes))
    mapping = {node: int(permutation[index]) for index, node in enumerate(nodes)}
    return normalize_topology_graph(nx.relabel_nodes(graph, mapping, copy=True))


def build_topology_teacher_states(
    degree_sequence: Sequence[int] | None = None,
    *,
    source_graph: nx.Graph | None = None,
    target_graphlet: np.ndarray,
    target_graphlet_mass: np.ndarray,
    target_clustering: np.ndarray | None = None,
    target_orbit: np.ndarray | None = None,
    graphlet_basis: TopologyGraphletBasis,
    summary_config: SummaryConfig | dict[str, Any],
    steps: int,
    proposal_budget: int,
    valid_candidate_budget: int,
    preserve_connectivity: bool,
    ensure_connected_source: bool,
    max_repair_trials: int,
    random_relabel_source: bool,
    source_randomization_steps: int,
    teacher_mode: str,
    teacher_temperature: float,
    teacher_top_k: int,
    teacher_sample_actions: bool,
    teacher_graphlet_weight: float = 1.0,
    teacher_graphlet_mass_weight: float = 0.0,
    teacher_clustering_weight: float = 0.0,
    teacher_orbit_weight: float = 0.0,
    teacher_min_improvement: float = 0.0,
    target_tolerance: float = 0.0,
    rng: np.random.Generator,
) -> tuple[list[nx.Graph], dict[str, Any]]:
    """Build a summary-guided teacher path from a completed base output.

    ``source_graph`` is the required path for post-correction training.  The
    positional ``degree_sequence`` path is retained only as an explicit legacy
    compatibility bridge for old experiments and unit tests.
    """

    if not preserve_connectivity or not ensure_connected_source:
        raise ValueError(
            "Topology teacher trajectories require a connected source and "
            "connectivity-preserving swaps."
        )
    weights = {
        "graphlet": float(teacher_graphlet_weight),
        "graphlet_mass": float(teacher_graphlet_mass_weight),
        "clustering": float(teacher_clustering_weight),
        "orbit": float(teacher_orbit_weight),
    }
    for name, value in weights.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"teacher_{name}_weight must be finite and nonnegative.")
    if not any(
        weights[name] > 0.0 for name in ("graphlet", "clustering", "orbit")
    ):
        raise ValueError("At least one topology teacher summary weight must be active.")
    if not np.isfinite(teacher_min_improvement) or (
        float(teacher_min_improvement) < 0.0
    ):
        raise ValueError("teacher_min_improvement must be finite and nonnegative.")
    if not np.isfinite(target_tolerance) or float(target_tolerance) < 0.0:
        raise ValueError("target_tolerance must be finite and nonnegative.")
    if int(steps) < 0:
        raise ValueError("Topology teacher steps must be nonnegative.")
    if int(teacher_top_k) < 0:
        raise ValueError("teacher_top_k must be nonnegative.")

    summary_cfg = (
        summary_config
        if isinstance(summary_config, SummaryConfig)
        else SummaryConfig.from_dict(summary_config or {})
    )
    target_clustering_array = np.asarray(
        target_clustering if target_clustering is not None else [],
        dtype=np.float64,
    ).reshape(-1)
    target_orbit_array = np.asarray(
        target_orbit if target_orbit is not None else [],
        dtype=np.float64,
    ).reshape(-1)
    if weights["clustering"] > 0.0 and target_clustering_array.size == 0:
        raise ValueError(
            "teacher_clustering_weight is active but no clustering target was supplied."
        )
    if weights["orbit"] > 0.0 and target_orbit_array.size == 0:
        raise ValueError(
            "teacher_orbit_weight is active but no orbit target was supplied."
        )

    if source_graph is not None:
        source = normalize_topology_graph(source_graph)
        source_mode = "completed_base_output"
        if int(source_randomization_steps) != 0:
            raise ValueError(
                "source_randomization_steps must be 0 for completed base outputs; "
                "training must start from the declared generator's finished sample."
            )
        if degree_sequence is not None:
            expected = sorted((int(value) for value in degree_sequence), reverse=True)
            actual = sorted((int(degree) for _, degree in source.degree()), reverse=True)
            if actual != expected:
                raise ValueError(
                    "The supplied completed source does not match degree_sequence."
                )
        if random_relabel_source:
            source = _randomly_relabel_topology_graph(source, rng=rng)
    else:
        if degree_sequence is None:
            raise ValueError(
                "Topology teacher training requires source_graph from a declared "
                "base generator. Set training_sources.mode: completed_base_outputs."
            )
        source_mode = "legacy_havel_hakimi"
        source = _construct_source_from_degree_sequence(
            degree_sequence,
            ensure_connected=ensure_connected_source,
            max_repair_trials=max_repair_trials,
            random_relabel=random_relabel_source,
            source_randomization_steps=source_randomization_steps,
            rng=rng,
        )
    if source.number_of_nodes() > 1 and not nx.is_connected(source):
        raise ValueError("Topology teacher training requires a connected source graph.")

    initial_indexed_degrees = [
        int(source.degree(node)) for node in sorted(source.nodes())
    ]
    current = source.copy()
    states = [current.copy()]
    mode = str(teacher_mode).lower()
    if mode not in {"hard", "soft"}:
        raise ValueError("teacher_mode must be 'hard' or 'soft'.")
    if not np.isfinite(teacher_temperature) or float(teacher_temperature) <= 0.0:
        raise ValueError("teacher_temperature must be finite and positive.")

    def score(
        graph: nx.Graph,
        counts: dict[str, dict[str, int]],
    ) -> dict[str, float]:
        return topology_structural_discrepancy_from_counts(
            graph,
            counts,
            graphlet_target=np.asarray(target_graphlet, dtype=np.float64),
            graphlet_mass_target=np.asarray(
                target_graphlet_mass, dtype=np.float64
            ),
            clustering_target=target_clustering_array,
            orbit_target=target_orbit_array,
            graphlet_basis=graphlet_basis,
            graphlet_weight=weights["graphlet"],
            graphlet_mass_weight=weights["graphlet_mass"],
            clustering_weight=weights["clustering"],
            orbit_weight=weights["orbit"],
        )

    initial_counts = extract_topology_graphlet_counts(
        current,
        graphlet_basis=graphlet_basis,
    )
    initial_score = score(current, initial_counts)
    decisions: list[dict[str, Any]] = []
    accepted = 0
    stop_reason = "max_steps"
    for step in range(max(int(steps), 0)):
        current_counts = extract_topology_graphlet_counts(
            current,
            graphlet_basis=graphlet_basis,
        )
        current_score = score(current, current_counts)
        current_distance = float(current_score["total"])
        if current_distance <= float(target_tolerance):
            decisions.append(
                {
                    "step": step,
                    "actions": [],
                    "improvements": [],
                    "distribution": [1.0],
                    "selected_index": 0,
                    "stop_index": 0,
                    "current_structural_discrepancy": current_distance,
                    "current_graphlet_discrepancy": float(
                        current_score["graphlet"]
                    ),
                    "current_components": dict(current_score),
                }
            )
            stop_reason = (
                "target_graphlet_tolerance"
                if weights["clustering"] == 0.0 and weights["orbit"] == 0.0
                else "target_structural_tolerance"
            )
            break

        candidates, candidate_graphs, proposal_diagnostics = (
            propose_valid_topology_swaps(
                current,
                proposal_budget=int(proposal_budget),
                valid_candidate_budget=int(valid_candidate_budget),
                preserve_connectivity=bool(preserve_connectivity),
                rng=rng,
            )
        )
        improvements: list[float] = []
        candidate_scores: list[dict[str, float]] = []
        for action in candidates:
            candidate = candidate_graphs[action]
            candidate_counts = candidate_topology_graphlet_counts(
                current,
                candidate,
                action,
                current_counts=current_counts,
                graphlet_basis=graphlet_basis,
            )
            candidate_score = score(candidate, candidate_counts)
            candidate_scores.append(candidate_score)
            improvements.append(current_distance - float(candidate_score["total"]))

        improving = [
            index
            for index, value in enumerate(improvements)
            if value > float(teacher_min_improvement)
        ]
        if int(teacher_top_k) > 0 and len(improving) > int(teacher_top_k):
            ranked = sorted(
                improving,
                key=lambda index: improvements[index],
                reverse=True,
            )
            cutoff = improvements[ranked[int(teacher_top_k) - 1]]
            improving = [
                index
                for index in improving
                if improvements[index] >= cutoff - 1.0e-12
            ]

        stop_index = len(candidates)
        distribution = np.zeros(stop_index + 1, dtype=np.float64)
        if not improving:
            distribution[stop_index] = 1.0
        elif mode == "hard":
            best = max(improvements[index] for index in improving)
            maximizers = [
                index
                for index in improving
                if abs(improvements[index] - best) <= 1.0e-12
            ]
            distribution[int(rng.choice(maximizers))] = 1.0
        else:
            logits = np.asarray(
                [improvements[index] for index in improving],
                dtype=np.float64,
            ) / float(teacher_temperature)
            logits -= float(np.max(logits))
            probabilities = np.exp(logits)
            probabilities /= float(probabilities.sum())
            distribution[np.asarray(improving, dtype=np.int64)] = probabilities
        if teacher_sample_actions:
            selected_index = int(rng.choice(len(distribution), p=distribution))
        else:
            maxima = np.flatnonzero(
                np.isclose(distribution, distribution.max(), atol=1.0e-12)
            )
            selected_index = int(rng.choice(maxima))
        decisions.append(
            {
                "step": step,
                "actions": [
                    [[list(edge) for edge in removed], [list(edge) for edge in added]]
                    for removed, added in candidates
                ],
                "improvements": [float(value) for value in improvements],
                "distribution": distribution.tolist(),
                "selected_index": selected_index,
                "stop_index": stop_index,
                "current_structural_discrepancy": current_distance,
                "current_graphlet_discrepancy": float(current_score["graphlet"]),
                "current_components": dict(current_score),
                "candidate_components": candidate_scores,
                **proposal_diagnostics,
            }
        )
        if selected_index == stop_index:
            stop_reason = (
                "no_improving_graphlet_swap"
                if weights["clustering"] == 0.0 and weights["orbit"] == 0.0
                else "no_improving_structural_swap"
            )
            break
        current = candidate_graphs[candidates[selected_index]]
        if [int(current.degree(node)) for node in sorted(current.nodes())] != (
            initial_indexed_degrees
        ):
            raise AssertionError("A topology teacher action changed indexed degrees.")
        states.append(current.copy())
        accepted += 1

    final_counts = extract_topology_graphlet_counts(
        current,
        graphlet_basis=graphlet_basis,
    )
    final_score = score(current, final_counts)
    report = {
        "source_mode": source_mode,
        "initial_structural_discrepancy": float(initial_score["total"]),
        "final_teacher_structural_discrepancy": float(final_score["total"]),
        "teacher_structural_reduction": float(
            initial_score["total"] - final_score["total"]
        ),
        # Preserve the legacy report keys for downstream diagnostics.
        "initial_graphlet_discrepancy": float(initial_score["graphlet"]),
        "final_teacher_graphlet_discrepancy": float(final_score["graphlet"]),
        "teacher_graphlet_reduction": float(
            initial_score["graphlet"] - final_score["graphlet"]
        ),
        "initial_components": dict(initial_score),
        "final_components": dict(final_score),
        "teacher_weights": weights,
        "accepted_teacher_steps": int(accepted),
        "teacher_stop_reason": stop_reason,
        "teacher_stop_selected": stop_reason != "max_steps",
        "teacher_decisions": decisions,
        "mean_valid_candidates": (
            float(
                np.mean(
                    [
                        row["num_valid_candidates"]
                        for row in decisions
                        if "num_valid_candidates" in row
                    ]
                )
            )
            if any("num_valid_candidates" in row for row in decisions)
            else 0.0
        ),
    }
    return states, report

def build_topology_examples(
    graphs: Sequence[nx.Graph | TopologyTrainingPair],
    *,
    summary_config: SummaryConfig | dict[str, Any],
    graphlet_basis: TopologyGraphletBasis,
    trajectory_config: dict[str, Any] | None = None,
    seed: int = 0,
) -> tuple[list[TopologyGraphletExample], dict[str, Any]]:
    """Create graph-level structural supervision from explicit source/target pairs."""

    cfg = dict(trajectory_config or {})
    summary_cfg = (
        summary_config
        if isinstance(summary_config, SummaryConfig)
        else SummaryConfig.from_dict(summary_config or {})
    )
    rng = np.random.default_rng(int(seed))
    examples: list[TopologyGraphletExample] = []
    reports: list[dict[str, Any]] = []
    trajectory_id = 0
    paths_per_graph = max(int(cfg.get("paths_per_graph", 1)), 1)
    valid_budget = int(
        cfg.get("valid_candidate_budget", cfg.get("candidate_budget", 64))
    )
    proposal_budget = int(
        cfg.get(
            "proposal_budget",
            valid_budget if valid_budget < 0 else max(valid_budget, 1) * 4,
        )
    )

    for raw_item in graphs:
        if isinstance(raw_item, TopologyTrainingPair):
            source = normalize_topology_graph(raw_item.source_graph)
            target = normalize_topology_graph(raw_item.target_graph)
            base_generator = str(raw_item.base_generator)
            source_index = int(raw_item.source_index)
            target_index = int(raw_item.target_index)
            matching_cost = float(raw_item.matching_cost)
            if source.number_of_nodes() != target.number_of_nodes():
                raise ValueError(
                    "Completed source/target pairs must have identical graph size."
                )
        else:
            source = None
            target = normalize_topology_graph(raw_item)
            base_generator = "legacy_havel_hakimi"
            source_index = -1
            target_index = -1
            matching_cost = 0.0

        if target.number_of_nodes() > 1 and not nx.is_connected(target):
            raise ValueError("Topology training targets must be connected graphs.")
        if source is not None and source.number_of_nodes() > 1 and not nx.is_connected(
            source
        ):
            raise ValueError("Topology training sources must be connected graphs.")
        degree_sequence = [int(target.degree(node)) for node in target.nodes()]
        (
            target_graphlet,
            target_mass,
            target_clustering,
            target_orbit,
        ) = extract_topology_structural_target(
            target,
            graphlet_basis=graphlet_basis,
            summary_config=summary_cfg,
        )
        for _path in range(paths_per_graph):
            states, report = build_topology_teacher_states(
                degree_sequence if source is None else None,
                source_graph=source,
                target_graphlet=target_graphlet,
                target_graphlet_mass=target_mass,
                target_clustering=target_clustering,
                target_orbit=target_orbit,
                graphlet_basis=graphlet_basis,
                summary_config=summary_cfg,
                steps=int(cfg.get("steps", 32)),
                proposal_budget=proposal_budget,
                valid_candidate_budget=valid_budget,
                preserve_connectivity=bool(cfg.get("preserve_connectivity", True)),
                ensure_connected_source=bool(cfg.get("ensure_connected_source", True)),
                max_repair_trials=int(cfg.get("max_repair_trials", 10000)),
                random_relabel_source=bool(cfg.get("random_relabel_source", True)),
                source_randomization_steps=int(
                    cfg.get("source_randomization_steps", 0)
                ),
                teacher_mode=str(cfg.get("teacher_mode", "hard")),
                teacher_temperature=float(cfg.get("teacher_temperature", 1.0)),
                teacher_top_k=int(cfg.get("teacher_top_k", 0)),
                teacher_sample_actions=bool(cfg.get("teacher_sample_actions", False)),
                teacher_graphlet_weight=float(
                    cfg.get("teacher_graphlet_weight", 1.0)
                ),
                teacher_graphlet_mass_weight=float(
                    cfg.get("teacher_graphlet_mass_weight", 0.0)
                ),
                teacher_clustering_weight=float(
                    cfg.get("teacher_clustering_weight", 0.0)
                ),
                teacher_orbit_weight=float(cfg.get("teacher_orbit_weight", 0.0)),
                teacher_min_improvement=float(
                    cfg.get("teacher_min_improvement", 0.0)
                ),
                target_tolerance=float(cfg.get("target_tolerance", 0.0)),
                rng=rng,
            )
            selected_count = min(
                max(int(cfg.get("states_per_graph", 8)), 1),
                len(states),
            )
            indices = (
                list(range(len(states)))
                if selected_count == len(states)
                else sorted(
                    {
                        int(round(value))
                        for value in np.linspace(
                            0,
                            len(states) - 1,
                            num=selected_count,
                        )
                    }
                )
            )
            horizon = max(int(cfg.get("steps", 32)), 1)
            decisions = report.get("teacher_decisions", [])
            for step in indices:
                decision = decisions[step] if step < len(decisions) else None
                actions: tuple[Action, ...] = ()
                distribution = None
                selected_index = -1
                if decision is not None:
                    actions = tuple(
                        make_action(value[0], value[1])
                        for value in decision.get("actions", [])
                    )
                    distribution = np.asarray(
                        decision.get("distribution", [1.0]),
                        dtype=np.float32,
                    )
                    selected_index = int(decision.get("selected_index", -1))
                examples.append(
                    TopologyGraphletExample(
                        current_graph=states[step],
                        time=float(step / horizon),
                        graphlet_target=target_graphlet.astype(np.float32).copy(),
                        graphlet_mass_target=target_mass.astype(np.float32).copy(),
                        clustering_target=target_clustering.astype(np.float32).copy(),
                        orbit_target=target_orbit.astype(np.float32).copy(),
                        base_generator=base_generator,
                        source_index=source_index,
                        target_index=target_index,
                        matching_cost=matching_cost,
                        trajectory_id=trajectory_id,
                        step=step,
                        teacher_actions=actions,
                        teacher_distribution=distribution,
                        teacher_selected_index=selected_index,
                    )
                )
            report = {
                **report,
                "base_generator": base_generator,
                "source_index": source_index,
                "target_index": target_index,
                "matching_cost": matching_cost,
            }
            reports.append(report)
            trajectory_id += 1

    diagnostics = {
        "num_graphs": len(graphs),
        "num_paths": len(reports),
        "num_examples": len(examples),
        "source_modes": sorted({str(r.get("source_mode")) for r in reports}),
        "base_generators": sorted({str(r.get("base_generator")) for r in reports}),
        "mean_matching_cost": (
            float(np.mean([r["matching_cost"] for r in reports]))
            if reports
            else 0.0
        ),
        "mean_initial_structural_discrepancy": (
            float(np.mean([r["initial_structural_discrepancy"] for r in reports]))
            if reports
            else 0.0
        ),
        "mean_final_teacher_structural_discrepancy": (
            float(
                np.mean(
                    [r["final_teacher_structural_discrepancy"] for r in reports]
                )
            )
            if reports
            else 0.0
        ),
        "mean_initial_graphlet_discrepancy": (
            float(np.mean([r["initial_graphlet_discrepancy"] for r in reports]))
            if reports
            else 0.0
        ),
        "mean_final_teacher_graphlet_discrepancy": (
            float(
                np.mean(
                    [r["final_teacher_graphlet_discrepancy"] for r in reports]
                )
            )
            if reports
            else 0.0
        ),
        "mean_accepted_teacher_steps": (
            float(np.mean([r["accepted_teacher_steps"] for r in reports]))
            if reports
            else 0.0
        ),
        "teacher_stop_rate": (
            float(np.mean([bool(r["teacher_stop_selected"]) for r in reports]))
            if reports
            else 0.0
        ),
        "mean_valid_candidates": (
            float(np.mean([r["mean_valid_candidates"] for r in reports]))
            if reports
            else 0.0
        ),
    }
    return examples, diagnostics


class TopologyTrajectoryIterableDataset(torch.utils.data.IterableDataset):
    """Generate topology teacher examples lazily with one graph in memory."""

    def __init__(
        self,
        graphs: Sequence[nx.Graph | TopologyTrainingPair],
        *,
        summary_config: SummaryConfig | dict[str, Any],
        graphlet_basis: TopologyGraphletBasis,
        trajectory_config: dict[str, Any] | None = None,
        seed: int = 0,
        shuffle_graphs: bool = True,
    ) -> None:
        super().__init__()
        self.graphs = tuple(graphs)
        self.summary_config = summary_config
        self.graphlet_basis = graphlet_basis
        self.trajectory_config = dict(trajectory_config or {})
        self.seed = int(seed)
        self.shuffle_graphs = bool(shuffle_graphs)
        self.epoch = 0
        self.last_diagnostics: list[dict[str, Any]] = []

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def estimated_examples(self) -> int:
        return (
            len(self.graphs)
            * max(int(self.trajectory_config.get("states_per_graph", 8)), 1)
            * max(int(self.trajectory_config.get("paths_per_graph", 1)), 1)
        )

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        indices = np.arange(len(self.graphs), dtype=np.int64)
        generator = np.random.default_rng(self.seed + 1_000_003 * self.epoch)
        if self.shuffle_graphs:
            generator.shuffle(indices)
        indices = indices[worker_id::worker_count]
        if worker is None:
            self.last_diagnostics = []
        for position, graph_index in enumerate(indices):
            examples, diagnostics = build_topology_examples(
                [self.graphs[int(graph_index)]],
                summary_config=self.summary_config,
                graphlet_basis=self.graphlet_basis,
                trajectory_config=self.trajectory_config,
                seed=(
                    self.seed
                    + 1_000_003 * self.epoch
                    + 10_007 * int(graph_index)
                    + position
                ),
            )
            if worker is None:
                self.last_diagnostics.append(diagnostics)
            yield from examples
