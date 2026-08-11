from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import networkx as nx
import numpy as np
import torch

from grapher.construction.coarse import (
    assert_constructor_validity,
    construct_coarse_graph,
)
from grapher.properties.summary import SummaryConfig
from grapher.refinement.rewiring import Action, make_action
from grapher.topology.graphlets import (
    extract_topology_graphlet_counts,
    extract_topology_graphlet_target,
    topology_candidate_graphlet_discrepancy,
    topology_graphlet_discrepancy,
    topology_graphlet_discrepancy_from_counts,
)
from grapher.topology.basis import TopologyGraphletBasis
from grapher.topology.rewiring import propose_valid_topology_swaps


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
class TopologyGraphletExample:
    current_graph: nx.Graph
    time: float
    graphlet_target: np.ndarray
    graphlet_mass_target: np.ndarray
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

    adjacency = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.bool_)
    node_mask = np.zeros((batch_size, max_nodes), dtype=np.bool_)
    pair_mask = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.bool_)
    degrees = np.zeros((batch_size, max_nodes), dtype=np.float32)
    graph_sizes = np.zeros(batch_size, dtype=np.float32)
    times = np.zeros(batch_size, dtype=np.float32)
    graphlets = np.zeros((batch_size, graphlet_width), dtype=np.float32)
    masses = np.zeros((batch_size, mass_width), dtype=np.float32)

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
        if graphlet.size != graphlet_width or mass.size != mass_width:
            raise ValueError("Topology examples use inconsistent graphlet targets.")
        graphlets[index] = graphlet
        masses[index] = mass

    return TopologyGraphletBatch(
        adjacency=torch.from_numpy(adjacency),
        node_mask=torch.from_numpy(node_mask),
        pair_mask=torch.from_numpy(pair_mask),
        degrees=torch.from_numpy(degrees),
        graph_size=torch.from_numpy(graph_sizes),
        time=torch.from_numpy(times),
        graphlet_target=torch.from_numpy(graphlets),
        graphlet_mass_target=torch.from_numpy(masses),
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


def build_topology_teacher_states(
    degree_sequence: Sequence[int],
    *,
    target_graphlet: np.ndarray,
    target_graphlet_mass: np.ndarray,
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
    teacher_graphlet_mass_weight: float,
    teacher_min_improvement: float,
    target_tolerance: float,
    rng: np.random.Generator,
) -> tuple[list[nx.Graph], dict[str, Any]]:
    """Build a graphlet-only teacher path inside one ordinary-degree fibre."""

    if not preserve_connectivity or not ensure_connected_source:
        raise ValueError(
            "Topology teacher trajectories require a connected source and "
            "connectivity-preserving swaps."
        )
    if not np.isfinite(teacher_graphlet_mass_weight) or (
        float(teacher_graphlet_mass_weight) < 0.0
    ):
        raise ValueError("teacher_graphlet_mass_weight must be finite and nonnegative.")
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

    source = _construct_source_from_degree_sequence(
        degree_sequence,
        ensure_connected=ensure_connected_source,
        max_repair_trials=max_repair_trials,
        random_relabel=random_relabel_source,
        source_randomization_steps=source_randomization_steps,
        rng=rng,
    )
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

    initial_distance, _, _ = topology_graphlet_discrepancy(
        current,
        target_graphlet,
        target_graphlet_mass,
        graphlet_basis=graphlet_basis,
        summary_config=summary_config,
        mass_weight=teacher_graphlet_mass_weight,
    )
    decisions: list[dict[str, Any]] = []
    accepted = 0
    stop_reason = "max_steps"
    for step in range(max(int(steps), 0)):
        current_counts = extract_topology_graphlet_counts(
            current,
            graphlet_basis=graphlet_basis,
        )
        current_distance, _, _ = topology_graphlet_discrepancy_from_counts(
            current_counts,
            num_nodes=current.number_of_nodes(),
            target=target_graphlet,
            target_mass=target_graphlet_mass,
            graphlet_basis=graphlet_basis,
            mass_weight=teacher_graphlet_mass_weight,
        )
        if current_distance <= float(target_tolerance):
            decisions.append(
                {
                    "step": step,
                    "actions": [],
                    "improvements": [],
                    "distribution": [1.0],
                    "selected_index": 0,
                    "stop_index": 0,
                    "current_graphlet_discrepancy": float(current_distance),
                }
            )
            stop_reason = "target_graphlet_tolerance"
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
        for action in candidates:
            candidate_distance, _, _ = topology_candidate_graphlet_discrepancy(
                current,
                candidate_graphs[action],
                action,
                target_graphlet,
                target_graphlet_mass,
                current_counts=current_counts,
                graphlet_basis=graphlet_basis,
                mass_weight=teacher_graphlet_mass_weight,
            )
            improvements.append(float(current_distance - candidate_distance))

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
            support = improving
            logits = np.asarray(
                [improvements[index] for index in improving],
                dtype=np.float64,
            ) / float(teacher_temperature)
            logits -= float(np.max(logits))
            probabilities = np.exp(logits)
            probabilities /= float(probabilities.sum())
            distribution[np.asarray(support, dtype=np.int64)] = probabilities
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
                "current_graphlet_discrepancy": float(current_distance),
                **proposal_diagnostics,
            }
        )
        if selected_index == stop_index:
            stop_reason = "no_improving_graphlet_swap"
            break
        current = candidate_graphs[candidates[selected_index]]
        if [int(current.degree(node)) for node in sorted(current.nodes())] != (
            initial_indexed_degrees
        ):
            raise AssertionError("A topology teacher action changed indexed degrees.")
        states.append(current.copy())
        accepted += 1

    final_distance, _, _ = topology_graphlet_discrepancy(
        current,
        target_graphlet,
        target_graphlet_mass,
        graphlet_basis=graphlet_basis,
        summary_config=summary_config,
        mass_weight=teacher_graphlet_mass_weight,
    )
    report = {
        "initial_graphlet_discrepancy": float(initial_distance),
        "final_teacher_graphlet_discrepancy": float(final_distance),
        "teacher_graphlet_reduction": float(initial_distance - final_distance),
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
    graphs: Sequence[nx.Graph],
    *,
    summary_config: SummaryConfig | dict[str, Any],
    graphlet_basis: TopologyGraphletBasis,
    trajectory_config: dict[str, Any] | None = None,
    seed: int = 0,
) -> tuple[list[TopologyGraphletExample], dict[str, Any]]:
    """Create graphlet-supervised states without terminal adjacency labels."""

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

    for raw_target in graphs:
        target = normalize_topology_graph(raw_target)
        if target.number_of_nodes() > 1 and not nx.is_connected(target):
            raise ValueError("Topology training targets must be connected graphs.")
        degree_sequence = [int(target.degree(node)) for node in target.nodes()]
        # One cached target is reused for every path, every teacher comparison,
        # and every supervised state derived from this terminal graph.
        target_graphlet, target_mass = extract_topology_graphlet_target(
            target,
            graphlet_basis=graphlet_basis,
            summary_config=summary_cfg,
        )
        for _path in range(paths_per_graph):
            states, report = build_topology_teacher_states(
                degree_sequence,
                target_graphlet=target_graphlet,
                target_graphlet_mass=target_mass,
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
                teacher_graphlet_mass_weight=float(
                    cfg.get("teacher_graphlet_mass_weight", 0.0)
                ),
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
                        trajectory_id=trajectory_id,
                        step=step,
                        teacher_actions=actions,
                        teacher_distribution=distribution,
                        teacher_selected_index=selected_index,
                    )
                )
            reports.append(report)
            trajectory_id += 1

    diagnostics = {
        "num_graphs": len(graphs),
        "num_paths": len(reports),
        "num_examples": len(examples),
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
        graphs: Sequence[nx.Graph],
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
