from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import networkx as nx
import numpy as np
import torch

from grapher.rewiring_mlp.core.rewiring import Action, make_action
from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
from grapher.rewiring_mlp.generic.data import (
    TopologyTrainingPair,
    _construct_source_from_degree_sequence,
    _randomly_relabel_topology_graph,
    normalize_topology_graph,
)
from grapher.rewiring_mlp.generic.graphlet_diffusion import (
    extract_topology_graphlet_simplex,
    graphlet_simplex_to_clr,
)
from grapher.rewiring_mlp.generic.rewiring import (
    propose_valid_topology_swaps,
    topology_state_key,
)
from grapher.rewiring_mlp.generic.spectral import (
    degree_spectral_moments,
    laplacian_eigenvalues,
    spectral_distance,
    spectral_scale,
    spectrum_moments,
)


@dataclass
class TopologySpectralExample:
    current_graph: nx.Graph
    time: float
    clean_spectrum_target: np.ndarray
    # Optional graphlet-logit diffusion supervision. Each graphlet order is a
    # probability simplex over connected graphlet classes plus one disconnected
    # subset bin; CLR coordinates are the Euclidean diffusion variables.
    current_graphlet_probabilities: np.ndarray | None = None
    clean_graphlet_probabilities_target: np.ndarray | None = None
    current_graphlet_logits: np.ndarray | None = None
    clean_graphlet_logits_target: np.ndarray | None = None
    graphlet_coordinate_mask: np.ndarray | None = None
    base_generator: str = "target_degree_havel_hakimi"
    source_index: int = -1
    target_index: int = -1
    matching_cost: float = 0.0
    trajectory_id: int = -1
    step: int = -1
    teacher_actions: tuple[Action, ...] = ()
    teacher_distribution: np.ndarray | None = None
    teacher_selected_index: int = -1


@dataclass
class TopologySpectralBatch:
    adjacency: torch.Tensor
    node_mask: torch.Tensor
    pair_mask: torch.Tensor
    degrees: torch.Tensor
    graph_size: torch.Tensor
    time: torch.Tensor
    current_spectrum: torch.Tensor
    clean_spectrum_target: torch.Tensor
    spectrum_mask: torch.Tensor
    current_graphlet_probabilities: torch.Tensor | None = None
    clean_graphlet_probabilities_target: torch.Tensor | None = None
    current_graphlet_logits: torch.Tensor | None = None
    clean_graphlet_logits_target: torch.Tensor | None = None
    graphlet_coordinate_mask: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "TopologySpectralBatch":
        return TopologySpectralBatch(
            **{
                key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                for key, value in self.__dict__.items()
            }
        )


def collate_spectral_examples(
    examples: Sequence[TopologySpectralExample],
) -> TopologySpectralBatch:
    """Pad graph states and variable-length eigenvalue targets for one batch."""

    if not examples:
        raise ValueError("Cannot collate an empty spectral batch.")
    max_nodes = max(example.current_graph.number_of_nodes() for example in examples)
    batch_size = len(examples)

    adjacency = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.bool_)
    node_mask = np.zeros((batch_size, max_nodes), dtype=np.bool_)
    pair_mask = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.bool_)
    degrees = np.zeros((batch_size, max_nodes), dtype=np.float32)
    graph_sizes = np.zeros(batch_size, dtype=np.float32)
    times = np.zeros(batch_size, dtype=np.float32)
    current_spectra = np.zeros((batch_size, max_nodes), dtype=np.float32)
    clean_spectra = np.zeros((batch_size, max_nodes), dtype=np.float32)
    spectrum_mask = np.zeros((batch_size, max_nodes), dtype=np.bool_)

    graphlet_widths = {
        int(np.asarray(example.current_graphlet_logits).size)
        for example in examples
        if example.current_graphlet_logits is not None
    }
    if len(graphlet_widths) > 1:
        raise ValueError("Graphlet-logit examples in one batch must share a fixed width.")
    graphlet_width = next(iter(graphlet_widths), 0)
    graphlet_enabled = graphlet_width > 0
    if graphlet_enabled and any(example.current_graphlet_logits is None for example in examples):
        raise ValueError("Cannot mix spectral-only and spectral+graphlet examples in one batch.")
    current_graphlet_probabilities = (
        np.zeros((batch_size, graphlet_width), dtype=np.float32)
        if graphlet_enabled else None
    )
    clean_graphlet_probabilities = (
        np.zeros((batch_size, graphlet_width), dtype=np.float32)
        if graphlet_enabled else None
    )
    current_graphlet_logits = (
        np.zeros((batch_size, graphlet_width), dtype=np.float32)
        if graphlet_enabled else None
    )
    clean_graphlet_logits = (
        np.zeros((batch_size, graphlet_width), dtype=np.float32)
        if graphlet_enabled else None
    )
    graphlet_coordinate_mask = (
        np.zeros((batch_size, graphlet_width), dtype=np.bool_)
        if graphlet_enabled else None
    )

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
        current = laplacian_eigenvalues(graph).astype(np.float32)
        target = np.asarray(
            example.clean_spectrum_target,
            dtype=np.float32,
        ).reshape(-1)
        if current.size != n or target.size != n:
            raise ValueError(
                "Spectral examples must contain exactly one eigenvalue per node: "
                f"current={current.size}, target={target.size}, n={n}."
            )
        current_spectra[index, :n] = current
        clean_spectra[index, :n] = target
        spectrum_mask[index, :n] = True
        if graphlet_enabled:
            assert current_graphlet_probabilities is not None
            assert clean_graphlet_probabilities is not None
            assert current_graphlet_logits is not None
            assert clean_graphlet_logits is not None
            assert graphlet_coordinate_mask is not None
            arrays = [
                np.asarray(example.current_graphlet_probabilities, dtype=np.float32).reshape(-1),
                np.asarray(example.clean_graphlet_probabilities_target, dtype=np.float32).reshape(-1),
                np.asarray(example.current_graphlet_logits, dtype=np.float32).reshape(-1),
                np.asarray(example.clean_graphlet_logits_target, dtype=np.float32).reshape(-1),
                np.asarray(example.graphlet_coordinate_mask, dtype=np.bool_).reshape(-1),
            ]
            if any(array.size != graphlet_width for array in arrays):
                raise ValueError("Graphlet-logit target width mismatch during collation.")
            current_graphlet_probabilities[index] = arrays[0]
            clean_graphlet_probabilities[index] = arrays[1]
            current_graphlet_logits[index] = arrays[2]
            clean_graphlet_logits[index] = arrays[3]
            graphlet_coordinate_mask[index] = arrays[4]

    return TopologySpectralBatch(
        adjacency=torch.from_numpy(adjacency),
        node_mask=torch.from_numpy(node_mask),
        pair_mask=torch.from_numpy(pair_mask),
        degrees=torch.from_numpy(degrees),
        graph_size=torch.from_numpy(graph_sizes),
        time=torch.from_numpy(times),
        current_spectrum=torch.from_numpy(current_spectra),
        clean_spectrum_target=torch.from_numpy(clean_spectra),
        spectrum_mask=torch.from_numpy(spectrum_mask),
        current_graphlet_probabilities=(
            torch.from_numpy(current_graphlet_probabilities)
            if current_graphlet_probabilities is not None else None
        ),
        clean_graphlet_probabilities_target=(
            torch.from_numpy(clean_graphlet_probabilities)
            if clean_graphlet_probabilities is not None else None
        ),
        current_graphlet_logits=(
            torch.from_numpy(current_graphlet_logits)
            if current_graphlet_logits is not None else None
        ),
        clean_graphlet_logits_target=(
            torch.from_numpy(clean_graphlet_logits)
            if clean_graphlet_logits is not None else None
        ),
        graphlet_coordinate_mask=(
            torch.from_numpy(graphlet_coordinate_mask)
            if graphlet_coordinate_mask is not None else None
        ),
    )


def _sorted_degree_sequence(graph: nx.Graph) -> list[int]:
    return sorted((int(degree) for _, degree in graph.degree()), reverse=True)


def assert_same_degree_fibre(source: nx.Graph, target: nx.Graph) -> None:
    """Require exact sorted degree equality for a clean spectral endpoint.

    Rewiring preserves all node degrees, so a terminal spectrum from a graph in a
    different degree fibre violates not only ``sum(lambda)=2m`` but also the
    fixed second Laplacian moment ``sum(lambda^2)=sum(d^2)+2m``.
    """

    left = _sorted_degree_sequence(source)
    right = _sorted_degree_sequence(target)
    if left != right:
        raise ValueError(
            "Spectral-guided training requires source and target graphs to have "
            "the same degree sequence. A clean target spectrum from another "
            "degree fibre is unreachable by degree-preserving rewiring."
        )


def build_spectral_teacher_states(
    degree_sequence: Sequence[int] | None = None,
    *,
    source_graph: nx.Graph | None = None,
    target_graph: nx.Graph,
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
    teacher_min_improvement: float = 0.0,
    target_tolerance: float = 0.0,
    teacher_allow_sideways: bool = False,
    teacher_max_consecutive_sideways: int = 0,
    teacher_sideways_tolerance: float = 0.0,
    teacher_tabu: bool = True,
    distance_metric: str = "rmse",
    distance_normalization: str = "mean_degree",
    low_frequency_weight: float = 1.0,
    low_frequency_cutoff: int = 0,
    require_same_degree_sequence: bool = True,
    rng: np.random.Generator,
) -> tuple[list[nx.Graph], dict[str, Any]]:
    """Build actual graph states using an oracle clean-spectrum teacher.

    The teacher is used only to create state coverage for x0 prediction.  It
    chooses valid degree-preserving swaps that move the actual graph spectrum
    toward the clean target spectrum.  Generation later converts the predicted
    clean spectrum into a scheduled *next* target before projecting by rewiring.
    """

    if not preserve_connectivity or not ensure_connected_source:
        raise ValueError(
            "Spectral teacher trajectories require a connected source and "
            "connectivity-preserving swaps."
        )
    if int(steps) < 0:
        raise ValueError("Spectral teacher steps must be nonnegative.")
    if int(teacher_top_k) < 0:
        raise ValueError("teacher_top_k must be nonnegative.")
    if not np.isfinite(teacher_temperature) or float(teacher_temperature) <= 0.0:
        raise ValueError("teacher_temperature must be finite and positive.")
    if not np.isfinite(teacher_min_improvement) or teacher_min_improvement < 0.0:
        raise ValueError("teacher_min_improvement must be finite and nonnegative.")
    if not np.isfinite(target_tolerance) or target_tolerance < 0.0:
        raise ValueError("target_tolerance must be finite and nonnegative.")
    if not np.isfinite(low_frequency_weight) or low_frequency_weight <= 0.0:
        raise ValueError("low_frequency_weight must be finite and positive.")
    if int(teacher_max_consecutive_sideways) < 0:
        raise ValueError("teacher_max_consecutive_sideways must be nonnegative.")
    if (
        not np.isfinite(teacher_sideways_tolerance)
        or teacher_sideways_tolerance < 0.0
    ):
        raise ValueError(
            "teacher_sideways_tolerance must be finite and nonnegative."
        )

    target = normalize_topology_graph(target_graph)
    if target.number_of_nodes() > 1 and not nx.is_connected(target):
        raise ValueError("Spectral training targets must be connected graphs.")

    if source_graph is not None:
        source = normalize_topology_graph(source_graph)
        source_mode = "completed_base_output"
        if int(source_randomization_steps) != 0:
            raise ValueError(
                "source_randomization_steps must be 0 for completed base outputs."
            )
        if random_relabel_source:
            source = _randomly_relabel_topology_graph(source, rng=rng)
    else:
        if degree_sequence is None:
            raise ValueError(
                "Spectral teacher training requires source_graph or degree_sequence."
            )
        source_mode = "target_degree_havel_hakimi"
        source = _construct_source_from_degree_sequence(
            degree_sequence,
            ensure_connected=ensure_connected_source,
            max_repair_trials=max_repair_trials,
            random_relabel=random_relabel_source,
            source_randomization_steps=source_randomization_steps,
            rng=rng,
        )
    if source.number_of_nodes() != target.number_of_nodes():
        raise ValueError("Spectral source and target must have identical graph size.")
    if source.number_of_nodes() > 1 and not nx.is_connected(source):
        raise ValueError("Spectral teacher training requires a connected source graph.")
    if require_same_degree_sequence:
        assert_same_degree_fibre(source, target)

    target_spectrum = laplacian_eigenvalues(target)
    target_first, target_second = spectrum_moments(target_spectrum)
    source_fixed_first, source_fixed_second = degree_spectral_moments(source)
    moment_residual = {
        "trace": float(target_first - source_fixed_first),
        "second": float(target_second - source_fixed_second),
    }
    if require_same_degree_sequence and (
        abs(moment_residual["trace"]) > 1.0e-7
        or abs(moment_residual["second"]) > 1.0e-6
    ):
        raise AssertionError(
            "Target Laplacian moments disagree with the source degree invariant."
        )

    scale = spectral_scale(source, mode=distance_normalization)

    def distance(graph: nx.Graph) -> float:
        return spectral_distance(
            laplacian_eigenvalues(graph),
            target_spectrum,
            metric=distance_metric,
            scale=scale,
            low_frequency_weight=low_frequency_weight,
            low_frequency_cutoff=low_frequency_cutoff,
        )

    initial_distance = distance(source)
    current = source.copy()
    states = [current.copy()]
    decisions: list[dict[str, Any]] = []
    accepted = 0
    stop_reason = "max_steps"
    mode = str(teacher_mode).lower()
    if mode not in {"hard", "soft"}:
        raise ValueError("teacher_mode must be 'hard' or 'soft'.")

    # F1: greedy one-swap descent on the spectral distance reaches a local
    # minimum after a handful of moves.  A tabu set plus bounded sideways moves
    # lets the teacher traverse flat regions instead of terminating there.
    visited: set[bytes] = {topology_state_key(current)}
    consecutive_sideways = 0
    sideways_accepted = 0
    # Sideways moves can end on a worse state than the best one seen, so the
    # best state is tracked explicitly and reported alongside the final one.
    best_distance = initial_distance
    best_state_index = 0

    for step in range(max(int(steps), 0)):
        current_spectrum = laplacian_eigenvalues(current)
        current_distance = spectral_distance(
            current_spectrum,
            target_spectrum,
            metric=distance_metric,
            scale=scale,
            low_frequency_weight=low_frequency_weight,
            low_frequency_cutoff=low_frequency_cutoff,
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
                    "current_spectral_discrepancy": current_distance,
                }
            )
            stop_reason = "target_spectral_tolerance"
            break

        candidates, candidate_graphs, proposal_diagnostics = (
            propose_valid_topology_swaps(
                current,
                proposal_budget=int(proposal_budget),
                valid_candidate_budget=int(valid_candidate_budget),
                preserve_connectivity=bool(preserve_connectivity),
                rng=rng,
                excluded_states=visited if teacher_tabu else None,
            )
        )
        candidate_distances: list[float] = []
        improvements: list[float] = []
        for action in candidates:
            candidate_distance = spectral_distance(
                laplacian_eigenvalues(candidate_graphs[action]),
                target_spectrum,
                metric=distance_metric,
                scale=scale,
                low_frequency_weight=low_frequency_weight,
                low_frequency_cutoff=low_frequency_cutoff,
            )
            candidate_distances.append(candidate_distance)
            improvements.append(current_distance - candidate_distance)

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
        move_kind = "improving"
        if not improving:
            # F1: no strictly improving swap.  Rather than terminating at the
            # local minimum, optionally take the least-harmful valid move so
            # the walk can cross a plateau.  `sideways_tolerance` is expressed
            # in the same normalized units as the spectral distance.
            sideways = [
                index
                for index, value in enumerate(improvements)
                if value >= -float(teacher_sideways_tolerance)
            ]
            can_step_sideways = (
                bool(teacher_allow_sideways)
                and bool(sideways)
                and consecutive_sideways < int(teacher_max_consecutive_sideways)
            )
            if can_step_sideways:
                best_sideways = max(
                    sideways, key=lambda index: improvements[index]
                )
                distribution[best_sideways] = 1.0
                move_kind = "sideways"
            else:
                distribution[stop_index] = 1.0
        elif mode == "hard":
            best = max(improvements[index] for index in improving)
            maxima = [
                index
                for index in improving
                if abs(improvements[index] - best) <= 1.0e-12
            ]
            distribution[int(rng.choice(maxima))] = 1.0
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
                "candidate_spectral_discrepancies": [
                    float(value) for value in candidate_distances
                ],
                "distribution": distribution.tolist(),
                "selected_index": selected_index,
                "stop_index": stop_index,
                "move_kind": move_kind,
                "current_spectral_discrepancy": current_distance,
                **proposal_diagnostics,
            }
        )
        if selected_index == stop_index:
            stop_reason = (
                "no_improving_spectral_swap"
                if not teacher_allow_sideways
                else "no_improving_or_sideways_spectral_swap"
            )
            break
        current = candidate_graphs[candidates[selected_index]]
        states.append(current.copy())
        accepted += 1
        visited.add(topology_state_key(current))
        if move_kind == "sideways":
            consecutive_sideways += 1
            sideways_accepted += 1
        else:
            consecutive_sideways = 0
        step_distance = current_distance - improvements[selected_index]
        if step_distance < best_distance:
            best_distance = float(step_distance)
            best_state_index = len(states) - 1

    final_distance = distance(current)
    report = {
        "source_mode": source_mode,
        "initial_spectral_discrepancy": float(initial_distance),
        "final_teacher_spectral_discrepancy": float(final_distance),
        "teacher_spectral_reduction": float(initial_distance - final_distance),
        "accepted_teacher_steps": int(accepted),
        "accepted_sideways_steps": int(sideways_accepted),
        "best_teacher_spectral_discrepancy": float(best_distance),
        "best_teacher_state_index": int(best_state_index),
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
        "spectral_target_moment_residual": moment_residual,
        "spectral_distance_metric": str(distance_metric),
        "spectral_distance_normalization": str(distance_normalization),
    }
    return states, report


def build_spectral_examples(
    graphs: Sequence[nx.Graph | TopologyTrainingPair],
    *,
    trajectory_config: dict[str, Any] | None = None,
    spectral_config: dict[str, Any] | None = None,
    graphlet_basis: TopologyGraphletBasis | None = None,
    graphlet_logit_epsilon: float = 1.0e-5,
    seed: int = 0,
) -> tuple[list[TopologySpectralExample], dict[str, Any]]:
    """Create variable-length clean-spectrum supervision from actual graph states."""

    cfg = dict(trajectory_config or {})
    spec_cfg = dict(spectral_config or {})
    rng = np.random.default_rng(int(seed))
    examples: list[TopologySpectralExample] = []
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
    require_same_degree_sequence = bool(
        spec_cfg.get("require_same_degree_sequence", True)
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
                    "Completed spectral source/target pairs must have identical size."
                )
            if require_same_degree_sequence:
                assert_same_degree_fibre(source, target)
        else:
            source = None
            target = normalize_topology_graph(raw_item)
            base_generator = "target_degree_havel_hakimi"
            source_index = -1
            target_index = -1
            matching_cost = 0.0

        target_spectrum = laplacian_eigenvalues(target)
        target_graphlet_probabilities: np.ndarray | None = None
        target_graphlet_logits: np.ndarray | None = None
        target_graphlet_mask: np.ndarray | None = None
        if graphlet_basis is not None:
            target_graphlet_probabilities, target_graphlet_mask, _ = (
                extract_topology_graphlet_simplex(
                    target,
                    graphlet_basis=graphlet_basis,
                )
            )
            target_graphlet_logits = graphlet_simplex_to_clr(
                target_graphlet_probabilities,
                graphlet_basis=graphlet_basis,
                epsilon=float(graphlet_logit_epsilon),
                coordinate_mask=target_graphlet_mask,
            )
        degree_sequence = [int(target.degree(node)) for node in target.nodes()]
        for _path in range(paths_per_graph):
            states, report = build_spectral_teacher_states(
                degree_sequence if source is None else None,
                source_graph=source,
                target_graph=target,
                steps=int(cfg.get("steps", 32)),
                proposal_budget=proposal_budget,
                valid_candidate_budget=valid_budget,
                preserve_connectivity=bool(cfg.get("preserve_connectivity", True)),
                ensure_connected_source=bool(cfg.get("ensure_connected_source", True)),
                max_repair_trials=int(cfg.get("max_repair_trials", 10000)),
                random_relabel_source=bool(cfg.get("random_relabel_source", True)),
                source_randomization_steps=int(cfg.get("source_randomization_steps", 0)),
                teacher_mode=str(cfg.get("teacher_mode", "hard")),
                teacher_temperature=float(cfg.get("teacher_temperature", 1.0)),
                teacher_top_k=int(cfg.get("teacher_top_k", 0)),
                teacher_sample_actions=bool(cfg.get("teacher_sample_actions", False)),
                teacher_min_improvement=float(cfg.get("teacher_min_improvement", 0.0)),
                target_tolerance=float(cfg.get("target_tolerance", 0.0)),
                teacher_allow_sideways=bool(
                    cfg.get("teacher_allow_sideways", False)
                ),
                teacher_max_consecutive_sideways=int(
                    cfg.get("teacher_max_consecutive_sideways", 0)
                ),
                teacher_sideways_tolerance=float(
                    cfg.get("teacher_sideways_tolerance", 0.0)
                ),
                teacher_tabu=bool(cfg.get("teacher_tabu", True)),
                distance_metric=str(spec_cfg.get("distance", "rmse")),
                distance_normalization=str(
                    spec_cfg.get("normalization", "mean_degree")
                ),
                low_frequency_weight=float(
                    spec_cfg.get("low_frequency_weight", 1.0)
                ),
                low_frequency_cutoff=int(
                    spec_cfg.get("low_frequency_cutoff", 0)
                ),
                require_same_degree_sequence=require_same_degree_sequence,
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
                current_graphlet_probabilities = None
                current_graphlet_logits = None
                current_graphlet_mask = None
                if graphlet_basis is not None:
                    current_graphlet_probabilities, current_graphlet_mask, _ = (
                        extract_topology_graphlet_simplex(
                            states[step],
                            graphlet_basis=graphlet_basis,
                        )
                    )
                    current_graphlet_logits = graphlet_simplex_to_clr(
                        current_graphlet_probabilities,
                        graphlet_basis=graphlet_basis,
                        epsilon=float(graphlet_logit_epsilon),
                        coordinate_mask=current_graphlet_mask,
                    )
                    if target_graphlet_mask is None or not np.array_equal(
                        current_graphlet_mask, target_graphlet_mask
                    ):
                        raise AssertionError(
                            "Source and clean graphlet coordinate masks must agree for equal-size graphs."
                        )
                examples.append(
                    TopologySpectralExample(
                        current_graph=states[step],
                        time=float(step / horizon),
                        clean_spectrum_target=target_spectrum.astype(np.float32).copy(),
                        current_graphlet_probabilities=(
                            None if current_graphlet_probabilities is None
                            else current_graphlet_probabilities.astype(np.float32).copy()
                        ),
                        clean_graphlet_probabilities_target=(
                            None if target_graphlet_probabilities is None
                            else target_graphlet_probabilities.astype(np.float32).copy()
                        ),
                        current_graphlet_logits=(
                            None if current_graphlet_logits is None
                            else current_graphlet_logits.astype(np.float32).copy()
                        ),
                        clean_graphlet_logits_target=(
                            None if target_graphlet_logits is None
                            else target_graphlet_logits.astype(np.float32).copy()
                        ),
                        graphlet_coordinate_mask=(
                            None if current_graphlet_mask is None
                            else current_graphlet_mask.astype(np.bool_).copy()
                        ),
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
            float(np.mean([r["matching_cost"] for r in reports])) if reports else 0.0
        ),
        "mean_initial_spectral_discrepancy": (
            float(np.mean([r["initial_spectral_discrepancy"] for r in reports]))
            if reports
            else 0.0
        ),
        "mean_final_teacher_spectral_discrepancy": (
            float(
                np.mean([r["final_teacher_spectral_discrepancy"] for r in reports])
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


class TopologySpectralTrajectoryIterableDataset(torch.utils.data.IterableDataset):
    """Generate clean-spectrum teacher examples lazily."""

    def __init__(
        self,
        graphs: Sequence[nx.Graph | TopologyTrainingPair],
        *,
        trajectory_config: dict[str, Any] | None = None,
        spectral_config: dict[str, Any] | None = None,
        graphlet_basis: TopologyGraphletBasis | None = None,
        graphlet_logit_epsilon: float = 1.0e-5,
        seed: int = 0,
        shuffle_graphs: bool = True,
    ) -> None:
        super().__init__()
        self.graphs = tuple(graphs)
        self.trajectory_config = dict(trajectory_config or {})
        self.spectral_config = dict(spectral_config or {})
        self.graphlet_basis = graphlet_basis
        self.graphlet_logit_epsilon = float(graphlet_logit_epsilon)
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
            examples, diagnostics = build_spectral_examples(
                [self.graphs[int(graph_index)]],
                trajectory_config=self.trajectory_config,
                spectral_config=self.spectral_config,
                graphlet_basis=self.graphlet_basis,
                graphlet_logit_epsilon=self.graphlet_logit_epsilon,
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
