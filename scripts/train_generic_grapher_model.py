from __future__ import annotations

import argparse
from dataclasses import dataclass
import random
import sys
import time
from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grapher.evaluation.data_io import load_dataset_splits
from grapher.evaluation.run_utils import make_model_run_config, run_output_dir
from grapher.generation.rewiring import (
    RewireAction,
    action_new_edges,
    action_removed_edges,
    action_signature,
    action_structural_delta,
    connected_sequence_feasible,
    degree_sequence,
    deterministic_connected_havel_hakimi,
    edge_symmetric_difference_size,
    enumerate_rewire_actions,
    graph_to_data,
    merge_action_sets,
    rewire_action,
)
from grapher.models.model_grapher import GraphER
from grapher.registry import available_datasets
from grapher.utils.compute import CudaTrainingDeviceError, PeakMemoryMonitor, compute_report, require_cuda_training_device
from grapher.utils.io import load_yaml, save_json, save_yaml, stable_hash
from grapher.utils.logging import configure_logging, get_logger
from grapher.utils.numerics import assert_model_tensors_finite
from grapher.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class CachedTeacherExample:
    """One cached hard-label rewiring flow-matching example.

    Candidate construction and NetworkX local-feature extraction are CPU-heavy,
    so the training script builds these examples once, then reuses them across
    epochs.  The neural forward pass still runs on the requested CUDA device.
    """

    x: torch.Tensor
    edge_index: torch.Tensor
    actions: list[RewireAction]
    label_idx: int
    t: float
    degree_sequence: list[int]
    action_local_features: torch.Tensor
    train_candidate_size: int
    offline_candidate_size: int
    improvement: float



def _edge_set(graph: nx.Graph) -> set[tuple[int, int]]:
    return {tuple(sorted((int(u), int(v)))) for u, v in graph.edges()}


def _action_is_target_aware(action: RewireAction, wrong_edges: set[tuple[int, int]], missing_edges: set[tuple[int, int]]) -> bool:
    removed = set(action_removed_edges(action))
    created = set(action_new_edges(action))
    return bool(removed & wrong_edges) and bool(created & missing_edges)


@dataclass(frozen=True)
class TeacherDiscrepancyConfig:
    mode: str = "edge_symmetric_difference"
    edge_weight: float = 1.0
    clustering_weight: float = 0.0
    triangle_weight: float = 0.0
    spectral_weight: float = 0.0
    spectral_shortlist: int = 8

    @property
    def edge_only(self) -> bool:
        return self.mode in {"edge", "edge_symmetric_difference", "symmetric_difference"}


@dataclass
class TeacherGraphStats:
    edges: set[tuple[int, int]]
    edge_symdiff: int
    triangle_count: int
    average_clustering: float
    spectrum: np.ndarray | None = None


@dataclass(frozen=True)
class TeacherTargetStats:
    edges: set[tuple[int, int]]
    edge_count: int
    triangle_count: int
    triangle_scale: float
    average_clustering: float
    spectrum: np.ndarray | None = None


def _triangle_count(graph: nx.Graph) -> int:
    return int(sum(int(value) for value in nx.triangles(graph).values()) // 3)


def _triangle_scale(graph: nx.Graph) -> float:
    # The degree sequence is invariant along a trajectory, so the number of
    # wedges is a fixed normalization shared by source, target, and candidates.
    wedges = sum(int(degree) * (int(degree) - 1) / 2.0 for _, degree in graph.degree())
    return max(float(wedges) / 3.0, 1.0)


def _normalized_laplacian_spectrum(graph: nx.Graph) -> np.ndarray:
    node_order = sorted(int(node) for node in graph.nodes())
    n = len(node_order)
    if n == 0:
        return np.empty((0,), dtype=np.float64)
    adjacency = nx.to_numpy_array(graph, nodelist=node_order, dtype=np.float64)
    degrees = adjacency.sum(axis=1)
    inv_sqrt = np.zeros_like(degrees)
    positive = degrees > 0
    inv_sqrt[positive] = 1.0 / np.sqrt(degrees[positive])
    laplacian = np.eye(n, dtype=np.float64) - np.diag(inv_sqrt) @ adjacency @ np.diag(inv_sqrt)
    try:
        return np.linalg.eigvalsh(laplacian)
    except np.linalg.LinAlgError:
        return np.sort(np.real(np.linalg.eigvals(laplacian)))


def _spectral_distance(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    width = min(int(left.size), int(right.size))
    if width <= 0:
        return 0.0
    return float(np.linalg.norm(left[:width] - right[:width]) / np.sqrt(float(width)))


def _make_target_stats(target: nx.Graph, *, include_spectrum: bool) -> TeacherTargetStats:
    edges = _edge_set(target)
    return TeacherTargetStats(
        edges=edges,
        edge_count=int(target.number_of_edges()),
        triangle_count=_triangle_count(target),
        triangle_scale=_triangle_scale(target),
        average_clustering=float(nx.average_clustering(target)),
        spectrum=_normalized_laplacian_spectrum(target) if include_spectrum else None,
    )


def _make_current_stats(
    current: nx.Graph,
    target_stats: TeacherTargetStats,
    *,
    include_spectrum: bool,
) -> TeacherGraphStats:
    edges = _edge_set(current)
    return TeacherGraphStats(
        edges=edges,
        edge_symdiff=int(len(edges ^ target_stats.edges)),
        triangle_count=_triangle_count(current),
        average_clustering=float(nx.average_clustering(current)),
        spectrum=_normalized_laplacian_spectrum(current) if include_spectrum else None,
    )


def _teacher_discrepancy_value(
    state: TeacherGraphStats,
    target: TeacherTargetStats,
    config: TeacherDiscrepancyConfig,
    *,
    include_spectral: bool,
) -> float:
    """Evaluate the configured offline teacher discrepancy.

    Edge-only mode preserves the original raw symmetric-difference scale.  The
    structural mode normalizes all components before weighting them so one term
    cannot dominate merely because it has a larger numerical range.
    """

    if config.edge_only:
        return float(state.edge_symdiff)

    edge_distance = float(state.edge_symdiff) / float(max(2 * target.edge_count, 1))
    clustering_distance = abs(float(state.average_clustering) - float(target.average_clustering))
    triangle_distance = abs(float(state.triangle_count - target.triangle_count)) / float(target.triangle_scale)
    spectral_distance = (
        _spectral_distance(state.spectrum, target.spectrum)
        if include_spectral and config.spectral_weight > 0.0
        else 0.0
    )
    return float(
        config.edge_weight * edge_distance
        + config.clustering_weight * clustering_distance
        + config.triangle_weight * triangle_distance
        + config.spectral_weight * spectral_distance
    )




def _edge_discrepancy_improvement(
    action: RewireAction,
    *,
    current_edges: set[tuple[int, int]],
    target_edges: set[tuple[int, int]],
) -> int:
    """Fast delta for |E(G) symmetric_difference E(target)|.

    Candidate actions have already passed validity checks, so the created edges
    are not present in ``current_edges``.  This avoids copying a graph and
    rebuilding two edge sets for every candidate during teacher selection.
    """

    improvement = 0
    for edge in action_removed_edges(action):
        edge = tuple(sorted((int(edge[0]), int(edge[1]))))
        improvement += 1 if edge not in target_edges else -1
    for edge in action_new_edges(action):
        edge = tuple(sorted((int(edge[0]), int(edge[1]))))
        # Valid candidates should create missing edges, but be defensive.
        if edge in current_edges:
            continue
        improvement += 1 if edge in target_edges else -1
    return int(improvement)


def _candidate_state_from_action(
    current: nx.Graph,
    current_stats: TeacherGraphStats,
    target_stats: TeacherTargetStats,
    action: RewireAction,
    *,
    neighbours: dict[int, set[int]],
    degrees: dict[int, int],
) -> TeacherGraphStats:
    edge_improvement = _edge_discrepancy_improvement(
        action,
        current_edges=current_stats.edges,
        target_edges=target_stats.edges,
    )
    structural = action_structural_delta(
        current,
        action,
        neighbours=neighbours,
        degrees=degrees,
    )
    next_edges = set(current_stats.edges)
    next_edges.difference_update(action_removed_edges(action))
    next_edges.update(action_new_edges(action))
    return TeacherGraphStats(
        edges=next_edges,
        edge_symdiff=max(int(current_stats.edge_symdiff - edge_improvement), 0),
        triangle_count=max(int(current_stats.triangle_count + structural.triangle_delta), 0),
        average_clustering=float(
            min(max(current_stats.average_clustering + structural.average_clustering_delta, 0.0), 1.0)
        ),
        spectrum=None,
    )


def _split_candidate_budget(total: int, target_fraction: float, global_fraction: float) -> tuple[int, int, int]:
    total = max(int(total), 1)
    target_fraction = max(0.0, min(1.0, float(target_fraction)))
    global_fraction = max(0.0, min(1.0, float(global_fraction)))
    target_budget = min(int(round(total * target_fraction)), total)
    global_budget = min(int(round(total * global_fraction)), total - target_budget)
    local_budget = max(total - target_budget - global_budget, 0)
    return target_budget, local_budget, global_budget


def _offline_candidate_actions(
    current: nx.Graph,
    target: nx.Graph,
    *,
    candidate_budget: int,
    offline_candidate_budget: int,
    k_hop: int | None,
    ensure_connected: bool,
    target_candidate_fraction: float,
    global_candidate_fraction: float,
    rng: random.Random,
) -> list[RewireAction]:
    """Build C_off = C_target union C_local union C_rand for teacher search.
    """

    current_edges = _edge_set(current)
    target_edges = _edge_set(target)
    wrong_edges = current_edges - target_edges
    missing_edges = target_edges - current_edges

    target_budget, local_budget, global_budget = _split_candidate_budget(
        int(offline_candidate_budget),
        target_candidate_fraction,
        global_candidate_fraction,
    )

    target_aware: list[RewireAction] = []
    if target_budget > 0 and wrong_edges and missing_edges:
        target_seed = enumerate_rewire_actions(
            current,
            ensure_connected=ensure_connected,
            k_hop=k_hop,
            max_candidates=max(target_budget * 4, target_budget),
            anchor_edges=sorted(wrong_edges),
            rng=rng,
            shuffle=True,
        )
        target_aware = [
            action
            for action in target_seed
            if _action_is_target_aware(action, wrong_edges, missing_edges)
        ][:target_budget]

    local_actions = enumerate_rewire_actions(
        current,
        ensure_connected=ensure_connected,
        k_hop=k_hop,
        max_candidates=local_budget,
        rng=rng,
        shuffle=True,
    ) if local_budget > 0 else []
    global_actions = enumerate_rewire_actions(
        current,
        ensure_connected=ensure_connected,
        k_hop=None,
        max_candidates=global_budget,
        rng=rng,
        shuffle=True,
    ) if global_budget > 0 else []

    actions = merge_action_sets(
        target_aware,
        local_actions,
        global_actions,
        max_candidates=int(offline_candidate_budget),
    )
    # Duplicates can leave the merged set below budget. Fill any remainder with
    # unrestricted random valid actions rather than silently shrinking C_off.
    remaining = int(offline_candidate_budget) - len(actions)
    if remaining > 0:
        fallback = enumerate_rewire_actions(
            current,
            ensure_connected=ensure_connected,
            k_hop=None if global_candidate_fraction > 0.0 else k_hop,
            max_candidates=max(remaining * 2, remaining),
            rng=rng,
            shuffle=True,
        )
        actions = merge_action_sets(actions, fallback, max_candidates=int(offline_candidate_budget))
    return actions


def _training_candidate_actions(
    current: nx.Graph,
    teacher_action: RewireAction,
    *,
    candidate_budget: int,
    k_hop: int | None,
    ensure_connected: bool,
    global_candidate_fraction: float,
    rng: random.Random,
    shuffle: bool,
) -> tuple[list[RewireAction], int]:
    """Build C_train = {a*} union C_local union C_rand with no target graph."""

    total_budget = max(int(candidate_budget), 1)
    negative_budget = max(total_budget - 1, 0)
    global_budget = min(
        int(round(negative_budget * max(0.0, min(1.0, float(global_candidate_fraction))))),
        negative_budget,
    )
    local_budget = max(negative_budget - global_budget, 0)
    local_negatives = enumerate_rewire_actions(
        current,
        ensure_connected=ensure_connected,
        k_hop=k_hop,
        max_candidates=max(local_budget * 2, local_budget),
        rng=rng,
        shuffle=True,
    ) if local_budget > 0 else []
    global_negatives = enumerate_rewire_actions(
        current,
        ensure_connected=ensure_connected,
        k_hop=None,
        max_candidates=max(global_budget * 2, global_budget),
        rng=rng,
        shuffle=True,
    ) if global_budget > 0 else []
    actions = merge_action_sets(
        [teacher_action],
        local_negatives,
        global_negatives,
        max_candidates=total_budget,
    )
    remaining = total_budget - len(actions)
    if remaining > 0:
        fallback = enumerate_rewire_actions(
            current,
            ensure_connected=ensure_connected,
            k_hop=None if global_candidate_fraction > 0.0 else k_hop,
            max_candidates=max(remaining * 2, remaining),
            rng=rng,
            shuffle=True,
        )
        actions = merge_action_sets(actions, fallback, max_candidates=total_budget)
    if shuffle:
        rng.shuffle(actions)
    label_idx = next(i for i, action in enumerate(actions) if action_signature(action) == action_signature(teacher_action))
    return actions, label_idx


def _candidate_teacher_step(
    current: nx.Graph,
    target: nx.Graph,
    *,
    candidate_budget: int,
    offline_candidate_budget: int,
    k_hop: int | None,
    ensure_connected: bool,
    discrepancy: TeacherDiscrepancyConfig,
    target_stats: TeacherTargetStats,
    current_stats: TeacherGraphStats,
    target_candidate_fraction: float,
    global_candidate_fraction: float,
    require_positive_improvement: bool,
    rng: random.Random,
) -> tuple[list[RewireAction], int, RewireAction, float, nx.Graph, TeacherGraphStats, int] | None:
    """Return one hard-label rewiring flow-matching example.

    The target graph is used only to select the teacher action from C_off.  The
    returned training candidate set is target-free and contains the teacher
    action by construction.
    """

    offline_actions = _offline_candidate_actions(
        current,
        target,
        candidate_budget=candidate_budget,
        offline_candidate_budget=offline_candidate_budget,
        k_hop=k_hop,
        ensure_connected=ensure_connected,
        target_candidate_fraction=target_candidate_fraction,
        global_candidate_fraction=global_candidate_fraction,
        rng=rng,
    )
    if not offline_actions:
        return None
    neighbours = {int(node): {int(v) for v in current.neighbors(node)} for node in current.nodes()}
    degrees = {int(node): int(value) for node, value in current.degree()}
    current_score_cheap = _teacher_discrepancy_value(
        current_stats,
        target_stats,
        discrepancy,
        include_spectral=False,
    )
    evaluations: list[tuple[float, RewireAction, TeacherGraphStats]] = []
    for action in offline_actions:
        next_stats = _candidate_state_from_action(
            current,
            current_stats,
            target_stats,
            action,
            neighbours=neighbours,
            degrees=degrees,
        )
        next_score_cheap = _teacher_discrepancy_value(
            next_stats,
            target_stats,
            discrepancy,
            include_spectral=False,
        )
        evaluations.append((float(current_score_cheap - next_score_cheap), action, next_stats))

    if not evaluations:
        return None

    if discrepancy.spectral_weight > 0.0:
        if current_stats.spectrum is None:
            current_stats.spectrum = _normalized_laplacian_spectrum(current)
        current_score_full = _teacher_discrepancy_value(
            current_stats,
            target_stats,
            discrepancy,
            include_spectral=True,
        )
        shortlist_size = min(max(int(discrepancy.spectral_shortlist), 1), len(evaluations))
        shortlist = sorted(evaluations, key=lambda item: item[0], reverse=True)[:shortlist_size]
        best_full: tuple[float, RewireAction, TeacherGraphStats, nx.Graph] | None = None
        for _, action, next_stats in shortlist:
            out = rewire_action(current, action, ensure_connected=ensure_connected)
            if out is None:
                continue
            candidate_graph = out[0]
            next_stats.spectrum = _normalized_laplacian_spectrum(candidate_graph)
            next_score_full = _teacher_discrepancy_value(
                next_stats,
                target_stats,
                discrepancy,
                include_spectral=True,
            )
            full_improvement = float(current_score_full - next_score_full)
            if best_full is None or full_improvement > best_full[0]:
                best_full = (full_improvement, action, next_stats, candidate_graph)
        if best_full is None:
            return None
        improvement, teacher_action, next_stats, next_graph = best_full
    else:
        improvement, teacher_action, next_stats = max(evaluations, key=lambda item: item[0])
        out = rewire_action(current, teacher_action, ensure_connected=ensure_connected)
        if out is None:
            return None
        next_graph = out[0]

    if require_positive_improvement and improvement <= 0.0:
        return None
    train_actions, label_idx = _training_candidate_actions(
        current,
        teacher_action,
        candidate_budget=candidate_budget,
        k_hop=k_hop,
        ensure_connected=ensure_connected,
        global_candidate_fraction=global_candidate_fraction,
        rng=rng,
        shuffle=True,
    )
    return (
        train_actions,
        label_idx,
        teacher_action,
        float(improvement),
        next_graph,
        next_stats,
        len(offline_actions),
    )


def _usable_training_graphs(graphs: Sequence[nx.Graph], max_graphs: int | None) -> tuple[list[nx.Graph], dict[str, int]]:
    usable: list[nx.Graph] = []
    skipped: dict[str, int] = {}
    for graph in graphs:
        g = nx.convert_node_labels_to_integers(nx.Graph(graph), ordering="sorted")
        if g.number_of_nodes() <= 1 or g.number_of_edges() < 2:
            skipped["too_small"] = skipped.get("too_small", 0) + 1
            continue
        seq = degree_sequence(g)
        feasible, reason = connected_sequence_feasible(seq)
        if not feasible:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        if not nx.is_connected(g):
            skipped["target_graph_disconnected"] = skipped.get("target_graph_disconnected", 0) + 1
            continue
        usable.append(g)
        if max_graphs is not None and len(usable) >= int(max_graphs):
            break
    return usable, skipped




def _build_teacher_cache(
    *,
    train_graphs: Sequence[nx.Graph],
    model: GraphER,
    k_eigen: int,
    max_steps: int,
    candidate_budget: int,
    offline_candidate_budget: int,
    k_hop: int | None,
    ensure_connected: bool,
    discrepancy: TeacherDiscrepancyConfig,
    target_candidate_fraction: float,
    global_candidate_fraction: float,
    require_positive_improvement: bool,
    rng: random.Random,
) -> tuple[list[CachedTeacherExample], dict[str, float | int]]:
    """Construct offline teacher examples once before neural optimization.

    This follows the paper's offline trajectory cache.  It removes the largest
    training bottleneck in the original script, which rebuilt the same
    Havel-Hakimi-to-data teacher paths in every epoch.
    """

    cache: list[CachedTeacherExample] = []
    num_exact = 0
    num_positive = 0
    num_hh_failures = 0
    num_no_teacher_step = 0
    num_graphs_with_examples = 0
    train_candidate_sizes: list[int] = []
    offline_candidate_sizes: list[int] = []
    improvements: list[float] = []
    path_lengths: list[int] = []
    minimum_swap_lower_bounds: list[int] = []
    initial_edge_symdiffs: list[int] = []
    final_edge_symdiffs: list[int] = []
    initial_clustering_gaps: list[float] = []
    final_clustering_gaps: list[float] = []
    initial_triangle_gaps: list[float] = []
    final_triangle_gaps: list[float] = []
    initial_discrepancies: list[float] = []
    final_discrepancies: list[float] = []
    num_reached_exact_target = 0
    cpu_device = torch.device("cpu")

    for graph_idx, target_raw in enumerate(train_graphs, start=1):
        target = nx.convert_node_labels_to_integers(nx.Graph(target_raw), ordering="sorted")
        try:
            current = deterministic_connected_havel_hakimi(G=target)
        except Exception:
            num_hh_failures += 1
            logger.debug(
                "GraphER cache graph=%d HH construction failed nodes=%d edges=%d",
                graph_idx,
                target.number_of_nodes(),
                target.number_of_edges(),
            )
            continue

        target_stats = _make_target_stats(
            target,
            include_spectrum=discrepancy.spectral_weight > 0.0,
        )
        current_stats = _make_current_stats(
            current,
            target_stats,
            include_spectrum=discrepancy.spectral_weight > 0.0,
        )
        initial_edge_symdiff = int(current_stats.edge_symdiff)
        initial_clustering_gap = abs(current_stats.average_clustering - target_stats.average_clustering)
        initial_triangle_gap = abs(current_stats.triangle_count - target_stats.triangle_count) / target_stats.triangle_scale
        initial_discrepancy = _teacher_discrepancy_value(
            current_stats,
            target_stats,
            discrepancy,
            include_spectral=discrepancy.spectral_weight > 0.0,
        )

        if initial_edge_symdiff == 0:
            num_exact += 1
            num_reached_exact_target += 1
            path_lengths.append(0)
            minimum_swap_lower_bounds.append(0)
            initial_edge_symdiffs.append(0)
            final_edge_symdiffs.append(0)
            initial_clustering_gaps.append(float(initial_clustering_gap))
            final_clustering_gaps.append(float(initial_clustering_gap))
            initial_triangle_gaps.append(float(initial_triangle_gap))
            final_triangle_gaps.append(float(initial_triangle_gap))
            initial_discrepancies.append(float(initial_discrepancy))
            final_discrepancies.append(float(initial_discrepancy))
            logger.debug("GraphER cache graph=%d HH source already exact", graph_idx)
            continue

        target_degree_sequence = degree_sequence(target)
        examples_before = len(cache)
        for step in range(max_steps):
            current_distance = int(current_stats.edge_symdiff)
            example = _candidate_teacher_step(
                current,
                target,
                candidate_budget=candidate_budget,
                offline_candidate_budget=offline_candidate_budget,
                k_hop=k_hop,
                ensure_connected=ensure_connected,
                discrepancy=discrepancy,
                target_stats=target_stats,
                current_stats=current_stats,
                target_candidate_fraction=target_candidate_fraction,
                global_candidate_fraction=global_candidate_fraction,
                require_positive_improvement=require_positive_improvement,
                rng=rng,
            )
            if example is None:
                num_no_teacher_step += 1
                logger.debug(
                    "GraphER cache graph=%d step=%d no teacher action current_distance=%d",
                    graph_idx,
                    step,
                    current_distance,
                )
                break

            (
                train_actions,
                label_idx,
                _teacher_action,
                improvement,
                next_graph,
                next_stats,
                offline_size,
            ) = example
            data = graph_to_data(current, k_eigen=k_eigen, include_edge_pairs=False)
            with torch.no_grad():
                local_features = model.action_local_feature_matrix(current, train_actions, device=cpu_device).cpu()
            cache.append(
                CachedTeacherExample(
                    x=data.x.detach().cpu(),
                    edge_index=data.edge_index.detach().cpu(),
                    actions=list(train_actions),
                    label_idx=int(label_idx),
                    # Normalized after the realized trajectory length is known.
                    t=0.0,
                    degree_sequence=list(map(int, target_degree_sequence)),
                    action_local_features=local_features,
                    train_candidate_size=len(train_actions),
                    offline_candidate_size=int(offline_size),
                    improvement=float(improvement),
                )
            )
            num_positive += int(improvement > 0.0)
            train_candidate_sizes.append(len(train_actions))
            offline_candidate_sizes.append(int(offline_size))
            improvements.append(float(improvement))
            current = nx.convert_node_labels_to_integers(next_graph, ordering="sorted")
            current_stats = next_stats
            if current_stats.edge_symdiff == 0:
                break

        path_length = len(cache) - examples_before
        path_lengths.append(int(path_length))
        minimum_swap_lower_bounds.append(int((initial_edge_symdiff + 3) // 4))
        initial_edge_symdiffs.append(int(initial_edge_symdiff))
        final_edge_symdiffs.append(int(current_stats.edge_symdiff))
        initial_clustering_gaps.append(float(initial_clustering_gap))
        final_clustering_gaps.append(
            float(abs(current_stats.average_clustering - target_stats.average_clustering))
        )
        initial_triangle_gaps.append(float(initial_triangle_gap))
        final_triangle_gaps.append(
            float(abs(current_stats.triangle_count - target_stats.triangle_count) / target_stats.triangle_scale)
        )
        initial_discrepancies.append(float(initial_discrepancy))
        final_discrepancies.append(
            float(
                _teacher_discrepancy_value(
                    current_stats,
                    target_stats,
                    discrepancy,
                    include_spectral=discrepancy.spectral_weight > 0.0,
                )
            )
        )
        num_reached_exact_target += int(current_stats.edge_symdiff == 0)

        # The paper uses t_s=s/L for each realized path.  Using s/L_max leaves
        # short trajectories concentrated near t=0 and creates a train/sample
        # mismatch because generation spans the whole [0,1] interval.
        if path_length > 0:
            for local_step in range(path_length):
                cache[examples_before + local_step].t = float(local_step) / float(path_length)
            num_graphs_with_examples += 1
        if graph_idx == 1 or graph_idx % 10 == 0 or graph_idx == len(train_graphs):
            logger.info(
                "GraphER teacher cache progress graph=%d/%d examples=%d graphs_with_examples=%d",
                graph_idx,
                len(train_graphs),
                len(cache),
                num_graphs_with_examples,
            )

    def mean(values: Sequence[float | int]) -> float:
        return float(np.mean(values)) if values else 0.0

    def median(values: Sequence[float | int]) -> float:
        return float(np.median(values)) if values else 0.0

    def percentile(values: Sequence[float | int], q: float) -> float:
        return float(np.percentile(values, q)) if values else 0.0

    stats: dict[str, float | int] = {
        "num_teacher_examples": len(cache),
        "num_hh_already_exact": num_exact,
        "num_positive_teacher_steps": num_positive,
        "num_hh_failures": num_hh_failures,
        "num_no_teacher_step": num_no_teacher_step,
        "num_graphs_with_examples": num_graphs_with_examples,
        "avg_train_candidate_size": sum(train_candidate_sizes) / max(len(train_candidate_sizes), 1),
        "avg_offline_candidate_size": sum(offline_candidate_sizes) / max(len(offline_candidate_sizes), 1),
        "avg_teacher_improvement": sum(improvements) / max(len(improvements), 1),
        "avg_teacher_path_length": mean(path_lengths),
        "median_teacher_path_length": median(path_lengths),
        "p90_teacher_path_length": percentile(path_lengths, 90.0),
        "avg_minimum_swap_lower_bound": mean(minimum_swap_lower_bounds),
        "median_minimum_swap_lower_bound": median(minimum_swap_lower_bounds),
        "p90_minimum_swap_lower_bound": percentile(minimum_swap_lower_bounds, 90.0),
        "avg_initial_edge_symdiff": mean(initial_edge_symdiffs),
        "avg_final_edge_symdiff": mean(final_edge_symdiffs),
        "avg_edge_symdiff_reduction": mean(initial_edge_symdiffs) - mean(final_edge_symdiffs),
        "avg_initial_clustering_gap": mean(initial_clustering_gaps),
        "avg_final_clustering_gap": mean(final_clustering_gaps),
        "avg_clustering_gap_reduction": mean(initial_clustering_gaps) - mean(final_clustering_gaps),
        "avg_initial_triangle_gap": mean(initial_triangle_gaps),
        "avg_final_triangle_gap": mean(final_triangle_gaps),
        "avg_triangle_gap_reduction": mean(initial_triangle_gaps) - mean(final_triangle_gaps),
        "avg_initial_teacher_discrepancy": mean(initial_discrepancies),
        "avg_final_teacher_discrepancy": mean(final_discrepancies),
        "avg_teacher_discrepancy_reduction": mean(initial_discrepancies) - mean(final_discrepancies),
        "num_reached_exact_target": int(num_reached_exact_target),
        "fraction_reached_exact_target": float(num_reached_exact_target) / float(max(len(path_lengths), 1)),
    }
    return cache, stats


def _num_trainable_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def train_grapher(
    *,
    dataset: str,
    model_config: dict,
    dataset_config_path: Path,
    dataset_root: str,
    seed: int,
    run_id: int | None,
    device: str,
) -> dict:
    device = require_cuda_training_device(device)
    logger.info(
        "Starting GraphER training dataset=%s run_id=%s seed=%s device=%s dataset_root=%s dataset_config=%s",
        dataset,
        run_id,
        seed,
        device,
        dataset_root,
        dataset_config_path,
    )
    set_seed(seed, include_torch=True)
    rng = random.Random(int(seed))
    cfg = make_model_run_config(model_config, dataset=dataset, model="grapher", run_id=run_id, seed=seed, use_run_paths=run_id is not None)
    logger.debug("Resolved GraphER config: %s", cfg)
    splits = load_dataset_splits(dataset, output_root=dataset_root, build_if_missing=True, config_path=dataset_config_path)
    logger.info("Loaded dataset splits: %s", {split: len(graphs) for split, graphs in splits.items()})
    max_graphs = cfg.get("max_graphs")
    max_graphs = None if max_graphs in (None, "") else int(max_graphs)
    train_graphs, skipped = _usable_training_graphs(splits["train"], max_graphs=max_graphs)
    if not train_graphs:
        raise ValueError(f"No usable connected training graphs for GraphER. Skipped counts: {skipped}")
    node_counts = [g.number_of_nodes() for g in train_graphs]
    edge_counts = [g.number_of_edges() for g in train_graphs]
    logger.info(
        "Usable GraphER train graphs used=%d skipped=%d skipped_by_reason=%s nodes=(%d,%d) edges=(%d,%d)",
        len(train_graphs),
        int(sum(skipped.values())),
        skipped,
        int(min(node_counts)),
        int(max(node_counts)),
        int(min(edge_counts)),
        int(max(edge_counts)),
    )

    k_eigen = int(cfg.get("k_eigen", 4))
    first_data = graph_to_data(train_graphs[0], k_eigen=k_eigen)
    node_in_dim = int(first_data.x.size(1))
    max_nodes = int(cfg.get("max_nodes") or max(g.number_of_nodes() for g in train_graphs))
    degree_histogram_dim = max_nodes
    logger.info(
        "GraphER feature dimensions node_in_dim=%d k_eigen=%d max_nodes=%d degree_histogram_dim=%d first_edge_index=%s",
        node_in_dim,
        k_eigen,
        max_nodes,
        degree_histogram_dim,
        tuple(first_data.edge_index.shape),
    )
    model = GraphER(
        node_in_dim=node_in_dim,
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        num_layer=int(cfg.get("num_layers", cfg.get("num_layer", 3))),
        T=int(cfg.get("num_steps", cfg.get("T", 32))),
        max_nodes=max_nodes,
        degree_histogram_dim=degree_histogram_dim,
        time_embedding_dim=int(cfg.get("time_embedding_dim", cfg.get("hidden_dim", 128))),
        local_feature_dim=int(cfg.get("local_feature_dim", 8)),
        dropout=float(cfg.get("dropout", 0.0)),
    )
    model.to(device)
    logger.info(
        "GraphER model params=%d hidden_dim=%d layers=%d T=%d local_feature_dim=%d",
        _num_trainable_parameters(model),
        int(cfg.get("hidden_dim", 128)),
        int(cfg.get("num_layers", cfg.get("num_layer", 3))),
        int(cfg.get("num_steps", cfg.get("T", 32))),
        int(cfg.get("local_feature_dim", 8)),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)), weight_decay=float(cfg.get("weight_decay", 0.0)))
    epochs = int(cfg.get("epochs", 5))
    max_steps = int(cfg.get("max_steps_per_graph", 8))
    candidate_budget = int(cfg.get("candidate_budget", 64))
    offline_candidate_budget = int(cfg.get("offline_candidate_budget", max(candidate_budget * 4, candidate_budget + 16)))
    k_hop_raw = cfg.get("k_hop", 2)
    k_hop = None if k_hop_raw in (None, "", "none", "None") else int(k_hop_raw)
    ensure_connected = bool(cfg.get("ensure_connected", True))
    T = int(cfg.get("num_steps", cfg.get("T", 32)))
    discrepancy_mode = str(cfg.get("teacher_discrepancy", "edge_symmetric_difference")).lower()
    supported_discrepancies = {
        "edge",
        "edge_symmetric_difference",
        "symmetric_difference",
        "structural",
        "edge_clustering",
        "edge_triangle_clustering",
    }
    if discrepancy_mode not in supported_discrepancies:
        raise ValueError(
            f"Unsupported teacher_discrepancy={discrepancy_mode!r}; "
            f"choose one of {sorted(supported_discrepancies)}."
        )
    if discrepancy_mode in {"edge", "edge_symmetric_difference", "symmetric_difference"}:
        discrepancy = TeacherDiscrepancyConfig(mode=discrepancy_mode)
    else:
        default_triangle_weight = 0.0 if discrepancy_mode == "edge_clustering" else 0.5
        discrepancy = TeacherDiscrepancyConfig(
            mode=discrepancy_mode,
            edge_weight=float(cfg.get("teacher_edge_weight", 0.10)),
            clustering_weight=float(cfg.get("teacher_clustering_weight", 1.0)),
            triangle_weight=float(cfg.get("teacher_triangle_weight", default_triangle_weight)),
            spectral_weight=float(cfg.get("teacher_spectral_weight", 0.0)),
            spectral_shortlist=max(int(cfg.get("teacher_spectral_shortlist", 8)), 1),
        )
        if (
            discrepancy.edge_weight <= 0.0
            and discrepancy.clustering_weight <= 0.0
            and discrepancy.triangle_weight <= 0.0
            and discrepancy.spectral_weight <= 0.0
        ):
            raise ValueError("At least one structural teacher discrepancy weight must be positive.")
    target_candidate_fraction = float(cfg.get("target_candidate_fraction", 0.25 if not discrepancy.edge_only else 0.5))
    global_candidate_fraction = float(cfg.get("global_candidate_fraction", 0.0))
    require_positive_improvement = bool(cfg.get("require_positive_teacher_improvement", False))
    logger.info(
        "GraphER optimization epochs=%d max_steps_per_graph=%d lr=%g weight_decay=%g grad_clip=%g candidate_budget=%d offline_candidate_budget=%d k_hop=%s ensure_connected=%s discrepancy=%s weights=(edge=%.3f,cluster=%.3f,triangle=%.3f,spectral=%.3f) target_fraction=%.2f global_fraction=%.2f require_positive=%s",
        epochs,
        max_steps,
        float(cfg.get("learning_rate", 1e-3)),
        float(cfg.get("weight_decay", 0.0)),
        float(cfg.get("grad_clip", 5.0)),
        candidate_budget,
        offline_candidate_budget,
        k_hop,
        ensure_connected,
        discrepancy.mode,
        discrepancy.edge_weight,
        discrepancy.clustering_weight,
        discrepancy.triangle_weight,
        discrepancy.spectral_weight,
        target_candidate_fraction,
        global_candidate_fraction,
        require_positive_improvement,
    )

    grad_accum_steps = max(int(cfg.get("grad_accum_steps", cfg.get("batch_size", 16))), 1)
    use_amp = bool(cfg.get("use_amp", True)) and str(device).startswith("cuda") and torch.cuda.is_available()
    examples_per_epoch_raw = cfg.get("examples_per_epoch")
    examples_per_epoch = None if examples_per_epoch_raw in (None, "") else int(examples_per_epoch_raw)
    logger.info(
        "GraphER training runtime grad_accum_steps=%d use_amp=%s examples_per_epoch=%s",
        grad_accum_steps,
        use_amp,
        examples_per_epoch,
    )

    history: list[dict] = []
    start = time.perf_counter()
    with PeakMemoryMonitor() as memory_monitor:
        cache_start = time.perf_counter()
        teacher_cache, cache_stats = _build_teacher_cache(
            train_graphs=train_graphs,
            model=model,
            k_eigen=k_eigen,
            max_steps=max_steps,
            candidate_budget=candidate_budget,
            offline_candidate_budget=offline_candidate_budget,
            k_hop=k_hop,
            ensure_connected=ensure_connected,
            discrepancy=discrepancy,
            target_candidate_fraction=target_candidate_fraction,
            global_candidate_fraction=global_candidate_fraction,
            require_positive_improvement=require_positive_improvement,
            rng=rng,
        )
        cache_seconds = time.perf_counter() - cache_start
        if not teacher_cache:
            raise ValueError(f"No GraphER teacher examples were constructed. Cache stats: {cache_stats}")
        logger.info(
            "GraphER teacher cache built examples=%d graphs_with_examples=%d exact=%d hh_fail=%d no_teacher=%d avgC=%.1f avgCoff=%.1f avg_delta=%.4f avg_path=%.1f min_swap_lb=%.1f edge_symdiff=%.1f->%.1f clustering_gap=%.4f->%.4f seconds=%.2f",
            len(teacher_cache),
            int(cache_stats["num_graphs_with_examples"]),
            int(cache_stats["num_hh_already_exact"]),
            int(cache_stats["num_hh_failures"]),
            int(cache_stats["num_no_teacher_step"]),
            float(cache_stats["avg_train_candidate_size"]),
            float(cache_stats["avg_offline_candidate_size"]),
            float(cache_stats["avg_teacher_improvement"]),
            float(cache_stats["avg_teacher_path_length"]),
            float(cache_stats["avg_minimum_swap_lower_bound"]),
            float(cache_stats["avg_initial_edge_symdiff"]),
            float(cache_stats["avg_final_edge_symdiff"]),
            float(cache_stats["avg_initial_clustering_gap"]),
            float(cache_stats["avg_final_clustering_gap"]),
            cache_seconds,
        )

        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            epoch_examples = list(teacher_cache)
            rng.shuffle(epoch_examples)
            if examples_per_epoch is not None:
                epoch_examples = epoch_examples[: max(1, int(examples_per_epoch))]
            logger.info("GraphER epoch %d/%d starting cached_examples=%d", epoch, epochs, len(epoch_examples))

            total_loss = 0.0
            num_examples = 0
            num_zero_logits = 0
            num_top1 = 0
            num_top5 = 0
            total_reciprocal_rank = 0.0
            total_action_entropy = 0.0
            total_teacher_margin = 0.0
            optimizer.zero_grad(set_to_none=True)
            pending_grads = 0
            for example_idx, example in enumerate(epoch_examples, start=1):
                x = example.x.to(device, non_blocking=True)
                edge_index = example.edge_index.to(device, non_blocking=True)
                local_features = example.action_local_features.to(device, non_blocking=True)
                target_idx = torch.tensor([int(example.label_idx)], dtype=torch.long, device=device)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model.score_actions(
                        x,
                        edge_index,
                        example.actions,
                        t=example.t,
                        degree_sequence=example.degree_sequence,
                        graph=None,
                        action_local_features=local_features,
                    )
                    if logits.numel() == 0:
                        num_zero_logits += 1
                        continue
                    raw_loss = F.cross_entropy(logits.view(1, -1), target_idx)
                    loss = raw_loss / float(grad_accum_steps)
                if not torch.isfinite(raw_loss.detach()):
                    raise FloatingPointError(f"Non-finite GraphER loss at epoch {epoch} example {example_idx}.")

                with torch.no_grad():
                    detached_logits = logits.detach().float()
                    label = int(example.label_idx)
                    teacher_logit = detached_logits[label]
                    rank = 1 + int((detached_logits > teacher_logit).sum().item())
                    num_top1 += int(rank == 1)
                    num_top5 += int(rank <= min(5, int(detached_logits.numel())))
                    total_reciprocal_rank += 1.0 / float(rank)
                    probabilities = F.softmax(detached_logits, dim=0)
                    total_action_entropy += float(
                        (-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item()
                    )
                    if detached_logits.numel() > 1:
                        negative_mask = torch.ones_like(detached_logits, dtype=torch.bool)
                        negative_mask[label] = False
                        best_negative = detached_logits[negative_mask].max()
                        total_teacher_margin += float((teacher_logit - best_negative).item())

                scaler.scale(loss).backward()
                total_loss += float(raw_loss.detach().item())
                num_examples += 1
                pending_grads += 1
                should_step = pending_grads >= grad_accum_steps or example_idx == len(epoch_examples)
                if should_step and pending_grads > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.get("grad_clip", 5.0)))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    pending_grads = 0

            row = {
                "epoch": epoch,
                "loss": total_loss / max(num_examples, 1),
                "num_teacher_examples": num_examples,
                "num_cached_teacher_examples": len(teacher_cache),
                "num_hh_already_exact": int(cache_stats["num_hh_already_exact"]),
                "num_positive_teacher_steps": int(cache_stats["num_positive_teacher_steps"]),
                "num_hh_failures": int(cache_stats["num_hh_failures"]),
                "num_no_teacher_step": int(cache_stats["num_no_teacher_step"]),
                "num_zero_logits": num_zero_logits,
                "num_graphs_with_examples": int(cache_stats["num_graphs_with_examples"]),
                "avg_train_candidate_size": float(cache_stats["avg_train_candidate_size"]),
                "avg_offline_candidate_size": float(cache_stats["avg_offline_candidate_size"]),
                "avg_teacher_improvement": float(cache_stats["avg_teacher_improvement"]),
                "teacher_top1_accuracy": float(num_top1) / float(max(num_examples, 1)),
                "teacher_top5_accuracy": float(num_top5) / float(max(num_examples, 1)),
                "teacher_mean_reciprocal_rank": total_reciprocal_rank / float(max(num_examples, 1)),
                "avg_action_entropy": total_action_entropy / float(max(num_examples, 1)),
                "avg_teacher_logit_margin": total_teacher_margin / float(max(num_examples, 1)),
                "avg_teacher_path_length": float(cache_stats["avg_teacher_path_length"]),
                "avg_minimum_swap_lower_bound": float(cache_stats["avg_minimum_swap_lower_bound"]),
                "avg_initial_edge_symdiff": float(cache_stats["avg_initial_edge_symdiff"]),
                "avg_final_edge_symdiff": float(cache_stats["avg_final_edge_symdiff"]),
                "avg_initial_clustering_gap": float(cache_stats["avg_initial_clustering_gap"]),
                "avg_final_clustering_gap": float(cache_stats["avg_final_clustering_gap"]),
                "cache_seconds": cache_seconds,
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
            history.append(row)
            logger.info(
                "GraphER epoch %d/%d loss=%.4f top1=%.3f top5=%.3f mrr=%.3f entropy=%.3f margin=%.3f examples=%d cached=%d zero_logits=%d avgC=%.1f avgCoff=%.1f avg_delta=%.4f seconds=%.2f",
                epoch,
                epochs,
                row["loss"],
                row["teacher_top1_accuracy"],
                row["teacher_top5_accuracy"],
                row["teacher_mean_reciprocal_rank"],
                row["avg_action_entropy"],
                row["avg_teacher_logit_margin"],
                num_examples,
                len(teacher_cache),
                num_zero_logits,
                row["avg_train_candidate_size"],
                row["avg_offline_candidate_size"],
                row["avg_teacher_improvement"],
                row["epoch_seconds"],
            )
        assert_model_tensors_finite(model, context=f"grapher/{dataset}")
    elapsed = time.perf_counter() - start

    checkpoint_path = Path(cfg.get("checkpoint_path") or f"outputs/checkpoints/{dataset}/grapher/grapher.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": "grapher",
        "model_architecture": GraphER.architecture,
        "model_state_dict": model.state_dict(),
        "model_params": {
            "architecture": GraphER.architecture,
            "node_in_dim": node_in_dim,
            "hidden_dim": int(cfg.get("hidden_dim", 128)),
            "num_layer": int(cfg.get("num_layers", cfg.get("num_layer", 3))),
            "T": T,
            "k_eigen": k_eigen,
            "max_nodes": max_nodes,
            "degree_histogram_dim": degree_histogram_dim,
            "time_embedding_dim": int(cfg.get("time_embedding_dim", cfg.get("hidden_dim", 128))),
            "local_feature_dim": int(cfg.get("local_feature_dim", 8)),
            "dropout": float(cfg.get("dropout", 0.0)),
        },
        "dataset": dataset,
        "seed": seed,
        "run_id": run_id,
        "training_config": cfg,
        "training_history": history,
        "training_graph_stats": {
            "num_train_graphs": len(splits["train"]),
            "num_used_graphs": len(train_graphs),
            "num_skipped": int(sum(skipped.values())),
            "skipped_by_reason": skipped,
            "max_nodes": max_nodes,
            "degree_histogram_dim": degree_histogram_dim,
            "candidate_action_type": "complete_double_edge_swap_(e1,e2,r)",
            "teacher_candidate_set": "target-aware plus target-free local/global valid candidates",
            "training_candidate_set": "teacher action plus target-free local/global negatives",
            "teacher_discrepancy": {
                "mode": discrepancy.mode,
                "edge_weight": discrepancy.edge_weight,
                "clustering_weight": discrepancy.clustering_weight,
                "triangle_weight": discrepancy.triangle_weight,
                "spectral_weight": discrepancy.spectral_weight,
                "spectral_shortlist": discrepancy.spectral_shortlist,
            },
            "target_candidate_fraction": target_candidate_fraction,
            "global_candidate_fraction": global_candidate_fraction,
            "num_cached_teacher_examples": len(teacher_cache),
            "teacher_cache_seconds": cache_seconds,
            "teacher_cache_stats": cache_stats,
            "gradient_accumulation_steps": grad_accum_steps,
            "mixed_precision_amp": use_amp,
            "gpu_required_by_training_script": True,
        },
    }
    torch.save(payload, checkpoint_path)
    logger.info("Wrote GraphER checkpoint payload keys=%s", sorted(payload.keys()))

    run_dir = run_output_dir(dataset, "grapher", run_id=run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, run_dir / "resolved_model_config.yaml", force=True)
    compute = compute_report(operation="training", runtime_seconds=elapsed, num_graphs=len(train_graphs), memory=memory_monitor.to_dict())
    metadata = {
        "dataset": dataset,
        "model": "grapher",
        "model_architecture": GraphER.architecture,
        "seed": seed,
        "run_id": run_id,
        "runtime_seconds": elapsed,
        "checkpoint_path": str(checkpoint_path),
        "model_config_hash": stable_hash(cfg),
        "compute": compute,
        "history_tail": history[-5:],
        "training_graph_stats": payload["training_graph_stats"],
    }
    save_json(metadata, run_dir / "train_metadata.json", force=True)
    logger.info("Saved GraphER checkpoint to %s", checkpoint_path)
    logger.info("Saved GraphER metadata to %s", run_dir / "train_metadata.json")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the generic GraphER complete-action rewiring scorer from HH-to-data teacher steps."
    )
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--model-config", type=str, default="configs/models/grapher_generic.yaml")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default="outputs/datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    args = parser.parse_args()

    if args.debug:
        configure_logging("DEBUG")

    dataset_root = args.dataset_root
    dataset_cfg = (
        Path(args.dataset_config)
        if args.dataset_config
        else Path("configs/datasets") / f"{args.dataset}.yaml"
    )

    try:
        train_grapher(
            dataset=args.dataset,
            model_config=load_yaml(args.model_config),
            dataset_config_path=dataset_cfg,
            dataset_root=dataset_root,
            seed=args.seed,
            run_id=args.run_id,
            device=args.device,
        )
    except CudaTrainingDeviceError as exc:
        parser.exit(status=2, message=f"error: {exc}\n")


if __name__ == "__main__":
    main()

