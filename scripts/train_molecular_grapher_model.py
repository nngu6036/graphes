from __future__ import annotations

import argparse
from dataclasses import dataclass
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from grapher.generation.molecular_rewiring import (
    EmpiricalBondTypePrior,
    MolecularRewireAction,
    apply_molecular_rewire,
    attributed_edge_discrepancy,
    edge_type_value,
    enumerate_molecular_rewire_actions,
    initialize_molecular_havel_hakimi,
    merge_molecular_action_sets,
    molecular_action_signature,
    molecular_graph_to_data,
    molecular_local_feature_matrix,
    node_type_value,
    parse_bond_order_mapping,
)
from grapher.generation.rewiring import (
    action_new_edges,
    action_removed_edges,
    connected_sequence_feasible,
    degree_sequence,
)
from grapher.models.model_molecular_grapher import MolecularGraphER
from grapher.utils.compute import (
    CudaTrainingDeviceError,
    PeakMemoryMonitor,
    compute_report,
    require_cuda_training_device,
)
from grapher.utils.io import load_yaml, save_json, save_yaml, stable_hash
from grapher.utils.logging import configure_logging, get_logger
from grapher.utils.numerics import assert_model_tensors_finite
from grapher.utils.seed import set_seed

logger = get_logger(__name__)


def _torch_load_compat(path: str | Path, *, map_location: str | torch.device = "cpu") -> Any:
    """Load torch payloads across old/new PyTorch versions."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError as exc:
        message = str(exc).lower()
        if "weights_only" not in message and "unexpected keyword" not in message and "invalid keyword" not in message:
            raise
        return torch.load(path, map_location=map_location)


def _atomic_torch_save(payload: Any, path: str | Path) -> None:
    """Write a torch payload atomically to avoid corrupt resume files."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)


def _teacher_cache_key(*, dataset: str, seed: int, cfg: Mapping[str, Any], model: MolecularGraphER, train_graph_count: int) -> str:
    """Hash the settings that determine molecular teacher-cache contents."""

    relevant_keys = [
        "max_graphs",
        "max_steps_per_graph",
        "candidate_budget",
        "offline_candidate_budget",
        "k_hop",
        "ensure_connected",
        "teacher_topology_weight",
        "teacher_bond_type_weight",
        "teacher_clustering_weight",
        "target_candidate_fraction",
        "global_candidate_fraction",
        "bond_type_proposals_per_edge",
        "bond_type_proposal_mode",
        "source_edge_type_strategy",
        "allow_global_bond_backoff",
        "reject_unseen_endpoint_pairs",
        "valence_tolerance",
        "require_positive_teacher_improvement",
        "empirical_bond_smoothing",
        "bond_order_by_type",
        "max_valence_by_node_type",
        "k_eigen",
        "local_feature_dim",
    ]
    payload = {key: cfg.get(key) for key in relevant_keys}
    payload.update(
        {
            "dataset": dataset,
            "seed": int(seed),
            "train_graph_count": int(train_graph_count),
            "node_type_values": list(map(int, model.node_type_values)),
            "edge_type_values": list(map(int, model.edge_type_values)),
            "max_nodes": int(model.max_nodes),
            "degree_histogram_dim": int(model.degree_histogram_dim),
        }
    )
    return stable_hash(payload)


@dataclass
class CachedMolecularTeacherExample:
    node_types: torch.Tensor
    degree_features: torch.Tensor
    pe: torch.Tensor
    edge_index: torch.Tensor
    edge_types: torch.Tensor
    actions: list[MolecularRewireAction]
    label_idx: int
    t: float
    degree_sequence: list[int]
    action_local_features: torch.Tensor
    train_candidate_size: int
    offline_candidate_size: int
    improvement: float


def _edge_set(graph: nx.Graph) -> set[tuple[int, int]]:
    return {tuple(sorted((int(u), int(v)))) for u, v in graph.edges()}


def _topology_is_target_aware(
    action: MolecularRewireAction,
    wrong_edges: set[tuple[int, int]],
    missing_edges: set[tuple[int, int]],
) -> bool:
    removed = {tuple(sorted(edge)) for edge in action_removed_edges(action.topology)}
    created = {tuple(sorted(edge)) for edge in action_new_edges(action.topology)}
    return bool(removed & wrong_edges) and bool(created & missing_edges)


def _split_candidate_budget(
    total: int,
    target_fraction: float,
    global_fraction: float,
) -> tuple[int, int, int]:
    total = max(int(total), 1)
    target_fraction = min(max(float(target_fraction), 0.0), 1.0)
    global_fraction = min(max(float(global_fraction), 0.0), 1.0)
    target_budget = min(int(round(total * target_fraction)), total)
    global_budget = min(int(round(total * global_fraction)), total - target_budget)
    local_budget = max(total - target_budget - global_budget, 0)
    return target_budget, local_budget, global_budget


def _usable_molecular_graphs(
    graphs: Sequence[nx.Graph],
    *,
    max_graphs: int | None,
) -> tuple[list[nx.Graph], dict[str, int]]:
    usable: list[nx.Graph] = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for raw_graph in graphs:
        graph = nx.convert_node_labels_to_integers(nx.Graph(raw_graph), ordering="sorted")
        if graph.number_of_nodes() <= 1 or graph.number_of_edges() < 2:
            skip("too_small_for_double_edge_swap")
            continue
        if not nx.is_connected(graph):
            skip("target_graph_disconnected")
            continue
        feasible, reason = connected_sequence_feasible(degree_sequence(graph))
        if not feasible:
            skip(reason)
            continue
        try:
            for node in graph.nodes():
                node_type_value(graph, int(node))
            for u, v in graph.edges():
                edge_type_value(graph, int(u), int(v))
        except (KeyError, TypeError, ValueError) as exc:
            skip(f"missing_or_invalid_molecular_attributes:{type(exc).__name__}")
            continue
        usable.append(graph)
        if max_graphs is not None and len(usable) >= int(max_graphs):
            break
    return usable, skipped


def _offline_candidate_actions(
    current: nx.Graph,
    target: nx.Graph,
    prior: EmpiricalBondTypePrior,
    *,
    offline_candidate_budget: int,
    k_hop: int | None,
    ensure_connected: bool,
    target_candidate_fraction: float,
    global_candidate_fraction: float,
    proposals_per_edge: int,
    proposal_mode: str,
    allow_global_backoff: bool,
    reject_unseen_endpoint_pairs: bool,
    valence_tolerance: float,
    rng: random.Random,
) -> list[MolecularRewireAction]:
    current_edges = _edge_set(current)
    target_edges = _edge_set(target)
    wrong_edges = current_edges - target_edges
    missing_edges = target_edges - current_edges
    target_budget, local_budget, global_budget = _split_candidate_budget(
        offline_candidate_budget,
        target_candidate_fraction,
        global_candidate_fraction,
    )

    target_actions: list[MolecularRewireAction] = []
    if target_budget > 0 and wrong_edges and missing_edges:
        target_actions = enumerate_molecular_rewire_actions(
            current,
            prior,
            rng=rng,
            ensure_connected=ensure_connected,
            k_hop=k_hop,
            max_candidates=max(target_budget * 3, target_budget),
            anchor_edges=sorted(wrong_edges),
            proposals_per_edge=proposals_per_edge,
            proposal_mode=proposal_mode,
            target=target,
            include_target_edge_types=True,
            allow_global_backoff=allow_global_backoff,
            reject_unseen_endpoint_pairs=reject_unseen_endpoint_pairs,
            valence_tolerance=valence_tolerance,
            shuffle=True,
        )
        target_actions = [
            action
            for action in target_actions
            if _topology_is_target_aware(action, wrong_edges, missing_edges)
        ][:target_budget]

    local_actions = (
        enumerate_molecular_rewire_actions(
            current,
            prior,
            rng=rng,
            ensure_connected=ensure_connected,
            k_hop=k_hop,
            max_candidates=local_budget,
            proposals_per_edge=proposals_per_edge,
            proposal_mode=proposal_mode,
            allow_global_backoff=allow_global_backoff,
            reject_unseen_endpoint_pairs=reject_unseen_endpoint_pairs,
            valence_tolerance=valence_tolerance,
            shuffle=True,
        )
        if local_budget > 0
        else []
    )
    global_actions = (
        enumerate_molecular_rewire_actions(
            current,
            prior,
            rng=rng,
            ensure_connected=ensure_connected,
            k_hop=None,
            max_candidates=global_budget,
            proposals_per_edge=proposals_per_edge,
            proposal_mode=proposal_mode,
            allow_global_backoff=allow_global_backoff,
            reject_unseen_endpoint_pairs=reject_unseen_endpoint_pairs,
            valence_tolerance=valence_tolerance,
            shuffle=True,
        )
        if global_budget > 0
        else []
    )
    actions = merge_molecular_action_sets(
        target_actions,
        local_actions,
        global_actions,
        max_candidates=int(offline_candidate_budget),
    )
    remaining = int(offline_candidate_budget) - len(actions)
    if remaining > 0:
        fallback = enumerate_molecular_rewire_actions(
            current,
            prior,
            rng=rng,
            ensure_connected=ensure_connected,
            k_hop=None if global_candidate_fraction > 0.0 else k_hop,
            max_candidates=remaining,
            proposals_per_edge=proposals_per_edge,
            proposal_mode=proposal_mode,
            allow_global_backoff=allow_global_backoff,
            reject_unseen_endpoint_pairs=reject_unseen_endpoint_pairs,
            valence_tolerance=valence_tolerance,
            shuffle=True,
        )
        actions = merge_molecular_action_sets(
            actions,
            fallback,
            max_candidates=int(offline_candidate_budget),
        )
    return actions


def _training_candidate_actions(
    current: nx.Graph,
    teacher_action: MolecularRewireAction,
    prior: EmpiricalBondTypePrior,
    *,
    candidate_budget: int,
    k_hop: int | None,
    ensure_connected: bool,
    global_candidate_fraction: float,
    proposals_per_edge: int,
    proposal_mode: str,
    allow_global_backoff: bool,
    reject_unseen_endpoint_pairs: bool,
    valence_tolerance: float,
    rng: random.Random,
) -> tuple[list[MolecularRewireAction], int]:
    total_budget = max(int(candidate_budget), 1)
    negative_budget = max(total_budget - 1, 0)
    global_budget = min(
        int(round(negative_budget * min(max(float(global_candidate_fraction), 0.0), 1.0))),
        negative_budget,
    )
    local_budget = max(negative_budget - global_budget, 0)
    local_negatives = (
        enumerate_molecular_rewire_actions(
            current,
            prior,
            rng=rng,
            ensure_connected=ensure_connected,
            k_hop=k_hop,
            max_candidates=local_budget,
            proposals_per_edge=proposals_per_edge,
            proposal_mode=proposal_mode,
            allow_global_backoff=allow_global_backoff,
            reject_unseen_endpoint_pairs=reject_unseen_endpoint_pairs,
            valence_tolerance=valence_tolerance,
            shuffle=True,
        )
        if local_budget > 0
        else []
    )
    global_negatives = (
        enumerate_molecular_rewire_actions(
            current,
            prior,
            rng=rng,
            ensure_connected=ensure_connected,
            k_hop=None,
            max_candidates=global_budget,
            proposals_per_edge=proposals_per_edge,
            proposal_mode=proposal_mode,
            allow_global_backoff=allow_global_backoff,
            reject_unseen_endpoint_pairs=reject_unseen_endpoint_pairs,
            valence_tolerance=valence_tolerance,
            shuffle=True,
        )
        if global_budget > 0
        else []
    )
    actions = merge_molecular_action_sets(
        [teacher_action],
        local_negatives,
        global_negatives,
        max_candidates=total_budget,
    )
    if len(actions) < total_budget:
        fallback = enumerate_molecular_rewire_actions(
            current,
            prior,
            rng=rng,
            ensure_connected=ensure_connected,
            k_hop=None if global_candidate_fraction > 0.0 else k_hop,
            max_candidates=total_budget - len(actions),
            proposals_per_edge=proposals_per_edge,
            proposal_mode=proposal_mode,
            allow_global_backoff=allow_global_backoff,
            reject_unseen_endpoint_pairs=reject_unseen_endpoint_pairs,
            valence_tolerance=valence_tolerance,
            shuffle=True,
        )
        actions = merge_molecular_action_sets(
            actions,
            fallback,
            max_candidates=total_budget,
        )
    rng.shuffle(actions)
    teacher_signature = molecular_action_signature(teacher_action)
    label_idx = next(
        index
        for index, action in enumerate(actions)
        if molecular_action_signature(action) == teacher_signature
    )
    return actions, int(label_idx)


def _candidate_teacher_step(
    current: nx.Graph,
    target: nx.Graph,
    prior: EmpiricalBondTypePrior,
    *,
    candidate_budget: int,
    offline_candidate_budget: int,
    k_hop: int | None,
    ensure_connected: bool,
    topology_weight: float,
    bond_type_weight: float,
    clustering_weight: float,
    target_candidate_fraction: float,
    global_candidate_fraction: float,
    proposals_per_edge: int,
    proposal_mode: str,
    allow_global_backoff: bool,
    reject_unseen_endpoint_pairs: bool,
    valence_tolerance: float,
    require_positive_improvement: bool,
    rng: random.Random,
) -> tuple[
    list[MolecularRewireAction],
    int,
    MolecularRewireAction,
    float,
    nx.Graph,
    int,
] | None:
    offline_actions = _offline_candidate_actions(
        current,
        target,
        prior,
        offline_candidate_budget=offline_candidate_budget,
        k_hop=k_hop,
        ensure_connected=ensure_connected,
        target_candidate_fraction=target_candidate_fraction,
        global_candidate_fraction=global_candidate_fraction,
        proposals_per_edge=proposals_per_edge,
        proposal_mode=proposal_mode,
        allow_global_backoff=allow_global_backoff,
        reject_unseen_endpoint_pairs=reject_unseen_endpoint_pairs,
        valence_tolerance=valence_tolerance,
        rng=rng,
    )
    if not offline_actions:
        return None
    current_score = attributed_edge_discrepancy(
        current,
        target,
        topology_weight=topology_weight,
        bond_type_weight=bond_type_weight,
        clustering_weight=clustering_weight,
    )
    best: tuple[float, MolecularRewireAction, nx.Graph] | None = None
    for action in offline_actions:
        candidate = apply_molecular_rewire(
            current,
            action,
            prior,
            ensure_connected=ensure_connected,
            valence_tolerance=valence_tolerance,
        )
        if candidate is None:
            continue
        candidate_score = attributed_edge_discrepancy(
            candidate,
            target,
            topology_weight=topology_weight,
            bond_type_weight=bond_type_weight,
            clustering_weight=clustering_weight,
        )
        improvement = float(current_score - candidate_score)
        if best is None or improvement > best[0]:
            best = (improvement, action, candidate)
    if best is None:
        return None
    improvement, teacher_action, next_graph = best
    if require_positive_improvement and improvement <= 0.0:
        return None
    train_actions, label_idx = _training_candidate_actions(
        current,
        teacher_action,
        prior,
        candidate_budget=candidate_budget,
        k_hop=k_hop,
        ensure_connected=ensure_connected,
        global_candidate_fraction=global_candidate_fraction,
        proposals_per_edge=proposals_per_edge,
        proposal_mode=proposal_mode,
        allow_global_backoff=allow_global_backoff,
        reject_unseen_endpoint_pairs=reject_unseen_endpoint_pairs,
        valence_tolerance=valence_tolerance,
        rng=rng,
    )
    return (
        train_actions,
        label_idx,
        teacher_action,
        float(improvement),
        next_graph,
        len(offline_actions),
    )


def _build_teacher_cache(
    *,
    train_graphs: Sequence[nx.Graph],
    model: MolecularGraphER,
    prior: EmpiricalBondTypePrior,
    k_eigen: int,
    max_steps: int,
    candidate_budget: int,
    offline_candidate_budget: int,
    k_hop: int | None,
    ensure_connected: bool,
    topology_weight: float,
    bond_type_weight: float,
    clustering_weight: float,
    target_candidate_fraction: float,
    global_candidate_fraction: float,
    proposals_per_edge: int,
    proposal_mode: str,
    source_proposal_mode: str,
    allow_global_backoff: bool,
    reject_unseen_endpoint_pairs: bool,
    valence_tolerance: float,
    require_positive_improvement: bool,
    rng: random.Random,
    cache_checkpoint_path: Path | None = None,
    cache_key: str = "",
    dataset: str = "",
    seed: int = 0,
    run_id: int | None = None,
    training_config_hash: str = "",
    resume_payload: Mapping[str, Any] | None = None,
    checkpoint_interval: int = 25,
) -> tuple[list[CachedMolecularTeacherExample], dict[str, float | int]]:
    cache: list[CachedMolecularTeacherExample] = []
    path_lengths: list[int] = []
    initial_discrepancies: list[float] = []
    final_discrepancies: list[float] = []
    improvements: list[float] = []
    train_candidate_sizes: list[int] = []
    offline_candidate_sizes: list[int] = []
    num_source_failures = 0
    num_no_teacher_step = 0
    num_exact = 0
    num_graphs_with_examples = 0
    start_index = 0
    checkpoint_interval = max(int(checkpoint_interval), 1)
    build_start = time.perf_counter()
    last_completed_graph_index = start_index

    if resume_payload is not None:
        saved_key = str(resume_payload.get("cache_key", ""))
        if saved_key and cache_key and saved_key != cache_key:
            logger.warning(
                "Ignoring partial molecular teacher cache because cache key does not match saved=%s current=%s",
                saved_key,
                cache_key,
            )
        else:
            cache = list(resume_payload.get("teacher_cache", []))
            state = dict(resume_payload.get("builder_state", {}))
            start_index = int(state.get("next_graph_index", resume_payload.get("next_graph_index", 0)))
            start_index = max(0, min(start_index, len(train_graphs)))
            last_completed_graph_index = start_index
            path_lengths = list(map(int, state.get("path_lengths", [])))
            initial_discrepancies = list(map(float, state.get("initial_discrepancies", [])))
            final_discrepancies = list(map(float, state.get("final_discrepancies", [])))
            improvements = list(map(float, state.get("improvements", [])))
            train_candidate_sizes = list(map(int, state.get("train_candidate_sizes", [])))
            offline_candidate_sizes = list(map(int, state.get("offline_candidate_sizes", [])))
            num_source_failures = int(state.get("num_source_failures", 0))
            num_no_teacher_step = int(state.get("num_no_teacher_step", 0))
            num_exact = int(state.get("num_source_already_exact", 0))
            num_graphs_with_examples = int(state.get("num_graphs_with_examples", 0))
            rng_state = state.get("rng_state")
            if rng_state is not None:
                try:
                    rng.setstate(rng_state)
                except Exception as exc:
                    logger.warning("Could not restore molecular teacher-cache RNG state: %s", exc)
            logger.info(
                "Resuming molecular teacher cache from graph=%d/%d examples=%d graphs_with_examples=%d",
                start_index + 1 if start_index < len(train_graphs) else len(train_graphs),
                len(train_graphs),
                len(cache),
                num_graphs_with_examples,
            )

    def builder_state(next_graph_index: int) -> dict[str, Any]:
        return {
            "next_graph_index": int(next_graph_index),
            "rng_state": rng.getstate(),
            "path_lengths": list(path_lengths),
            "initial_discrepancies": list(initial_discrepancies),
            "final_discrepancies": list(final_discrepancies),
            "improvements": list(improvements),
            "train_candidate_sizes": list(train_candidate_sizes),
            "offline_candidate_sizes": list(offline_candidate_sizes),
            "num_source_failures": int(num_source_failures),
            "num_no_teacher_step": int(num_no_teacher_step),
            "num_source_already_exact": int(num_exact),
            "num_graphs_with_examples": int(num_graphs_with_examples),
        }

    def save_progress(next_graph_index: int, *, complete: bool = False) -> None:
        if cache_checkpoint_path is None:
            return
        _save_incremental_teacher_cache(
            path=cache_checkpoint_path,
            teacher_cache=cache,
            builder_state=builder_state(next_graph_index),
            cache_key=cache_key,
            dataset=dataset,
            seed=seed,
            run_id=run_id,
            training_config_hash=training_config_hash,
            complete=complete,
            cache_seconds=time.perf_counter() - build_start,
        )

    try:
        for raw_index in range(start_index, len(train_graphs)):
            graph_index = raw_index + 1
            raw_target = train_graphs[raw_index]
            target = nx.convert_node_labels_to_integers(nx.Graph(raw_target), ordering="sorted")
            source_rng = random.Random(rng.randrange(0, 2**31 - 1))
            try:
                current = initialize_molecular_havel_hakimi(
                    target,
                    prior,
                    rng=source_rng,
                    proposal_mode=source_proposal_mode,
                    allow_global_backoff=allow_global_backoff,
                    valence_tolerance=valence_tolerance,
                )
            except Exception as exc:
                num_source_failures += 1
                logger.debug("Molecular GraphER source failed graph=%d error=%s", graph_index, exc)
                last_completed_graph_index = graph_index
                if graph_index % checkpoint_interval == 0:
                    save_progress(graph_index, complete=False)
                continue

            initial_score = attributed_edge_discrepancy(
                current,
                target,
                topology_weight=topology_weight,
                bond_type_weight=bond_type_weight,
                clustering_weight=clustering_weight,
            )
            initial_discrepancies.append(float(initial_score))
            examples_before = len(cache)
            target_degree_sequence = degree_sequence(target)
            if initial_score <= 1e-12:
                num_exact += 1
                path_lengths.append(0)
                final_discrepancies.append(0.0)
                last_completed_graph_index = graph_index
                if graph_index % checkpoint_interval == 0:
                    save_progress(graph_index, complete=False)
                continue

            for _step in range(max(int(max_steps), 0)):
                result = _candidate_teacher_step(
                    current,
                    target,
                    prior,
                    candidate_budget=candidate_budget,
                    offline_candidate_budget=offline_candidate_budget,
                    k_hop=k_hop,
                    ensure_connected=ensure_connected,
                    topology_weight=topology_weight,
                    bond_type_weight=bond_type_weight,
                    clustering_weight=clustering_weight,
                    target_candidate_fraction=target_candidate_fraction,
                    global_candidate_fraction=global_candidate_fraction,
                    proposals_per_edge=proposals_per_edge,
                    proposal_mode=proposal_mode,
                    allow_global_backoff=allow_global_backoff,
                    reject_unseen_endpoint_pairs=reject_unseen_endpoint_pairs,
                    valence_tolerance=valence_tolerance,
                    require_positive_improvement=require_positive_improvement,
                    rng=rng,
                )
                if result is None:
                    num_no_teacher_step += 1
                    break
                (
                    train_actions,
                    label_idx,
                    _teacher_action,
                    improvement,
                    next_graph,
                    offline_size,
                ) = result
                data = molecular_graph_to_data(
                    current,
                    node_type_to_index=model.node_type_to_index,
                    edge_type_to_index=model.edge_type_to_index,
                    k_eigen=k_eigen,
                )
                local_features = molecular_local_feature_matrix(
                    current,
                    train_actions,
                    prior,
                    local_feature_dim=model.local_feature_dim,
                )
                cache.append(
                    CachedMolecularTeacherExample(
                        node_types=data.node_types.cpu(),
                        degree_features=data.degree_features.cpu(),
                        pe=data.pe.cpu(),
                        edge_index=data.edge_index.cpu(),
                        edge_types=data.edge_types.cpu(),
                        actions=list(train_actions),
                        label_idx=int(label_idx),
                        t=0.0,
                        degree_sequence=list(map(int, target_degree_sequence)),
                        action_local_features=local_features.cpu(),
                        train_candidate_size=len(train_actions),
                        offline_candidate_size=int(offline_size),
                        improvement=float(improvement),
                    )
                )
                improvements.append(float(improvement))
                train_candidate_sizes.append(len(train_actions))
                offline_candidate_sizes.append(int(offline_size))
                current = nx.convert_node_labels_to_integers(next_graph, ordering="sorted")
                if attributed_edge_discrepancy(
                    current,
                    target,
                    topology_weight=topology_weight,
                    bond_type_weight=bond_type_weight,
                    clustering_weight=clustering_weight,
                ) <= 1e-12:
                    break

            path_length = len(cache) - examples_before
            path_lengths.append(int(path_length))
            final_score = attributed_edge_discrepancy(
                current,
                target,
                topology_weight=topology_weight,
                bond_type_weight=bond_type_weight,
                clustering_weight=clustering_weight,
            )
            final_discrepancies.append(float(final_score))
            if path_length > 0:
                for local_step in range(path_length):
                    cache[examples_before + local_step].t = float(local_step) / float(path_length)
                num_graphs_with_examples += 1

            last_completed_graph_index = graph_index
            if graph_index == 1 or graph_index % 25 == 0 or graph_index == len(train_graphs):
                logger.info(
                    "Molecular GraphER cache graph=%d/%d examples=%d graphs_with_examples=%d",
                    graph_index,
                    len(train_graphs),
                    len(cache),
                    num_graphs_with_examples,
                )
            if graph_index % checkpoint_interval == 0 or graph_index == len(train_graphs):
                save_progress(graph_index, complete=False)
    except KeyboardInterrupt:
        logger.warning(
            "Interrupted while building molecular teacher cache; saving partial cache at graph=%d/%d examples=%d",
            int(last_completed_graph_index),
            len(train_graphs),
            len(cache),
        )
        # Save through the last fully completed graph. On resume the current graph, if interrupted mid-graph,
        # is retried rather than partially used.
        save_progress(last_completed_graph_index, complete=False)
        raise

    state = builder_state(len(train_graphs))
    stats = _cache_stats_from_builder_state(cache, state)
    save_progress(len(train_graphs), complete=True)
    return cache, stats


def _num_trainable_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def train_molecular_grapher(
    *,
    dataset: str,
    model_config: dict[str, Any],
    dataset_config_path: Path,
    dataset_root: str,
    seed: int,
    run_id: int | None,
    device: str,
) -> dict[str, Any]:
    if dataset not in {"qm9", "zinc"}:
        raise ValueError("train_molecular_grapher_model supports only --dataset qm9 or zinc.")
    device = require_cuda_training_device(device)
    set_seed(seed, include_torch=True)
    rng = random.Random(int(seed))
    cfg = make_model_run_config(
        model_config,
        dataset=dataset,
        model="grapher_molecular",
        run_id=run_id,
        seed=seed,
        use_run_paths=run_id is not None,
    )
    run_dir = run_output_dir(dataset, "grapher_molecular", run_id=run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    resume = bool(cfg.get("resume", True))
    force_rebuild_cache = bool(cfg.get("force_rebuild_teacher_cache", False))
    teacher_cache_path = Path(cfg.get("teacher_cache_path") or (run_dir / "teacher_cache.pt"))
    training_state_path = Path(cfg.get("training_state_path") or (run_dir / "training_state.pt"))
    checkpoint_path = Path(
        cfg.get("checkpoint_path")
        or f"outputs/checkpoints/{dataset}/grapher_molecular/grapher_molecular.pt"
    )
    logger.info(
        "Starting molecular GraphER training dataset=%s run_id=%s seed=%s device=%s resume=%s",
        dataset,
        run_id,
        seed,
        device,
        resume,
    )
    splits = load_dataset_splits(
        dataset,
        output_root=dataset_root,
        build_if_missing=dataset != "zinc",
        config_path=dataset_config_path,
    )
    max_graphs_raw = cfg.get("max_graphs")
    max_graphs = None if max_graphs_raw in (None, "") else int(max_graphs_raw)
    train_graphs, skipped = _usable_molecular_graphs(
        splits["train"],
        max_graphs=max_graphs,
    )
    if not train_graphs:
        raise ValueError(f"No usable molecular training graphs. Skipped: {skipped}")

    observed_atomic_numbers = sorted(
        {
            int(graph.nodes[node].get("atomic_number", graph.nodes[node].get("z", -1)))
            for graph in train_graphs
            for node in graph.nodes()
        }
    )
    if any(value <= 0 for value in observed_atomic_numbers):
        raise ValueError(
            "Molecular training requires positive atomic_number or z attributes on every node. "
            "Re-prepare QM9 with scripts/prepare_dataset.py or ZINC with "
            "scripts/prepare_zinc_from_smiles.py."
        )
    if dataset == "qm9":
        expected_qm9_atomic_numbers = {1, 6, 7, 8, 9}
        unexpected = sorted(set(observed_atomic_numbers) - expected_qm9_atomic_numbers)
        if unexpected:
            raise ValueError(
                "Prepared QM9 contains invalid atomic numbers "
                f"{unexpected}. This usually means it was created by the old converter that "
                "treated data.x[:,0] as atomic number. Remove outputs/datasets/qm9 and re-run "
                "scripts/prepare_dataset.py --dataset qm9 --force."
            )
    logger.info(
        "Loaded molecular splits=%s usable_train=%d skipped=%s",
        {name: len(graphs) for name, graphs in splits.items()},
        len(train_graphs),
        skipped,
    )

    bond_order_mapping = parse_bond_order_mapping(cfg.get("bond_order_by_type"))
    raw_valence_overrides = cfg.get("max_valence_by_node_type") or {}
    valence_overrides = {
        int(node_type): float(value)
        for node_type, value in dict(raw_valence_overrides).items()
    }
    prior = EmpiricalBondTypePrior.fit(
        train_graphs,
        bond_order_by_type=bond_order_mapping,
        max_valence_by_node_type=valence_overrides,
        smoothing=float(cfg.get("empirical_bond_smoothing", 0.1)),
    )
    logger.info(
        "Fitted empirical bond prior node_types=%s edge_types=%s endpoint_pairs=%d max_valence=%s",
        prior.node_types,
        prior.edge_types,
        len(prior.pair_counts),
        prior.max_valence_by_node_type,
    )

    max_nodes = int(cfg.get("max_nodes") or max(graph.number_of_nodes() for graph in train_graphs))
    k_eigen = int(cfg.get("k_eigen", 4))
    local_feature_dim = int(cfg.get("local_feature_dim", 24))
    T = int(cfg.get("num_steps", cfg.get("T", 24)))
    model = MolecularGraphER(
        node_type_values=prior.node_types,
        edge_type_values=prior.edge_types,
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        num_layer=int(cfg.get("num_layers", cfg.get("num_layer", 4))),
        T=T,
        max_nodes=max_nodes,
        degree_histogram_dim=int(cfg.get("degree_histogram_dim") or max_nodes),
        k_eigen=k_eigen,
        time_embedding_dim=int(cfg.get("time_embedding_dim", cfg.get("hidden_dim", 128))),
        local_feature_dim=local_feature_dim,
        dropout=float(cfg.get("dropout", 0.05)),
    ).to(device)
    logger.info(
        "Molecular GraphER params=%d hidden=%d layers=%d max_nodes=%d local_features=%d",
        _num_trainable_parameters(model),
        model.hidden_dim,
        model.num_layer,
        max_nodes,
        local_feature_dim,
    )

    epochs = int(cfg.get("epochs", 80))
    max_steps = int(cfg.get("max_steps_per_graph", 16))
    candidate_budget = int(cfg.get("candidate_budget", 48))
    offline_candidate_budget = int(cfg.get("offline_candidate_budget", 192))
    k_hop_raw = cfg.get("k_hop", 2)
    k_hop = None if k_hop_raw in (None, "", "none", "None") else int(k_hop_raw)
    ensure_connected = bool(cfg.get("ensure_connected", True))
    topology_weight = float(cfg.get("teacher_topology_weight", 1.0))
    bond_type_weight = float(cfg.get("teacher_bond_type_weight", 1.0))
    clustering_weight = float(cfg.get("teacher_clustering_weight", 0.0))
    target_candidate_fraction = float(cfg.get("target_candidate_fraction", 0.5))
    global_candidate_fraction = float(cfg.get("global_candidate_fraction", 0.1))
    proposals_per_edge = max(int(cfg.get("bond_type_proposals_per_edge", 2)), 1)
    proposal_mode = str(cfg.get("bond_type_proposal_mode", "sample"))
    source_proposal_mode = str(cfg.get("source_edge_type_strategy", "sample"))
    allow_global_backoff = bool(cfg.get("allow_global_bond_backoff", True))
    reject_unseen_endpoint_pairs = bool(cfg.get("reject_unseen_endpoint_pairs", False))
    valence_tolerance = float(cfg.get("valence_tolerance", 1e-6))
    require_positive = bool(cfg.get("require_positive_teacher_improvement", True))
    grad_accum_steps = max(int(cfg.get("grad_accum_steps", cfg.get("batch_size", 16))), 1)
    examples_per_epoch_raw = cfg.get("examples_per_epoch")
    examples_per_epoch = (
        None if examples_per_epoch_raw in (None, "") else int(examples_per_epoch_raw)
    )
    use_amp = bool(cfg.get("use_amp", True)) and str(device).startswith("cuda")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 5e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
    )
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    cache_seconds = 0.0
    cache_key = _teacher_cache_key(
        dataset=dataset,
        seed=seed,
        cfg=cfg,
        model=model,
        train_graph_count=len(train_graphs),
    )
    with PeakMemoryMonitor() as memory_monitor:
        teacher_cache: list[CachedMolecularTeacherExample] | None = None
        cache_stats: dict[str, float | int] | None = None
        partial_cache_payload: Mapping[str, Any] | None = None

        if resume and teacher_cache_path.exists() and not force_rebuild_cache:
            logger.info("Loading molecular teacher cache from %s", teacher_cache_path)
            cache_payload = _torch_load_compat(teacher_cache_path, map_location="cpu")
            if isinstance(cache_payload, dict) and "teacher_cache" in cache_payload:
                saved_key = str(cache_payload.get("cache_key", ""))
                if saved_key == cache_key:
                    cache_seconds = float(cache_payload.get("cache_seconds", 0.0))
                    if bool(cache_payload.get("complete", True)):
                        teacher_cache = list(cache_payload["teacher_cache"])
                        cache_stats = dict(cache_payload.get("cache_stats", {}))
                        logger.info(
                            "Loaded complete molecular teacher cache examples=%d cache_seconds=%.2f",
                            len(teacher_cache),
                            cache_seconds,
                        )
                    else:
                        partial_cache_payload = cache_payload
                        logger.info(
                            "Loaded partial molecular teacher cache examples=%d next_graph=%s cache_seconds=%.2f; resuming cache build",
                            len(cache_payload.get("teacher_cache", [])),
                            cache_payload.get("next_graph_index", cache_payload.get("builder_state", {}).get("next_graph_index", "?")),
                            cache_seconds,
                        )
                else:
                    logger.warning(
                        "Ignoring stale molecular teacher cache at %s: cache key mismatch saved=%s current=%s",
                        teacher_cache_path,
                        saved_key,
                        cache_key,
                    )
            elif isinstance(cache_payload, list):
                # Backward-compatible fallback for manually saved cache lists.
                teacher_cache = list(cache_payload)
                cache_stats = {"num_teacher_examples": len(teacher_cache)}
                cache_seconds = 0.0
                logger.warning(
                    "Loaded legacy molecular teacher cache list without a cache key from %s.",
                    teacher_cache_path,
                )
            else:
                logger.warning("Ignoring unrecognized teacher cache payload at %s", teacher_cache_path)

        if teacher_cache is None or cache_stats is None:
            cache_start = time.perf_counter()
            teacher_cache, cache_stats = _build_teacher_cache(
                train_graphs=train_graphs,
                model=model,
                prior=prior,
                k_eigen=k_eigen,
                max_steps=max_steps,
                candidate_budget=candidate_budget,
                offline_candidate_budget=offline_candidate_budget,
                k_hop=k_hop,
                ensure_connected=ensure_connected,
                topology_weight=topology_weight,
                bond_type_weight=bond_type_weight,
                clustering_weight=clustering_weight,
                target_candidate_fraction=target_candidate_fraction,
                global_candidate_fraction=global_candidate_fraction,
                proposals_per_edge=proposals_per_edge,
                proposal_mode=proposal_mode,
                source_proposal_mode=source_proposal_mode,
                allow_global_backoff=allow_global_backoff,
                reject_unseen_endpoint_pairs=reject_unseen_endpoint_pairs,
                valence_tolerance=valence_tolerance,
                require_positive_improvement=require_positive,
                rng=rng,
                cache_checkpoint_path=teacher_cache_path if bool(cfg.get("save_teacher_cache", True)) else None,
                cache_key=cache_key,
                dataset=dataset,
                seed=seed,
                run_id=run_id,
                training_config_hash=stable_hash(cfg),
                resume_payload=partial_cache_payload,
                checkpoint_interval=int(cfg.get("teacher_cache_checkpoint_interval", 25)),
            )
            cache_seconds = cache_seconds + (time.perf_counter() - cache_start)
            if not teacher_cache:
                raise ValueError(f"No molecular GraphER teacher examples were built: {cache_stats}")
            if bool(cfg.get("save_teacher_cache", True)):
                _atomic_torch_save(
                    {
                        "teacher_cache": teacher_cache,
                        "cache_stats": cache_stats,
                        "cache_seconds": cache_seconds,
                        "cache_key": cache_key,
                        "dataset": dataset,
                        "seed": seed,
                        "run_id": run_id,
                        "training_config_hash": stable_hash(cfg),
                        "complete": True,
                        "next_graph_index": len(train_graphs),
                    },
                    teacher_cache_path,
                )
                logger.info(
                    "Saved molecular teacher cache examples=%d to %s",
                    len(teacher_cache),
                    teacher_cache_path,
                )

        if not teacher_cache:
            raise ValueError(f"No molecular GraphER teacher examples were available: {cache_stats}")
        cache_stats = dict(cache_stats)
        cache_stats.setdefault("num_teacher_examples", len(teacher_cache))
        logger.info(
            "Molecular teacher cache examples=%d graphs=%d source_failures=%d no_teacher=%d avg_path=%.2f discrepancy=%.4f->%.4f seconds=%.2f",
            len(teacher_cache),
            int(cache_stats.get("num_graphs_with_examples", 0)),
            int(cache_stats.get("num_source_failures", 0)),
            int(cache_stats.get("num_no_teacher_step", 0)),
            float(cache_stats.get("avg_teacher_path_length", 0.0)),
            float(cache_stats.get("avg_initial_attributed_discrepancy", 0.0)),
            float(cache_stats.get("avg_final_attributed_discrepancy", 0.0)),
            cache_seconds,
        )

        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        start_epoch = 1
        if resume and training_state_path.exists():
            logger.info("Loading molecular GraphER training state from %s", training_state_path)
            state = _torch_load_compat(training_state_path, map_location=device)
            if str(state.get("model_architecture", MolecularGraphER.architecture)) != MolecularGraphER.architecture:
                logger.warning("Ignoring training state with incompatible architecture at %s", training_state_path)
            else:
                model.load_state_dict(state["model_state_dict"])
                optimizer.load_state_dict(state["optimizer_state_dict"])
                if "scaler_state_dict" in state:
                    try:
                        scaler.load_state_dict(state["scaler_state_dict"])
                    except Exception as exc:  # pragma: no cover - defensive across torch versions
                        logger.warning("Could not restore AMP scaler state: %s", exc)
                history = list(state.get("history", []))
                start_epoch = int(state.get("epoch", 0)) + 1
                if state.get("rng_state") is not None:
                    try:
                        rng.setstate(state["rng_state"])
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.warning("Could not restore Python RNG state: %s", exc)
                logger.info(
                    "Resuming molecular GraphER from epoch %d/%d with %d prior history rows.",
                    start_epoch,
                    epochs,
                    len(history),
                )

        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            epoch_examples = list(teacher_cache)
            rng.shuffle(epoch_examples)
            if examples_per_epoch is not None:
                epoch_examples = epoch_examples[: max(int(examples_per_epoch), 1)]
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            total_loss = 0.0
            num_examples = 0
            top1 = 0
            top5 = 0
            reciprocal_rank = 0.0
            entropy = 0.0

            for index, example in enumerate(epoch_examples, start=1):
                target_idx = torch.tensor([example.label_idx], dtype=torch.long, device=device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model.score_actions(
                        node_types=example.node_types.to(device, non_blocking=True),
                        degree_features=example.degree_features.to(device, non_blocking=True),
                        pe=example.pe.to(device, non_blocking=True),
                        edge_index=example.edge_index.to(device, non_blocking=True),
                        edge_types=example.edge_types.to(device, non_blocking=True),
                        actions=example.actions,
                        t=example.t,
                        degree_sequence=example.degree_sequence,
                        action_local_features=example.action_local_features.to(
                            device,
                            non_blocking=True,
                        ),
                    )
                    raw_loss = F.cross_entropy(logits.view(1, -1), target_idx)
                    loss = raw_loss / float(grad_accum_steps)
                if not torch.isfinite(raw_loss.detach()):
                    raise FloatingPointError(
                        f"Non-finite molecular GraphER loss at epoch={epoch} example={index}."
                    )
                with torch.no_grad():
                    detached = logits.detach().float()
                    teacher_logit = detached[example.label_idx]
                    rank = 1 + int((detached > teacher_logit).sum().item())
                    top1 += int(rank == 1)
                    top5 += int(rank <= min(5, detached.numel()))
                    reciprocal_rank += 1.0 / float(rank)
                    probabilities = F.softmax(detached, dim=0)
                    entropy += float(
                        (-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item()
                    )
                scaler.scale(loss).backward()
                pending += 1
                total_loss += float(raw_loss.detach().item())
                num_examples += 1
                if pending >= grad_accum_steps or index == len(epoch_examples):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=float(cfg.get("grad_clip", 5.0)),
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    pending = 0

            row = {
                "epoch": epoch,
                "loss": total_loss / max(num_examples, 1),
                "teacher_top1_accuracy": float(top1) / max(num_examples, 1),
                "teacher_top5_accuracy": float(top5) / max(num_examples, 1),
                "teacher_mean_reciprocal_rank": reciprocal_rank / max(num_examples, 1),
                "avg_action_entropy": entropy / max(num_examples, 1),
                "num_teacher_examples": num_examples,
                "num_cached_teacher_examples": len(teacher_cache),
                "cache_seconds": cache_seconds,
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
            history.append(row)
            _atomic_torch_save(
                {
                    "epoch": epoch,
                    "model_architecture": MolecularGraphER.architecture,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "history": history,
                    "rng_state": rng.getstate(),
                    "dataset": dataset,
                    "seed": seed,
                    "run_id": run_id,
                    "model_config_hash": stable_hash(cfg),
                    "teacher_cache_path": str(teacher_cache_path),
                },
                training_state_path,
            )
            logger.info(
                "Molecular GraphER epoch %d/%d loss=%.4f top1=%.3f top5=%.3f mrr=%.3f entropy=%.3f examples=%d seconds=%.2f saved_state=%s",
                epoch,
                epochs,
                row["loss"],
                row["teacher_top1_accuracy"],
                row["teacher_top5_accuracy"],
                row["teacher_mean_reciprocal_rank"],
                row["avg_action_entropy"],
                num_examples,
                row["epoch_seconds"],
                training_state_path,
            )
        assert_model_tensors_finite(model, context=f"grapher_molecular/{dataset}")
    elapsed = time.perf_counter() - start

    checkpoint_path = Path(
        cfg.get("checkpoint_path")
        or f"outputs/checkpoints/{dataset}/grapher_molecular/grapher_molecular.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": "grapher_molecular",
        "model_architecture": MolecularGraphER.architecture,
        "model_state_dict": model.state_dict(),
        "model_params": {
            "architecture": MolecularGraphER.architecture,
            "node_type_values": model.node_type_values,
            "edge_type_values": model.edge_type_values,
            "hidden_dim": model.hidden_dim,
            "num_layer": model.num_layer,
            "T": model.T,
            "max_nodes": model.max_nodes,
            "degree_histogram_dim": model.degree_histogram_dim,
            "k_eigen": model.k_eigen,
            "time_embedding_dim": model.time_embedding_dim,
            "local_feature_dim": model.local_feature_dim,
            "dropout": model.dropout,
        },
        "empirical_bond_prior": prior.to_dict(),
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
            "candidate_action_type": "typed_double_edge_swap_(e1,e2,r,c1,c2)",
            "node_types_fixed": True,
            "edge_type_proposal": "empirical_p(edge_type|unordered_endpoint_node_types)",
            "valence_filter": "empirical_max_bond_order_sum_by_node_type",
            "teacher_cache_seconds": cache_seconds,
            "teacher_cache_stats": cache_stats,
            "teacher_cache_path": str(teacher_cache_path),
            "training_state_path": str(training_state_path),
            "resume_enabled": resume,
            "bond_prior": prior.to_dict(),
            "gpu_required_by_training_script": True,
            "mixed_precision_amp": use_amp,
            "gradient_accumulation_steps": grad_accum_steps,
        },
    }
    torch.save(payload, checkpoint_path)

    run_dir = run_output_dir(dataset, "grapher_molecular", run_id=run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, run_dir / "resolved_model_config.yaml", force=True)
    compute = compute_report(
        operation="training",
        runtime_seconds=elapsed,
        num_graphs=len(train_graphs),
        memory=memory_monitor.to_dict(),
    )
    metadata = {
        "dataset": dataset,
        "model": "grapher_molecular",
        "model_architecture": MolecularGraphER.architecture,
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
    logger.info("Saved molecular GraphER checkpoint to %s", checkpoint_path)
    logger.info("Saved molecular GraphER metadata to %s", run_dir / "train_metadata.json")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train attributed GraphER on QM9 or ZINC. Node types stay fixed; "
            "new bond types are proposed from the empirical endpoint-type prior, "
            "and valence-invalid rewiring actions are rejected."
        )
    )
    parser.add_argument("--dataset", required=True, choices=["qm9", "zinc"])
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default="outputs/datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from saved teacher cache or training state.")
    parser.add_argument("--force-rebuild-cache", action="store_true", help="Ignore any saved molecular teacher cache and rebuild it.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        configure_logging("DEBUG")
    model_config_path = Path(
        args.model_config
        or (
            "configs/models/grapher_molecular_qm9.yaml"
            if args.dataset == "qm9"
            else "configs/models/grapher_molecular_zinc.yaml"
        )
    )
    dataset_config_path = Path(
        args.dataset_config or f"configs/datasets/{args.dataset}.yaml"
    )
    model_config = load_yaml(model_config_path)
    if args.no_resume:
        model_config["resume"] = False
    if args.force_rebuild_cache:
        model_config["force_rebuild_teacher_cache"] = True

    try:
        train_molecular_grapher(
            dataset=args.dataset,
            model_config=model_config,
            dataset_config_path=dataset_config_path,
            dataset_root=args.dataset_root,
            seed=args.seed,
            run_id=args.run_id,
            device=args.device,
        )
    except CudaTrainingDeviceError as exc:
        parser.exit(status=2, message=f"error: {exc}\n")


if __name__ == "__main__":
    main()
