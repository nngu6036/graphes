from __future__ import annotations

import argparse
from dataclasses import dataclass
import random
import sys
import time
from pathlib import Path
from typing import Sequence

import networkx as nx
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


def _teacher_discrepancy(current: nx.Graph, target: nx.Graph, *, mode: str = "edge_symmetric_difference") -> float:
    """Offline teacher discrepancy rho(G, G_data).

    The generic implementation uses the paper's edge-set discrepancy term by
    default.  The function is isolated so spectral/motif terms can be added
    later without changing model inputs.
    """

    mode = str(mode or "edge_symmetric_difference").lower()
    if mode not in {"edge", "edge_symmetric_difference", "symmetric_difference"}:
        raise ValueError(f"Unsupported teacher_discrepancy={mode!r}. Currently supported: edge_symmetric_difference.")
    return float(edge_symmetric_difference_size(current, target))




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


def _offline_candidate_actions(
    current: nx.Graph,
    target: nx.Graph,
    *,
    candidate_budget: int,
    offline_candidate_budget: int,
    k_hop: int | None,
    ensure_connected: bool,
    rng: random.Random,
) -> list[RewireAction]:
    """Build C_off = C_target union C_local union C_rand for teacher search.
    """

    current_edges = _edge_set(current)
    target_edges = _edge_set(target)
    wrong_edges = current_edges - target_edges
    missing_edges = target_edges - current_edges

    target_seed = enumerate_rewire_actions(
        current,
        ensure_connected=ensure_connected,
        k_hop=k_hop,
        max_candidates=max(int(offline_candidate_budget), int(candidate_budget)),
        anchor_edges=sorted(wrong_edges),
        rng=rng,
        shuffle=True,
    )
    target_aware = [a for a in target_seed if _action_is_target_aware(a, wrong_edges, missing_edges)]
    local_random = enumerate_rewire_actions(
        current,
        ensure_connected=ensure_connected,
        k_hop=k_hop,
        max_candidates=max(int(candidate_budget), 1),
        rng=rng,
        shuffle=True,
    )
    # Keep target-aware candidates first so the greedy teacher sees promising
    # moves even under tight budgets, but include target-free local/random moves
    # as fallbacks for reachability.
    return merge_action_sets(target_aware, target_seed, local_random, max_candidates=int(offline_candidate_budget))


def _training_candidate_actions(
    current: nx.Graph,
    teacher_action: RewireAction,
    *,
    candidate_budget: int,
    k_hop: int | None,
    ensure_connected: bool,
    rng: random.Random,
    shuffle: bool,
) -> tuple[list[RewireAction], int]:
    """Build C_train = {a*} union C_local union C_rand with no target graph."""

    negatives = enumerate_rewire_actions(
        current,
        ensure_connected=ensure_connected,
        k_hop=k_hop,
        max_candidates=max(int(candidate_budget) * 2, int(candidate_budget)),
        rng=rng,
        shuffle=True,
    )
    actions = merge_action_sets([teacher_action], negatives, max_candidates=max(int(candidate_budget), 1))
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
    discrepancy: str,
    require_positive_improvement: bool,
    rng: random.Random,
) -> tuple[list[RewireAction], int, RewireAction, float, nx.Graph, int] | None:
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
        rng=rng,
    )
    if not offline_actions:
        return None
    mode = str(discrepancy or "edge_symmetric_difference").lower()
    best: tuple[float, RewireAction] | None = None
    if mode in {"edge", "edge_symmetric_difference", "symmetric_difference"}:
        current_edges = _edge_set(current)
        target_edges = _edge_set(target)
        for action in offline_actions:
            improvement = float(_edge_discrepancy_improvement(action, current_edges=current_edges, target_edges=target_edges))
            if best is None or improvement > best[0]:
                best = (improvement, action)
    else:
        current_score = _teacher_discrepancy(current, target, mode=discrepancy)
        for action in offline_actions:
            out = rewire_action(current, action, ensure_connected=ensure_connected)
            if out is None:
                continue
            candidate_graph = out[0]
            next_score = _teacher_discrepancy(candidate_graph, target, mode=discrepancy)
            improvement = current_score - next_score
            if best is None or improvement > best[0]:
                best = (float(improvement), action)
    if best is None:
        return None
    improvement, teacher_action = best
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
        rng=rng,
        shuffle=True,
    )
    return train_actions, label_idx, teacher_action, float(improvement), next_graph, len(offline_actions)


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
    discrepancy: str,
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

        if edge_symmetric_difference_size(current, target) == 0:
            num_exact += 1
            logger.debug("GraphER cache graph=%d HH source already exact", graph_idx)
            continue

        target_degree_sequence = degree_sequence(target)
        examples_before = len(cache)
        for step in range(max_steps):
            current_distance = edge_symmetric_difference_size(current, target)
            example = _candidate_teacher_step(
                current,
                target,
                candidate_budget=candidate_budget,
                offline_candidate_budget=offline_candidate_budget,
                k_hop=k_hop,
                ensure_connected=ensure_connected,
                discrepancy=discrepancy,
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

            train_actions, label_idx, _teacher_action, improvement, next_graph, offline_size = example
            data = graph_to_data(current, k_eigen=k_eigen, include_edge_pairs=False)
            with torch.no_grad():
                local_features = model.action_local_feature_matrix(current, train_actions, device=cpu_device).cpu()
            cache.append(
                CachedTeacherExample(
                    x=data.x.detach().cpu(),
                    edge_index=data.edge_index.detach().cpu(),
                    actions=list(train_actions),
                    label_idx=int(label_idx),
                    t=float(step) / max(float(max_steps), 1.0),
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
            if edge_symmetric_difference_size(current, target) == 0:
                break

        if len(cache) > examples_before:
            num_graphs_with_examples += 1
        if graph_idx == 1 or graph_idx % 10 == 0 or graph_idx == len(train_graphs):
            logger.info(
                "GraphER teacher cache progress graph=%d/%d examples=%d graphs_with_examples=%d",
                graph_idx,
                len(train_graphs),
                len(cache),
                num_graphs_with_examples,
            )

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
    k_hop = int(cfg.get("k_hop", 2))
    ensure_connected = bool(cfg.get("ensure_connected", True))
    T = int(cfg.get("num_steps", cfg.get("T", 32)))
    discrepancy = str(cfg.get("teacher_discrepancy", "edge_symmetric_difference"))
    require_positive_improvement = bool(cfg.get("require_positive_teacher_improvement", False))
    logger.info(
        "GraphER optimization epochs=%d max_steps_per_graph=%d lr=%g weight_decay=%g grad_clip=%g candidate_budget=%d offline_candidate_budget=%d k_hop=%s ensure_connected=%s discrepancy=%s require_positive=%s",
        epochs,
        max_steps,
        float(cfg.get("learning_rate", 1e-3)),
        float(cfg.get("weight_decay", 0.0)),
        float(cfg.get("grad_clip", 5.0)),
        candidate_budget,
        offline_candidate_budget,
        k_hop,
        ensure_connected,
        discrepancy,
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
            require_positive_improvement=require_positive_improvement,
            rng=rng,
        )
        cache_seconds = time.perf_counter() - cache_start
        if not teacher_cache:
            raise ValueError(f"No GraphER teacher examples were constructed. Cache stats: {cache_stats}")
        logger.info(
            "GraphER teacher cache built examples=%d graphs_with_examples=%d exact=%d hh_fail=%d no_teacher=%d avgC=%.1f avgCoff=%.1f avg_delta=%.3f seconds=%.2f",
            len(teacher_cache),
            int(cache_stats["num_graphs_with_examples"]),
            int(cache_stats["num_hh_already_exact"]),
            int(cache_stats["num_hh_failures"]),
            int(cache_stats["num_no_teacher_step"]),
            float(cache_stats["avg_train_candidate_size"]),
            float(cache_stats["avg_offline_candidate_size"]),
            float(cache_stats["avg_teacher_improvement"]),
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
                "cache_seconds": cache_seconds,
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
            history.append(row)
            logger.info(
                "GraphER epoch %d/%d loss=%.4f examples=%d cached=%d zero_logits=%d avgC=%.1f avgCoff=%.1f avg_delta=%.3f seconds=%.2f",
                epoch,
                epochs,
                row["loss"],
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
            "teacher_candidate_set": "target-aware offline candidates plus target-free local/random candidates",
            "training_candidate_set": "teacher action plus target-free local/random negatives",
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

