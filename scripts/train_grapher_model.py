from __future__ import annotations

import argparse
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
from grapher.utils.compute import PeakMemoryMonitor, compute_report
from grapher.utils.io import load_yaml, save_json, save_yaml, stable_hash
from grapher.utils.logging import get_logger
from grapher.utils.numerics import assert_model_tensors_finite
from grapher.utils.seed import set_seed

logger = get_logger(__name__)


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

    Target-aware actions are used only here, never as model input unless they are
    the selected teacher action.  This mirrors the paper's separation between
    target-aware offline path construction and target-free training/generation
    candidates.
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
        anchor_edges=sorted(wrong_edges) if wrong_edges else None,
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
    current_score = _teacher_discrepancy(current, target, mode=discrepancy)
    best: tuple[float, RewireAction, nx.Graph] | None = None
    for action in offline_actions:
        out = rewire_action(current, action, ensure_connected=ensure_connected)
        if out is None:
            continue
        candidate_graph = out[0]
        next_score = _teacher_discrepancy(candidate_graph, target, mode=discrepancy)
        improvement = current_score - next_score
        if best is None or improvement > best[0]:
            best = (float(improvement), action, candidate_graph)
    if best is None:
        return None
    improvement, teacher_action, next_graph = best
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
    set_seed(seed, include_torch=True)
    rng = random.Random(int(seed))
    cfg = make_model_run_config(model_config, dataset=dataset, model="grapher", run_id=run_id, seed=seed, use_run_paths=run_id is not None)
    splits = load_dataset_splits(dataset, output_root=dataset_root, build_if_missing=True, config_path=dataset_config_path)
    max_graphs = cfg.get("max_graphs")
    max_graphs = None if max_graphs in (None, "") else int(max_graphs)
    train_graphs, skipped = _usable_training_graphs(splits["train"], max_graphs=max_graphs)
    if not train_graphs:
        raise ValueError(f"No usable connected training graphs for GraphER. Skipped counts: {skipped}")

    k_eigen = int(cfg.get("k_eigen", 4))
    first_data = graph_to_data(train_graphs[0], k_eigen=k_eigen)
    node_in_dim = int(first_data.x.size(1))
    max_nodes = int(cfg.get("max_nodes") or max(g.number_of_nodes() for g in train_graphs))
    degree_histogram_dim = int(cfg.get("degree_histogram_dim") or max_nodes)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)), weight_decay=float(cfg.get("weight_decay", 0.0)))
    epochs = int(cfg.get("epochs", 5))
    max_steps = int(cfg.get("max_steps_per_graph", 8))
    candidate_budget = int(cfg.get("candidate_budget", 64))
    offline_candidate_budget = int(cfg.get("offline_candidate_budget", max(candidate_budget * 4, candidate_budget + 16)))
    k_hop_value = cfg.get("k_hop", 2)
    k_hop = None if k_hop_value in (None, "none", "None") else int(k_hop_value)
    ensure_connected = bool(cfg.get("ensure_connected", True))
    T = int(cfg.get("num_steps", cfg.get("T", 32)))
    discrepancy = str(cfg.get("teacher_discrepancy", "edge_symmetric_difference"))
    require_positive_improvement = bool(cfg.get("require_positive_teacher_improvement", False))

    history: list[dict] = []
    start = time.perf_counter()
    with PeakMemoryMonitor() as memory_monitor:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            num_examples = 0
            num_exact = 0
            num_positive = 0
            train_candidate_sizes: list[int] = []
            offline_candidate_sizes: list[int] = []
            improvements: list[float] = []
            epoch_graphs = list(train_graphs)
            rng.shuffle(epoch_graphs)
            for target in epoch_graphs:
                target = nx.convert_node_labels_to_integers(nx.Graph(target), ordering="sorted")
                try:
                    current = deterministic_connected_havel_hakimi(G=target)
                except Exception:
                    continue
                if edge_symmetric_difference_size(current, target) == 0:
                    num_exact += 1
                    continue
                target_degree_sequence = degree_sequence(target)
                for step in range(max_steps):
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
                        break
                    train_actions, label_idx, _teacher_action, improvement, next_graph, offline_size = example
                    data = graph_to_data(current, k_eigen=k_eigen).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    t_norm = float(step) / max(float(max_steps), 1.0)
                    logits = model.score_actions(
                        data.x,
                        data.edge_index,
                        train_actions,
                        t=t_norm,
                        degree_sequence=target_degree_sequence,
                        graph=current,
                    )
                    if logits.numel() == 0:
                        break
                    target_idx = torch.tensor([int(label_idx)], dtype=torch.long, device=device)
                    loss = F.cross_entropy(logits.view(1, -1), target_idx)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite GraphER loss at epoch {epoch} step {step}.")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.get("grad_clip", 5.0)))
                    optimizer.step()
                    total_loss += float(loss.item())
                    num_examples += 1
                    num_positive += int(improvement > 0.0)
                    train_candidate_sizes.append(len(train_actions))
                    offline_candidate_sizes.append(int(offline_size))
                    improvements.append(float(improvement))
                    current = nx.convert_node_labels_to_integers(next_graph, ordering="sorted")
                    if edge_symmetric_difference_size(current, target) == 0:
                        break
            row = {
                "epoch": epoch,
                "loss": total_loss / max(num_examples, 1),
                "num_teacher_examples": num_examples,
                "num_hh_already_exact": num_exact,
                "num_positive_teacher_steps": num_positive,
                "avg_train_candidate_size": sum(train_candidate_sizes) / max(len(train_candidate_sizes), 1),
                "avg_offline_candidate_size": sum(offline_candidate_sizes) / max(len(offline_candidate_sizes), 1),
                "avg_teacher_improvement": sum(improvements) / max(len(improvements), 1),
            }
            history.append(row)
            logger.info(
                "GraphER epoch %d/%d loss=%.4f examples=%d avgC=%.1f avgCoff=%.1f avgΔ=%.3f",
                epoch,
                epochs,
                row["loss"],
                num_examples,
                row["avg_train_candidate_size"],
                row["avg_offline_candidate_size"],
                row["avg_teacher_improvement"],
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
        },
    }
    torch.save(payload, checkpoint_path)

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
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the generic GraphER complete-action rewiring scorer from HH-to-data teacher steps.")
    parser.add_argument("--dataset", required=True, choices=available_datasets())
    parser.add_argument("--model-config", type=str, default="configs/models/grapher_generic.yaml")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default="outputs/datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    dataset_cfg = Path(args.dataset_config) if args.dataset_config else Path("configs/datasets") / f"{args.dataset}.yaml"
    train_grapher(
        dataset=args.dataset,
        model_config=load_yaml(args.model_config),
        dataset_config_path=dataset_cfg,
        dataset_root=args.dataset_root,
        seed=args.seed,
        run_id=args.run_id,
        device=args.device,
    )


if __name__ == "__main__":
    main()
