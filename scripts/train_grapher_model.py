from __future__ import annotations

import argparse
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
    build_candidates,
    connected_sequence_feasible,
    degree_sequence,
    deterministic_connected_havel_hakimi,
    edge_symmetric_difference_size,
    graph_to_data,
    rewire,
)
from grapher.models.model_grapher import GraphER
from grapher.registry import available_datasets
from grapher.utils.compute import PeakMemoryMonitor, compute_report
from grapher.utils.io import load_yaml, save_json, save_yaml, stable_hash
from grapher.utils.logging import get_logger
from grapher.utils.numerics import assert_model_tensors_finite
from grapher.utils.seed import set_seed

logger = get_logger(__name__)


def _candidate_teacher_step(
    current: nx.Graph,
    target: nx.Graph,
    *,
    candidate_budget: int,
    k_hop: int | None,
    ensure_connected: bool,
) -> tuple[tuple[int, int], list[tuple[int, int]], int, int, nx.Graph] | None:
    """Return one supervised edge-pair prediction example.

    The candidate set itself is target-free.  The target graph is used only to
    select the teacher label, matching the paper's separation between offline
    teacher construction and target-free model inputs.
    """

    edges = [tuple(e) for e in current.edges()]
    if len(edges) < 2:
        return None
    target_edges = {tuple(sorted(e)) for e in target.edges()}
    wrong_edges = [e for e in edges if tuple(sorted(e)) not in target_edges]
    anchors = wrong_edges + [e for e in edges if e not in wrong_edges]
    current_score = edge_symmetric_difference_size(current, target)

    best: tuple[float, tuple[int, int], list[tuple[int, int]], int, int, nx.Graph] | None = None
    for anchor in anchors:
        candidates = build_candidates(
            current,
            anchor,
            ensure_connected=ensure_connected,
            k_hop=k_hop,
            max_candidates=candidate_budget,
        )
        if not candidates:
            continue
        for idx, e2 in enumerate(candidates):
            for orient in (0, 1):
                out = rewire(current, anchor, e2, orient, ensure_connected=ensure_connected)
                if out is None:
                    continue
                candidate_graph = out[0]
                next_score = edge_symmetric_difference_size(candidate_graph, target)
                improvement = current_score - next_score
                key = (float(improvement), anchor, idx, orient)
                if best is None or key[0] > best[0]:
                    best = (float(improvement), anchor, candidates, idx, orient, candidate_graph)
        # If we found a strictly improving move for a wrong-edge anchor, use it.
        if best is not None and best[0] > 0:
            break
    if best is None:
        return None
    _, anchor, candidates, idx, orient, next_graph = best
    return anchor, candidates, idx, orient, next_graph


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
    model = GraphER(
        node_in_dim=node_in_dim,
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        num_layer=int(cfg.get("num_layers", cfg.get("num_layer", 3))),
        T=int(cfg.get("num_steps", cfg.get("T", 32))),
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)), weight_decay=float(cfg.get("weight_decay", 0.0)))
    epochs = int(cfg.get("epochs", 5))
    max_steps = int(cfg.get("max_steps_per_graph", 8))
    candidate_budget = int(cfg.get("candidate_budget", 64))
    k_hop_value = cfg.get("k_hop", 2)
    k_hop = None if k_hop_value in (None, "none", "None") else int(k_hop_value)
    ensure_connected = bool(cfg.get("ensure_connected", True))
    T = int(cfg.get("num_steps", cfg.get("T", 32)))

    history: list[dict] = []
    start = time.perf_counter()
    with PeakMemoryMonitor() as memory_monitor:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            num_examples = 0
            num_exact = 0
            for target in train_graphs:
                try:
                    current = deterministic_connected_havel_hakimi(G=target)
                except Exception:
                    continue
                if edge_symmetric_difference_size(current, target) == 0:
                    num_exact += 1
                    continue
                for step in range(max_steps):
                    example = _candidate_teacher_step(
                        current,
                        target,
                        candidate_budget=candidate_budget,
                        k_hop=k_hop,
                        ensure_connected=ensure_connected,
                    )
                    if example is None:
                        break
                    anchor, candidates, label_idx, _orient, next_graph = example
                    data = graph_to_data(current, k_eigen=k_eigen).to(device)
                    if data.x.size(1) != node_in_dim:
                        # Keep the model architecture unchanged; pad/truncate features to match.
                        if data.x.size(1) < node_in_dim:
                            data.x = torch.cat([data.x, data.x.new_zeros((data.x.size(0), node_in_dim - data.x.size(1)))], dim=1)
                        else:
                            data.x = data.x[:, :node_in_dim]
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(data.x, data.edge_index, anchor, candidates, t=min(step, T))
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
                    current = next_graph
                    if edge_symmetric_difference_size(current, target) == 0:
                        break
            row = {
                "epoch": epoch,
                "loss": total_loss / max(num_examples, 1),
                "num_teacher_examples": num_examples,
                "num_hh_already_exact": num_exact,
            }
            history.append(row)
            logger.info("GraphER epoch %d/%d loss=%.4f examples=%d", epoch, epochs, row["loss"], num_examples)
        assert_model_tensors_finite(model, context=f"grapher/{dataset}")
    elapsed = time.perf_counter() - start

    checkpoint_path = Path(cfg.get("checkpoint_path") or f"outputs/checkpoints/{dataset}/grapher/grapher.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_params": {
            "node_in_dim": node_in_dim,
            "hidden_dim": int(cfg.get("hidden_dim", 128)),
            "num_layer": int(cfg.get("num_layers", cfg.get("num_layer", 3))),
            "T": T,
            "k_eigen": k_eigen,
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
    parser = argparse.ArgumentParser(description="Train the GraphER rewiring scorer from HH-to-data teacher steps.")
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
