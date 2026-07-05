#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from grapher.construction.coarse import construct_coarse_graph
from grapher.data.io import load_dataset_splits
from grapher.properties.sampler import EmpiricalSummarySampler, LearnedSummarySampler
from grapher.properties.summary import (
    SummaryConfig,
    distance_to_summary,
    extract_summary,
    summary_to_jsonable,
)
from grapher.refinement.rewiring import (
    Action,
    apply_action,
    sample_valid_double_edge_swaps,
)
from grapher.utils.io import ensure_dir, load_yaml, save_json


def _json_edge(edge: tuple[int, int]) -> list[int]:
    u, v = edge
    return [int(u), int(v)]


def _json_edges(edges) -> list[list[int]]:
    return [_json_edge(e) for e in sorted(edges)]


def _graph_to_edges(graph: nx.Graph) -> list[list[int]]:
    return _json_edges((min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges())


def _action_to_json(action: Action, delta_energy: float, candidate_energy: float) -> dict[str, Any]:
    removed, added = action
    return {
        "removed": _json_edges(removed),
        "added": _json_edges(added),
        "delta_energy": float(delta_energy),
        "candidate_energy": float(candidate_energy),
    }


def _softmax_masked(
    values: np.ndarray,
    *,
    temperature: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if mask is None:
        mask = np.ones(values.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    probs = np.zeros_like(values, dtype=np.float64)

    if not np.any(mask):
        return probs

    temp = max(float(temperature), 1.0e-12)
    logits = values[mask] / temp
    logits = logits - np.max(logits)
    p = np.exp(logits)
    p = p / np.sum(p)
    probs[mask] = p
    return probs


def _choose_action(
    deltas: np.ndarray,
    probs: np.ndarray,
    *,
    selection: str,
    rng: np.random.Generator,
) -> int:
    selection = str(selection).lower()

    if selection in {"greedy", "argmax"}:
        return int(np.argmax(deltas))

    if selection in {"soft", "softmax", "sample"}:
        if probs.sum() <= 0:
            return int(np.argmax(deltas))
        return int(rng.choice(np.arange(len(deltas)), p=probs))

    raise ValueError(f"Unknown teacher.selection: {selection!r}")


def _load_graphs(config: dict[str, Any]) -> dict[str, list[nx.Graph]]:
    dataset_cfg = config.get("dataset", {}) or {}
    splits = load_dataset_splits(
        dataset_cfg.get("name", "sbm_spectre"),
        root=dataset_cfg.get("root", "outputs/datasets"),
        build_if_missing=bool(dataset_cfg.get("build_if_missing", True)),
        config_path=dataset_cfg.get("config_path"),
    )

    max_train = dataset_cfg.get("max_train_graphs")
    if max_train:
        splits["train"] = list(splits["train"])[: int(max_train)]

    return splits


def _build_empirical_sampler(
    graphs: list[nx.Graph],
    summary_config: SummaryConfig,
    *,
    seed: int,
) -> EmpiricalSummarySampler:
    return EmpiricalSummarySampler.fit(
        graphs,
        summary_config,
        seed=seed,
    )


def _maybe_build_learned_sampler(
    config: dict[str, Any],
    *,
    seed: int,
) -> LearnedSummarySampler | None:
    generator_cfg = config.get("summary_generator", {}) or {}
    generator_type = str(generator_cfg.get("type", "empirical")).lower()

    if generator_type not in {"learned", "summary_vae", "vae"}:
        return None

    checkpoint_path = generator_cfg.get("checkpoint_path") or generator_cfg.get("checkpoint")
    if not checkpoint_path:
        return None

    if not Path(checkpoint_path).exists():
        return None

    return LearnedSummarySampler.from_config(generator_cfg, seed=seed)


def _sample_target_summary(
    *,
    empirical_sampler: EmpiricalSummarySampler,
    learned_sampler: LearnedSummarySampler | None,
    teacher_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[dict[str, Any], str]:
    target_source = str(teacher_cfg.get("target_source", "mixed")).lower()

    if target_source == "empirical":
        return empirical_sampler.sample(rng), "empirical"

    if target_source == "learned":
        if learned_sampler is None:
            raise RuntimeError(
                "teacher.target_source=learned but no learned summary sampler is available."
            )
        return learned_sampler.sample(rng), "learned"

    if target_source == "mixed":
        learned_prob = float(teacher_cfg.get("learned_target_prob", 0.5))
        empirical_prob = float(teacher_cfg.get("empirical_target_prob", 0.5))
        total = max(learned_prob + empirical_prob, 1.0e-12)
        learned_prob = learned_prob / total

        if learned_sampler is not None and float(rng.random()) < learned_prob:
            return learned_sampler.sample(rng), "learned"

        return empirical_sampler.sample(rng), "empirical"

    raise ValueError(f"Unknown teacher.target_source: {target_source!r}")


def build_one_trajectory(
    *,
    trajectory_id: int,
    target_summary: dict[str, Any],
    target_source: str,
    summary_config: SummaryConfig,
    constructor_cfg: dict[str, Any],
    energy_cfg: dict[str, Any],
    teacher_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    steps = int(teacher_cfg.get("steps", 20))
    candidate_budget = int(teacher_cfg.get("candidate_budget", 64))
    preserve_connectivity = bool(teacher_cfg.get("preserve_connectivity", True))
    selection = str(teacher_cfg.get("selection", "soft")).lower()
    temperature = float(teacher_cfg.get("temperature", 0.05))
    accept_only_improving = bool(teacher_cfg.get("accept_only_improving", True))
    min_improvement = float(teacher_cfg.get("min_improvement", 1.0e-12))

    graph = construct_coarse_graph(target_summary, constructor_cfg, rng)
    records: list[dict[str, Any]] = []

    current_energy = distance_to_summary(
        graph,
        target_summary,
        summary_config,
        energy_cfg,
    )

    for step in range(steps):
        candidates = sample_valid_double_edge_swaps(
            graph,
            candidate_budget,
            rng,
            preserve_connectivity=preserve_connectivity,
        )

        if not candidates:
            break

        candidate_energies: list[float] = []
        deltas: list[float] = []

        for action in candidates:
            next_graph = apply_action(graph, action)
            e_next = distance_to_summary(
                next_graph,
                target_summary,
                summary_config,
                energy_cfg,
            )
            candidate_energies.append(float(e_next))
            deltas.append(float(current_energy - e_next))

        deltas_np = np.asarray(deltas, dtype=np.float64)

        if accept_only_improving:
            valid_mask = deltas_np > min_improvement
        else:
            valid_mask = np.ones_like(deltas_np, dtype=bool)

        if not np.any(valid_mask):
            break

        teacher_probs = _softmax_masked(
            deltas_np,
            temperature=temperature,
            mask=valid_mask,
        )

        chosen_idx = _choose_action(
            deltas_np,
            teacher_probs,
            selection=selection,
            rng=rng,
        )

        chosen_action = candidates[chosen_idx]
        chosen_delta = float(deltas_np[chosen_idx])

        action_records = [
            _action_to_json(action, delta, energy)
            for action, delta, energy in zip(candidates, deltas, candidate_energies)
        ]

        record = {
            "trajectory_id": int(trajectory_id),
            "step": int(step),
            "target_source": str(target_source),
            "num_nodes": int(graph.number_of_nodes()),
            "num_edges": int(graph.number_of_edges()),
            "edges": _graph_to_edges(graph),
            "target_summary": summary_to_jsonable(target_summary),
            "current_energy": float(current_energy),
            "actions": action_records,
            "teacher_probs": [float(x) for x in teacher_probs.tolist()],
            "chosen_index": int(chosen_idx),
            "chosen_delta": float(chosen_delta),
            "best_delta": float(np.max(deltas_np)),
            "num_candidates": int(len(candidates)),
        }

        records.append(record)

        graph = apply_action(graph, chosen_action)
        current_energy = float(current_energy - chosen_delta)

    return records


def build_teacher_cache(
    config: dict[str, Any],
    *,
    output_dir: str | Path,
    num_trajectories: int | None = None,
    seed: int | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    seed = int(config.get("seed", 0) if seed is None else seed)
    rng = np.random.default_rng(seed)

    output_dir = ensure_dir(output_dir)

    splits = _load_graphs(config)
    train_graphs = list(splits["train"])

    summary_config = SummaryConfig.from_dict(
        config.get("summary", {}) or {},
        train_graphs,
    )

    teacher_cfg = dict(config.get("teacher", {}) or {})
    if num_trajectories is not None:
        teacher_cfg["num_trajectories"] = int(num_trajectories)

    n_trajectories = int(teacher_cfg.get("num_trajectories", 512))
    val_fraction = float(teacher_cfg.get("val_fraction", 0.1))

    constructor_cfg = config.get("constructor", {}) or {}
    energy_cfg = config.get("energy", {}) or {}

    empirical_sampler = _build_empirical_sampler(
        train_graphs,
        summary_config,
        seed=seed,
    )
    learned_sampler = _maybe_build_learned_sampler(config, seed=seed)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    stats = {
        "seed": seed,
        "num_requested_trajectories": n_trajectories,
        "num_trajectories": 0,
        "num_train_records": 0,
        "num_val_records": 0,
        "num_empty_trajectories": 0,
        "num_records": 0,
        "mean_best_delta": 0.0,
        "mean_chosen_delta": 0.0,
        "mean_num_candidates": 0.0,
        "target_source_counts": {},
        "train_path": str(train_path),
        "val_path": str(val_path),
    }

    all_best_deltas: list[float] = []
    all_chosen_deltas: list[float] = []
    all_num_candidates: list[int] = []

    with train_path.open("w", encoding="utf-8") as f_train, val_path.open("w", encoding="utf-8") as f_val:
        for trajectory_id in range(n_trajectories):
            target_summary, target_source = _sample_target_summary(
                empirical_sampler=empirical_sampler,
                learned_sampler=learned_sampler,
                teacher_cfg=teacher_cfg,
                rng=rng,
            )

            stats["target_source_counts"][target_source] = (
                stats["target_source_counts"].get(target_source, 0) + 1
            )

            is_val = bool(rng.random() < val_fraction)

            try:
                records = build_one_trajectory(
                    trajectory_id=trajectory_id,
                    target_summary=target_summary,
                    target_source=target_source,
                    summary_config=summary_config,
                    constructor_cfg=constructor_cfg,
                    energy_cfg=energy_cfg,
                    teacher_cfg=teacher_cfg,
                    rng=rng,
                )
            except Exception as exc:
                stats["num_empty_trajectories"] += 1
                if debug:
                    print(f"[WARN] trajectory={trajectory_id} failed: {exc}")
                continue

            if not records:
                stats["num_empty_trajectories"] += 1
                continue

            out = f_val if is_val else f_train

            for rec in records:
                out.write(json.dumps(rec, sort_keys=True) + "\n")
                all_best_deltas.append(float(rec["best_delta"]))
                all_chosen_deltas.append(float(rec["chosen_delta"]))
                all_num_candidates.append(int(rec["num_candidates"]))

            stats["num_trajectories"] += 1
            stats["num_records"] += len(records)

            if is_val:
                stats["num_val_records"] += len(records)
            else:
                stats["num_train_records"] += len(records)

            if debug and (trajectory_id + 1) % 25 == 0:
                print(
                    f"[INFO] trajectory={trajectory_id + 1}/{n_trajectories} "
                    f"records={stats['num_records']} "
                    f"empty={stats['num_empty_trajectories']}"
                )

    if all_best_deltas:
        stats["mean_best_delta"] = float(np.mean(all_best_deltas))
        stats["mean_chosen_delta"] = float(np.mean(all_chosen_deltas))
        stats["mean_num_candidates"] = float(np.mean(all_num_candidates))

    save_json(stats, output_dir / "teacher_report.json")

    print("Teacher cache built")
    print(f"  output_dir: {output_dir}")
    print(f"  train_path: {train_path}")
    print(f"  val_path:   {val_path}")
    print(f"  records:    {stats['num_records']}")
    print(f"  train:      {stats['num_train_records']}")
    print(f"  val:        {stats['num_val_records']}")
    print(f"  empty traj: {stats['num_empty_trajectories']}")
    print(f"  mean best delta:   {stats['mean_best_delta']:.6f}")
    print(f"  mean chosen delta: {stats['mean_chosen_delta']:.6f}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build GraphER-Opt energy-guided rewiring teacher cache."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-trajectories", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)

    teacher_cfg = config.get("teacher", {}) or {}
    output_dir = (
        args.output_dir
        or teacher_cfg.get("output_dir")
        or "outputs/teachers/sbm_report"
    )

    build_teacher_cache(
        config,
        output_dir=output_dir,
        num_trajectories=args.num_trajectories,
        seed=args.seed,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
