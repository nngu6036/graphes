from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from grapher.properties.summary import SummaryConfig, distance_to_summary
from grapher.refinement.rewiring import (
    Action,
    apply_action,
    sample_valid_double_edge_swaps,
)


@dataclass(frozen=True)
class RefinerConfig:
    steps: int = 20
    candidate_budget: int = 128
    k_hop: int = -1  # Placeholder. The first branch samples globally.
    preserve_connectivity: bool = True
    selection: str = "greedy"  # greedy or softmax
    temperature: float = 0.05
    accept_only_improving: bool = True
    min_improvement: float = 1e-12
    target_energy_threshold: float = -1.0
    patience: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None = None) -> RefinerConfig:
        data = data or {}
        return cls(
            steps=int(data.get("steps", 20)),
            candidate_budget=int(data.get("candidate_budget", 128)),
            k_hop=int(data.get("k_hop", -1)),
            preserve_connectivity=bool(data.get("preserve_connectivity", True)),
            selection=str(data.get("selection", "greedy")),
            temperature=float(data.get("temperature", 0.05)),
            accept_only_improving=bool(data.get("accept_only_improving", True)),
            min_improvement=float(data.get("min_improvement", 1e-12)),
            target_energy_threshold=float(data.get("target_energy_threshold", -1.0)),
            patience=max(int(data.get("patience", 1)), 1),
        )


def _choose_action(
    actions: list[Action],
    deltas: np.ndarray,
    cfg: RefinerConfig,
    rng: np.random.Generator,
) -> tuple[Action | None, float]:
    if not actions:
        return None, 0.0
    if cfg.accept_only_improving:
        mask = deltas > cfg.min_improvement
        if not np.any(mask):
            return None, float(np.max(deltas))
        actions = [a for a, keep in zip(actions, mask) if bool(keep)]
        deltas = deltas[mask]
    if cfg.selection == "greedy":
        idx = int(np.argmax(deltas))
        return actions[idx], float(deltas[idx])
    if cfg.selection == "softmax":
        temp = max(float(cfg.temperature), 1e-8)
        logits = deltas / temp
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / np.sum(probs)
        idx = int(rng.choice(len(actions), p=probs))
        return actions[idx], float(deltas[idx])
    raise ValueError(f"Unknown action selection mode: {cfg.selection}")


def refine_graph(
    graph: nx.Graph,
    target_summary: dict[str, Any],
    *,
    summary_config: SummaryConfig | dict[str, Any] | None = None,
    energy_weights: dict[str, float] | None = None,
    refiner_config: RefinerConfig | dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
    return_trace: bool = False,
) -> nx.Graph | tuple[nx.Graph, list[dict[str, Any]]]:
    cfg = (
        refiner_config
        if isinstance(refiner_config, RefinerConfig)
        else RefinerConfig.from_dict(refiner_config)
    )
    generator = rng if rng is not None else np.random.default_rng(0)
    g = graph.copy()
    trace: list[dict[str, Any]] = []
    current_energy = distance_to_summary(
        g, target_summary, summary_config, energy_weights
    )
    stalled_steps = 0

    for step in range(cfg.steps):
        if (
            cfg.target_energy_threshold >= 0.0
            and current_energy <= cfg.target_energy_threshold
        ):
            trace.append(
                {
                    "step": step,
                    "energy": current_energy,
                    "accepted": False,
                    "reason": "target_reached",
                }
            )
            break
        candidates = sample_valid_double_edge_swaps(
            g,
            cfg.candidate_budget,
            generator,
            preserve_connectivity=cfg.preserve_connectivity,
        )
        if not candidates:
            trace.append(
                {
                    "step": step,
                    "energy": current_energy,
                    "accepted": False,
                    "reason": "no_candidates",
                }
            )
            break
        deltas: list[float] = []
        for action in candidates:
            candidate_graph = apply_action(g, action)
            candidate_energy = distance_to_summary(
                candidate_graph, target_summary, summary_config, energy_weights
            )
            deltas.append(current_energy - candidate_energy)
        action, chosen_delta = _choose_action(
            candidates, np.asarray(deltas, dtype=np.float64), cfg, generator
        )
        if action is None:
            trace.append(
                {
                    "step": step,
                    "energy": current_energy,
                    "accepted": False,
                    "best_delta": float(np.max(deltas)) if deltas else 0.0,
                    "reason": "no_improvement",
                }
            )
            break
        g = apply_action(g, action)
        current_energy = current_energy - chosen_delta
        if chosen_delta <= cfg.min_improvement:
            stalled_steps += 1
        else:
            stalled_steps = 0
        trace.append(
            {
                "step": step,
                "energy": current_energy,
                "accepted": True,
                "delta": float(chosen_delta),
                "num_candidates": len(candidates),
            }
        )
        if stalled_steps >= cfg.patience:
            trace.append(
                {
                    "step": step,
                    "energy": current_energy,
                    "accepted": False,
                    "reason": "patience_exhausted",
                    "stalled_steps": stalled_steps,
                }
            )
            break
    if return_trace:
        return g, trace
    return g


def random_rewire_graph(
    graph: nx.Graph,
    *,
    steps: int,
    candidate_budget: int,
    preserve_connectivity: bool,
    rng: np.random.Generator,
) -> nx.Graph:
    g = graph.copy()
    for _ in range(int(steps)):
        candidates = sample_valid_double_edge_swaps(
            g,
            int(candidate_budget),
            rng,
            preserve_connectivity=preserve_connectivity,
        )
        if not candidates:
            break
        action = candidates[int(rng.integers(0, len(candidates)))]
        g = apply_action(g, action)
    return g
