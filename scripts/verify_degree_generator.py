#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from grapher.construction.coarse import construct_coarse_graph
from grapher.data.io import load_dataset_splits
from grapher.evaluation.metrics import descriptor_matrix, mmd_rbf
from grapher.generators.degree_sampler import DegreeVAESampler, EmpiricalDegreeSampler
from grapher.generators.degree_vectorizer import connected_feasible_degree_sequence
from grapher.properties.summary import degree_histogram
from grapher.utils.io import load_yaml, save_json


def _limit(items: list[Any], limit: int | None) -> list[Any]:
    if limit is None or int(limit) <= 0:
        return items
    return items[: int(limit)]


def _degree_sequence_to_graph(summary: dict[str, Any], constructor_cfg: dict[str, Any], rng: np.random.Generator):
    try:
        graph = construct_coarse_graph(summary, constructor_cfg, rng)
        return graph, True
    except Exception:
        return None, False


def _metrics(reference_graphs: list[nx.Graph], summaries: list[dict[str, Any]], generated_graphs: list[nx.Graph]) -> dict[str, float]:
    max_degree = 0
    for g in reference_graphs:
        if g.number_of_nodes():
            max_degree = max(max_degree, max(dict(g.degree()).values()))
    for s in summaries:
        seq = [int(d) for d in s["degree_sequence"]]
        if seq:
            max_degree = max(max_degree, max(seq))
    ref_deg = descriptor_matrix(reference_graphs, lambda g: degree_histogram(g, max_degree))
    gen_deg = descriptor_matrix(generated_graphs, lambda g: degree_histogram(g, max_degree)) if generated_graphs else np.zeros((0, max_degree + 1))
    return {
        "degree_mmd": mmd_rbf(ref_deg, gen_deg) if generated_graphs else float("nan"),
        "num_nodes_mean": float(np.mean([s["num_nodes"] for s in summaries])) if summaries else 0.0,
        "num_edges_mean": float(np.mean([s["num_edges"] for s in summaries])) if summaries else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify sampled degree sequences from the degree generator.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--empirical", action="store_true", help="Use empirical degree sampler instead of the learned DegreeVAE.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    rng = np.random.default_rng(seed)
    dataset_cfg = config.get("dataset", {}) or {}
    splits = load_dataset_splits(
        dataset_cfg.get("name", "sbm_spectre"),
        root=dataset_cfg.get("root", "outputs/datasets"),
        build_if_missing=bool(dataset_cfg.get("build_if_missing", True)),
        config_path=dataset_cfg.get("config_path"),
    )
    train_graphs = _limit(list(splits["train"]), dataset_cfg.get("max_train_graphs"))
    reference_graphs = _limit(list(splits["test"]), dataset_cfg.get("max_reference_graphs"))
    constructor_cfg = config.get("constructor", {}) or {}

    if args.empirical:
        sampler = EmpiricalDegreeSampler.fit_from_graphs(train_graphs, seed=seed)
    else:
        degree_cfg = config.get("degree_generator", {}) or {}
        sampler = DegreeVAESampler.from_config(degree_cfg, seed=seed)

    summaries = []
    graphs = []
    graphical = []
    connected_feasible = []
    construct_valid = []

    for _ in range(int(args.num_samples)):
        s = sampler.sample(rng)
        seq = [int(d) for d in s["degree_sequence"]]
        summaries.append(s)
        graphical.append(nx.is_graphical(seq, method="eg"))
        connected_feasible.append(connected_feasible_degree_sequence(seq))
        g, ok = _degree_sequence_to_graph(s, constructor_cfg, rng)
        construct_valid.append(ok)
        if ok and g is not None:
            graphs.append(g)

    report = {
        "num_samples": int(args.num_samples),
        "graphical_rate": float(np.mean(graphical)),
        "connected_feasible_rate": float(np.mean(connected_feasible)),
        "constructor_valid_rate": float(np.mean(construct_valid)),
        "num_constructed": int(len(graphs)),
        **_metrics(reference_graphs, summaries, graphs),
    }
    print(
        "Degree generator verification "
        f"graphical_rate={report['graphical_rate']:.3f} "
        f"connected_feasible_rate={report['connected_feasible_rate']:.3f} "
        f"constructor_valid_rate={report['constructor_valid_rate']:.3f} "
        f"degree_mmd={report['degree_mmd']:.6f}"
    )
    if args.output:
        save_json(report, Path(args.output))


if __name__ == "__main__":
    main()
