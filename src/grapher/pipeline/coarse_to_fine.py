from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from grapher.construction.coarse import assert_constructor_validity, construct_coarse_graph
from grapher.data.io import load_dataset_splits
from grapher.evaluation.metrics import degree_preservation_rate, evaluate_graph_sets
from grapher.properties.sampler import EmpiricalSummarySampler
from grapher.properties.summary import SummaryConfig, distance_to_summary, summary_to_jsonable
from grapher.refinement.grapher_opt import random_rewire_graph, refine_graph
from grapher.utils.io import ensure_dir, save_json, save_pickle


def _limit(graphs: list[nx.Graph], limit: int | None) -> list[nx.Graph]:
    if limit is None or limit <= 0:
        return graphs
    return graphs[: int(limit)]


def run_coarse_to_fine(config: dict[str, Any], *, output_dir: str | Path, num_generate: int | None = None) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)
    out_dir = ensure_dir(output_dir)

    dataset_cfg = config.get("dataset", {}) or {}
    dataset_name = str(dataset_cfg.get("name", "sbm_spectre"))
    splits = load_dataset_splits(
        dataset_name,
        root=dataset_cfg.get("root", "outputs/datasets"),
        build_if_missing=bool(dataset_cfg.get("build_if_missing", True)),
        config_path=dataset_cfg.get("config_path"),
    )
    train_graphs = _limit(list(splits["train"]), dataset_cfg.get("max_train_graphs"))
    reference_graphs = _limit(list(splits["test"]), dataset_cfg.get("max_reference_graphs"))

    summary_cfg = SummaryConfig.from_dict(config.get("summary", {}) or {}, train_graphs)
    sampler = EmpiricalSummarySampler.fit(train_graphs, summary_cfg, seed=seed)
    n_generate = int(num_generate or config.get("evaluation", {}).get("num_generate", len(reference_graphs)))

    coarse_graphs: list[nx.Graph] = []
    refined_graphs: list[nx.Graph] = []
    random_graphs: list[nx.Graph] = []
    initial_energies: list[float] = []
    final_energies: list[float] = []
    sampled_summaries: list[dict[str, Any]] = []

    refiner_cfg = config.get("refiner", {}) or {}
    energy_cfg = config.get("energy", {}) or {}
    constructor_cfg = config.get("constructor", {}) or {}
    for _ in range(n_generate):
        target_summary = sampler.sample(rng)
        sampled_summaries.append(summary_to_jsonable(target_summary))
        coarse = construct_coarse_graph(target_summary, constructor_cfg, rng)
        assert_constructor_validity(coarse, target_summary, require_connected=bool(constructor_cfg.get("ensure_connected", True)))
        initial_energy = distance_to_summary(coarse, target_summary, summary_cfg, energy_cfg)
        refined = refine_graph(
            coarse,
            target_summary,
            summary_config=summary_cfg,
            energy_weights=energy_cfg,
            refiner_config=refiner_cfg,
            rng=rng,
        )
        assert isinstance(refined, nx.Graph)
        final_energy = distance_to_summary(refined, target_summary, summary_cfg, energy_cfg)
        coarse_graphs.append(coarse)
        refined_graphs.append(refined)
        initial_energies.append(float(initial_energy))
        final_energies.append(float(final_energy))
        if bool(config.get("evaluation", {}).get("random_rewire_baseline", True)):
            random_graphs.append(
                random_rewire_graph(
                    coarse,
                    steps=int(refiner_cfg.get("steps", 20)),
                    candidate_budget=int(refiner_cfg.get("candidate_budget", 128)),
                    preserve_connectivity=bool(refiner_cfg.get("preserve_connectivity", True)),
                    rng=rng,
                )
            )

    metrics = {
        "coarse": evaluate_graph_sets(reference_graphs, coarse_graphs, train_graphs),
        "grapher_opt": evaluate_graph_sets(reference_graphs, refined_graphs, train_graphs),
    }
    metrics["grapher_opt"]["degree_preservation_from_coarse_rate"] = degree_preservation_rate(coarse_graphs, refined_graphs)
    if random_graphs:
        metrics["random_rewire"] = evaluate_graph_sets(reference_graphs, random_graphs, train_graphs)
        metrics["random_rewire"]["degree_preservation_from_coarse_rate"] = degree_preservation_rate(coarse_graphs, random_graphs)

    diagnostics = {
        "initial_energy_mean": float(np.mean(initial_energies)) if initial_energies else 0.0,
        "final_energy_mean": float(np.mean(final_energies)) if final_energies else 0.0,
        "energy_improvement_rate": float(np.mean([f < i for i, f in zip(initial_energies, final_energies)])) if initial_energies else 0.0,
        "num_generated": n_generate,
        "num_train_graphs": len(train_graphs),
        "num_reference_graphs": len(reference_graphs),
    }
    result = {
        "experiment": config.get("experiment", "coarse_to_fine"),
        "dataset": dataset_name,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "config": config,
    }
    save_json(result, out_dir / "metrics.json")
    save_pickle(coarse_graphs, out_dir / "coarse_graphs.pkl")
    save_pickle(refined_graphs, out_dir / "refined_graphs.pkl")
    if random_graphs:
        save_pickle(random_graphs, out_dir / "random_rewire_graphs.pkl")
    save_json(sampled_summaries, out_dir / "sampled_summaries.json")
    return result
