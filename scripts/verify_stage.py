#!/usr/bin/env python
from __future__ import annotations

import argparse
import math

import networkx as nx
import numpy as np

from grapher.construction.coarse import assert_constructor_validity, construct_coarse_graph
from grapher.data.io import load_dataset_splits
from grapher.properties.sampler import EmpiricalSummarySampler
from grapher.properties.summary import SummaryConfig, distance_to_summary, extract_summary
from grapher.refinement.grapher_opt import refine_graph
from grapher.refinement.rewiring import enumerate_valid_double_edge_swaps, permute_action, apply_action
from grapher.utils.io import load_yaml


def _load_graphs(config):
    dataset_cfg = config.get("dataset", {}) or {}
    splits = load_dataset_splits(
        dataset_cfg.get("name", "sbm_spectre"),
        root=dataset_cfg.get("root", "outputs/datasets"),
        build_if_missing=bool(dataset_cfg.get("build_if_missing", True)),
        config_path=dataset_cfg.get("config_path"),
    )
    train = list(splits["train"])
    limit = dataset_cfg.get("max_train_graphs")
    if limit:
        train = train[: int(limit)]
    return train


def _permute_graph(g: nx.Graph, rng: np.random.Generator):
    nodes = list(g.nodes())
    perm = rng.permutation(nodes).tolist()
    mapping = {old: int(new) for old, new in zip(nodes, perm)}
    inv = {v: k for k, v in mapping.items()}
    return nx.relabel_nodes(g, mapping, copy=True), mapping, inv


def verify_summary(config) -> None:
    rng = np.random.default_rng(int(config.get("seed", 0)))
    graphs = _load_graphs(config)
    summary_cfg = SummaryConfig.from_dict(config.get("summary", {}) or {}, graphs)
    for g in graphs[:10]:
        gp, _, _ = _permute_graph(g, rng)
        s1 = extract_summary(g, summary_cfg)
        s2 = extract_summary(gp, summary_cfg)
        for key in ["degree_hist", "clustering_hist", "spectral_hist", "motif_proxy"]:
            if not np.allclose(s1[key], s2[key], atol=1e-6):
                raise AssertionError(f"Summary invariance failed for {key}")
        for key in ["num_nodes", "num_edges", "density", "triangle_count_norm", "orbit_count"]:
            if not math.isclose(float(s1[key]), float(s2[key]), rel_tol=1e-8, abs_tol=1e-8):
                raise AssertionError(f"Summary invariance failed for {key}")
    print("PASS summary invariance")


def verify_constructor(config) -> None:
    rng = np.random.default_rng(int(config.get("seed", 0)))
    graphs = _load_graphs(config)
    summary_cfg = SummaryConfig.from_dict(config.get("summary", {}) or {}, graphs)
    sampler = EmpiricalSummarySampler.fit(graphs, summary_cfg, seed=int(config.get("seed", 0)))
    for _ in range(20):
        s = sampler.sample(rng)
        g = construct_coarse_graph(s, config.get("constructor", {}) or {}, rng)
        assert_constructor_validity(g, s, require_connected=bool(config.get("constructor", {}).get("ensure_connected", True)))
    print("PASS constructor validity")


def verify_refiner(config) -> None:
    rng = np.random.default_rng(int(config.get("seed", 0)))
    graphs = _load_graphs(config)
    summary_cfg = SummaryConfig.from_dict(config.get("summary", {}) or {}, graphs)
    sampler = EmpiricalSummarySampler.fit(graphs, summary_cfg, seed=int(config.get("seed", 0)))
    improved = []
    fast_refiner_cfg = dict(config.get("refiner", {}) or {})
    fast_refiner_cfg["steps"] = min(int(fast_refiner_cfg.get("steps", 20)), 3)
    fast_refiner_cfg["candidate_budget"] = min(int(fast_refiner_cfg.get("candidate_budget", 128)), 32)
    for _ in range(5):
        s = sampler.sample(rng)
        g0 = construct_coarse_graph(s, config.get("constructor", {}) or {}, rng)
        e0 = distance_to_summary(g0, s, summary_cfg, config.get("energy", {}) or {})
        g1 = refine_graph(g0, s, summary_config=summary_cfg, energy_weights=config.get("energy", {}) or {}, refiner_config=fast_refiner_cfg, rng=rng)
        assert isinstance(g1, nx.Graph)
        assert sorted(dict(g0.degree()).values()) == sorted(dict(g1.degree()).values())
        assert nx.is_connected(g1)
        e1 = distance_to_summary(g1, s, summary_cfg, config.get("energy", {}) or {})
        improved.append(e1 < e0)
    rate = float(np.mean(improved))
    if rate < 0.5:
        raise AssertionError(f"Refiner improvement rate too low: {rate:.3f}")
    print(f"PASS refiner invariants and energy improvement rate={rate:.3f}")


def verify_equivariance(config) -> None:
    rng = np.random.default_rng(int(config.get("seed", 0)))
    # Use a small deterministic graph for the exact candidate-set test.
    # Full SBM graphs can have millions of valid swaps, so exhaustive
    # equivariance is kept as a unit-scale gate.
    g = nx.connected_watts_strogatz_graph(14, 4, 0.35, seed=0)
    summary_cfg = SummaryConfig(degree_hist_max_degree=8, clustering_bins=10, spectral_bins=10)
    target = extract_summary(g, summary_cfg)
    gp, mapping, inv = _permute_graph(g, rng)
    c1 = enumerate_valid_double_edge_swaps(g, preserve_connectivity=True)
    c2 = enumerate_valid_double_edge_swaps(gp, preserve_connectivity=True)
    c2_unpermuted = {permute_action(a, inv) for a in c2}
    c1_set = set(c1)
    if c1_set != c2_unpermuted:
        raise AssertionError("Candidate set is not permutation-equivariant.")
    weights = config.get("energy", {}) or {}
    for action in c1[: min(100, len(c1))]:
        ap = permute_action(action, mapping)
        delta = distance_to_summary(g, target, summary_cfg, weights) - distance_to_summary(apply_action(g, action), target, summary_cfg, weights)
        delta_p = distance_to_summary(gp, target, summary_cfg, weights) - distance_to_summary(apply_action(gp, ap), target, summary_cfg, weights)
        if not math.isclose(delta, delta_p, rel_tol=1e-6, abs_tol=1e-6):
            raise AssertionError("Action improvement score is not permutation-equivariant.")
    print("PASS GraphER-Opt candidate and score equivariance")

def main() -> None:
    parser = argparse.ArgumentParser(description="Run verification gates for the fresh proposal branch.")
    parser.add_argument("--stage", choices=["summary", "constructor", "refiner", "equivariance"], required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.stage == "summary":
        verify_summary(config)
    elif args.stage == "constructor":
        verify_constructor(config)
    elif args.stage == "refiner":
        verify_refiner(config)
    elif args.stage == "equivariance":
        verify_equivariance(config)


if __name__ == "__main__":
    main()
