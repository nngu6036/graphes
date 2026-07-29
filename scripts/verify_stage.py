#!/usr/bin/env python
from __future__ import annotations

import argparse
import math

import networkx as nx
import numpy as np

from grapher.construction.coarse import (
    assert_constructor_validity,
    construct_coarse_graph,
)
from grapher.data.io import load_dataset_splits
from grapher.properties.kernel_residual import KernelResidualSummarySampler
from grapher.properties.sampler import (
    EmpiricalSummarySampler,
    LearnedSummarySampler,
    maybe_wrap_with_degree_sampler,
)
from grapher.properties.summary import (
    SummaryConfig,
    distance_to_summary,
    extract_summary,
)
from grapher.refinement.grapher_opt import refine_graph
from grapher.refinement.rewiring import (
    apply_action,
    enumerate_valid_double_edge_swaps,
    permute_action,
)
from grapher.utils.io import load_yaml


def _load_graphs(config):
    dataset_cfg = config.get("dataset", {}) or {}

    splits = load_dataset_splits(
        dataset_cfg.get("name", "sbm"),
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

        for key in [
            "degree_hist",
            "clustering_hist",
            "spectral_hist",
            "motif_proxy",
            "orbit_count",
        ]:
            if key not in s1 or key not in s2:
                continue

            if not np.allclose(s1[key], s2[key], atol=1e-6):
                raise AssertionError(f"Summary invariance failed for {key}")

        for key in [
            "num_nodes",
            "num_edges",
            "density",
            "triangle_count_norm",
        ]:
            if key not in s1 or key not in s2:
                continue

            if not math.isclose(
                float(s1[key]),
                float(s2[key]),
                rel_tol=1e-8,
                abs_tol=1e-8,
            ):
                raise AssertionError(f"Summary invariance failed for {key}")

    print("PASS summary invariance")


def verify_constructor(config) -> None:
    rng = np.random.default_rng(int(config.get("seed", 0)))

    graphs = _load_graphs(config)
    summary_cfg = SummaryConfig.from_dict(config.get("summary", {}) or {}, graphs)

    sampler = EmpiricalSummarySampler.fit(
        graphs,
        summary_cfg,
        seed=int(config.get("seed", 0)),
    )

    constructor_cfg = config.get("constructor", {}) or {}
    require_connected = bool(constructor_cfg.get("ensure_connected", True))

    for _ in range(20):
        s = sampler.sample(rng)
        g = construct_coarse_graph(s, constructor_cfg, rng)
        assert_constructor_validity(
            g,
            s,
            require_connected=require_connected,
        )

    print("PASS constructor validity")


def verify_refiner(config) -> None:
    rng = np.random.default_rng(int(config.get("seed", 0)))

    graphs = _load_graphs(config)
    summary_cfg = SummaryConfig.from_dict(config.get("summary", {}) or {}, graphs)

    sampler = EmpiricalSummarySampler.fit(
        graphs,
        summary_cfg,
        seed=int(config.get("seed", 0)),
    )

    constructor_cfg = config.get("constructor", {}) or {}
    energy_cfg = config.get("energy", {}) or {}

    fast_refiner_cfg = dict(config.get("refiner", {}) or {})
    fast_refiner_cfg["steps"] = min(int(fast_refiner_cfg.get("steps", 20)), 3)
    fast_refiner_cfg["candidate_budget"] = min(
        int(fast_refiner_cfg.get("candidate_budget", 128)),
        32,
    )

    improved = []

    for _ in range(5):
        s = sampler.sample(rng)

        g0 = construct_coarse_graph(s, constructor_cfg, rng)
        e0 = distance_to_summary(g0, s, summary_cfg, energy_cfg)

        g1 = refine_graph(
            g0,
            s,
            summary_config=summary_cfg,
            energy_weights=energy_cfg,
            refiner_config=fast_refiner_cfg,
            rng=rng,
        )

        if not isinstance(g1, nx.Graph):
            raise AssertionError("Refiner did not return a NetworkX graph.")

        if sorted(dict(g0.degree()).values()) != sorted(dict(g1.degree()).values()):
            raise AssertionError("Refiner changed the degree sequence.")

        if g1.number_of_nodes() > 1 and not nx.is_connected(g1):
            raise AssertionError("Refiner produced a disconnected graph.")

        e1 = distance_to_summary(g1, s, summary_cfg, energy_cfg)
        improved.append(e1 < e0)

    rate = float(np.mean(improved))

    if rate < 0.5:
        raise AssertionError(f"Refiner improvement rate too low: {rate:.3f}")

    print(f"PASS refiner invariants and energy improvement rate={rate:.3f}")


def verify_equivariance(config) -> None:
    rng = np.random.default_rng(int(config.get("seed", 0)))

    # Use a small deterministic graph for the exact candidate-set test.
    # Full SBM graphs can have too many valid swaps for exhaustive checking.
    g = nx.connected_watts_strogatz_graph(14, 4, 0.35, seed=0)

    summary_cfg = SummaryConfig(
        degree_hist_max_degree=8,
        clustering_bins=10,
        spectral_bins=10,
    )

    target = extract_summary(g, summary_cfg)

    gp, mapping, inv = _permute_graph(g, rng)

    c1 = enumerate_valid_double_edge_swaps(
        g,
        preserve_connectivity=True,
    )

    c2 = enumerate_valid_double_edge_swaps(
        gp,
        preserve_connectivity=True,
    )

    c1_set = set(c1)
    c2_unpermuted = {permute_action(a, inv) for a in c2}

    if c1_set != c2_unpermuted:
        raise AssertionError("Candidate set is not permutation-equivariant.")

    weights = config.get("energy", {}) or {}

    base_e = distance_to_summary(g, target, summary_cfg, weights)
    base_e_perm = distance_to_summary(gp, target, summary_cfg, weights)

    for action in c1[: min(100, len(c1))]:
        action_perm = permute_action(action, mapping)

        g_next = apply_action(g, action)
        gp_next = apply_action(gp, action_perm)

        delta = base_e - distance_to_summary(g_next, target, summary_cfg, weights)
        delta_perm = base_e_perm - distance_to_summary(
            gp_next,
            target,
            summary_cfg,
            weights,
        )

        if not math.isclose(delta, delta_perm, rel_tol=1e-6, abs_tol=1e-6):
            raise AssertionError(
                "Action improvement score is not permutation-equivariant."
            )

    print("PASS GraphER-Opt candidate and score equivariance")


def verify_summary_generator(config) -> None:
    """Verify configured target-summary sampler compatibility.

    This checks whether sampled summaries contain graphical degree sequences
    and whether the coarse constructor can build valid graphs from them.
    """

    rng = np.random.default_rng(int(config.get("seed", 0)))

    graphs = _load_graphs(config)
    summary_cfg = SummaryConfig.from_dict(config.get("summary", {}) or {}, graphs)

    generator_cfg = config.get("summary_generator", {}) or {}
    generator_type = str(generator_cfg.get("type", "empirical")).lower()

    if generator_type in {"empirical", "empirical_sampler"}:
        sampler = EmpiricalSummarySampler.fit(
            graphs,
            summary_cfg,
            seed=int(config.get("seed", 0)),
        )
    elif generator_type in {"learned", "summary_vae", "vae"}:
        sampler = LearnedSummarySampler.from_config(
            generator_cfg,
            seed=int(config.get("seed", 0)),
        )
    elif generator_type in {
        "kernel_residual",
        "kernel_conditioned",
        "weighted_kernel",
    }:
        sampler = KernelResidualSummarySampler.from_config(
            graphs,
            summary_cfg,
            config,
            seed=int(config.get("seed", 0)),
        )
    else:
        raise AssertionError(
            f"Unknown summary_generator.type: {generator_type!r}"
        )
    sampler = maybe_wrap_with_degree_sampler(sampler, config, graphs, seed=int(config.get("seed", 0)))

    constructor_cfg = config.get("constructor", {}) or {}
    require_connected = bool(constructor_cfg.get("ensure_connected", True))

    graphical = []
    valid = []

    for _ in range(10):
        s = sampler.sample(rng)

        degree_sequence = [int(d) for d in s["degree_sequence"]]
        graphical.append(nx.is_graphical(degree_sequence, method="eg"))

        try:
            g = construct_coarse_graph(s, constructor_cfg, rng)
            assert_constructor_validity(
                g,
                s,
                require_connected=require_connected,
            )
            valid.append(True)
        except Exception:
            valid.append(False)

    graphical_rate = float(np.mean(graphical))
    valid_rate = float(np.mean(valid))

    if graphical_rate < 0.9 or valid_rate < 0.9:
        raise AssertionError(
            "Summary generator compatibility failed: "
            f"graphical_rate={graphical_rate:.3f}, "
            f"valid_rate={valid_rate:.3f}"
        )

    print(
        "PASS summary generator compatibility "
        f"graphical_rate={graphical_rate:.3f} "
        f"valid_rate={valid_rate:.3f}"
    )


def verify_degree_generator(config) -> None:
    """Verify sampled degree sequences from an optional DegreeVAE."""
    from grapher.generators.degree_sampler import DegreeVAESampler
    from grapher.generators.degree_vae import connected_feasible_degree_sequence

    rng = np.random.default_rng(int(config.get("seed", 0)))
    graphs = _load_graphs(config)
    degree_cfg = config.get("degree_generator", {}) or {}
    if not bool(degree_cfg.get("enabled", False)):
        raise AssertionError("degree_generator.enabled is false; nothing to verify.")

    sampler = DegreeVAESampler.from_config(degree_cfg, seed=int(config.get("seed", 0)))
    constructor_cfg = config.get("constructor", {}) or {}
    require_connected = bool(constructor_cfg.get("ensure_connected", True))

    graphical = []
    feasible = []
    valid = []
    for _ in range(25):
        s = sampler.sample(rng)
        degree_sequence = [int(d) for d in s["degree_sequence"]]
        graphical.append(nx.is_graphical(degree_sequence, method="eg"))
        feasible.append(connected_feasible_degree_sequence(degree_sequence))
        try:
            g = construct_coarse_graph(s, constructor_cfg, rng)
            assert_constructor_validity(g, s, require_connected=require_connected)
            valid.append(True)
        except Exception:
            valid.append(False)

    graphical_rate = float(np.mean(graphical))
    feasible_rate = float(np.mean(feasible))
    valid_rate = float(np.mean(valid))
    if graphical_rate < 0.9 or feasible_rate < 0.9 or valid_rate < 0.9:
        raise AssertionError(
            "Degree generator compatibility failed: "
            f"graphical_rate={graphical_rate:.3f}, "
            f"feasible_rate={feasible_rate:.3f}, "
            f"valid_rate={valid_rate:.3f}"
        )
    print(
        "PASS degree generator compatibility "
        f"graphical_rate={graphical_rate:.3f} "
        f"feasible_rate={feasible_rate:.3f} "
        f"valid_rate={valid_rate:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run verification gates for the fresh proposal branch."
    )

    parser.add_argument(
        "--stage",
        choices=[
            "summary",
            "constructor",
            "refiner",
            "equivariance",
            "summary_generator",
            "degree_generator",
        ],
        required=True,
    )

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
    elif args.stage == "summary_generator":
        verify_summary_generator(config)
    elif args.stage == "degree_generator":
        verify_degree_generator(config)
    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
