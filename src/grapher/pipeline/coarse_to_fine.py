from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from grapher.construction.coarse import (
    assert_constructor_validity,
    construct_coarse_graph,
)
from grapher.data.io import load_dataset_splits
from grapher.evaluation.metrics import degree_preservation_rate, evaluate_graph_sets
from grapher.properties.sampler import EmpiricalSummarySampler, LearnedSummarySampler
from grapher.properties.summary import (
    SummaryConfig,
    distance_to_summary,
    summary_to_jsonable,
)
from grapher.refinement.grapher_opt import random_rewire_graph, refine_graph
from grapher.utils.io import ensure_dir, save_json, save_pickle

# Learned selector is optional for older configs, but required when
# refiner.type = learned_selector.
try:
    from grapher.refinement.learned_selector import (
        load_learned_selector,
        refine_graph_with_selector,
    )
except Exception:  # pragma: no cover
    load_learned_selector = None
    refine_graph_with_selector = None


def _limit(graphs: list[nx.Graph], limit: int | None) -> list[nx.Graph]:
    if limit is None or int(limit) <= 0:
        return graphs
    return graphs[: int(limit)]


def _debug_print(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[debug] {message}", flush=True)


def _graph_stats(graph: nx.Graph) -> str:
    connected = nx.is_connected(graph) if graph.number_of_nodes() > 0 else False
    return (
        f"n={graph.number_of_nodes()} "
        f"m={graph.number_of_edges()} "
        f"connected={connected}"
    )


def _build_summary_sampler(
    config: dict[str, Any],
    train_graphs: list[nx.Graph],
    summary_cfg: SummaryConfig,
    *,
    seed: int,
    debug: bool = False,
):
    """Build either the empirical sampler or the learned SummaryVAE sampler."""

    generator_cfg = config.get("summary_generator", {}) or {}
    generator_type = str(generator_cfg.get("type", "empirical")).lower()

    if generator_type in {"empirical", "empirical_sampler"}:
        _debug_print(debug, "summary_generator.type=empirical")
        return EmpiricalSummarySampler.fit(train_graphs, summary_cfg, seed=seed)

    if generator_type in {"learned", "summary_vae", "vae"}:
        checkpoint_path = generator_cfg.get("checkpoint_path") or generator_cfg.get("checkpoint")
        _debug_print(debug, f"summary_generator.type=learned checkpoint_path={checkpoint_path}")

        if checkpoint_path is None:
            raise ValueError(
                "summary_generator.type=learned requires "
                "summary_generator.checkpoint_path"
            )

        return LearnedSummarySampler.from_config(generator_cfg, seed=seed)

    raise ValueError(f"Unknown summary_generator.type: {generator_type!r}")


def _main_refiner_method_name(refiner_cfg: dict[str, Any]) -> str:
    refiner_type = str(refiner_cfg.get("type", "grapher_opt")).lower()

    if refiner_type in {"learned_selector", "selector", "graph_action_selector"}:
        return "learned_selector"

    if refiner_type in {"grapher_opt", "energy_guided", "oracle"}:
        return "grapher_opt"

    raise ValueError(f"Unknown refiner.type: {refiner_type!r}")


def _run_main_refiner(
    coarse: nx.Graph,
    target_summary: dict[str, Any],
    *,
    method_name: str,
    selector: Any,
    summary_cfg: SummaryConfig,
    energy_cfg: dict[str, Any],
    refiner_cfg: dict[str, Any],
    rng: np.random.Generator,
    debug: bool,
):
    """Run either the learned selector or the energy-guided GraphER-Opt refiner."""

    if method_name == "learned_selector":
        if load_learned_selector is None or refine_graph_with_selector is None:
            raise ImportError(
                "refiner.type=learned_selector was requested, but "
                "grapher.refinement.learned_selector is not available. "
                "Add src/grapher/refinement/learned_selector.py first."
            )

        return refine_graph_with_selector(
            coarse,
            target_summary,
            selector=selector,
            summary_config=summary_cfg,
            energy_weights=energy_cfg,
            refiner_config=refiner_cfg,
            rng=rng,
            return_trace=debug,
        )

    if method_name == "grapher_opt":
        return refine_graph(
            coarse,
            target_summary,
            summary_config=summary_cfg,
            energy_weights=energy_cfg,
            refiner_config=refiner_cfg,
            rng=rng,
            return_trace=debug,
        )

    raise ValueError(f"Unknown method_name: {method_name!r}")


def _run_oracle_refiner(
    coarse: nx.Graph,
    target_summary: dict[str, Any],
    *,
    summary_cfg: SummaryConfig,
    energy_cfg: dict[str, Any],
    oracle_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> nx.Graph:
    """Run the energy-guided GraphER-Opt oracle baseline."""

    oracle_result = refine_graph(
        coarse,
        target_summary,
        summary_config=summary_cfg,
        energy_weights=energy_cfg,
        refiner_config=oracle_cfg,
        rng=rng,
        return_trace=False,
    )

    if not isinstance(oracle_result, nx.Graph):
        raise TypeError("Oracle GraphER-Opt did not return a NetworkX graph.")

    return oracle_result


def _trace_delta_key(item: dict[str, Any]) -> float:
    """Handle trace key names from both GraphER-Opt and learned selector."""

    if "delta" in item:
        return float(item.get("delta", 0.0))
    if "actual_delta" in item and item.get("actual_delta") is not None:
        return float(item.get("actual_delta", 0.0))
    return 0.0


def _trace_energy_key(item: dict[str, Any]) -> float | None:
    if "energy" in item:
        return float(item["energy"])
    return None


def _print_trace(
    *,
    debug: bool,
    graph_idx: int,
    trace: list[dict[str, Any]],
) -> None:
    if not debug:
        return

    if not trace:
        _debug_print(debug, f"graph {graph_idx + 1}: no refiner steps recorded")
        return

    for item in trace:
        step = int(item.get("step", -1))

        if item.get("accepted"):
            delta = _trace_delta_key(item)
            energy = _trace_energy_key(item)

            if energy is None:
                _debug_print(
                    debug,
                    f"graph {graph_idx + 1}: step={step} accepted "
                    f"delta={delta:.6g} "
                    f"candidates={int(item.get('num_candidates', 0))}",
                )
            else:
                _debug_print(
                    debug,
                    f"graph {graph_idx + 1}: step={step} accepted "
                    f"delta={delta:.6g} energy={energy:.6g} "
                    f"candidates={int(item.get('num_candidates', 0))}",
                )
        else:
            reason = item.get("reason", "unknown")
            best_delta = float(item.get("best_delta", 0.0))
            energy = _trace_energy_key(item)

            if energy is None:
                _debug_print(
                    debug,
                    f"graph {graph_idx + 1}: step={step} stopped "
                    f"reason={reason} best_delta={best_delta:.6g}",
                )
            else:
                _debug_print(
                    debug,
                    f"graph {graph_idx + 1}: step={step} stopped "
                    f"reason={reason} best_delta={best_delta:.6g} "
                    f"energy={energy:.6g}",
                )


def run_coarse_to_fine(
    config: dict[str, Any],
    *,
    output_dir: str | Path,
    num_generate: int | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)

    out_dir = ensure_dir(output_dir)

    _debug_print(
        debug,
        f"starting run experiment={config.get('experiment', 'coarse_to_fine')} "
        f"seed={seed} output_dir={out_dir}",
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset_cfg = config.get("dataset", {}) or {}
    dataset_name = str(dataset_cfg.get("name", "sbm_spectre"))

    _debug_print(
        debug,
        f"loading dataset name={dataset_name} "
        f"root={dataset_cfg.get('root', 'outputs/datasets')}",
    )

    splits = load_dataset_splits(
        dataset_name,
        root=dataset_cfg.get("root", "outputs/datasets"),
        build_if_missing=bool(dataset_cfg.get("build_if_missing", True)),
        config_path=dataset_cfg.get("config_path"),
    )

    train_graphs = _limit(
        list(splits["train"]),
        dataset_cfg.get("max_train_graphs"),
    )

    reference_graphs = _limit(
        list(splits["test"]),
        dataset_cfg.get("max_reference_graphs"),
    )

    # ------------------------------------------------------------------
    # Summary configuration and summary sampler
    # ------------------------------------------------------------------
    summary_cfg = SummaryConfig.from_dict(config.get("summary", {}) or {}, train_graphs)

    sampler = _build_summary_sampler(
        config,
        train_graphs,
        summary_cfg,
        seed=seed,
        debug=debug,
    )

    eval_cfg = config.get("evaluation", {}) or {}

    n_generate = int(
        num_generate
        or eval_cfg.get("num_generate")
        or len(reference_graphs)
    )

    _debug_print(
        debug,
        f"dataset ready train={len(train_graphs)} "
        f"reference={len(reference_graphs)} generate={n_generate}",
    )

    # ------------------------------------------------------------------
    # Config sections
    # ------------------------------------------------------------------
    constructor_cfg = config.get("constructor", {}) or {}
    energy_cfg = config.get("energy", {}) or {}
    refiner_cfg = config.get("refiner", {}) or {}
    oracle_cfg = config.get("oracle_refiner", {}) or {}

    method_name = _main_refiner_method_name(refiner_cfg)

    _debug_print(debug, f"constructor={constructor_cfg}")
    _debug_print(debug, f"energy={energy_cfg}")
    _debug_print(debug, f"refiner={refiner_cfg}")
    _debug_print(debug, f"main_method={method_name}")

    # ------------------------------------------------------------------
    # Optional learned selector
    # ------------------------------------------------------------------
    selector = None
    if method_name == "learned_selector":
        if load_learned_selector is None:
            raise ImportError(
                "refiner.type=learned_selector was requested, but "
                "load_learned_selector is not importable."
            )

        selector_cfg = config.get("selector", {}) or {}
        _debug_print(
            debug,
            f"loading learned selector checkpoint={selector_cfg.get('checkpoint_path')}",
        )
        selector = load_learned_selector(selector_cfg)
        _debug_print(debug, f"learned selector device={selector.device}")

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    coarse_graphs: list[nx.Graph] = []
    refined_graphs: list[nx.Graph] = []
    oracle_graphs: list[nx.Graph] = []
    random_graphs: list[nx.Graph] = []

    initial_energies: list[float] = []
    final_energies: list[float] = []
    oracle_energies: list[float] = []

    sampled_summaries: list[dict[str, Any]] = []
    refinement_traces: list[list[dict[str, Any]]] = []

    # ------------------------------------------------------------------
    # Generate graphs
    # ------------------------------------------------------------------
    for graph_idx in range(n_generate):
        _debug_print(
            debug,
            f"graph {graph_idx + 1}/{n_generate}: sampling target summary",
        )

        target_summary = sampler.sample(rng)
        sampled_summaries.append(summary_to_jsonable(target_summary))

        _debug_print(
            debug,
            f"graph {graph_idx + 1}/{n_generate}: "
            f"target n={int(target_summary['num_nodes'])} "
            f"m={int(target_summary['num_edges'])} "
            f"density={float(target_summary['density']):.6f}",
        )

        coarse = construct_coarse_graph(target_summary, constructor_cfg, rng)

        assert_constructor_validity(
            coarse,
            target_summary,
            require_connected=bool(constructor_cfg.get("ensure_connected", True)),
        )

        initial_energy = distance_to_summary(
            coarse,
            target_summary,
            summary_cfg,
            energy_cfg,
        )

        _debug_print(
            debug,
            f"graph {graph_idx + 1}/{n_generate}: "
            f"coarse {_graph_stats(coarse)} "
            f"initial_energy={initial_energy:.6g}",
        )

        refined_result = _run_main_refiner(
            coarse,
            target_summary,
            method_name=method_name,
            selector=selector,
            summary_cfg=summary_cfg,
            energy_cfg=energy_cfg,
            refiner_cfg=refiner_cfg,
            rng=rng,
            debug=debug,
        )

        if debug:
            refined, trace = refined_result
        else:
            refined = refined_result
            trace = []

        if not isinstance(refined, nx.Graph):
            raise TypeError(f"{method_name} did not return a NetworkX graph.")

        final_energy = distance_to_summary(
            refined,
            target_summary,
            summary_cfg,
            energy_cfg,
        )

        refinement_traces.append(trace)

        if debug:
            _print_trace(debug=debug, graph_idx=graph_idx, trace=trace)
            _debug_print(
                debug,
                f"graph {graph_idx + 1}/{n_generate}: "
                f"{method_name} {_graph_stats(refined)} "
                f"final_energy={final_energy:.6g} "
                f"improvement={initial_energy - final_energy:.6g}",
            )

        coarse_graphs.append(coarse)
        refined_graphs.append(refined)
        initial_energies.append(float(initial_energy))
        final_energies.append(float(final_energy))

        # --------------------------------------------------------------
        # Optional oracle GraphER-Opt baseline
        # --------------------------------------------------------------
        run_oracle = bool(eval_cfg.get("oracle_grapher_opt_baseline", False))

        # If the main method is already grapher_opt, the refined graph is the oracle.
        if run_oracle and method_name != "grapher_opt":
            _debug_print(
                debug,
                f"graph {graph_idx + 1}/{n_generate}: "
                "running oracle GraphER-Opt baseline",
            )

            oracle = _run_oracle_refiner(
                coarse,
                target_summary,
                summary_cfg=summary_cfg,
                energy_cfg=energy_cfg,
                oracle_cfg=oracle_cfg,
                rng=rng,
            )

            oracle_energy = distance_to_summary(
                oracle,
                target_summary,
                summary_cfg,
                energy_cfg,
            )

            oracle_graphs.append(oracle)
            oracle_energies.append(float(oracle_energy))

            _debug_print(
                debug,
                f"graph {graph_idx + 1}/{n_generate}: "
                f"oracle grapher_opt {_graph_stats(oracle)} "
                f"oracle_energy={oracle_energy:.6g} "
                f"improvement={initial_energy - oracle_energy:.6g}",
            )

        # --------------------------------------------------------------
        # Random rewiring baseline
        # --------------------------------------------------------------
        if bool(eval_cfg.get("random_rewire_baseline", True)):
            _debug_print(
                debug,
                f"graph {graph_idx + 1}/{n_generate}: "
                "running random rewire baseline",
            )

            random_graphs.append(
                random_rewire_graph(
                    coarse,
                    steps=int(refiner_cfg.get("steps", 20)),
                    candidate_budget=int(refiner_cfg.get("candidate_budget", 128)),
                    preserve_connectivity=bool(
                        refiner_cfg.get("preserve_connectivity", True)
                    ),
                    rng=rng,
                )
            )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    _debug_print(debug, "evaluating generated graph sets")

    compute_orbit = bool(eval_cfg.get("compute_orbit", True))

    metrics: dict[str, Any] = {}

    metrics["coarse"] = evaluate_graph_sets(
        reference_graphs,
        coarse_graphs,
        train_graphs,
        compute_orbit=compute_orbit,
    )

    metrics[method_name] = evaluate_graph_sets(
        reference_graphs,
        refined_graphs,
        train_graphs,
        compute_orbit=compute_orbit,
    )

    metrics[method_name]["degree_preservation_from_coarse_rate"] = (
        degree_preservation_rate(coarse_graphs, refined_graphs)
    )

    # If the main method is learned_selector and oracle baseline is enabled,
    # report GraphER-Opt as a separate row.
    if oracle_graphs:
        metrics["grapher_opt"] = evaluate_graph_sets(
            reference_graphs,
            oracle_graphs,
            train_graphs,
            compute_orbit=compute_orbit,
        )
        metrics["grapher_opt"]["degree_preservation_from_coarse_rate"] = (
            degree_preservation_rate(coarse_graphs, oracle_graphs)
        )

    # If the main method is GraphER-Opt, keep the row name as grapher_opt.
    if method_name == "grapher_opt":
        metrics["grapher_opt"] = metrics.pop("grapher_opt")

    if random_graphs:
        metrics["random_rewire"] = evaluate_graph_sets(
            reference_graphs,
            random_graphs,
            train_graphs,
            compute_orbit=compute_orbit,
        )
        metrics["random_rewire"]["degree_preservation_from_coarse_rate"] = (
            degree_preservation_rate(coarse_graphs, random_graphs)
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    diagnostics = {
        "main_method": method_name,
        "initial_energy_mean": float(np.mean(initial_energies)) if initial_energies else 0.0,
        "final_energy_mean": float(np.mean(final_energies)) if final_energies else 0.0,
        "energy_improvement_rate": (
            float(np.mean([f < i for i, f in zip(initial_energies, final_energies)]))
            if initial_energies
            else 0.0
        ),
        "num_generated": int(n_generate),
        "num_train_graphs": int(len(train_graphs)),
        "num_reference_graphs": int(len(reference_graphs)),
    }

    if oracle_energies:
        diagnostics["oracle_energy_mean"] = float(np.mean(oracle_energies))
        diagnostics["oracle_energy_improvement_rate"] = float(
            np.mean([f < i for i, f in zip(initial_energies, oracle_energies)])
        )

    result = {
        "experiment": config.get("experiment", "coarse_to_fine"),
        "dataset": dataset_name,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "config": config,
    }

    if debug:
        result["refinement_traces"] = refinement_traces

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    _debug_print(debug, f"saving outputs to {out_dir}")

    save_json(result, out_dir / "metrics.json")

    save_pickle(coarse_graphs, out_dir / "coarse_graphs.pkl")
    save_pickle(refined_graphs, out_dir / f"{method_name}_graphs.pkl")

    # Backward-compatible filename for older scripts.
    save_pickle(refined_graphs, out_dir / "refined_graphs.pkl")

    if oracle_graphs:
        save_pickle(oracle_graphs, out_dir / "grapher_opt_graphs.pkl")

    if random_graphs:
        save_pickle(random_graphs, out_dir / "random_rewire_graphs.pkl")

    save_json(sampled_summaries, out_dir / "sampled_summaries.json")

    if debug:
        save_json(refinement_traces, out_dir / "refinement_traces.json")

    _debug_print(debug, "run complete")

    return result
