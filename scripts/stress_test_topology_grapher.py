#!/usr/bin/env python
"""Controlled compute-scaling study for structural-summary topology GraphER.

The study deliberately tunes on the validation split.  It constructs one fixed
set of oracle-degree Havel-Hakimi source graphs and reuses the exact graph bytes
and per-graph refinement seeds for every variant.  This prevents a larger
candidate budget from silently changing the degree samples or initial graphs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from grapher.utils.io import (
    ensure_dir,
    load_pickle,
    load_yaml,
    save_json,
    save_pickle,
    save_yaml,
)

if TYPE_CHECKING:
    import networkx as nx


def _load_runtime_dependencies() -> None:
    """Load GraphER's heavy numerical stack only when execution needs it.

    Keeping these imports out of module initialization lets ``--dry-run``
    validate a study plan before a GPU/GraphER environment is activated.
    """

    global nx, np
    global assert_constructor_validity, construct_coarse_graph
    global load_dataset_splits, degree_preservation_rate, evaluate_graph_sets
    global configure_orca_executable, load_topology_checkpoint
    global refine_graph_with_topology_predictions, propose_valid_topology_swaps

    import networkx as nx_module
    import numpy as np_module

    from grapher.models.dhvae_hh.havel_hakimi import (
        assert_constructor_validity as assert_constructor_validity_fn,
    )
    from grapher.models.dhvae_hh.havel_hakimi import (
        construct_coarse_graph as construct_coarse_graph_fn,
    )
    from grapher.data.io import load_dataset_splits as load_dataset_splits_fn
    from grapher.rewiring_mlp.evaluation.metrics import (
        degree_preservation_rate as degree_preservation_rate_fn,
    )
    from grapher.rewiring_mlp.evaluation.metrics import evaluate_graph_sets as evaluate_graph_sets_fn
    from grapher.properties.summary import (
        configure_orca_executable as configure_orca_executable_fn,
    )
    from grapher.rewiring_mlp.generic.model import (
        load_topology_checkpoint as load_topology_checkpoint_fn,
    )
    from grapher.rewiring_mlp.generic.refiner import (
        refine_graph_with_topology_predictions as refine_graph_fn,
    )
    from grapher.rewiring_mlp.generic.rewiring import (
        propose_valid_topology_swaps as propose_swaps_fn,
    )

    nx = nx_module
    np = np_module
    assert_constructor_validity = assert_constructor_validity_fn
    construct_coarse_graph = construct_coarse_graph_fn
    load_dataset_splits = load_dataset_splits_fn
    degree_preservation_rate = degree_preservation_rate_fn
    evaluate_graph_sets = evaluate_graph_sets_fn
    configure_orca_executable = configure_orca_executable_fn
    load_topology_checkpoint = load_topology_checkpoint_fn
    refine_graph_with_topology_predictions = refine_graph_fn
    propose_valid_topology_swaps = propose_swaps_fn


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating either input."""

    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _format_seed(value: Any, seed: int) -> Any:
    """Resolve ``{seed}`` placeholders recursively in a configuration."""

    if isinstance(value, dict):
        return {key: _format_seed(item, seed) for key, item in value.items()}
    if isinstance(value, list):
        return [_format_seed(item, seed) for item in value]
    if isinstance(value, tuple):
        return tuple(_format_seed(item, seed) for item in value)
    if isinstance(value, str):
        return value.format(seed=seed)
    return value


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _degree_summary(graph: nx.Graph) -> dict[str, Any]:
    sequence = sorted((int(degree) for _, degree in graph.degree()), reverse=True)
    return {
        "num_nodes": len(sequence),
        "num_edges": int(sum(sequence) // 2),
        "degree_sequence": sequence,
    }


def _randomize_source(
    graph: nx.Graph,
    *,
    steps: int,
    rng: np.random.Generator,
) -> nx.Graph:
    """Optionally diversify a source with constraint-preserving random swaps."""

    current = graph.copy()
    for _ in range(max(int(steps), 0)):
        actions, candidates, _ = propose_valid_topology_swaps(
            current,
            proposal_budget=128,
            valid_candidate_budget=32,
            preserve_connectivity=True,
            rng=rng,
        )
        if not actions:
            break
        action = actions[int(rng.integers(0, len(actions)))]
        current = candidates[action]
    return current


def _graph_collection_fingerprint(graphs: Sequence[nx.Graph]) -> str:
    digest = hashlib.sha256()
    for graph in graphs:
        nodes = sorted(int(node) for node in graph.nodes())
        edges = sorted(
            (min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges()
        )
        digest.update(json.dumps([nodes, edges], separators=(",", ":")).encode())
    return digest.hexdigest()


def _source_spec_hash(
    split: str,
    source_cfg: dict[str, Any],
    constructor_cfg: dict[str, Any],
    reference_graphs: Sequence[nx.Graph],
) -> str:
    payload = {
        "split": split,
        "source_seed": int(source_cfg.get("source_seed", 20260812)),
        "replicates": int(source_cfg.get("replicates", 1)),
        "source_randomization_steps": int(
            source_cfg.get("source_randomization_steps", 0)
        ),
        "constructor": constructor_cfg,
        "reference_fingerprint": _graph_collection_fingerprint(reference_graphs),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _fixed_sources(
    reference_graphs: Sequence[nx.Graph],
    *,
    split: str,
    source_cfg: dict[str, Any],
    constructor_cfg: dict[str, Any],
    cache_path: Path,
    force: bool,
) -> dict[str, Any]:
    expected_spec = _source_spec_hash(
        split,
        source_cfg,
        constructor_cfg,
        reference_graphs,
    )
    if cache_path.is_file() and not force:
        payload = load_pickle(cache_path)
        if payload.get("source_spec_hash") != expected_spec:
            raise RuntimeError(
                f"Fixed-source cache {cache_path} was created from a different "
                "configuration. Re-run with --force-sources."
            )
        graphs = list(payload.get("graphs", []))
        if not graphs:
            raise RuntimeError(f"Fixed-source cache is empty: {cache_path}")
        actual = _graph_collection_fingerprint(graphs)
        if actual != payload.get("source_fingerprint"):
            raise RuntimeError(f"Fixed-source cache fingerprint failed: {cache_path}")
        return payload

    if not reference_graphs:
        raise ValueError(f"Reference split {split!r} is empty.")
    rng = np.random.default_rng(int(source_cfg.get("source_seed", 20260812)))
    replicates = int(source_cfg.get("replicates", 1))
    if replicates <= 0:
        raise ValueError("fixed_sources.replicates must be positive.")
    randomization_steps = int(source_cfg.get("source_randomization_steps", 0))
    sources: list[nx.Graph] = []
    reference_indices: list[int] = []
    for _replica in range(replicates):
        for reference_index, reference in enumerate(reference_graphs):
            summary = _degree_summary(reference)
            source = construct_coarse_graph(summary, constructor_cfg, rng)
            source = _randomize_source(
                source,
                steps=randomization_steps,
                rng=rng,
            )
            assert_constructor_validity(source, summary, require_connected=True)
            sources.append(source)
            reference_indices.append(reference_index)
    payload = {
        "format": "topology_stress_fixed_sources_v1",
        "reference_split": split,
        "source_spec_hash": expected_spec,
        "source_fingerprint": _graph_collection_fingerprint(sources),
        "reference_indices": reference_indices,
        "graphs": sources,
    }
    save_pickle(payload, cache_path)
    return payload


def _metric_kwargs(evaluation_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "compute_orbit": bool(evaluation_cfg.get("compute_orbit", True)),
        "compute_graphlet_history": bool(
            evaluation_cfg.get("compute_graphlet_history", True)
        ),
        "graphlet_k_min": int(evaluation_cfg.get("graphlet_k_min", 3)),
        "graphlet_k_max": int(evaluation_cfg.get("graphlet_k_max", 5)),
        "graphlet_connected_only": bool(
            evaluation_cfg.get("graphlet_connected_only", True)
        ),
        "graphlet_num_samples": evaluation_cfg.get("graphlet_num_samples", 8192),
        "graphlet_backend": str(evaluation_cfg.get("graphlet_backend", "sampled")),
    }


def _finite_mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _training_metadata(checkpoint_path: Path) -> dict[str, Any]:
    report_path = checkpoint_path.parent / "training_report.json"
    if not report_path.is_file():
        return {}
    with report_path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    return {
        "best_epoch": report.get("best_epoch"),
        "best_val_loss": report.get("best_val_loss"),
        "num_train_examples": report.get("num_train_examples"),
        "num_val_examples": report.get("num_val_examples"),
    }


def _evaluate_variant(
    *,
    phase: str,
    variant: str,
    seed: int,
    checkpoint_path: Path,
    sources: Sequence[nx.Graph],
    source_fingerprint: str,
    reference_graphs: Sequence[nx.Graph],
    train_graphs: Sequence[nx.Graph],
    coarse_metrics: dict[str, float],
    refiner_cfg: dict[str, Any],
    evaluation_cfg: dict[str, Any],
    output_dir: Path,
    device: str,
    save_graphs: bool,
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing topology checkpoint: {checkpoint_path}")
    model, graphlet_basis, summary_config, checkpoint = load_topology_checkpoint(
        checkpoint_path,
        device=device,
    )
    model_device = next(model.parameters()).device
    refinement_seed = int(evaluation_cfg.get("refinement_seed", 20260813))
    refined_graphs: list[nx.Graph] = []
    traces: list[list[dict[str, Any]]] = []
    runtimes: list[float] = []
    for graph_index, source in enumerate(sources):
        graph_rng = np.random.default_rng(
            refinement_seed + graph_index * 1_000_003
        )
        started = time.perf_counter()
        refined, trace = refine_graph_with_topology_predictions(
            source.copy(),
            model=model,
            graphlet_basis=graphlet_basis,
            summary_config=summary_config,
            refiner_config=refiner_cfg,
            device=model_device,
            rng=graph_rng,
            return_trace=True,
        )
        runtimes.append(float(time.perf_counter() - started))
        refined_graphs.append(refined)
        traces.append(trace)
        print(
            f"[{phase}/{variant}/seed_{seed}] "
            f"graph={graph_index + 1}/{len(sources)} "
            f"accepted={sum(bool(row.get('accepted')) for row in trace)}",
            flush=True,
        )

    degree_rate = degree_preservation_rate(sources, refined_graphs)
    connected_rate = float(
        np.mean(
            [
                graph.number_of_nodes() == 1 or nx.is_connected(graph)
                for graph in refined_graphs
            ]
        )
    )
    simple_rate = float(
        np.mean(
            [
                not graph.is_directed()
                and not graph.is_multigraph()
                and nx.number_of_selfloops(graph) == 0
                for graph in refined_graphs
            ]
        )
    )
    if degree_rate != 1.0 or connected_rate != 1.0 or simple_rate != 1.0:
        raise AssertionError(
            "Stress-test refinement violated an invariant: "
            f"degree={degree_rate}, connected={connected_rate}, simple={simple_rate}."
        )
    if _graph_collection_fingerprint(sources) != source_fingerprint:
        raise AssertionError("A stress-test variant mutated the cached source graphs.")

    refined_metrics = evaluate_graph_sets(
        reference_graphs,
        refined_graphs,
        train_graphs,
        **_metric_kwargs(evaluation_cfg),
    )
    selection_names = [
        str(name)
        for name in evaluation_cfg.get(
            "selection_metrics",
            ["clustering_mmd", "orbit_mmd", "graphlet_history_mmd"],
        )
    ]
    used_names = [
        name
        for name in selection_names
        if name in refined_metrics and np.isfinite(float(refined_metrics[name]))
    ]
    if not used_names:
        raise RuntimeError("No finite validation selection metrics were produced.")
    selection_score = _finite_mean([refined_metrics[name] for name in used_names])
    relative_score = _finite_mean(
        [
            float(refined_metrics[name])
            / max(abs(float(coarse_metrics[name])), 1.0e-12)
            for name in used_names
        ]
    )
    relative_improvement = _finite_mean(
        [
            (float(coarse_metrics[name]) - float(refined_metrics[name]))
            / max(abs(float(coarse_metrics[name])), 1.0e-12)
            for name in used_names
        ]
    )

    accepted_counts = [
        sum(bool(row.get("accepted")) for row in trace) for trace in traces
    ]
    trace_rows = [row for trace in traces for row in trace]
    decision_rows = [row for row in trace_rows if "num_proposals" in row]
    accepted_rows = [row for row in trace_rows if bool(row.get("accepted"))]
    steps = int(refiner_cfg.get("steps", 0))
    total_proposals = sum(int(row.get("num_proposals", 0)) for row in decision_rows)
    total_valid = sum(
        int(row.get("num_valid_candidates", 0)) for row in decision_rows
    )
    total_accepted = sum(accepted_counts)
    training = _training_metadata(checkpoint_path)
    predictor_report = dict(checkpoint.get("report", {}) or {})
    report = {
        "format": "topology_stress_variant_v1",
        "phase": phase,
        "variant": variant,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "source_fingerprint": source_fingerprint,
        "num_sources": len(sources),
        "reference_split": str(evaluation_cfg.get("reference_split", "val")),
        "refiner": refiner_cfg,
        "coarse_metrics": coarse_metrics,
        "refined_metrics": refined_metrics,
        "selection_metrics_requested": selection_names,
        "selection_metrics_used": used_names,
        "selection_score": selection_score,
        "relative_score_vs_coarse": relative_score,
        "relative_improvement_vs_coarse": relative_improvement,
        "degree_preservation_rate": degree_rate,
        "connectedness_rate": connected_rate,
        "simple_graph_rate": simple_rate,
        "mean_accepted_steps": float(np.mean(accepted_counts)),
        "fraction_hitting_step_limit": float(
            np.mean([count == steps for count in accepted_counts])
        ),
        "mean_graph_runtime_seconds": float(np.mean(runtimes)),
        "runtime_seconds": float(sum(runtimes)),
        "candidate_proposals": total_proposals,
        "valid_candidates": total_valid,
        "candidate_pass_rate": float(total_valid / max(total_proposals, 1)),
        "proposals_per_accepted_swap": (
            float(total_proposals / total_accepted)
            if total_accepted > 0
            else float("nan")
        ),
        "all_accepted_moves_improve_frozen_energy": bool(
            all(float(row.get("energy_improvement", 0.0)) > 0.0 for row in accepted_rows)
        ),
        "predictor_val_graphlet_mae": predictor_report.get("val_graphlet_mae"),
        "training": training,
    }
    ensure_dir(output_dir)
    save_json(report, output_dir / "report.json")
    if save_graphs:
        save_pickle(refined_graphs, output_dir / "refined_graphs.pkl")
        save_pickle(traces, output_dir / "traces.pkl")
    return report


def _flatten_report(report: dict[str, Any]) -> dict[str, Any]:
    row = {
        "phase": report["phase"],
        "variant": report["variant"],
        "seed": report["seed"],
        "selection_score": report["selection_score"],
        "relative_score_vs_coarse": report["relative_score_vs_coarse"],
        "relative_improvement_vs_coarse": report[
            "relative_improvement_vs_coarse"
        ],
        "degree_preservation_rate": report["degree_preservation_rate"],
        "connectedness_rate": report["connectedness_rate"],
        "mean_accepted_steps": report["mean_accepted_steps"],
        "fraction_hitting_step_limit": report["fraction_hitting_step_limit"],
        "mean_graph_runtime_seconds": report["mean_graph_runtime_seconds"],
        "proposals_per_accepted_swap": report["proposals_per_accepted_swap"],
        "best_epoch": report.get("training", {}).get("best_epoch"),
        "best_val_loss": report.get("training", {}).get("best_val_loss"),
        "predictor_val_graphlet_mae": report.get("predictor_val_graphlet_mae"),
        "checkpoint": report["checkpoint"],
        "source_fingerprint": report["source_fingerprint"],
    }
    for prefix in ("coarse", "refined"):
        for key, value in report[f"{prefix}_metrics"].items():
            row[f"{prefix}_{key}"] = value
    return row


def _update_leaderboard(phase_dir: Path, reports: Sequence[dict[str, Any]]) -> None:
    json_path = phase_dir / "leaderboard.json"
    existing: list[dict[str, Any]] = []
    if json_path.is_file():
        with json_path.open("r", encoding="utf-8") as stream:
            existing = list(json.load(stream))
    keyed = {
        (str(row["phase"]), str(row["variant"]), int(row["seed"])): row
        for row in existing
    }
    for report in reports:
        row = _flatten_report(report)
        keyed[(row["phase"], row["variant"], int(row["seed"]))] = row
    rows = sorted(
        keyed.values(),
        key=lambda row: (
            float(row.get("selection_score", float("inf"))),
            str(row.get("variant", "")),
            int(row.get("seed", 0)),
        ),
    )
    save_json(rows, json_path)
    fields = sorted({key for row in rows for key in row})
    csv_path = phase_dir / "leaderboard.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _variant_rows(
    phase_cfg: dict[str, Any], selected: set[str] | None
) -> list[dict[str, Any]]:
    variants = [dict(item) for item in phase_cfg.get("variants", [])]
    if selected:
        known = {str(item.get("name")) for item in variants}
        missing = sorted(selected - known)
        if missing:
            raise ValueError(f"Unknown requested variants: {missing}; known={sorted(known)}")
        return [item for item in variants if str(item.get("name")) in selected]
    return [item for item in variants if bool(item.get("enabled", True))]


def _resolved_training_config(
    base_config: dict[str, Any],
    phase_cfg: dict[str, Any],
    variant_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    resolved = _deep_merge(base_config, phase_cfg.get("shared_overrides", {}))
    resolved = _deep_merge(resolved, variant_cfg.get("overrides", {}))
    resolved["seed"] = int(seed)
    resolved["topology_refiner"] = _deep_merge(
        dict(resolved.get("topology_refiner", {}) or {}),
        dict(phase_cfg.get("evaluation_refiner", {}) or {}),
    )
    return _format_seed(resolved, seed)


def _train_variant(
    *,
    repo_root: Path,
    config: dict[str, Any],
    output_dir: Path,
    seed: int,
    device: str,
    force: bool,
) -> Path:
    checkpoint = output_dir / "checkpoint.pt"
    training_report = output_dir / "training_report.json"
    resolved_path = output_dir / "resolved_config.yaml"
    ensure_dir(output_dir)
    save_yaml(config, resolved_path)
    if checkpoint.is_file() and training_report.is_file() and not force:
        print(f"Reusing completed training run: {checkpoint}", flush=True)
        return checkpoint
    command = [
        sys.executable,
        "scripts/train_topology_grapher.py",
        "--config",
        str(resolved_path),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    environment = os.environ.copy()
    source_path = str(repo_root / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not environment.get("PYTHONPATH")
        else source_path + os.pathsep + environment["PYTHONPATH"]
    )
    environment["PYTHONUNBUFFERED"] = "1"
    print("Training command:", " ".join(command), flush=True)
    subprocess.run(command, cwd=repo_root, env=environment, check=True)
    if not checkpoint.is_file() or not training_report.is_file():
        raise RuntimeError(f"Training did not produce a complete run in {output_dir}")
    return checkpoint


def _print_plan(
    phases: Sequence[str],
    stress_config: dict[str, Any],
    selected: set[str] | None,
    seeds: Sequence[int],
    output_root: Path,
) -> None:
    print(f"Output root: {output_root}")
    print(f"Seeds: {list(seeds)}")
    for phase in phases:
        phase_cfg = dict(stress_config.get(phase, {}) or {})
        rows = _variant_rows(phase_cfg, selected)
        print(f"{phase}: {[row.get('name') for row in rows]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run controlled Community-small topology GraphER stress tests."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--phase",
        choices=[
            "inference_budget",
            "training_budget",
            "capacity",
            "graphlet_order",
            "all",
        ],
        required=True,
    )
    parser.add_argument(
        "--action",
        choices=["train-eval", "train-only", "eval-only"],
        default="train-eval",
    )
    parser.add_argument("--variant", action="append", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-source-graphs", type=int, default=None)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-sources", action="store_true")
    parser.add_argument("--no-save-graphs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    stress_path = _repo_path(repo_root, args.config).resolve()
    stress_config = load_yaml(stress_path)
    base_path = _repo_path(repo_root, stress_config["base_config"]).resolve()
    base_config = load_yaml(base_path)
    output_root = _repo_path(
        repo_root,
        args.output_dir or stress_config.get("output_root", "outputs/stress/topology"),
    ).resolve()
    seeds = list(
        args.seeds
        if args.seeds is not None
        else stress_config.get("development_seeds", [42])
    )
    selected = set(args.variant or []) or None
    if args.phase == "all":
        phases = ["inference_budget", "training_budget"]
        phases.extend(
            phase
            for phase in ("capacity", "graphlet_order")
            if bool((stress_config.get(phase, {}) or {}).get("enabled", False))
        )
    else:
        phases = [args.phase]
    if args.dry_run:
        _print_plan(phases, stress_config, selected, seeds, output_root)
        return
    if "inference_budget" in phases and args.action == "train-only":
        raise ValueError("inference_budget has no training stage; use --action eval-only.")

    needs_evaluation = args.action != "train-only" or "inference_budget" in phases
    if needs_evaluation:
        _load_runtime_dependencies()
    train_graphs: list[nx.Graph] = []
    reference_graphs: list[nx.Graph] = []
    sources: list[nx.Graph] = []
    source_fingerprint = ""
    coarse_metrics: dict[str, float] = {}
    evaluation_cfg = dict(stress_config.get("evaluation", {}) or {})
    if needs_evaluation:
        dataset_cfg = dict(base_config.get("dataset", {}) or {})
        splits = load_dataset_splits(
            str(dataset_cfg.get("name", "sbm")),
            root=_repo_path(repo_root, dataset_cfg.get("root", "outputs/datasets")),
            build_if_missing=bool(dataset_cfg.get("build_if_missing", False)),
            config_path=_repo_path(
                repo_root,
                dataset_cfg.get("config_path", "configs/datasets/community_small.yaml"),
            ),
        )
        train_graphs = list(splits["train"])
        reference_split = str(evaluation_cfg.get("reference_split", "val"))
        if reference_split != "val":
            raise ValueError(
                "Stress-test model selection must use evaluation.reference_split: val."
            )
        reference_graphs = list(splits.get(reference_split, []))
        evaluation_cfg["reference_split"] = reference_split
        source_cfg = dict(stress_config.get("fixed_sources", {}) or {})
        constructor_cfg = _deep_merge(
            dict(base_config.get("constructor", {}) or {}),
            dict(source_cfg.get("constructor_overrides", {}) or {}),
        )
        cache_name = str(source_cfg.get("cache_name", "fixed_val_sources.pkl"))
        source_payload = _fixed_sources(
            reference_graphs,
            split=reference_split,
            source_cfg=source_cfg,
            constructor_cfg=constructor_cfg,
            cache_path=output_root / cache_name,
            force=args.force_sources,
        )
        sources = list(source_payload["graphs"])
        if args.max_source_graphs is not None:
            if args.max_source_graphs <= 0:
                raise ValueError("--max-source-graphs must be positive.")
            sources = sources[: args.max_source_graphs]
        source_fingerprint = _graph_collection_fingerprint(sources)
        if bool(evaluation_cfg.get("compute_orbit", True)):
            configure_orca_executable(
                evaluation_cfg.get("orca_exec"),
                required=True,
            )
        coarse_metrics = evaluate_graph_sets(
            reference_graphs,
            sources,
            train_graphs,
            **_metric_kwargs(evaluation_cfg),
        )
        save_json(
            {
                "reference_split": reference_split,
                "num_sources": len(sources),
                "source_fingerprint": source_fingerprint,
                "coarse_metrics": coarse_metrics,
            },
            output_root / "fixed_source_report.json",
        )

    for phase in phases:
        phase_cfg = dict(stress_config.get(phase, {}) or {})
        variants = _variant_rows(phase_cfg, selected)
        phase_dir = ensure_dir(output_root / phase)
        new_reports: list[dict[str, Any]] = []
        for variant_cfg in variants:
            name = str(variant_cfg["name"])
            for seed in seeds:
                output_dir = phase_dir / name / f"seed_{seed}"
                if phase == "inference_budget":
                    checkpoint_template = str(phase_cfg["checkpoint"])
                    checkpoint_path = _repo_path(
                        repo_root,
                        checkpoint_template.format(seed=seed),
                    ).resolve()
                    refiner_cfg = _deep_merge(
                        dict(base_config.get("topology_refiner", {}) or {}),
                        dict(variant_cfg.get("refiner", {}) or {}),
                    )
                else:
                    resolved = _resolved_training_config(
                        base_config,
                        phase_cfg,
                        variant_cfg,
                        seed,
                    )
                    refiner_cfg = dict(resolved.get("topology_refiner", {}) or {})
                    checkpoint_path = output_dir / "checkpoint.pt"
                    if args.action != "eval-only":
                        checkpoint_path = _train_variant(
                            repo_root=repo_root,
                            config=resolved,
                            output_dir=output_dir,
                            seed=seed,
                            device=args.device,
                            force=args.force_train,
                        )
                    elif not (output_dir / "resolved_config.yaml").is_file():
                        ensure_dir(output_dir)
                        save_yaml(resolved, output_dir / "resolved_config.yaml")
                if args.action == "train-only":
                    continue
                report = _evaluate_variant(
                    phase=phase,
                    variant=name,
                    seed=seed,
                    checkpoint_path=checkpoint_path,
                    sources=sources,
                    source_fingerprint=source_fingerprint,
                    reference_graphs=reference_graphs,
                    train_graphs=train_graphs,
                    coarse_metrics=coarse_metrics,
                    refiner_cfg=refiner_cfg,
                    evaluation_cfg=evaluation_cfg,
                    output_dir=output_dir,
                    device=args.device,
                    save_graphs=not args.no_save_graphs,
                )
                new_reports.append(report)
        if new_reports:
            _update_leaderboard(phase_dir, new_reports)
            best = min(new_reports, key=lambda row: float(row["selection_score"]))
            print(
                f"Best newly evaluated {phase} variant: {best['variant']} "
                f"seed={best['seed']} score={best['selection_score']:.6f}",
                flush=True,
            )
            print(f"Leaderboard: {phase_dir / 'leaderboard.csv'}", flush=True)


if __name__ == "__main__":
    main()
