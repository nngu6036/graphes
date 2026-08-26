#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import pickle
import time
from typing import Any

import networkx as nx
import numpy as np
import torch

from grapher.data.io import load_dataset_splits
from grapher.models.dhvae_hh.degree_sampler import TypedDegreeVAESampler
from grapher.models.dhvae_hh.typed_constructor import (
    TypedConstructionError,
    construct_typed_graph,
)
from grapher.rewiring_mlp.attributed.spectral_graphlet_refiner import (
    AttributedSpectralGraphletRefinerConfig,
    refine_attributed_graph_with_spectral_graphlet_diffusion,
)
from grapher.rewiring_mlp.attributed.spectral_model import (
    ATTRIBUTED_SPECTRAL_GRAPHLET_CHECKPOINT_FORMAT,
    load_attributed_spectral_graphlet_checkpoint,
)
from grapher.rewiring_mlp.molecular.constraints import bond_order
from grapher.rewiring_mlp.molecular.graph_io import (
    graph_to_smiles,
    is_valid_molecular_graph,
)
from grapher.rewiring_mlp.molecular.typed_invariants import (
    TypedInvariant,
    extract_typed_invariant,
    typed_invariant_matches_graph,
)
from grapher.utils.io import (
    apply_config_overrides,
    ensure_dir,
    load_yaml,
    save_json,
    save_pickle,
)


def _limited(values: list[Any], limit: int | None) -> list[Any]:
    return values if limit is None or int(limit) <= 0 else values[: int(limit)]


def _complete_molecular_aliases(
    graph: nx.Graph,
    *,
    node_attribute: str,
    edge_attribute: str,
) -> nx.Graph:
    """Normalize labels and provide the aliases used by RDKit helpers."""

    out = nx.convert_node_labels_to_integers(
        nx.Graph(graph), first_label=0, ordering="sorted"
    )
    for node, data in out.nodes(data=True):
        if node_attribute not in data:
            raise KeyError(f"Node {node!r} is missing {node_attribute!r}.")
        value = data[node_attribute]
        data.setdefault("atomic_num", int(value))
        data.setdefault("atom_type", int(value))
    for u, v, data in out.edges(data=True):
        if edge_attribute not in data:
            raise KeyError(f"Edge {(u, v)!r} is missing {edge_attribute!r}.")
        value = int(data[edge_attribute])
        data.setdefault("bond_type", value)
        data.setdefault("bond_order", float(bond_order(value)))
    return out


def _atomic_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(obj, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else 0.0


def _weighted_valence(graph: nx.Graph) -> tuple[float, ...]:
    values = {int(node): 0.0 for node in graph.nodes()}
    for u, v, data in graph.edges(data=True):
        category = int(data.get("bond_type", data.get("bond_order", 1)))
        weight = float(bond_order(category))
        values[int(u)] += weight
        values[int(v)] += weight
    return tuple(values[index] for index in sorted(values))


def _edge_type_counts(graph: nx.Graph, edge_attribute: str) -> Counter[Any]:
    return Counter(data[edge_attribute] for _, _, data in graph.edges(data=True))


def _preservation_audit(
    source: nx.Graph,
    final: nx.Graph,
    *,
    node_attribute: str,
    edge_attribute: str,
) -> dict[str, bool]:
    source_nodes = tuple(source.nodes[node][node_attribute] for node in sorted(source.nodes()))
    final_nodes = tuple(final.nodes[node][node_attribute] for node in sorted(final.nodes()))
    source_degrees = tuple(int(source.degree(node)) for node in sorted(source.nodes()))
    final_degrees = tuple(int(final.degree(node)) for node in sorted(final.nodes()))
    return {
        "node_type_preserved": source_nodes == final_nodes,
        "indexed_degree_preserved": source_degrees == final_degrees,
        "edge_type_counts_preserved": _edge_type_counts(
            source, edge_attribute
        )
        == _edge_type_counts(final, edge_attribute),
        "weighted_valence_preserved": bool(
            np.allclose(_weighted_valence(source), _weighted_valence(final), atol=1.0e-8)
        ),
    }


def _sample_invariant(
    source_mode: str,
    *,
    index: int,
    train_graphs: list[nx.Graph],
    test_graphs: list[nx.Graph],
    typed_sampler: TypedDegreeVAESampler | None,
    rng: np.random.Generator,
    edge_types: tuple[Any, ...],
    node_attribute: str,
    edge_attribute: str,
) -> tuple[TypedInvariant, dict[str, Any]]:
    if source_mode in {"oracle", "test_oracle"}:
        if not test_graphs:
            raise ValueError("Oracle typed-invariant generation requires a non-empty test split.")
        reference_index = int(index % len(test_graphs))
        graph = test_graphs[reference_index]
        return (
            extract_typed_invariant(
                graph,
                edge_types=edge_types,
                node_attribute=node_attribute,
                edge_attribute=edge_attribute,
            ),
            {"reference_split": "test", "reference_index": reference_index},
        )
    if source_mode in {"empirical", "train_empirical"}:
        if not train_graphs:
            raise ValueError("Empirical typed-invariant generation requires training graphs.")
        reference_index = int(rng.integers(0, len(train_graphs)))
        graph = train_graphs[reference_index]
        return (
            extract_typed_invariant(
                graph,
                edge_types=edge_types,
                node_attribute=node_attribute,
                edge_attribute=edge_attribute,
            ),
            {"reference_split": "train", "reference_index": reference_index},
        )
    if source_mode in {"learned", "typed_vae", "degree_vae"}:
        if typed_sampler is None:
            raise ValueError("Learned typed-invariant generation requires degree_generator checkpoint_path.")
        summary = typed_sampler.sample(rng)
        invariant = TypedInvariant.from_dict(summary["typed_invariant"])
        return invariant, {"reference_split": None, "reference_index": None}
    raise ValueError(f"Unknown generation.invariant_source: {source_mode!r}")


def _partial_report(
    *,
    generated: int,
    requested: int,
    attempts: list[int],
    started: float,
) -> dict[str, Any]:
    return {
        "format": "attributed_spectral_graphlet_partial_generation_v1",
        "generated": int(generated),
        "requested": int(requested),
        "generation_attempts": int(sum(attempts)),
        "end_to_end_yield_so_far": float(generated / max(sum(attempts), 1)),
        "runtime_seconds": float(time.perf_counter() - started),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate attributed molecular graphs using a typed-degree invariant, "
            "continuous topology/bond-spectrum and attributed graphlet-logit denoising, "
            "and same-bond-type degree-constrained rewiring projection."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-generate", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--set",
        "--override",
        dest="config_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a YAML option with a dotted path; repeat as needed.",
    )
    args = parser.parse_args()
    started = time.perf_counter()

    config = load_yaml(args.config)
    apply_config_overrides(config, args.config_overrides)
    stage = str((config.get("pipeline", {}) or {}).get("stage", "attributed")).lower()
    if stage not in {"attributed", "attributed_topology", "molecular"}:
        raise ValueError("run_attributed_grapher.py requires pipeline.stage: attributed.")

    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    torch.manual_seed(seed)
    seed_sequence = np.random.SeedSequence(seed)
    source_seed, refiner_seed = seed_sequence.spawn(2)
    source_rng = np.random.default_rng(source_seed)

    dataset_cfg = dict(config.get("dataset", {}) or {})
    splits = load_dataset_splits(
        str(dataset_cfg.get("name", "qm9_attributed")),
        root=dataset_cfg.get("root", "outputs/datasets"),
        build_if_missing=bool(dataset_cfg.get("build_if_missing", False)),
        config_path=dataset_cfg.get("config_path"),
    )
    train_graphs = _limited(
        list(splits["train"]), dataset_cfg.get("max_generation_train_graphs")
    )
    test_graphs = list(splits.get("test", []))

    predictor_cfg = dict(config.get("attributed_predictor", {}) or {})
    checkpoint_path = args.checkpoint or predictor_cfg.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("attributed_predictor.checkpoint_path is required.")
    model, vocabulary, graphlet_basis, summary_config, checkpoint = (
        load_attributed_spectral_graphlet_checkpoint(
            checkpoint_path,
            device=args.device or predictor_cfg.get("device", "auto"),
        )
    )
    if checkpoint.get("format") != ATTRIBUTED_SPECTRAL_GRAPHLET_CHECKPOINT_FORMAT:
        raise ValueError("Unsupported attributed predictor checkpoint format.")
    model_device = next(model.parameters()).device
    node_attribute = str(vocabulary.node_attribute)
    edge_attribute = str(vocabulary.edge_attribute)
    predictor_report = dict(checkpoint.get("report", {}) or {})

    categorical_cfg = dict(config.get("categorical_state", {}) or {})
    configured_nodes = categorical_cfg.get("node_categories")
    configured_edges = categorical_cfg.get("edge_categories")
    if configured_nodes is not None and tuple(configured_nodes) != tuple(vocabulary.node_values):
        raise ValueError(
            f"Configured node categories {tuple(configured_nodes)} do not match checkpoint {vocabulary.node_values}."
        )
    if configured_edges is not None and tuple(configured_edges) != tuple(vocabulary.edge_values):
        raise ValueError(
            f"Configured edge categories {tuple(configured_edges)} do not match checkpoint {vocabulary.edge_values}."
        )

    print(
        "[GraphER/AttributedSpectralGraphlet] loaded joint attributed checkpoint "
        f"format={checkpoint.get('format')} device={model_device}",
        flush=True,
    )
    print(
        "[GraphER/AttributedSpectralGraphlet] training=continuous stochastic summary diffusion "
        "without rewiring; generation=reverse summary target followed by same-bond-type "
        "discrete rewiring projection.",
        flush=True,
    )
    print(
        "[GraphER/AttributedSpectralGraphlet] guidance channels: unweighted topology spectrum, "
        "bond-order-weighted spectrum, and attributed graphlet CLR/logit diffusion.",
        flush=True,
    )

    generation_cfg = dict(config.get("generation", {}) or {})
    num_generate = int(
        args.num_generate
        if args.num_generate is not None
        else generation_cfg.get("num_generate", len(test_graphs) or 256)
    )
    if num_generate <= 0:
        raise ValueError("num_generate must be positive.")
    max_attempts = max(int(generation_cfg.get("max_attempts_per_graph", 32)), 1)
    invariant_source = str(
        generation_cfg.get(
            "invariant_source", generation_cfg.get("degree_source", "empirical")
        )
    ).lower()
    require_source_validity = bool(
        generation_cfg.get("require_rdkit_source_validity", False)
    )
    require_final_validity = bool(
        generation_cfg.get("require_rdkit_final_validity", False)
    )
    checkpoint_every = max(int(generation_cfg.get("checkpoint_every", 25)), 0)
    store_traces = bool(generation_cfg.get("store_traces", True))

    typed_sampler: TypedDegreeVAESampler | None = None
    if invariant_source in {"learned", "typed_vae", "degree_vae"}:
        degree_cfg = dict(config.get("degree_generator", {}) or {})
        degree_type = str(degree_cfg.get("type", "typed_degree_histogram_vae")).lower()
        if "typed" not in degree_type:
            raise ValueError("Attributed learned generation requires a typed-degree VAE.")
        typed_sampler = TypedDegreeVAESampler.from_config(degree_cfg, seed=seed)
    elif invariant_source not in {
        "empirical",
        "train_empirical",
        "oracle",
        "test_oracle",
    }:
        raise ValueError(f"Unknown generation.invariant_source: {invariant_source!r}")

    constructor_cfg = dict(config.get("constructor", {}) or {})
    constructor_type = str(constructor_cfg.get("type", "typed_backtracking")).lower()
    if constructor_type not in {"typed_backtracking", "typed", "typed_constructor"}:
        raise ValueError("Attributed generation requires constructor.type: typed_backtracking.")
    refiner_cfg = AttributedSpectralGraphletRefinerConfig.from_dict(
        dict(config.get("attributed_refiner", {}) or {})
    )
    print(
        "[GraphER/AttributedSpectralGraphlet] "
        f"debug={refiner_cfg.debug_enabled} steps={refiner_cfg.steps} "
        f"proposal_budget={refiner_cfg.proposal_budget} "
        f"valid_candidate_budget={refiner_cfg.valid_candidate_budget} "
        f"rdkit_shortlist={refiner_cfg.rdkit_shortlist}",
        flush=True,
    )

    output_dir = ensure_dir(args.output_dir)
    source_graphs: list[nx.Graph] = []
    final_graphs: list[nx.Graph] = []
    traces: list[list[dict[str, Any]]] = []
    graph_runtimes: list[float] = []
    attempts_per_graph: list[int] = []
    constructor_records: list[dict[str, Any]] = []
    source_metadata: list[dict[str, Any]] = []
    audits: list[dict[str, bool]] = []
    rejection_reasons: Counter[str] = Counter()

    for graph_index in range(num_generate):
        graph_started = time.perf_counter()
        succeeded = False
        for attempt in range(1, max_attempts + 1):
            try:
                invariant, invariant_metadata = _sample_invariant(
                    invariant_source,
                    index=graph_index,
                    train_graphs=train_graphs,
                    test_graphs=test_graphs,
                    typed_sampler=typed_sampler,
                    rng=source_rng,
                    edge_types=tuple(vocabulary.edge_values),
                    node_attribute=node_attribute,
                    edge_attribute=edge_attribute,
                )
                if tuple(invariant.edge_types) != tuple(vocabulary.edge_values):
                    raise ValueError(
                        "Typed invariant edge vocabulary does not match predictor checkpoint."
                    )
                if invariant.node_attribute != node_attribute or invariant.edge_attribute != edge_attribute:
                    raise ValueError(
                        "Typed invariant attribute names do not match predictor checkpoint."
                    )

                source, constructor_record = construct_typed_graph(
                    invariant,
                    constructor_cfg,
                    source_rng,
                )
                source = _complete_molecular_aliases(
                    source,
                    node_attribute=node_attribute,
                    edge_attribute=edge_attribute,
                )
                indexed_source_invariant = extract_typed_invariant(
                    source,
                    edge_types=vocabulary.edge_values,
                    node_attribute=node_attribute,
                    edge_attribute=edge_attribute,
                )
                source_is_valid = is_valid_molecular_graph(source)
                if require_source_validity and not source_is_valid:
                    rejection_reasons["rdkit_invalid_source"] += 1
                    continue

                attempt_rng = np.random.default_rng(
                    np.random.SeedSequence([seed, graph_index, attempt, 7919])
                )
                refined, trace = refine_attributed_graph_with_spectral_graphlet_diffusion(
                    model,
                    source,
                    vocabulary=vocabulary,
                    graphlet_basis=graphlet_basis,
                    config=refiner_cfg,
                    device=model_device,
                    rng=attempt_rng,
                    return_trace=True,
                    debug_context=f"graph={graph_index + 1}/{num_generate}",
                )
                refined = _complete_molecular_aliases(
                    refined,
                    node_attribute=node_attribute,
                    edge_attribute=edge_attribute,
                )
                if not typed_invariant_matches_graph(refined, indexed_source_invariant):
                    raise AssertionError(
                        "Refinement changed the indexed typed-degree invariant."
                    )
                if refined.number_of_nodes() > 1 and not nx.is_connected(refined):
                    raise AssertionError("Refinement returned a disconnected molecule.")
                final_is_valid = is_valid_molecular_graph(refined)
                if require_final_validity and not final_is_valid:
                    rejection_reasons["rdkit_invalid_final"] += 1
                    continue

                source_graphs.append(source)
                final_graphs.append(refined)
                traces.append(trace)
                graph_runtimes.append(float(time.perf_counter() - graph_started))
                attempts_per_graph.append(attempt)
                constructor_records.append(
                    {
                        **constructor_record,
                        "generation_attempt": attempt,
                        "source_rdkit_valid": source_is_valid,
                        "final_rdkit_valid": final_is_valid,
                    }
                )
                source_metadata.append(invariant_metadata)
                audits.append(
                    _preservation_audit(
                        source,
                        refined,
                        node_attribute=node_attribute,
                        edge_attribute=edge_attribute,
                    )
                )
                accepted = sum(bool(row.get("accepted")) for row in trace)
                prediction_calls = max(
                    (int(row.get("prediction_calls", 0)) for row in trace), default=0
                )
                print(
                    f"graph={graph_index + 1}/{num_generate} guidance=attributed_spectral_graphlet "
                    f"n={refined.number_of_nodes()} m={refined.number_of_edges()} "
                    f"accepted_steps={accepted} prediction_calls={prediction_calls} "
                    f"source_valid={source_is_valid} final_valid={final_is_valid} "
                    f"attempts={attempt} runtime={graph_runtimes[-1]:.3f}s",
                    flush=True,
                )
                succeeded = True
                break
            except TypedConstructionError as exc:
                reason = str(exc.diagnostics.get("failure_reason", "failed"))
                rejection_reasons[f"constructor:{reason}"] += 1
            except (AssertionError, KeyError, RuntimeError, TypeError, ValueError) as exc:
                rejection_reasons[f"generation:{type(exc).__name__}"] += 1
                if attempt == max_attempts:
                    raise RuntimeError(
                        f"Attributed generation failed for graph {graph_index + 1} "
                        f"after {max_attempts} attempts: {exc}"
                    ) from exc

        if not succeeded:
            raise RuntimeError(
                f"Attributed generation exhausted {max_attempts} attempts for graph {graph_index + 1}."
            )

        if checkpoint_every > 0 and len(final_graphs) % checkpoint_every == 0:
            _atomic_pickle(
                final_graphs, output_dir / "molecular_graphs.partial.pkl"
            )
            _atomic_pickle(
                source_graphs, output_dir / "typed_source_graphs.partial.pkl"
            )
            _atomic_json(
                _partial_report(
                    generated=len(final_graphs),
                    requested=num_generate,
                    attempts=attempts_per_graph,
                    started=started,
                ),
                output_dir / "partial_report.json",
            )
            print(
                f"Saved atomic partial generation checkpoint: {len(final_graphs)}/{num_generate}",
                flush=True,
            )

    trace_rows = [row for trace in traces for row in trace]
    accepted_rows = [row for row in trace_rows if bool(row.get("accepted"))]
    source_validity = [is_valid_molecular_graph(graph) for graph in source_graphs]
    final_validity = [is_valid_molecular_graph(graph) for graph in final_graphs]
    typed_preservation = [
        typed_invariant_matches_graph(
            final,
            extract_typed_invariant(
                source,
                edge_types=vocabulary.edge_values,
                node_attribute=node_attribute,
                edge_attribute=edge_attribute,
            ),
        )
        for source, final in zip(source_graphs, final_graphs)
    ]
    accepted_counts = [
        sum(bool(row.get("accepted")) for row in trace) for trace in traces
    ]
    prediction_counts = [
        max((int(row.get("prediction_calls", 0)) for row in trace), default=0)
        for trace in traces
    ]
    diagnostics = {
        "pipeline_mode": "attributed",
        "guidance_mode": "dual_spectral_attributed_graphlet",
        "invariant_source": invariant_source,
        "num_generated": len(final_graphs),
        "typed_invariant_preservation_rate": float(np.mean(typed_preservation)),
        "node_type_preservation_rate": float(
            np.mean([row["node_type_preserved"] for row in audits])
        ),
        "indexed_degree_preservation_rate": float(
            np.mean([row["indexed_degree_preserved"] for row in audits])
        ),
        "edge_type_count_preservation_rate": float(
            np.mean([row["edge_type_counts_preserved"] for row in audits])
        ),
        "weighted_valence_preservation_rate": float(
            np.mean([row["weighted_valence_preserved"] for row in audits])
        ),
        "connectedness_rate": float(
            np.mean(
                [
                    graph.number_of_nodes() <= 1 or nx.is_connected(graph)
                    for graph in final_graphs
                ]
            )
        ),
        "rdkit_valid_source_rate": float(np.mean(source_validity)),
        "rdkit_valid_final_rate": float(np.mean(final_validity)),
        "mean_accepted_steps": float(np.mean(accepted_counts)),
        "mean_prediction_calls": float(np.mean(prediction_counts)),
        "mean_accepted_swaps_per_prediction_call": float(
            sum(accepted_counts) / max(sum(prediction_counts), 1)
        ),
        "mean_accepted_spectral_gain": _mean(accepted_rows, "spectral_gain"),
        "mean_accepted_topology_spectral_gain": _mean(
            accepted_rows, "topology_spectral_gain"
        ),
        "mean_accepted_bond_spectral_gain": _mean(
            accepted_rows, "bond_spectral_gain"
        ),
        "mean_accepted_graphlet_gain": _mean(accepted_rows, "graphlet_gain"),
        "mean_projection_residual": _mean(accepted_rows, "projection_residual"),
        "mean_spectral_projection_residual": _mean(
            accepted_rows, "spectral_projection_residual"
        ),
        "mean_topology_spectral_projection_residual": _mean(
            accepted_rows, "topology_spectral_projection_residual"
        ),
        "mean_bond_spectral_projection_residual": _mean(
            accepted_rows, "bond_spectral_projection_residual"
        ),
        "mean_graphlet_projection_residual": _mean(
            accepted_rows, "graphlet_projection_residual"
        ),
        "predictor_topology_spectral_nrmse": predictor_report.get(
            "val_topology_spectral_nrmse",
            predictor_report.get("topology_spectral_nrmse"),
        ),
        "predictor_bond_spectral_nrmse": predictor_report.get(
            "val_bond_spectral_nrmse",
            predictor_report.get("bond_spectral_nrmse"),
        ),
        "predictor_graphlet_logit_rmse": predictor_report.get(
            "val_graphlet_logit_rmse",
            predictor_report.get("graphlet_logit_rmse"),
        ),
        "predictor_graphlet_probability_mae": predictor_report.get(
            "val_graphlet_probability_mae",
            predictor_report.get("graphlet_probability_mae"),
        ),
        "mean_candidate_pass_rate": _mean(trace_rows, "candidate_pass_rate"),
        "rdkit_candidates_checked": int(
            sum(int(row.get("rdkit_checked", 0)) for row in trace_rows)
        ),
        "rdkit_candidates_rejected": int(
            sum(int(row.get("rdkit_rejected", 0)) for row in trace_rows)
        ),
        "mean_graph_runtime_seconds": float(np.mean(graph_runtimes)),
        "generation_attempts": int(sum(attempts_per_graph)),
        "end_to_end_yield": float(
            len(final_graphs) / max(sum(attempts_per_graph), 1)
        ),
        "runtime_seconds": float(time.perf_counter() - started),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
    }

    save_pickle(source_graphs, output_dir / "typed_source_graphs.pkl")
    save_pickle(final_graphs, output_dir / "molecular_graphs.pkl")
    save_pickle(final_graphs, output_dir / "generated_graphs.pkl")
    valid_smiles = [
        smiles
        for graph in final_graphs
        for smiles in [graph_to_smiles(graph, canonical=True, sanitize=True)]
        if smiles is not None
    ]
    (output_dir / "generated.smi").write_text(
        "\n".join(valid_smiles) + ("\n" if valid_smiles else ""),
        encoding="utf-8",
    )
    report = {
        "format": "attributed_spectral_graphlet_generation_v1",
        "checkpoint_format": checkpoint.get("format"),
        "training_state_source": checkpoint.get("config", {})
        .get("summary_diffusion", {})
        .get("bridge", "continuous_summary_diffusion"),
        "rewiring_used_during_training": False,
        "rewiring_used_during_generation": True,
        "spectral_channels": ["topology", "bond_weighted"],
        "graphlet_orders": list(graphlet_basis.sizes),
        "vocabulary": vocabulary.to_dict(),
        "summary_config": dict(summary_config.__dict__),
        "diagnostics": diagnostics,
        "constructor_records": constructor_records,
        "source_metadata": source_metadata,
        "traces": traces if store_traces else [],
        "seed": seed,
        "config": config,
    }
    save_json(report, output_dir / "report.json")
    print("Attributed generation diagnostics", flush=True)
    for key, value in diagnostics.items():
        print(f"  {key}: {value}", flush=True)
    print(f"Saved results to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
