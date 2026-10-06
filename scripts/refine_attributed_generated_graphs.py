#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import pickle
import time
from typing import Any

import networkx as nx
import numpy as np

from grapher.rewiring_mlp.attributed.adjacency_diffusion import validate_model_config
from grapher.rewiring_mlp.attributed.induced_graphlets import validate_model_graphlets
from grapher.rewiring_mlp.attributed.joint_typed_edge_data import (
    graph_from_record,
    graph_record,
    record_hash,
)
from grapher.rewiring_mlp.attributed.joint_typed_edge_generation import (
    refine_typed_graph,
    sample_soft_endpoint,
    validate_refiner,
)
from grapher.rewiring_mlp.attributed.joint_typed_edge_model import load_checkpoint
from grapher.rewiring_mlp.molecular.graph_io import is_valid_molecular_graph
from grapher.rewiring_mlp.molecular.typed_invariants import (
    extract_typed_invariant,
    typed_invariant_matches_graph,
)
from grapher.utils.io import apply_config_overrides, load_yaml, save_pickle


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
    temporary.replace(path)


def _atomic_pickle(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle)
        handle.flush()
    temporary.replace(path)


def _normalize_graph(graph: nx.Graph) -> nx.Graph:
    # Reuse the same canonical graph serialization used by the joint attributed pipeline.
    return graph_from_record(graph_record(nx.Graph(graph)))


def _edge_type_counts(graph: nx.Graph) -> Counter[int]:
    return Counter(int(data["bond_type"]) for _, _, data in graph.edges(data=True))


def _node_types(graph: nx.Graph) -> tuple[int, ...]:
    return tuple(int(graph.nodes[node]["atomic_num"]) for node in sorted(graph.nodes()))


def _degrees(graph: nx.Graph) -> tuple[int, ...]:
    return tuple(int(graph.degree(node)) for node in sorted(graph.nodes()))


def _mean(values: list[float | int | bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def _unsupported_signatures(graph: nx.Graph, model) -> list[Any]:
    """Return typed signatures not represented by the trained checkpoint.

    The checkpoint vocabulary is fixed at training time.  External generated
    molecules may legitimately contain domain-feasible signatures that were not
    observed in the training split.  We must not expand/remap the vocabulary at
    inference because that would change the trained model.
    """
    invariant = extract_typed_invariant(
        graph,
        edge_types=model.edge_types,
        node_attribute=model.vectorizer.vocabulary.node_attribute,
        edge_attribute=model.vectorizer.vocabulary.edge_attribute,
    )
    support = set(model.vectorizer.vocabulary.signatures)
    return sorted(
        {signature for signature in invariant.signatures if signature not in support},
        key=lambda signature: (repr(signature.node_type), signature.edge_degrees),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the trained joint attributed GraphER refiner to an existing list of "
            "molecular graphs. The source graphs are never regenerated or filtered."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-graphs", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-graphs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--set", "--override", dest="config_overrides", action="append", default=[]
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    apply_config_overrides(config, args.config_overrides)
    external = dict(config.get("external_rewiring", {}) or {})

    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    input_path = Path(args.input_graphs or external.get("input_graphs", ""))
    checkpoint_path = Path(args.checkpoint or external.get("checkpoint", ""))
    output_dir = Path(args.output_dir or external.get("output_dir", ""))
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing input graph file: {input_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing GraphER checkpoint: {checkpoint_path}")
    if not str(output_dir):
        raise ValueError("Provide external_rewiring.output_dir or --output-dir.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("rb") as handle:
        loaded = pickle.load(handle)
    if not isinstance(loaded, (list, tuple)):
        raise TypeError("--input-graphs must contain a list/tuple of NetworkX graphs.")
    all_sources = [_normalize_graph(graph) for graph in loaded]
    requested = int(
        args.num_graphs
        if args.num_graphs is not None
        else external.get("num_graphs", len(all_sources))
    )
    if requested <= 0:
        requested = len(all_sources)
    if requested > len(all_sources):
        raise ValueError(
            f"Requested {requested} source graphs but input contains only {len(all_sources)}."
        )
    sources = all_sources[:requested]

    model, checkpoint = load_checkpoint(str(checkpoint_path), args.device or "auto")
    validate_refiner(config["attributed_refiner"], model)
    validate_model_config(model, config)
    validate_model_graphlets(model, config)

    categorical = dict(config.get("categorical_state", {}) or {})
    if tuple(categorical.get("edge_categories", [])) != tuple(model.edge_types):
        raise ValueError("Config/checkpoint edge categories differ.")
    if tuple(categorical.get("node_categories", [])) != tuple(model.atom_types):
        raise ValueError("Config/checkpoint node categories differ.")

    # Sampling semantics that define the trained soft bridge must remain unchanged.
    trained_diffusion = dict(checkpoint["config"]["edge_diffusion"])
    current_diffusion = dict(config.get("edge_diffusion", {}) or {})
    if not math.isclose(
        float(current_diffusion.get("smoothing", 0.01)), float(model.smoothing),
        rel_tol=0.0, abs_tol=1.0e-12,
    ):
        raise ValueError("edge_diffusion.smoothing differs from the checkpoint.")
    defaults = {
        "sigma": 1.0,
        "spectral_sigma": 0.15,
        "spectral_enabled": True,
        "bridge": "centered_logit_brownian",
    }
    for key, default in defaults.items():
        if current_diffusion.get(key, default) != trained_diffusion.get(key, default):
            raise ValueError(
                f"edge_diffusion.{key} differs from training; use the training value."
            )

    refiner_cfg = dict(config["attributed_refiner"])
    checkpoint_every = max(int(external.get("checkpoint_every", 64)), 0)
    unsupported_policy = str(external.get("unsupported_signature_policy", "retain")).lower()
    if unsupported_policy not in {"retain", "error"}:
        raise ValueError(
            "external_rewiring.unsupported_signature_policy must be retain or error."
        )
    finals: list[nx.Graph] = []
    records: list[dict[str, Any]] = []
    started = time.perf_counter()

    input_sha = _sha256(input_path)
    checkpoint_sha = _sha256(checkpoint_path)
    source_record_sha = record_hash([graph_record(graph) for graph in sources])

    def flush(complete: bool) -> None:
        _atomic_pickle(sources, output_dir / "source_graphs.pkl")
        _atomic_pickle(finals, output_dir / "base_graphs.partial.pkl" if not complete else output_dir / "base_graphs.pkl")
        report = {
            "format": "external_attributed_grapher_rewiring_v1",
            "complete": bool(complete),
            "seed": seed,
            "input_graphs": str(input_path),
            "input_graphs_sha256": input_sha,
            "source_record_sha256": source_record_sha,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "num_requested": requested,
            "num_completed": len(finals),
            "paired_one_to_one": len(finals) == requested if complete else None,
            "refiner": refiner_cfg,
            "unsupported_signature_policy": unsupported_policy,
            "runtime_seconds": float(time.perf_counter() - started),
            "records": records,
        }
        if complete:
            report["output_record_sha256"] = record_hash(
                [graph_record(graph) for graph in finals]
            )
            supported_records = [r for r in records if r.get("checkpoint_signature_supported", True)]
            unsupported_records = [r for r in records if not r.get("checkpoint_signature_supported", True)]
            report["diagnostics"] = {
                "source_raw_validity": _mean([r["source_raw_valid"] for r in records]),
                "final_raw_validity": _mean([r["final_raw_valid"] for r in records]),
                "changed_fraction": _mean([r["changed"] for r in records]),
                "mean_accepted_steps": _mean([r["accepted_steps"] for r in records]),
                "node_type_preservation_rate": _mean([r["node_types_preserved"] for r in records]),
                "ordinary_degree_preservation_rate": _mean([r["ordinary_degrees_preserved"] for r in records]),
                "bond_type_count_preservation_rate": _mean([r["bond_type_counts_preserved"] for r in records]),
                "typed_degree_preservation_rate": _mean([r["typed_degree_preserved"] for r in records]),
                "connected_source_rate": _mean([r["source_connected"] for r in records]),
                "connected_final_rate": _mean([r["final_connected"] for r in records]),
                "checkpoint_signature_support_rate": _mean([r.get("checkpoint_signature_supported", True) for r in records]),
                "supported_graph_count": len(supported_records),
                "unsupported_signature_graph_count": len(unsupported_records),
                "supported_changed_fraction": _mean([r["changed"] for r in supported_records]),
                "supported_mean_accepted_steps": _mean([r["accepted_steps"] for r in supported_records]),
            }
            supported_indices = [r["index"] for r in supported_records]
            _atomic_pickle(
                [sources[i] for i in supported_indices],
                output_dir / "supported_source_graphs.pkl",
            )
            _atomic_pickle(
                [finals[i] for i in supported_indices],
                output_dir / "supported_base_graphs.pkl",
            )
        _atomic_json(report, output_dir / "report.json")

    try:
        for index, source in enumerate(sources):
            source = _normalize_graph(source)
            source_invariant = extract_typed_invariant(source, edge_types=model.edge_types)
            source_valid = bool(is_valid_molecular_graph(source))
            source_connected = bool(len(source) <= 1 or nx.is_connected(source))

            unsupported = _unsupported_signatures(source, model)
            checkpoint_supported = not unsupported
            if unsupported and unsupported_policy == "error":
                raise ValueError(
                    f"Graph {index} contains checkpoint-OOV typed signatures: "
                    + ", ".join(repr(sig) for sig in unsupported)
                )

            if unsupported:
                # Safe selective-refinement policy: keep the graph exactly as it
                # was generated.  Do not remap an unseen typed signature into the
                # model vocabulary and do not filter/replace the sample.
                final = source.copy()
                bridge = {"prediction_calls": 0, "sampling_steps": 0}
                refinement = {
                    "accepted_steps": 0,
                    "stop_reason": "unsupported_typed_signature_retained",
                    "typed_degree_preserved": True,
                    "connected": source_connected,
                    "unsupported_typed_signatures": [sig.to_dict() for sig in unsupported],
                }
            else:
                targets, bridge = sample_soft_endpoint(
                    model,
                    source,
                    config,
                    seed=seed + index * 1009 + 7043,
                )
                final, refinement = refine_typed_graph(
                    source,
                    targets,
                    model,
                    config,
                    seed=seed + index * 1009 + 9049,
                )
            final = _normalize_graph(final)

            if _node_types(source) != _node_types(final):
                raise AssertionError(f"Graph {index}: node categories changed.")
            if _degrees(source) != _degrees(final):
                raise AssertionError(f"Graph {index}: indexed ordinary degrees changed.")
            if _edge_type_counts(source) != _edge_type_counts(final):
                raise AssertionError(f"Graph {index}: global bond-type counts changed.")
            if not typed_invariant_matches_graph(final, source_invariant):
                raise AssertionError(f"Graph {index}: indexed typed-degree invariant changed.")

            final_valid = bool(is_valid_molecular_graph(final))
            final_connected = bool(len(final) <= 1 or nx.is_connected(final))
            changed = graph_record(source) != graph_record(final)
            record = {
                "index": index,
                "source_raw_valid": source_valid,
                "final_raw_valid": final_valid,
                "source_connected": source_connected,
                "final_connected": final_connected,
                "changed": bool(changed),
                "accepted_steps": int(refinement.get("accepted_steps", 0)),
                "node_types_preserved": True,
                "ordinary_degrees_preserved": True,
                "bond_type_counts_preserved": True,
                "typed_degree_preserved": bool(refinement.get("typed_degree_preserved", True)),
                "checkpoint_signature_supported": bool(checkpoint_supported),
                "unsupported_typed_signatures": [sig.to_dict() for sig in unsupported],
                "bridge_prediction_calls": int(bridge.get("prediction_calls", 0)),
                "bridge_sampling_steps": int(bridge.get("sampling_steps", 0)),
                "refinement": refinement,
            }
            finals.append(final)
            records.append(record)

            print(
                f"[ExternalAttributedRewire] graph={index + 1}/{requested} "
                f"accepted={record['accepted_steps']} changed={record['changed']} "
                f"supported={record['checkpoint_signature_supported']} "
                f"valid={source_valid}->{final_valid}",
                flush=True,
            )
            if checkpoint_every > 0 and (index + 1) % checkpoint_every == 0:
                flush(False)
    except Exception:
        flush(False)
        raise

    flush(True)
    print(f"Saved paired sources: {output_dir / 'source_graphs.pkl'}", flush=True)
    print(f"Saved rewired graphs: {output_dir / 'base_graphs.pkl'}", flush=True)
    print(f"Saved report: {output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
