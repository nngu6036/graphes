"""Completed base-generator source pools for topology-corrector training.

The topology corrector is trained from *finished* samples emitted by declared
base-generator wrappers.  The saved pools are unconditional and therefore do
not carry a source-to-target index.  This module resolves and verifies their
manifests, creates a deterministic train/validation partition, and performs an
explicit one-to-one coupling to dataset targets within graph-size strata.

The matching cost deliberately uses only the sorted ordinary-degree profile.
Clustering, orbit counts, and graphlet targets are never used to create pairs;
those quantities remain prediction/supervision targets rather than leaked
matching features.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np

from grapher.rewiring_mlp.generic.data import (
    TopologyTrainingPair,
    normalize_topology_graph,
)
from grapher.utils.io import load_pickle

try:  # SciPy is used elsewhere by the evaluation stack, but keep a clear guard.
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - exercised only in incomplete environments.
    linear_sum_assignment = None


_SUPPORTED_MANIFEST_ARTIFACTS = (
    "estimated_graphs",
    "graphs",
    "base_graphs",
)


@dataclass(frozen=True)
class CompletedBasePool:
    generator_id: str
    graphs: tuple[nx.Graph, ...]
    source_indices: tuple[int, ...]
    graph_path: Path
    manifest_path: Path | None
    artifact_key: str
    manifest_format: str | None
    manifest_sha256: str | None
    graph_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_int(value: str) -> int:
    return int.from_bytes(
        hashlib.sha256(value.encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )


def _require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping.")
    return dict(value)


def _resolve_artifact_from_manifest(
    manifest_path: Path,
    *,
    artifact: str | None,
) -> tuple[Path, str, dict[str, Any], dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise TypeError(f"Base-generator manifest must be a mapping: {manifest_path}")

    requested = str(artifact or "auto").strip().lower()
    if requested == "auto":
        available = [key for key in _SUPPORTED_MANIFEST_ARTIFACTS if key in manifest]
        if len(available) != 1:
            raise ValueError(
                "Could not infer the completed graph artifact from manifest "
                f"{manifest_path}; available supported keys are {available}. "
                "Set training_sources.generators[].artifact explicitly."
            )
        key = available[0]
    else:
        aliases = {
            "training_estimates": "estimated_graphs",
            "generated_graphs": "graphs",
            "completed_graphs": "base_graphs",
        }
        key = aliases.get(requested, requested)
        if key not in _SUPPORTED_MANIFEST_ARTIFACTS:
            raise ValueError(
                f"Unsupported completed graph artifact {artifact!r}; expected one "
                f"of {_SUPPORTED_MANIFEST_ARTIFACTS}."
            )
        if key not in manifest:
            raise KeyError(
                f"Manifest {manifest_path} does not contain artifact key {key!r}."
            )

    record = _require_mapping(manifest[key], name=f"manifest.{key}")
    relative_path = record.get("path")
    if not relative_path:
        raise KeyError(f"manifest.{key}.path is required in {manifest_path}.")
    graph_path = Path(str(relative_path)).expanduser()
    if not graph_path.is_absolute():
        graph_path = (manifest_path.parent / graph_path).resolve()
    return graph_path, key, record, manifest


def _normalise_pool_graphs(
    values: Any,
    *,
    graph_path: Path,
    disconnected_policy: str,
) -> tuple[tuple[nx.Graph, ...], tuple[int, ...], dict[str, int]]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(
            "Completed base-generator graph artifact must contain a list or tuple; "
            f"got {type(values).__name__} from {graph_path}."
        )
    policy = str(disconnected_policy).lower()
    if policy not in {"drop", "error"}:
        raise ValueError("matching.disconnected_policy must be 'drop' or 'error'.")

    graphs: list[nx.Graph] = []
    indices: list[int] = []
    dropped_empty = 0
    dropped_disconnected = 0
    for index, raw in enumerate(values):
        if not isinstance(raw, nx.Graph):
            raise TypeError(
                f"Completed source index {index} in {graph_path} is not a NetworkX graph."
            )
        graph = normalize_topology_graph(raw)
        if graph.number_of_nodes() == 0:
            if policy == "error":
                raise ValueError(
                    f"Completed source index {index} in {graph_path} is empty."
                )
            dropped_empty += 1
            continue
        if graph.number_of_nodes() > 1 and not nx.is_connected(graph):
            if policy == "error":
                raise ValueError(
                    "Topology-corrector training requires connected completed "
                    f"sources; source index {index} in {graph_path} is disconnected."
                )
            dropped_disconnected += 1
            continue
        graphs.append(graph)
        indices.append(index)
    return (
        tuple(graphs),
        tuple(indices),
        {
            "loaded": len(values),
            "retained": len(graphs),
            "dropped_empty": dropped_empty,
            "dropped_disconnected": dropped_disconnected,
        },
    )


def load_completed_base_pool(
    declaration: dict[str, Any],
    *,
    disconnected_policy: str = "drop",
) -> tuple[CompletedBasePool, dict[str, Any]]:
    """Load one declared completed graph pool and verify its published checksum."""

    entry = _require_mapping(declaration, name="training_sources.generators[]")
    generator_id = str(
        entry.get("id", entry.get("base_generator", entry.get("type", "")))
    ).strip()
    if not generator_id:
        raise KeyError(
            "Each training source declaration requires id or base_generator."
        )

    manifest_raw = entry.get("manifest_path", entry.get("training_estimates_manifest"))
    generated_raw = entry.get("generated_path", entry.get("graph_path"))
    if bool(manifest_raw) == bool(generated_raw):
        raise ValueError(
            f"Training source {generator_id!r} must declare exactly one of "
            "manifest_path or generated_path."
        )

    manifest_path: Path | None = None
    manifest: dict[str, Any] | None = None
    manifest_hash: str | None = None
    artifact_key: str
    record: dict[str, Any] = {}
    if manifest_raw:
        manifest_path = Path(str(manifest_raw)).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Completed base-generator manifest not found: {manifest_path}"
            )
        graph_path, artifact_key, record, manifest = _resolve_artifact_from_manifest(
            manifest_path,
            artifact=entry.get("artifact"),
        )
        manifest_hash = _sha256(manifest_path)
    else:
        graph_path = Path(str(generated_raw)).expanduser().resolve()
        artifact_key = str(entry.get("artifact", "direct_completed_graphs"))

    if not graph_path.is_file():
        raise FileNotFoundError(
            f"Completed base-generator graph pool not found: {graph_path}"
        )
    graph_hash = _sha256(graph_path)
    verify_sha = bool(entry.get("verify_sha256", True))
    expected_hash = record.get("sha256")
    if verify_sha and expected_hash and graph_hash != str(expected_hash):
        raise RuntimeError(
            "Completed graph pool checksum mismatch for "
            f"{generator_id!r}: expected {expected_hash}, got {graph_hash}."
        )

    raw_graphs = load_pickle(graph_path)
    graphs, source_indices, filtering = _normalise_pool_graphs(
        raw_graphs,
        graph_path=graph_path,
        disconnected_policy=disconnected_policy,
    )
    expected_count = record.get(
        "count", record.get("returned", record.get("requested"))
    )
    if expected_count is not None and int(expected_count) != len(raw_graphs):
        raise RuntimeError(
            f"Manifest count mismatch for {generator_id!r}: manifest records "
            f"{expected_count}, artifact contains {len(raw_graphs)}."
        )

    max_graphs_raw = entry.get("max_graphs")
    if max_graphs_raw is not None and int(max_graphs_raw) > 0:
        limit = int(max_graphs_raw)
        graphs = graphs[:limit]
        source_indices = source_indices[:limit]

    if not graphs:
        raise ValueError(
            f"Completed source pool {generator_id!r} contains no eligible graphs."
        )

    pool = CompletedBasePool(
        generator_id=generator_id,
        graphs=graphs,
        source_indices=source_indices,
        graph_path=graph_path,
        manifest_path=manifest_path,
        artifact_key=artifact_key,
        manifest_format=(str(manifest.get("format")) if manifest else None),
        manifest_sha256=manifest_hash,
        graph_sha256=graph_hash,
    )
    report = {
        "base_generator": generator_id,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_format": pool.manifest_format,
        "manifest_sha256": manifest_hash,
        "artifact": artifact_key,
        "graph_path": str(graph_path),
        "graph_sha256": graph_hash,
        "checksum_verified": bool(verify_sha and expected_hash),
        **filtering,
        "retained_after_limit": len(graphs),
        "published_pairing": (
            dict(manifest.get("pairing", {}) or {}) if manifest else None
        ),
    }
    return pool, report


def _partition_pool(
    pool: CompletedBasePool,
    *,
    validation_fraction: float,
    partition_seed: int,
) -> dict[str, list[tuple[int, nx.Graph]]]:
    fraction = float(validation_fraction)
    if not np.isfinite(fraction) or not 0.0 <= fraction < 1.0:
        raise ValueError("training_sources.validation_fraction must be in [0, 1).")
    count = len(pool.graphs)
    order = np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(
        int(partition_seed) + (_stable_int(pool.generator_id) % 2_000_000_000)
    )
    rng.shuffle(order)
    if count >= 2 and fraction > 0.0:
        val_count = min(max(int(round(count * fraction)), 1), count - 1)
    else:
        val_count = 0
    val_positions = set(int(value) for value in order[:val_count])
    partitions: dict[str, list[tuple[int, nx.Graph]]] = {"train": [], "val": []}
    for position, (source_index, graph) in enumerate(
        zip(pool.source_indices, pool.graphs)
    ):
        split = "val" if position in val_positions else "train"
        partitions[split].append((int(source_index), graph))
    return partitions


def _degree_profile(graph: nx.Graph) -> np.ndarray:
    n = graph.number_of_nodes()
    scale = float(max(n - 1, 1))
    return np.asarray(
        sorted((int(degree) for _, degree in graph.degree()), reverse=True),
        dtype=np.float64,
    ) / scale


def _degree_profile_cost(source: nx.Graph, target: nx.Graph) -> float:
    if source.number_of_nodes() != target.number_of_nodes():
        return float("inf")
    left = _degree_profile(source)
    right = _degree_profile(target)
    return float(np.mean(np.abs(left - right))) if left.size else 0.0


def _match_one_pool(
    sources: Sequence[tuple[int, nx.Graph]],
    targets: Sequence[nx.Graph],
    *,
    generator_id: str,
    split: str,
    graph_path: Path,
    manifest_path: Path | None,
    require_exact_node_count: bool,
    max_degree_cost: float | None,
) -> tuple[list[TopologyTrainingPair], dict[str, Any]]:
    if linear_sum_assignment is None:
        raise RuntimeError(
            "Completed-source matching requires scipy.optimize.linear_sum_assignment."
        )
    normal_targets = [normalize_topology_graph(graph) for graph in targets]
    for target_index, graph in enumerate(normal_targets):
        if graph.number_of_nodes() == 0:
            raise ValueError(f"Target graph {target_index} in split {split} is empty.")
        if graph.number_of_nodes() > 1 and not nx.is_connected(graph):
            raise ValueError(
                f"Target graph {target_index} in split {split} is disconnected."
            )

    source_by_n: dict[int, list[tuple[int, int, nx.Graph]]] = defaultdict(list)
    target_by_n: dict[int, list[tuple[int, nx.Graph]]] = defaultdict(list)
    for local_index, (source_index, graph) in enumerate(sources):
        source_by_n[graph.number_of_nodes()].append(
            (local_index, int(source_index), graph)
        )
    for target_index, graph in enumerate(normal_targets):
        target_by_n[graph.number_of_nodes()].append((target_index, graph))

    if not require_exact_node_count:
        raise NotImplementedError(
            "Only exact-node-count coupling is implemented because rewiring cannot "
            "change graph size."
        )

    pairs: list[TopologyTrainingPair] = []
    stratum_rows: list[dict[str, Any]] = []
    unmatched_source = 0
    unmatched_target = 0
    for n in sorted(set(source_by_n) | set(target_by_n)):
        source_rows = source_by_n.get(n, [])
        target_rows = target_by_n.get(n, [])
        if not source_rows or not target_rows:
            unmatched_source += len(source_rows)
            unmatched_target += len(target_rows)
            stratum_rows.append(
                {
                    "num_nodes": n,
                    "num_sources": len(source_rows),
                    "num_targets": len(target_rows),
                    "num_pairs": 0,
                }
            )
            continue
        cost = np.empty((len(source_rows), len(target_rows)), dtype=np.float64)
        for row, (_, _, source) in enumerate(source_rows):
            for column, (_, target) in enumerate(target_rows):
                # A tiny deterministic tie breaker leaves the reported semantic
                # cost unchanged while making equal-cost assignment reproducible.
                semantic = _degree_profile_cost(source, target)
                cost[row, column] = semantic + 1.0e-12 * (
                    row * max(len(target_rows), 1) + column
                )
        selected_rows, selected_columns = linear_sum_assignment(cost)
        retained_costs: list[float] = []
        rejected_costs = 0
        for row, column in zip(selected_rows.tolist(), selected_columns.tolist()):
            _local_index, source_index, source = source_rows[row]
            target_index, target = target_rows[column]
            semantic_cost = _degree_profile_cost(source, target)
            if max_degree_cost is not None and semantic_cost > float(max_degree_cost):
                rejected_costs += 1
                continue
            retained_costs.append(semantic_cost)
            pairs.append(
                TopologyTrainingPair(
                    source_graph=source.copy(),
                    target_graph=target.copy(),
                    base_generator=generator_id,
                    source_index=int(source_index),
                    target_index=int(target_index),
                    split=str(split),
                    matching_method="hungarian_degree_profile",
                    matching_cost=float(semantic_cost),
                    source_graph_path=str(graph_path),
                    source_manifest_path=(
                        str(manifest_path) if manifest_path is not None else None
                    ),
                )
            )
        paired = len(retained_costs)
        unmatched_source += len(source_rows) - paired
        unmatched_target += len(target_rows) - paired
        stratum_rows.append(
            {
                "num_nodes": n,
                "num_sources": len(source_rows),
                "num_targets": len(target_rows),
                "num_pairs": paired,
                "rejected_by_max_cost": rejected_costs,
                "mean_degree_cost": (
                    float(np.mean(retained_costs)) if retained_costs else None
                ),
                "max_degree_cost": (
                    float(np.max(retained_costs)) if retained_costs else None
                ),
            }
        )

    pairs.sort(key=lambda pair: (pair.source_index, pair.target_index))
    costs = [pair.matching_cost for pair in pairs]
    report = {
        "base_generator": generator_id,
        "split": split,
        "method": "hungarian_degree_profile",
        "strata": ["num_nodes"],
        "target_features_used_for_matching": ["sorted_degree_profile"],
        "target_features_excluded_from_matching": [
            "clustering_histogram",
            "orbit_count",
            "graphlet_histogram",
        ],
        "one_to_one": True,
        "num_sources": len(sources),
        "num_targets": len(targets),
        "num_pairs": len(pairs),
        "source_retention": float(len(pairs) / len(sources)) if sources else 0.0,
        "target_retention": float(len(pairs) / len(targets)) if targets else 0.0,
        "unmatched_sources": int(unmatched_source),
        "unmatched_targets": int(unmatched_target),
        "mean_degree_cost": float(np.mean(costs)) if costs else None,
        "median_degree_cost": float(np.median(costs)) if costs else None,
        "max_degree_cost": float(np.max(costs)) if costs else None,
        "strata_report": stratum_rows,
    }
    return pairs, report


def build_completed_base_training_pairs(
    train_targets: Sequence[nx.Graph],
    val_targets: Sequence[nx.Graph],
    *,
    config: dict[str, Any],
    seed: int,
) -> tuple[list[TopologyTrainingPair], list[TopologyTrainingPair], dict[str, Any]]:
    """Resolve all declared base pools and construct auditable train/val pairs."""

    cfg = _require_mapping(config, name="training_sources")
    mode = str(cfg.get("mode", "completed_base_outputs")).lower()
    if mode != "completed_base_outputs":
        raise ValueError(
            "build_completed_base_training_pairs requires mode: "
            "completed_base_outputs."
        )
    declarations = cfg.get("generators")
    if not isinstance(declarations, list) or not declarations:
        raise ValueError(
            "training_sources.generators must declare at least one base generator."
        )
    validation_fraction = float(cfg.get("validation_fraction", 0.1))
    partition_seed = int(cfg.get("partition_seed", seed))
    matching_cfg = _require_mapping(
        cfg.get("matching", {}), name="training_sources.matching"
    )
    method = str(matching_cfg.get("method", "hungarian_degree_profile")).lower()
    if method not in {"hungarian_degree", "hungarian_degree_profile"}:
        raise ValueError(
            "training_sources.matching.method must be hungarian_degree_profile."
        )
    disconnected_policy = str(
        matching_cfg.get("disconnected_policy", "drop")
    ).lower()
    require_exact_node_count = bool(
        matching_cfg.get("require_exact_node_count", True)
    )
    max_degree_cost_raw = matching_cfg.get("max_degree_cost")
    max_degree_cost = (
        None if max_degree_cost_raw is None else float(max_degree_cost_raw)
    )
    if max_degree_cost is not None and (
        not np.isfinite(max_degree_cost) or max_degree_cost < 0.0
    ):
        raise ValueError("matching.max_degree_cost must be finite and nonnegative.")

    train_pairs: list[TopologyTrainingPair] = []
    val_pairs: list[TopologyTrainingPair] = []
    pool_reports: list[dict[str, Any]] = []
    matching_reports: list[dict[str, Any]] = []
    generator_ids: set[str] = set()
    for raw_declaration in declarations:
        declaration = _require_mapping(
            raw_declaration, name="training_sources.generators[]"
        )
        for path_key in (
            "manifest_path",
            "training_estimates_manifest",
            "generated_path",
            "graph_path",
        ):
            if path_key in declaration and isinstance(declaration[path_key], str):
                declaration[path_key] = declaration[path_key].format(seed=int(seed))
        pool, pool_report = load_completed_base_pool(
            declaration,
            disconnected_policy=disconnected_policy,
        )
        if pool.generator_id in generator_ids:
            raise ValueError(
                f"Duplicate training source generator id: {pool.generator_id!r}."
            )
        generator_ids.add(pool.generator_id)
        partitions = _partition_pool(
            pool,
            validation_fraction=validation_fraction,
            partition_seed=partition_seed,
        )
        pool_report["partition"] = {
            "seed": partition_seed,
            "validation_fraction": validation_fraction,
            "train_count": len(partitions["train"]),
            "val_count": len(partitions["val"]),
        }
        pool_reports.append(pool_report)

        generator_train_pairs, train_report = _match_one_pool(
            partitions["train"],
            train_targets,
            generator_id=pool.generator_id,
            split="train",
            graph_path=pool.graph_path,
            manifest_path=pool.manifest_path,
            require_exact_node_count=require_exact_node_count,
            max_degree_cost=max_degree_cost,
        )
        generator_val_pairs, val_report = _match_one_pool(
            partitions["val"],
            val_targets,
            generator_id=pool.generator_id,
            split="val",
            graph_path=pool.graph_path,
            manifest_path=pool.manifest_path,
            require_exact_node_count=require_exact_node_count,
            max_degree_cost=max_degree_cost,
        )
        if not generator_train_pairs:
            raise ValueError(
                f"Declared base generator {pool.generator_id!r} produced no train "
                "pairs after exact-size matching."
            )
        if not generator_val_pairs:
            raise ValueError(
                f"Declared base generator {pool.generator_id!r} produced no "
                "validation pairs. Increase its completed pool, lower the "
                "validation fraction, or ensure matching graph sizes exist."
            )
        train_pairs.extend(generator_train_pairs)
        val_pairs.extend(generator_val_pairs)
        matching_reports.extend([train_report, val_report])

    # Keep generator blocks balanced and reproducible; each declaration supplies
    # its own one-to-one coupling, so no generator can consume another's targets.
    train_pairs.sort(
        key=lambda pair: (pair.base_generator, pair.source_index, pair.target_index)
    )
    val_pairs.sort(
        key=lambda pair: (pair.base_generator, pair.source_index, pair.target_index)
    )
    report = {
        "format": "topology_completed_base_pairing_v1",
        "mode": mode,
        "declared_base_generators": sorted(generator_ids),
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "validation_fraction": validation_fraction,
        "partition_seed": partition_seed,
        "pool_reports": pool_reports,
        "matching_reports": matching_reports,
        "coupling_scope": "one_to_one_within_generator_and_split",
        "target_reuse_across_generators": len(generator_ids) > 1,
    }
    return train_pairs, val_pairs, report
