#!/usr/bin/env python
"""Create the paper-facing graph-generation metrics report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import networkx as nx

from grapher.data.io import load_dataset_splits, validate_prepared_dataset_contract
from grapher.rewiring_mlp.evaluation.metrics import (
    descriptor_matrix,
    mmd_gaussian_emd,
    mmd_graphlet_statistics,
    mmd_orbit,
    mmd_orbit_graphrnn,
    wl_novelty_rate,
)
from grapher.rewiring_mlp.evaluation.degree_sequences import evaluate_degree_sequence_sets
from grapher.rewiring_mlp.molecular.graph_io import nx_to_rdkit_mol, require_rdkit
from grapher.properties.summary import (
    clustering_histogram,
    configure_orca_executable,
    degree_histogram,
    spectral_histogram,
)
from grapher.utils.io import ensure_dir, load_pickle, load_yaml, save_json

REPORT_METRICS = ("degree_mmd", "clustering_mmd", "orbit_mmd")
GENERIC_REPORT_METRICS = (
    *REPORT_METRICS, "spectral_mmd", "graphlet_history_mmd", "novelty_rate",
)


def _load_graph_list(path: Path) -> list[nx.Graph]:
    value = load_pickle(path)
    if isinstance(value, dict):
        for key in (
            "graphs",
            "generated_graphs",
            "topology_refined_graphs",
            "hybrid_refined_graphs",
        ):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{path} does not contain a graph list.")
    graphs = list(value)
    if not all(isinstance(graph, nx.Graph) for graph in graphs):
        raise TypeError(f"{path} contains a non-NetworkX graph.")
    if not graphs:
        raise ValueError(f"{path} contains no graphs.")
    return graphs


def _load_degree_sequence_list(path: Path) -> list[list[int]]:
    value = load_pickle(path)
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{path} does not contain a degree-sequence list.")
    sequences: list[list[int]] = []
    for index, sequence in enumerate(value):
        if not isinstance(sequence, (list, tuple)):
            raise TypeError(f"Degree sequence {index} in {path} is not a list/tuple.")
        row = [int(v) for v in sequence]
        if any(v < 0 for v in row):
            raise ValueError(f"Degree sequence {index} in {path} contains a negative degree.")
        sequences.append(sorted(row, reverse=True))
    if not sequences:
        raise ValueError(f"{path} contains no degree sequences.")
    return sequences


def _graph_degree_sequences(graphs: Sequence[nx.Graph]) -> list[list[int]]:
    return [
        sorted([int(degree) for _, degree in graph.degree()], reverse=True)
        for graph in graphs
    ]


def _resolve_sampled_degree_sequences(
    generated_dir: Path,
    generation_manifest: dict[str, Any] | None,
) -> Path | None:
    record = (
        generation_manifest.get("sampled_degree_sequences")
        if isinstance(generation_manifest, dict)
        else None
    )
    if isinstance(record, dict) and record.get("path"):
        candidate = generated_dir / str(record["path"])
        if candidate.is_file():
            expected = record.get("sha256")
            if expected is not None and _sha256(candidate) != str(expected):
                raise RuntimeError(
                    f"Sampled degree-sequence artifact hash mismatch: {candidate}"
                )
            return candidate
    fallback = generated_dir / "sampled_degree_sequences.pkl"
    return fallback if fallback.is_file() else None


def _print_degree_sequence_prior(metrics: Mapping[str, Any], *, reference_split: str) -> None:
    print("\nAuxiliary generated degree-sequence MMD (lower is better)")
    print(f"{'Comparison':38s}{'Degree MMD':>14s}")
    print(
        f"{f'dhvae_degree_sequences_to_{reference_split}':38s}"
        f"{float(metrics['degree_histogram_mmd_graphrnn']):14.6f}"
    )


def _write_degree_sequence_csv(
    metrics: Mapping[str, Any], path: Path, *, reference_split: str
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("comparison", "degree_mmd", "num_candidate_sequences", "num_reference_sequences")
        )
        writer.writeheader()
        writer.writerow({
            "comparison": f"dhvae_degree_sequences_to_{reference_split}",
            "degree_mmd": float(metrics["degree_histogram_mmd_graphrnn"]),
            "num_candidate_sequences": int(metrics["num_candidate_sequences"]),
            "num_reference_sequences": int(metrics["num_reference_sequences"]),
        })


def _load_json_mapping(path: Path) -> dict[str, Any] | None:
    """Load an optional JSON object while rejecting malformed provenance."""

    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"JSON manifest {path} must contain an object.")
    return value



def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one on-disk artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_split_fingerprint(split_sha256: dict[str, str]) -> str:
    """Match the baseline-wrapper dataset fingerprint convention."""

    record = {
        split: {"path": f"{split}.pkl", "sha256": split_sha256[split]}
        for split in sorted(split_sha256)
    }
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataset_provenance(
    dataset_config: dict[str, Any],
    splits: dict[str, Sequence[nx.Graph]],
) -> dict[str, Any]:
    """Describe the exact prepared dataset used by this evaluation.

    Evaluation must be reproducible across machines.  The split pickle hashes
    are therefore treated as the primary dataset identity, while the protocol
    ID and config hashes provide human-readable provenance.
    """

    serialized_id = str(dataset_config.get("name", "sbm"))
    benchmark_id = str(dataset_config.get("benchmark", serialized_id))
    root = Path(dataset_config.get("root", "outputs/datasets"))
    directory = root / serialized_id
    split_paths = {
        split: directory / f"{split}.pkl" for split in ("train", "val", "test")
    }
    missing = [str(path) for path in split_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot fingerprint the evaluation dataset; missing prepared split "
            f"files: {missing}. Prepare the dataset before evaluation."
        )
    split_sha256 = {split: _sha256(path) for split, path in split_paths.items()}

    requested_config_path = dataset_config.get("config_path")
    requested_config = None
    requested_config_sha256 = None
    if requested_config_path is not None:
        requested_path = Path(str(requested_config_path))
        if requested_path.is_file():
            requested_config = load_yaml(requested_path)
            requested_config_sha256 = _sha256(requested_path)

    resolved_path = directory / "resolved_dataset_config.yaml"
    resolved_config = load_yaml(resolved_path) if resolved_path.is_file() else None
    resolved_config_sha256 = _sha256(resolved_path) if resolved_path.is_file() else None
    protocol_id = None
    for candidate in (resolved_config, requested_config, dataset_config):
        if isinstance(candidate, dict) and candidate.get("protocol_id"):
            protocol_id = str(candidate["protocol_id"])
            break

    return {
        "benchmark_id": benchmark_id,
        "serialized_id": serialized_id,
        "root": str(root),
        "dataset_dir": str(directory),
        "protocol_id": protocol_id,
        "split_sizes": {
            split: len(list(splits.get(split, [])))
            for split in ("train", "val", "test")
        },
        "split_paths": {split: str(path) for split, path in split_paths.items()},
        "split_sha256": split_sha256,
        "fingerprint": _combined_split_fingerprint(split_sha256),
        "requested_config_path": (
            str(requested_config_path) if requested_config_path is not None else None
        ),
        "requested_config_sha256": requested_config_sha256,
        "resolved_config_path": str(resolved_path) if resolved_path.is_file() else None,
        "resolved_config_sha256": resolved_config_sha256,
    }


def _find_training_manifest(start: Path) -> tuple[dict[str, Any] | None, Path | None]:
    """Find the managed baseline training manifest associated with a generation."""

    start = start.expanduser().resolve(strict=False)
    candidates = [start if start.is_dir() else start.parent, *(start.parents)]
    seen: set[str] = set()
    for directory in candidates[:8]:
        candidate = directory / "train" / "manifest.json"
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        value = _load_json_mapping(candidate)
        if value is not None:
            return value, candidate
    return None, None


def _manifest_dataset_record(
    generation_manifest: dict[str, Any] | None,
    training_manifest: dict[str, Any] | None,
) -> dict[str, Any] | None:
    # Prefer the training manifest because it records the exact prepared-split
    # fingerprint; generation manifests may only carry the benchmark name.
    for manifest in (training_manifest, generation_manifest):
        if not isinstance(manifest, dict):
            continue
        dataset = manifest.get("dataset")
        if isinstance(dataset, dict):
            return dataset
    return None


def validate_dataset_compatibility(
    provenance: dict[str, Any],
    *,
    generation_manifest: dict[str, Any] | None,
    training_manifest: dict[str, Any] | None,
    mismatch_policy: str,
) -> list[str]:
    """Validate available generation/training provenance against current splits."""

    policy = str(mismatch_policy).strip().lower()
    if policy not in {"error", "warn", "ignore"}:
        raise ValueError(f"Unknown dataset mismatch policy: {mismatch_policy!r}")
    dataset_record = _manifest_dataset_record(generation_manifest, training_manifest)
    if dataset_record is None:
        return []

    mismatches: list[str] = []
    expected_benchmark = dataset_record.get("benchmark_id")
    if expected_benchmark is not None and str(expected_benchmark) != str(
        provenance["benchmark_id"]
    ):
        mismatches.append(
            "benchmark_id differs: generation/training="
            f"{expected_benchmark!r}, evaluation={provenance['benchmark_id']!r}"
        )

    expected_fingerprint = dataset_record.get("fingerprint")
    if expected_fingerprint is not None and str(expected_fingerprint) != str(
        provenance["fingerprint"]
    ):
        mismatches.append(
            "dataset split fingerprint differs: generation/training="
            f"{expected_fingerprint}, evaluation={provenance['fingerprint']}"
        )

    expected_split_hashes = dataset_record.get("split_sha256")
    if isinstance(expected_split_hashes, dict):
        for split in ("train", "val", "test"):
            expected = expected_split_hashes.get(split)
            observed = provenance["split_sha256"].get(split)
            if expected is not None and str(expected) != str(observed):
                mismatches.append(
                    f"{split} split SHA-256 differs: generation/training="
                    f"{expected}, evaluation={observed}"
                )

    if mismatches and policy == "error":
        detail = "\n  - ".join(mismatches)
        raise RuntimeError(
            "Generated graphs were trained/prepared against a different dataset "
            "than the one selected for evaluation:\n  - "
            + detail
            + "\nUse --dataset-mismatch-policy warn only for an intentional "
            "cross-dataset diagnostic."
        )
    if mismatches and policy == "warn":
        print("WARNING: dataset provenance mismatch detected:", flush=True)
        for mismatch in mismatches:
            print(f"  - {mismatch}", flush=True)
    return mismatches


def resolve_evaluation_counts(
    *,
    num_reference: int,
    num_generated: int,
    configured_reference_cap: int | None,
    max_generated_graphs: int | None,
) -> tuple[int, int]:
    """Keep the held-out reference fixed while optionally capping candidates."""

    reference_count = int(num_reference)
    if configured_reference_cap is not None and int(configured_reference_cap) > 0:
        reference_count = min(reference_count, int(configured_reference_cap))
    generated_count = int(num_generated)
    if max_generated_graphs is not None:
        cap = int(max_generated_graphs)
        if cap <= 0:
            raise ValueError("--max-graphs must be positive when provided.")
        generated_count = min(generated_count, cap)
    return reference_count, generated_count


def _print_dataset_provenance(provenance: dict[str, Any]) -> None:
    sizes = provenance["split_sizes"]
    print(
        "Dataset: "
        f"benchmark={provenance['benchmark_id']} "
        f"serialized={provenance['serialized_id']} "
        f"protocol={provenance['protocol_id'] or 'unspecified'}",
        flush=True,
    )
    print(
        "Dataset splits: "
        f"train={sizes['train']} val={sizes['val']} test={sizes['test']}",
        flush=True,
    )
    print(f"Dataset fingerprint: {provenance['fingerprint']}", flush=True)
    for split in ("train", "val", "test"):
        print(
            f"  {split} SHA256: {provenance['split_sha256'][split]}",
            flush=True,
        )


def _normalize_stage_label(value: str, *, field: str) -> str:
    """Convert a human/model label into a stable comparison-table token."""

    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    normalized = (
        normalized.strip("_.-")
        .lower()
        .replace("-", "_")
        .replace(".", "_")
    )
    if not normalized:
        raise ValueError(f"{field} must contain at least one letter or digit.")
    return normalized


def _baseline_model_id(
    graph_path: Path,
    manifest: dict[str, Any] | None = None,
) -> str | None:
    """Infer a managed baseline model from its manifest or canonical path."""

    if manifest is None:
        manifest = _load_json_mapping(graph_path.parent / "manifest.json")
    if isinstance(manifest, dict) and manifest.get("model_id"):
        return _normalize_stage_label(
            str(manifest["model_id"]),
            field="generation manifest model_id",
        )

    parts = graph_path.expanduser().resolve(strict=False).parts
    lowered = [part.lower() for part in parts]
    if "baselines" in lowered:
        index = lowered.index("baselines")
        if index + 1 < len(parts):
            return _normalize_stage_label(parts[index + 1], field="baseline model")
    return None


def resolve_generated_graph_path(
    generated_dir: Path,
    *,
    generated_graphs: str | Path | None = None,
) -> tuple[Path, dict[str, Any] | None, Path | None]:
    """Resolve GraphER final outputs or a managed raw-baseline generation.

    Managed baseline wrappers publish ``base_graphs.pkl`` and ``manifest.json``
    in one generation directory.  GraphER correction runs instead publish
    ``topology_refined_graphs.pkl`` (or the historical hybrid alias).  This
    resolver supports both layouts without requiring callers to copy or rename
    the baseline batch.
    """

    generated_dir = Path(generated_dir)
    directory_manifest_path = generated_dir / "manifest.json"
    directory_manifest = _load_json_mapping(directory_manifest_path)

    if generated_graphs is not None:
        path = Path(generated_graphs)
        if not path.is_file():
            raise FileNotFoundError(f"Generated graph file does not exist: {path}")
        sibling_manifest_path = path.parent / "manifest.json"
        sibling_manifest = _load_json_mapping(sibling_manifest_path)
        if sibling_manifest is not None:
            return path, sibling_manifest, sibling_manifest_path
        if directory_manifest is not None:
            return path, directory_manifest, directory_manifest_path
        return path, None, None

    candidates: list[Path] = [
        generated_dir / "topology_refined_graphs.pkl",
        generated_dir / "hybrid_refined_graphs.pkl",
    ]
    if directory_manifest is not None:
        base_entry = directory_manifest.get("base_graphs")
        if isinstance(base_entry, dict) and base_entry.get("path"):
            manifest_graph_path = Path(str(base_entry["path"]))
            if not manifest_graph_path.is_absolute():
                manifest_graph_path = generated_dir / manifest_graph_path
            candidates.append(manifest_graph_path)
    candidates.append(generated_dir / "base_graphs.pkl")

    deduplicated: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.expanduser().resolve(strict=False))
        if key not in seen:
            deduplicated.append(candidate)
            seen.add(key)

    for candidate in deduplicated:
        if not candidate.is_file():
            continue
        sibling_manifest_path = candidate.parent / "manifest.json"
        sibling_manifest = _load_json_mapping(sibling_manifest_path)
        if sibling_manifest is not None:
            return candidate, sibling_manifest, sibling_manifest_path
        if directory_manifest is not None:
            return candidate, directory_manifest, directory_manifest_path
        return candidate, None, None

    attempted = "\n  - ".join(str(path) for path in deduplicated)
    raise FileNotFoundError(
        "Could not locate generated graphs. Checked:\n  - "
        + attempted
        + "\nPass --generated-graphs for a non-standard artifact path."
    )


def _semantic_graph_stage(
    graph_path: Path,
    generation_manifest: dict[str, Any] | None = None,
) -> str | None:
    """Return a semantic stage label for managed spectral-diversification artifacts."""

    name = graph_path.name
    sampling = (generation_manifest or {}).get("sampling", {})
    stochastic = (
        isinstance(sampling, dict)
        and sampling.get("stochastic_diversification") is not None
    )
    model_id = _baseline_model_id(graph_path, generation_manifest) or "generated"

    realization = (
        isinstance(sampling, dict)
        and isinstance(sampling.get("degree_constrained_realization"), dict)
        and bool(sampling["degree_constrained_realization"].get("enabled", False))
    )
    summary_selected_realization = bool(
        realization
        and (sampling.get("degree_constrained_realization", {}) or {}).get(
            "candidate_selection_enabled", False
        )
    )

    if name == "pre_realization_graphs.pkl":
        return f"{model_id}_threshold_control"
    if name == "pre_refinement_graphs.pkl":
        if realization:
            return f"{model_id}_degree_constrained_pre_rewire"
        return f"{model_id}_spectral_pre_rewire"
    if name == "base_graphs.pkl" and stochastic and summary_selected_realization:
        return f"{model_id}_degree_constrained_summary_selected_rewired"
    if name == "base_graphs.pkl" and stochastic and realization:
        return f"{model_id}_degree_constrained_rewired"
    if name == "base_graphs.pkl" and stochastic:
        return f"{model_id}_diversified"
    if name == "base_graphs.pkl" and summary_selected_realization:
        return f"{model_id}_degree_constrained_summary_selected_realization"
    if name == "base_graphs.pkl" and realization:
        return f"{model_id}_degree_constrained_realization"
    return None


def resolve_generated_stage(
    config: dict[str, Any],
    generated_path: Path,
    *,
    generation_manifest: dict[str, Any] | None = None,
    explicit_stage: str | None = None,
) -> str:
    """Choose the report row label for raw baselines or corrected outputs."""

    if explicit_stage is not None:
        return _normalize_stage_label(explicit_stage, field="--generated-stage")

    semantic_stage = _semantic_graph_stage(generated_path, generation_manifest)
    if semantic_stage is not None:
        return semantic_stage

    model_id = _baseline_model_id(generated_path, generation_manifest)
    if model_id is not None:
        return model_id

    pipeline_stage = str(
        (config.get("pipeline", {}) or {}).get("stage", "endpoint")
    ).lower()
    topology_stages = {"topology", "posthoc_correction", "topology_correction"}
    return "topology_final" if pipeline_stage in topology_stages else "hybrid_final"


def _same_artifact(left: Path | None, right: Path | None) -> bool:
    if left is None or right is None:
        return False
    return left.expanduser().resolve(strict=False) == right.expanduser().resolve(
        strict=False
    )


def resolve_base_graph_path(
    generated_dir: Path,
    *,
    generated_path: Path | None = None,
    generation_manifest: dict[str, Any] | None = None,
    base_graphs: str | Path | None = None,
    coarse_graphs: str | Path | None = None,
) -> tuple[Path | None, str | None]:
    """Resolve a semantically meaningful source stage for paired evaluation.

    For stochastic structural diversification, ``pre_refinement_graphs.pkl``
    is the spectral source.  For degree-constrained spectral realization,
    ``pre_realization_graphs.pkl`` is the deterministic threshold control.
    Prefer these paired sources automatically so report labels cannot invert
    the source and final populations merely because the endpoint is named
    ``base_graphs.pkl``.
    """

    if base_graphs is not None and coarse_graphs is not None:
        raise ValueError("Use --base-graphs or --coarse-graphs, not both.")
    if base_graphs is not None:
        path = Path(base_graphs)
        semantic = _semantic_graph_stage(path, generation_manifest)
        if semantic is not None:
            return path, semantic
        model_id = _baseline_model_id(path, generation_manifest)
        return path, f"{model_id}_base" if model_id else "base"
    if coarse_graphs is not None:
        return Path(coarse_graphs), "hh_source"

    candidates: list[tuple[Path, str | None]] = []
    pre_refinement = generated_dir / "pre_refinement_graphs.pkl"
    if pre_refinement.is_file():
        candidates.append(
            (pre_refinement, _semantic_graph_stage(pre_refinement, generation_manifest))
        )
    pre_realization = generated_dir / "pre_realization_graphs.pkl"
    if pre_realization.is_file():
        candidates.append(
            (pre_realization, _semantic_graph_stage(pre_realization, generation_manifest))
        )
    candidates.extend((
        (generated_dir / "defog_base_graphs.pkl", "defog_base"),
        (generated_dir / "base_graphs.pkl", None),
        (generated_dir / "coarse_graphs.pkl", "hh_source"),
    ))

    for path, declared_stage in candidates:
        if not path.is_file():
            continue
        if generated_path is not None and _same_artifact(path, generated_path):
            continue
        if declared_stage is not None:
            return path, declared_stage
        semantic = _semantic_graph_stage(path, generation_manifest)
        if semantic is not None:
            return path, semantic
        model_id = _baseline_model_id(path, generation_manifest)
        return path, f"{model_id}_base" if model_id else "base"
    return None, None


def resolve_additional_managed_stage_paths(
    generated_dir: Path,
    *,
    generated_path: Path | None = None,
    generation_manifest: dict[str, Any] | None = None,
    primary_base_path: Path | None = None,
) -> list[tuple[Path, str]]:
    """Return extra managed intermediate stages that should be reported.

    Most pipelines have only one useful paired source stage, which is returned
    by :func:`resolve_base_graph_path`.  The combined degree-constrained +
    rewiring pipeline has two meaningful controls:

    ``pre_realization_graphs.pkl``
        deterministic threshold decoding before stochastic exact-degree
        realization;
    ``pre_refinement_graphs.pkl``
        stochastic exact-degree realization before rewiring.

    When the latter is the primary paired source, this helper exposes the
    threshold control as an additional row instead of silently dropping it.
    """

    rows: list[tuple[Path, str]] = []
    for name in ("pre_realization_graphs.pkl", "pre_refinement_graphs.pkl"):
        path = generated_dir / name
        if not path.is_file():
            continue
        if generated_path is not None and _same_artifact(path, generated_path):
            continue
        if primary_base_path is not None and _same_artifact(path, primary_base_path):
            continue
        stage = _semantic_graph_stage(path, generation_manifest)
        if stage is None:
            continue
        rows.append((path, stage))
    return rows



def _topology_invariant_key(graph: nx.Graph) -> tuple[Any, ...]:
    """Return cheap exact-isomorphism invariants for topology-only bucketing.

    Equal keys are necessary, not sufficient, for isomorphism.  The key is
    intentionally attribute-free because the generic-graph novelty protocol
    measures topology only.  Exact ``nx.is_isomorphic`` checks are still
    performed within each bucket, so WL collisions cannot create false
    duplicate matches.
    """

    num_nodes = int(graph.number_of_nodes())
    num_edges = int(graph.number_of_edges())
    directed = bool(graph.is_directed())
    multigraph = bool(graph.is_multigraph())
    if directed:
        degree_signature = tuple(sorted(
            (int(graph.in_degree(node)), int(graph.out_degree(node)))
            for node in graph.nodes()
        ))
        component_sizes = tuple(sorted(
            (len(component) for component in nx.weakly_connected_components(graph)),
            reverse=True,
        ))
        triangle_count = -1
    else:
        degree_signature = tuple(sorted(
            (int(degree) for _, degree in graph.degree()), reverse=True
        ))
        component_sizes = tuple(sorted(
            (len(component) for component in nx.connected_components(graph)),
            reverse=True,
        ))
        # Triangle count is an exact isomorphism invariant and cheaply separates
        # many regular WL-collision buckets before VF2 is invoked.
        triangle_count = int(sum(nx.triangles(nx.Graph(graph)).values()) // 3)

    wl_hash = nx.weisfeiler_lehman_graph_hash(graph)
    return (
        directed,
        multigraph,
        num_nodes,
        num_edges,
        degree_signature,
        component_sizes,
        triangle_count,
        wl_hash,
    )


class _ExactTopologyClassIndex:
    """Incremental exact-isomorphism class index with invariant bucketing."""

    def __init__(self) -> None:
        self.representatives: list[nx.Graph] = []
        self.keys: list[tuple[Any, ...]] = []
        self.buckets: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        self.num_exact_checks = 0

    def classify(self, graph: nx.Graph) -> int:
        key = _topology_invariant_key(graph)
        for class_id in self.buckets.get(key, []):
            representative = self.representatives[class_id]
            self.num_exact_checks += 1
            if nx.is_isomorphic(graph, representative):
                return int(class_id)
        class_id = len(self.representatives)
        self.representatives.append(graph.copy())
        self.keys.append(key)
        self.buckets[key].append(class_id)
        return class_id


def _frequency_tv(
    left: Counter[int],
    right: Counter[int],
) -> float:
    """Total-variation distance between two empirical class distributions."""

    left_total = int(sum(left.values()))
    right_total = int(sum(right.values()))
    if left_total <= 0 or right_total <= 0:
        return float("nan")
    class_ids = set(left) | set(right)
    return 0.5 * sum(
        abs(
            float(left.get(class_id, 0)) / left_total
            - float(right.get(class_id, 0)) / right_total
        )
        for class_id in class_ids
    )


def exact_isomorphism_support_diagnostics(
    populations: Mapping[str, Sequence[nx.Graph]],
    *,
    train_name: str = "train",
    reference_name: str,
) -> dict[str, Any]:
    """Measure exact topology novelty and isomorphism-class frequency matching.

    All supplied populations are partitioned jointly into exact topology
    isomorphism classes.  WL hashes and additional graph invariants are used
    only as safe pre-filters; class membership is confirmed with
    ``networkx.is_isomorphic``.  This makes the diagnostic strictly stronger
    than WL-hash novelty while remaining practical for the small generic-graph
    benchmarks used by this project.

    The returned per-population metrics include:

    * WL-hash novelty against the complete training split;
    * exact-isomorphism novelty against training;
    * exact-isomorphism uniqueness;
    * coverage of training isomorphism classes; and
    * total-variation/overlap of class frequencies versus training and the
      held-out reference population.
    """

    if train_name not in populations:
        raise KeyError(f"Missing training population {train_name!r}.")
    if reference_name not in populations:
        raise KeyError(f"Missing reference population {reference_name!r}.")

    index = _ExactTopologyClassIndex()
    assignments: dict[str, list[int]] = {}
    class_counts: dict[str, Counter[int]] = {}

    # Train first so class IDs 0..K_train-1 are stable training-support classes.
    ordered_names = [train_name, reference_name] + [
        name for name in populations
        if name not in {train_name, reference_name}
    ]
    for name in ordered_names:
        graphs = list(populations[name])
        ids = [index.classify(graph) for graph in graphs]
        assignments[name] = ids
        class_counts[name] = Counter(ids)

    train_counts = class_counts[train_name]
    reference_counts = class_counts[reference_name]
    train_classes = set(train_counts)
    num_train_classes = len(train_classes)
    train_graphs = list(populations[train_name])

    summary_rows: list[dict[str, Any]] = []
    for name in ordered_names:
        counts = class_counts[name]
        total = int(sum(counts.values()))
        classes = set(counts)
        novel_classes = classes - train_classes
        novel_graphs = int(sum(counts[class_id] for class_id in novel_classes))
        represented_train_classes = classes & train_classes
        tv_train = _frequency_tv(counts, train_counts)
        tv_reference = _frequency_tv(counts, reference_counts)
        wl_novelty = (
            wl_novelty_rate(populations[name], train_graphs)
            if train_graphs and total > 0
            else float("nan")
        )
        summary_rows.append({
            "population": name,
            "num_graphs": total,
            "num_exact_isomorphism_classes": len(classes),
            "exact_isomorphism_uniqueness_rate": (
                len(classes) / total if total > 0 else float("nan")
            ),
            "wl_novelty_rate_against_train": wl_novelty,
            "exact_isomorphism_novelty_rate_against_train": (
                novel_graphs / total if total > 0 else float("nan")
            ),
            "num_exact_novel_graphs_against_train": novel_graphs,
            "num_exact_novel_classes_against_train": len(novel_classes),
            "train_isomorphism_class_coverage_rate": (
                len(represented_train_classes) / num_train_classes
                if num_train_classes > 0 else float("nan")
            ),
            "class_frequency_tv_to_train": tv_train,
            "class_frequency_overlap_to_train": (
                1.0 - tv_train if tv_train == tv_train else float("nan")
            ),
            "class_frequency_tv_to_reference": tv_reference,
            "class_frequency_overlap_to_reference": (
                1.0 - tv_reference
                if tv_reference == tv_reference else float("nan")
            ),
        })

    class_rows: list[dict[str, Any]] = []
    for class_id, representative in enumerate(index.representatives):
        key = index.keys[class_id]
        row: dict[str, Any] = {
            "class_id": f"iso_{class_id:06d}",
            "is_training_class": class_id in train_classes,
            "num_nodes": int(representative.number_of_nodes()),
            "num_edges": int(representative.number_of_edges()),
            "degree_sequence": json.dumps(
                sorted(
                    [int(degree) for _, degree in representative.degree()],
                    reverse=True,
                ),
                separators=(",", ":"),
            ),
            "wl_hash": str(key[-1]),
        }
        for name in ordered_names:
            total = max(int(sum(class_counts[name].values())), 1)
            count = int(class_counts[name].get(class_id, 0))
            row[f"{name}_count"] = count
            row[f"{name}_rate"] = count / total
        class_rows.append(row)

    return {
        "protocol": {
            "topology_only": True,
            "node_and_edge_attributes": False,
            "wl_hash_role": "safe prefilter only; never accepted as proof of isomorphism",
            "exact_test": "networkx.is_isomorphic",
            "frequency_distance": "total variation over jointly constructed exact-isomorphism classes",
            "train_population": train_name,
            "reference_population": reference_name,
        },
        "num_joint_isomorphism_classes": len(index.representatives),
        "num_exact_isomorphism_checks": int(index.num_exact_checks),
        "population_metrics": summary_rows,
        "class_frequencies": class_rows,
        "assignments": assignments,
    }




def _stable_wl_digest(parts: Sequence[str]) -> str:
    """Return a deterministic short digest for a WL neighborhood signature."""

    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _normalized_counter(counter: Counter[str] | Counter[int]) -> dict[str, float]:
    """Normalize a counter to a probability mass function."""

    total = float(sum(counter.values()))
    if total <= 0.0:
        return {}
    return {str(key): float(value) / total for key, value in counter.items()}


def _tv_histogram(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Total variation between two normalized sparse histograms."""

    keys = set(left) | set(right)
    return 0.5 * sum(abs(float(left.get(k, 0.0)) - float(right.get(k, 0.0))) for k in keys)


def _component_size_histogram(graph: nx.Graph) -> dict[str, float]:
    """Node-mass distribution over connected-component sizes."""

    if graph.number_of_nodes() <= 0:
        return {}
    if graph.is_directed():
        components = nx.weakly_connected_components(graph)
    else:
        components = nx.connected_components(graph)
    counts: Counter[int] = Counter()
    for component in components:
        size = len(component)
        counts[size] += size
    return _normalized_counter(counts)


def _wl_layer_histograms(graph: nx.Graph, *, iterations: int) -> list[dict[str, float]]:
    """Compute comparable topology-only WL color histograms.

    Colors are initialized from degree information and updated with deterministic
    SHA-256 neighborhood signatures.  Because the same textual signature maps to
    the same digest across graphs and processes, the sparse histograms can be
    compared directly.  Attributes are intentionally ignored.
    """

    if iterations < 0:
        raise ValueError("WL iterations must be non-negative.")
    nodes = list(graph.nodes())
    if not nodes:
        return [{} for _ in range(iterations + 1)]

    if graph.is_directed():
        labels = {
            node: f"io:{int(graph.in_degree(node))},{int(graph.out_degree(node))}"
            for node in nodes
        }
    else:
        labels = {node: f"d:{int(graph.degree(node))}" for node in nodes}

    layers: list[dict[str, float]] = [_normalized_counter(Counter(labels.values()))]
    for _ in range(iterations):
        updated: dict[Any, str] = {}
        for node in nodes:
            if graph.is_directed():
                incoming: list[str] = []
                outgoing: list[str] = []
                for nbr in graph.predecessors(node):
                    multiplicity = (
                        int(graph.number_of_edges(nbr, node)) if graph.is_multigraph() else 1
                    )
                    incoming.extend([labels[nbr]] * multiplicity)
                for nbr in graph.successors(node):
                    multiplicity = (
                        int(graph.number_of_edges(node, nbr)) if graph.is_multigraph() else 1
                    )
                    outgoing.extend([labels[nbr]] * multiplicity)
                signature = [
                    "self", labels[node], "in", *sorted(incoming), "out", *sorted(outgoing)
                ]
            else:
                neighbors: list[str] = []
                for nbr in graph.neighbors(node):
                    multiplicity = (
                        int(graph.number_of_edges(node, nbr)) if graph.is_multigraph() else 1
                    )
                    neighbors.extend([labels[nbr]] * multiplicity)
                signature = ["self", labels[node], "nbr", *sorted(neighbors)]
            updated[node] = _stable_wl_digest(signature)
        labels = updated
        layers.append(_normalized_counter(Counter(labels.values())))
    return layers


def _graded_topology_profile(graph: nx.Graph, *, wl_iterations: int) -> dict[str, Any]:
    """Cache the invariant features used by the graded novelty distance."""

    return {
        "num_nodes": int(graph.number_of_nodes()),
        "num_edges": int(graph.number_of_edges()),
        "component_histogram": _component_size_histogram(graph),
        "wl_layers": _wl_layer_histograms(graph, iterations=wl_iterations),
    }


def graded_topology_distance(
    left: nx.Graph | Mapping[str, Any],
    right: nx.Graph | Mapping[str, Any],
    *,
    wl_iterations: int = 3,
) -> float:
    r"""Return a fast permutation-invariant graded structural pseudodistance.

    Let :math:`p_G^{(h)}` be the normalized WL color histogram at iteration
    :math:`h`, and let :math:`q_G` be the node-mass histogram of connected
    component sizes.  We use

    .. math::

       d(G,H) = \frac{\delta_V + \delta_E + \operatorname{TV}(q_G,q_H)
       + \sum_{h=0}^{L}\operatorname{TV}(p_G^{(h)},p_H^{(h)})}{L+4},

    where :math:`\delta_V=|n_G-n_H|/\max(n_G,n_H,1)` and
    :math:`\delta_E=|m_G-m_H|/\max(m_G,m_H,1)`.  The result lies in [0,1].

    This is deliberately a scalable *pseudodistance*, not exact graph-edit
    distance: exact isomorphism implies zero distance, but 1-WL-equivalent
    non-isomorphic graphs can still have zero or very small distance.  Exact
    novelty is therefore reported separately.
    """

    left_profile = (
        _graded_topology_profile(left, wl_iterations=wl_iterations)
        if isinstance(left, nx.Graph)
        else dict(left)
    )
    right_profile = (
        _graded_topology_profile(right, wl_iterations=wl_iterations)
        if isinstance(right, nx.Graph)
        else dict(right)
    )
    n_left = int(left_profile["num_nodes"])
    n_right = int(right_profile["num_nodes"])
    m_left = int(left_profile["num_edges"])
    m_right = int(right_profile["num_edges"])
    delta_nodes = abs(n_left - n_right) / max(n_left, n_right, 1)
    delta_edges = abs(m_left - m_right) / max(m_left, m_right, 1)
    component_tv = _tv_histogram(
        left_profile["component_histogram"], right_profile["component_histogram"]
    )
    left_layers = list(left_profile["wl_layers"])
    right_layers = list(right_profile["wl_layers"])
    if len(left_layers) != len(right_layers):
        raise ValueError("Graded topology profiles use different WL depths.")
    wl_tvs = [_tv_histogram(a, b) for a, b in zip(left_layers, right_layers)]
    terms = [delta_nodes, delta_edges, component_tv, *wl_tvs]
    return float(sum(terms) / len(terms)) if terms else 0.0


def _quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated empirical quantile without NumPy dependency."""

    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(max(float(q), 0.0), 1.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _nearest_profile_distances(
    queries: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    *,
    wl_iterations: int,
) -> list[float]:
    if not queries:
        return []
    if not targets:
        return [float("nan")] * len(queries)
    distances: list[float] = []
    for query in queries:
        distances.append(min(
            graded_topology_distance(query, target, wl_iterations=wl_iterations)
            for target in targets
        ))
    return distances


def graded_diversity_diagnostics(
    populations: Mapping[str, Sequence[nx.Graph]],
    *,
    train_name: str = "train",
    reference_name: str,
    epsilons: Sequence[float] = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5),
    wl_iterations: int = 3,
) -> dict[str, Any]:
    r"""Measure graded novelty and held-out coverage for generic graph sets.

    For each graph :math:`G`, define its nearest-training distance
    :math:`D_T(G)=\min_{H\in T} d(G,H)`.  Population-level graded novelty is
    summarized by the mean/median/quantiles of :math:`D_T`.  For threshold
    :math:`\epsilon`,

    .. math::

       N_\epsilon(P)=\frac{1}{|P|}\sum_{G\in P}1[D_T(G)>\epsilon]

    is epsilon-novelty.  Held-out coverage of population :math:`P` is

    .. math::

       C_\epsilon(P;R)=\frac{1}{|R|}\sum_{R_i\in R}
       1[\min_{G\in P}d(R_i,G)\le\epsilon].

    The held-out reference's own nearest-training distance distribution is a
    calibration target: a useful generator need not maximize distance from
    training, but should ideally resemble legitimate held-out departure while
    covering held-out structure.
    """

    if train_name not in populations:
        raise KeyError(f"Missing training population {train_name!r}.")
    if reference_name not in populations:
        raise KeyError(f"Missing reference population {reference_name!r}.")
    if wl_iterations < 0:
        raise ValueError("wl_iterations must be non-negative.")
    epsilon_values = sorted({float(value) for value in epsilons})
    if not epsilon_values:
        raise ValueError("At least one diversity epsilon is required.")
    if any((not math.isfinite(value)) or value < 0.0 or value > 1.0 for value in epsilon_values):
        raise ValueError("Diversity epsilons must be finite values in [0, 1].")

    profiles: dict[str, list[dict[str, Any]]] = {
        name: [_graded_topology_profile(graph, wl_iterations=wl_iterations) for graph in graphs]
        for name, graphs in populations.items()
    }
    train_profiles = profiles[train_name]
    reference_profiles = profiles[reference_name]

    population_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    nearest_train_by_population: dict[str, list[float]] = {}

    for name, query_profiles in profiles.items():
        nearest_train = _nearest_profile_distances(
            query_profiles, train_profiles, wl_iterations=wl_iterations
        )
        nearest_train_by_population[name] = nearest_train
        finite = [value for value in nearest_train if math.isfinite(value)]
        coverage_distances = _nearest_profile_distances(
            reference_profiles, query_profiles, wl_iterations=wl_iterations
        )
        finite_coverage = [value for value in coverage_distances if math.isfinite(value)]
        population_rows.append({
            "population": name,
            "num_graphs": len(query_profiles),
            "mean_nearest_train_structural_distance": (
                statistics.fmean(finite) if finite else float("nan")
            ),
            "median_nearest_train_structural_distance": _quantile(finite, 0.5),
            "q10_nearest_train_structural_distance": _quantile(finite, 0.1),
            "q90_nearest_train_structural_distance": _quantile(finite, 0.9),
            "mean_heldout_to_population_distance": (
                statistics.fmean(finite_coverage) if finite_coverage else float("nan")
            ),
        })
        for epsilon in epsilon_values:
            epsilon_novelty = (
                sum(value > epsilon for value in finite) / len(finite)
                if finite else float("nan")
            )
            heldout_coverage = (
                sum(value <= epsilon for value in finite_coverage) / len(finite_coverage)
                if finite_coverage else float("nan")
            )
            curve_rows.append({
                "population": name,
                "epsilon": epsilon,
                "epsilon_novelty_rate_against_train": epsilon_novelty,
                "heldout_coverage_rate": heldout_coverage,
            })

    return {
        "protocol": {
            "topology_only": True,
            "node_and_edge_attributes": False,
            "distance_name": "wl_component_structural_pseudodistance_v1",
            "wl_iterations": int(wl_iterations),
            "distance_range": [0.0, 1.0],
            "distance_terms": (
                "relative node-count difference; relative edge-count difference; "
                "component-size node-mass TV; normalized WL-color-histogram TV at "
                f"iterations 0..{wl_iterations}, averaged equally"
            ),
            "caveat": (
                "Permutation-invariant scalable pseudodistance, not exact graph-edit distance; "
                "non-isomorphic 1-WL-equivalent graphs may have distance zero."
            ),
            "train_population": train_name,
            "reference_population": reference_name,
            "epsilons": epsilon_values,
        },
        "population_metrics": population_rows,
        "epsilon_curve": curve_rows,
        "nearest_train_distances": nearest_train_by_population,
    }


def _merge_generic_diversity_rows(
    exact_rows: Sequence[Mapping[str, Any]],
    graded_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge exact-support and graded-distance summaries by population."""

    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for source in (exact_rows, graded_rows):
        for row in source:
            name = str(row["population"])
            if name not in merged:
                merged[name] = {"population": name}
                order.append(name)
            merged[name].update(dict(row))
    return [merged[name] for name in order]


def _print_generic_diversity_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    curve_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Print the four diversity quantities used by the paper-facing report."""

    print("\nGeneric topology diversity diagnostics (attributes ignored)")
    print(
        f"{'Population':30s}"
        f"{'Exact novel':>14s}"
        f"{'Iso unique':>13s}"
        f"{'Mean NN-train':>16s}"
        f"{'Median NN-train':>18s}"
    )
    for row in rows:
        print(
            f"{str(row['population']):30s}"
            f"{float(row.get('exact_isomorphism_novelty_rate_against_train', float('nan'))):14.6f}"
            f"{float(row.get('exact_isomorphism_uniqueness_rate', float('nan'))):13.6f}"
            f"{float(row.get('mean_nearest_train_structural_distance', float('nan'))):16.6f}"
            f"{float(row.get('median_nearest_train_structural_distance', float('nan'))):18.6f}"
        )
    print(
        "Exact novelty detects unseen isomorphism classes; iso uniqueness measures "
        "distinct classes within a population; NN-train is a graded [0,1] structural "
        "pseudodistance to the nearest training graph."
    )

    grouped: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in curve_rows:
        grouped[float(row["epsilon"])].append(row)
    print("\nEpsilon-novelty / held-out coverage curve")
    print(f"{'epsilon':>9s}{'Population':30s}{'eps novel':>13s}{'coverage':>13s}")
    for epsilon in sorted(grouped):
        for row in grouped[epsilon]:
            print(
                f"{epsilon:9.4f}{str(row['population']):30s}"
                f"{float(row['epsilon_novelty_rate_against_train']):13.6f}"
                f"{float(row['heldout_coverage_rate']):13.6f}"
            )
    print(
        "Epsilon novelty = fraction farther than epsilon from every training graph; "
        "coverage = fraction of held-out graphs within epsilon of at least one graph "
        "from the population."
    )


def _write_generic_diversity_csv(
    rows: Sequence[Mapping[str, Any]], path: Path,
) -> None:
    _write_isomorphism_support_csv(rows, path)

def _write_isomorphism_support_csv(
    rows: Sequence[Mapping[str, Any]], path: Path,
) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_isomorphism_support_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    print("\nExact topology-support diagnostics (attributes ignored)")
    print(
        f"{'Population':30s}"
        f"{'WL novel':>12s}"
        f"{'Exact novel':>14s}"
        f"{'Iso unique':>13s}"
        f"{'TV->train':>12s}"
        f"{'TV->heldout':>14s}"
    )
    for row in rows:
        print(
            f"{str(row['population']):30s}"
            f"{float(row['wl_novelty_rate_against_train']):12.6f}"
            f"{float(row['exact_isomorphism_novelty_rate_against_train']):14.6f}"
            f"{float(row['exact_isomorphism_uniqueness_rate']):13.6f}"
            f"{float(row['class_frequency_tv_to_train']):12.6f}"
            f"{float(row['class_frequency_tv_to_reference']):14.6f}"
        )
    print(
        "Exact novelty = fraction of graphs not isomorphic to any training graph; "
        "TV=0 means identical empirical isomorphism-class frequencies."
    )

def _paper_mmd(
    reference: Sequence[nx.Graph],
    candidate: Sequence[nx.Graph],
    *,
    compute_orbit: bool = True,
    metric_protocol: str = "graphrnn",
    clustering_bins: int = 100,
    include_generic_metrics: bool = True,
    graphlet_options: dict[str, Any] | None = None,
    train: Sequence[nx.Graph] | None = None,
    novelty_candidate: Sequence[nx.Graph] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, float]:
    """Evaluate generic graph statistics under a declared metric protocol.

    ``graphrnn`` reproduces the historical GraphRNN/SPECTRE convention used
    by GraphRNN, HOG-Diff (Community/Ego), and DiGress/DeFoG Comm20:

    * degree histogram + Gaussian EMD, sigma=1;
    * 100-bin clustering histogram + Gaussian EMD, sigma=0.1, distance
      scaling equal to the number of bins; and
    * mean per-node four-node orbit vector + Gaussian L2, sigma=30.

    ``graphes_adaptive`` keeps the pre-alignment GraphES behavior for
    backwards diagnostics only.

    Additional generic metrics use the existing evaluation conventions:
    20-bin normalized-Laplacian spectral histograms with adaptive Gaussian
    EMD, and graphlet-composition MMD (connected orders 3--5 by default).
    Novelty is the fraction of topology WL hashes in ``novelty_candidate``
    absent from the complete training split.  By default ``novelty_candidate``
    is the MMD candidate population.  This override is used for the
    ``train_to_<split>`` reference row: its MMD compares train against held-out,
    while its novelty must describe the held-out population against train.
    Novelty is unavailable when training graphs are absent.
    """

    protocol = str(metric_protocol).strip().lower().replace("-", "_")
    graphrnn = protocol in {
        "graphrnn", "graphrnn_legacy", "spectre", "spectre_legacy",
    }
    if not graphrnn and protocol not in {"graphes_adaptive", "adaptive"}:
        raise ValueError(
            f"Unknown generic MMD protocol {metric_protocol!r}; expected "
            "'graphrnn' or 'graphes_adaptive'."
        )
    bins = int(clustering_bins) if graphrnn else 20
    if bins <= 0:
        raise ValueError("clustering_bins must be positive.")

    def announce(phase: str, completed: int) -> None:
        if progress is not None:
            progress(phase, completed, 1)

    metrics = {}
    if include_generic_metrics:
        announce("novelty", 0)
        novelty_population = (
            candidate if novelty_candidate is None else novelty_candidate
        )
        metrics["novelty_rate"] = (
            wl_novelty_rate(novelty_population, train) if train else float("nan")
        )
        announce("novelty", 1)
        announce("spectral MMD", 0)
        spectral_reference = descriptor_matrix(reference, spectral_histogram)
        spectral_candidate = descriptor_matrix(candidate, spectral_histogram)
        metrics["spectral_mmd"] = mmd_gaussian_emd(
            spectral_reference, spectral_candidate,
        )
        announce("spectral MMD", 1)
        graphlet_mmd, _ = mmd_graphlet_statistics(
            reference, candidate, progress=progress, **(graphlet_options or {})
        )
        metrics["graphlet_history_mmd"] = graphlet_mmd
    announce("degree MMD", 0)
    max_degree = max(
        (
            max((int(degree) for _, degree in graph.degree()), default=0)
            for graph in [*reference, *candidate]
        ),
        default=0,
    )
    degree_reference = descriptor_matrix(
        reference,
        lambda graph: degree_histogram(graph, max_degree),
    )
    degree_candidate = descriptor_matrix(
        candidate,
        lambda graph: degree_histogram(graph, max_degree),
    )
    metrics["degree_mmd"] = mmd_gaussian_emd(
        degree_reference, degree_candidate, sigma=1.0 if graphrnn else None,
    )
    announce("degree MMD", 1)
    announce("clustering MMD", 0)
    clustering_reference = descriptor_matrix(
        reference, lambda graph: clustering_histogram(graph, bins),
    )
    clustering_candidate = descriptor_matrix(
        candidate, lambda graph: clustering_histogram(graph, bins),
    )
    metrics["clustering_mmd"] = mmd_gaussian_emd(
        clustering_reference, clustering_candidate,
        sigma=0.1 if graphrnn else None,
        distance_scaling=float(bins) if graphrnn else 1.0,
    )
    announce("clustering MMD", 1)
    orbit_metric = mmd_orbit_graphrnn if graphrnn else mmd_orbit
    metrics["orbit_mmd"] = (
        orbit_metric(reference, candidate, progress=progress)
        if compute_orbit else float("nan")
    )
    return metrics


def _evaluate_comparison(
    reference: Sequence[nx.Graph],
    candidate: Sequence[nx.Graph],
    comparison: str,
    **options,
) -> dict[str, Any]:
    """Keep long descriptor extraction visible without logging every graph."""
    started = time.monotonic()
    last_phase = None
    last_output = started
    print(
        f"Evaluating {comparison}: reference={len(reference)} candidate={len(candidate)}",
        flush=True,
    )

    def progress(phase: str, completed: int, total: int) -> None:
        nonlocal last_phase, last_output
        now = time.monotonic()
        if phase != last_phase or completed == total or now - last_output >= 10.0:
            print(
                f"[{comparison}] {phase}: {completed}/{total}; elapsed={now - started:.1f}s",
                flush=True,
            )
            last_phase, last_output = phase, now

    metrics = _paper_mmd(reference, candidate, progress=progress, **options)
    print(f"Completed {comparison} in {time.monotonic() - started:.1f}s", flush=True)
    return {"comparison": comparison, **metrics}


def is_molecular_evaluation(
    dataset_config: dict[str, Any],
    graphs: Sequence[nx.Graph],
) -> bool:
    """Detect an attributed molecular dataset without relying on its name alone."""

    dataset_name = str(dataset_config.get("name", "")).lower()
    if any(token in dataset_name for token in ("qm9", "zinc", "molecule")):
        return True
    return any(
        "atomic_num" in data or "atom_type" in data
        for graph in graphs
        for _, data in graph.nodes(data=True)
    )


def _canonical_molecular_smiles(
    graph: nx.Graph,
) -> tuple[str | None, str | None]:
    """Return an RDKit-sanitized canonical SMILES and any conversion error."""

    if graph.number_of_nodes() == 0:
        return None, "EmptyMolecule"
    if any(
        "atomic_num" not in data and "atom_type" not in data
        for _, data in graph.nodes(data=True)
    ):
        return None, "MissingAtomType"
    if any(
        "bond_type" not in data and "bond_order" not in data
        for _, _, data in graph.edges(data=True)
    ):
        return None, "MissingBondType"
    Chem = require_rdkit()
    try:
        molecule = nx_to_rdkit_mol(graph, sanitize=True)
        smiles = str(
            Chem.MolToSmiles(
                molecule,
                canonical=True,
                isomericSmiles=False,
            )
        )
        if not smiles:
            return None, "EmptySMILES"
        return smiles, None
    except Exception as exc:
        return None, type(exc).__name__


def molecular_quality_metrics(
    generated_graphs: Sequence[nx.Graph],
    train_graphs: Sequence[nx.Graph],
) -> tuple[dict[str, Any], list[str], list[int], dict[str, int]]:
    """Compute RDKit validity, canonical-SMILES uniqueness, and novelty."""

    # Fail early with the existing installation guidance when RDKit is absent.
    require_rdkit()

    valid_smiles: list[str] = []
    invalid_indices: list[int] = []
    conversion_errors: Counter[str] = Counter()
    for index, graph in enumerate(generated_graphs):
        smiles, error = _canonical_molecular_smiles(graph)
        if smiles is None:
            invalid_indices.append(index)
            conversion_errors[str(error or "InvalidMolecule")] += 1
        else:
            valid_smiles.append(smiles)

    train_smiles = {
        smiles
        for graph in train_graphs
        if (smiles := _canonical_molecular_smiles(graph)[0]) is not None
    }
    unique_valid_smiles = sorted(set(valid_smiles))
    novel_smiles = [
        smiles for smiles in unique_valid_smiles if smiles not in train_smiles
    ]

    num_generated = len(generated_graphs)
    num_valid = len(valid_smiles)
    num_unique = len(unique_valid_smiles)
    novelty_rate: float | None
    if not train_smiles:
        novelty_rate = None
    elif num_unique == 0:
        novelty_rate = 0.0
    else:
        novelty_rate = len(novel_smiles) / num_unique

    metrics = {
        "num_generated_graphs": num_generated,
        "num_valid_generated_molecules": num_valid,
        "num_invalid_generated_molecules": num_generated - num_valid,
        "validity_without_correction": num_valid / max(num_generated, 1),
        "unique_valid_count": num_unique,
        "uniqueness_rate": num_unique / max(num_valid, 1),
        "num_valid_training_molecules": len(train_smiles),
        "novel_unique_valid_count": len(novel_smiles),
        "novelty_rate": novelty_rate,
    }
    return (
        metrics,
        valid_smiles,
        invalid_indices,
        dict(conversion_errors),
    )


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("comparison", *_report_metrics(rows)),
        )
        writer.writeheader()
        writer.writerows(rows)


def _report_metrics(rows: Sequence[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(
        metric for metric in GENERIC_REPORT_METRICS
        if metric in REPORT_METRICS or any(metric in row for row in rows)
    )


def _write_molecular_csv(
    rows: Sequence[dict[str, Any]],
    path: Path,
) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def resolve_reference_split(
    config: dict[str, Any], explicit_split: str | None = None,
) -> str:
    """CLI overrides evaluation.reference_split; legacy configs default to test.

    protocol.tune_on_split is intentionally not treated as a CLI instruction.
    """
    value = explicit_split
    if value is None:
        value = (config.get("evaluation", {}) or {}).get("reference_split", "test")
    split = str(value).strip().lower()
    if split not in {"val", "test"}:
        raise ValueError("evaluation.reference_split must be 'val' or 'test'.")
    return split


def validate_report_reference_split(output_dir: Path, reference_split: str) -> None:
    """Prevent validation and test reports from silently overwriting one another."""
    report = _load_json_mapping(output_dir / "graph_evaluation_report.json")
    if report is not None and report.get("reference_split", "test") != reference_split:
        raise ValueError(
            "This output directory already contains a report for a different "
            "reference split. Use separate --output-dir paths for validation "
            "and test (for example evaluation_val and evaluation_test)."
        )


def _print_table(rows: Sequence[dict[str, Any]], *, reference_split: str = "test") -> None:
    print(f"Graph-distribution MMD against held-out {reference_split} graphs (lower is better)")
    if any("novelty_rate" in row for row in rows):
        print("WL novelty against the complete training split (evaluated population; higher is better)")
        print("For train_to_* rows, novelty is held-out-vs-training; for model rows, novelty is generated-vs-training.")
    columns = {
        "degree_mmd": ("Degree MMD", 14),
        "clustering_mmd": ("Clustering MMD", 18),
        "orbit_mmd": ("Orbit MMD", 14),
        "spectral_mmd": ("Spectral MMD", 16),
        "graphlet_history_mmd": ("Graphlet MMD", 16),
        "novelty_rate": ("Novelty", 14),
    }
    metrics = _report_metrics(rows)
    print(f"{'Comparison':30s}" + "".join(
        f"{columns[key][0]:>{columns[key][1]}s}" for key in metrics
    ))
    for row in rows:
        print(f"{str(row['comparison']):30s}" + "".join(
            f"{float(row.get(key, float('nan'))):{columns[key][1]}.6f}"
            for key in metrics
        ))


def _print_molecular_metrics(rows: Sequence[dict[str, Any]]) -> None:
    print("\nMolecular quality by stage (RDKit; higher is better)")
    print(
        f"{'Stage':24s}"
        f"{'Validity':>12s}"
        f"{'Uniqueness':>14s}"
        f"{'Novelty':>12s}"
        f"{'Valid count':>16s}"
    )
    for row in rows:
        novelty = row["novelty_rate"]
        novelty_text = "n/a" if novelty is None else f"{float(novelty):.6f}"
        count = f"{row['num_valid_generated_molecules']}/{row['num_generated_graphs']}"
        print(
            f"{str(row['stage']):24s}"
            f"{float(row['validity_without_correction']):12.6f}"
            f"{float(row['uniqueness_rate']):14.6f}"
            f"{novelty_text:>12s}"
            f"{count:>16s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved generated graphs with degree and clustering MMD, "
            "spectral and graphlet MMD for generic graphs, and optional "
            "four-node ORCA orbit MMD. Save CSV and JSON metric reports."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--reference-split", choices=("val", "test"), default=None,
        help=("Reference graphs for all MMD rows. Overrides evaluation.reference_split; "
              "legacy configs default to test. Use val for weight selection."),
    )
    parser.add_argument(
        "--generic-mmd-protocol",
        choices=("config", "graphrnn", "graphes_adaptive"),
        default="config",
        help=(
            "Generic-graph MMD convention. 'config' reads "
            "evaluation.generic_mmd_protocol (default: graphrnn)."
        ),
    )
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument(
        "--generated-graphs",
        default=None,
        help=(
            "Optional explicit graph pickle. When omitted, the evaluator "
            "auto-detects GraphER refined outputs or a managed baseline "
            "generation's base_graphs.pkl."
        ),
    )
    parser.add_argument(
        "--generated-stage",
        default=None,
        help=(
            "Optional comparison-row stage label, for example 'defog_800k'. "
            "Managed baseline model IDs are inferred from manifest.json when "
            "this option is omitted."
        ),
    )
    parser.add_argument("--base-graphs", default=None)
    parser.add_argument("--coarse-graphs", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help=(
            "Cap generated/source candidate graphs only. The selected held-out "
            "reference remains fixed; use protocol.max_reference_graphs in "
            "the config to cap the reference set explicitly."
        ),
    )
    parser.add_argument(
        "--dataset-mismatch-policy",
        choices=("error", "warn", "ignore"),
        default="error",
        help=(
            "Behavior when managed generation/training provenance identifies "
            "different prepared dataset splits. Default: error."
        ),
    )
    parser.add_argument(
        "--build-dataset-if-missing",
        action="store_true",
        help=(
            "Opt in to rebuilding missing prepared dataset splits during "
            "evaluation. By default evaluation fails instead of silently "
            "creating a new reference dataset."
        ),
    )
    # Legacy figure options are accepted for command compatibility but ignored.
    parser.add_argument("--num-samples", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument(
        "--sample-selection",
        choices=("stratified", "random"),
        default="stratified",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--skip-exact-isomorphism-diagnostics",
        action="store_true",
        help=(
            "Skip topology-only exact-isomorphism novelty and isomorphism-class "
            "frequency diagnostics for generic graphs. They are enabled by "
            "default and use the complete training split."
        ),
    )
    parser.add_argument(
        "--skip-graded-diversity-diagnostics",
        action="store_true",
        help=(
            "Skip nearest-training structural distance and epsilon-novelty/coverage "
            "diagnostics for generic graphs. They are enabled by default."
        ),
    )
    parser.add_argument(
        "--diversity-wl-iterations",
        type=int,
        default=None,
        help=(
            "WL refinement depth for the graded structural pseudodistance. "
            "Overrides evaluation.diversity_wl_iterations; default: 3."
        ),
    )
    parser.add_argument(
        "--diversity-epsilons",
        type=float,
        nargs="+",
        default=None,
        metavar="EPS",
        help=(
            "Thresholds in [0,1] for epsilon-novelty and held-out coverage. "
            "Overrides evaluation.diversity_epsilons; default: "
            "0 0.01 0.025 0.05 0.1 0.2 0.3 0.5."
        ),
    )
    parser.add_argument("--sample-seed", type=int, default=42, help=argparse.SUPPRESS)
    parser.add_argument("--layout-seed", type=int, default=42, help=argparse.SUPPRESS)
    parser.add_argument("--columns", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--dpi", type=int, default=300, help=argparse.SUPPRESS)
    args = parser.parse_args()

    config = load_yaml(args.config)
    reference_split = resolve_reference_split(config, args.reference_split)
    generated_dir = Path(args.generated_dir)
    (
        generated_path,
        generation_manifest,
        generation_manifest_path,
    ) = resolve_generated_graph_path(
        generated_dir,
        generated_graphs=args.generated_graphs,
    )
    base_path, base_stage = resolve_base_graph_path(
        generated_dir,
        generated_path=generated_path,
        generation_manifest=generation_manifest,
        base_graphs=args.base_graphs,
        coarse_graphs=args.coarse_graphs,
    )
    # A managed raw baseline stores its evaluated batch as base_graphs.pkl.
    # Do not evaluate the same artifact twice as both the source and final row.
    if _same_artifact(base_path, generated_path):
        base_path, base_stage = None, None
    additional_managed_stage_paths = (
        []
        if args.base_graphs is not None or args.coarse_graphs is not None
        else resolve_additional_managed_stage_paths(
            generated_dir,
            generated_path=generated_path,
            generation_manifest=generation_manifest,
            primary_base_path=base_path,
        )
    )
    default_output_name = "evaluation_report" if reference_split == "test" else "evaluation_report_val"
    output_dir = ensure_dir(args.output_dir or generated_dir / default_output_name)
    validate_report_reference_split(output_dir, reference_split)

    evaluation_cfg = config.get("evaluation", {}) or {}
    generic_mmd_protocol = (
        str(evaluation_cfg.get("generic_mmd_protocol", "graphrnn"))
        if args.generic_mmd_protocol == "config"
        else str(args.generic_mmd_protocol)
    )
    generic_clustering_bins = int(
        evaluation_cfg.get("generic_clustering_bins", 100)
    )
    diversity_wl_iterations = int(
        args.diversity_wl_iterations
        if args.diversity_wl_iterations is not None
        else evaluation_cfg.get("diversity_wl_iterations", 3)
    )
    configured_diversity_epsilons = evaluation_cfg.get(
        "diversity_epsilons", (0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5)
    )
    diversity_epsilons = (
        list(args.diversity_epsilons)
        if args.diversity_epsilons is not None
        else [float(value) for value in configured_diversity_epsilons]
    )
    if diversity_wl_iterations < 0:
        raise ValueError("diversity_wl_iterations must be non-negative.")
    if not diversity_epsilons:
        raise ValueError("diversity_epsilons must contain at least one threshold.")
    if any(
        (not math.isfinite(float(value))) or float(value) < 0.0 or float(value) > 1.0
        for value in diversity_epsilons
    ):
        raise ValueError("All diversity_epsilons must be finite values in [0,1].")
    compute_orbit = bool(evaluation_cfg.get("compute_orbit", True))
    graphlet_backend = str(
        evaluation_cfg.get("graphlet_backend", "sampled")
    ).lower()
    exact_orca_backend = graphlet_backend in {
        "orca",
        "exact_orca",
        "exact",
    }
    orca_required = compute_orbit and exact_orca_backend
    orca_exec = (
        configure_orca_executable(
            evaluation_cfg.get("orca_exec"),
            required=orca_required,
        )
        if compute_orbit
        else None
    )
    if orca_exec:
        print(f"ORCA orbit evaluation enabled: {orca_exec}", flush=True)
    elif compute_orbit:
        print("Python four-node orbit evaluation enabled.", flush=True)
    else:
        print("ORCA evaluation disabled by configuration.", flush=True)

    print(
        "Generic MMD protocol: "
        f"{generic_mmd_protocol} (clustering_bins={generic_clustering_bins})",
        flush=True,
    )

    dataset_cfg = config.get("dataset", {}) or {}
    validate_prepared_dataset_contract(
        str(dataset_cfg.get("name", "sbm")),
        root=dataset_cfg.get("root", "outputs/datasets"),
        config_path=dataset_cfg.get("config_path"),
        mismatch_policy=args.dataset_mismatch_policy,
    )
    splits = load_dataset_splits(
        str(dataset_cfg.get("name", "sbm")),
        root=dataset_cfg.get("root", "outputs/datasets"),
        build_if_missing=bool(args.build_dataset_if_missing),
        config_path=dataset_cfg.get("config_path"),
    )
    train_graphs = list(splits.get("train", []))
    reference_graphs = list(splits.get(reference_split, []))
    provenance = dataset_provenance(dataset_cfg, splits)
    _print_dataset_provenance(provenance)
    training_manifest, training_manifest_path = _find_training_manifest(
        generated_path.parent
    )
    dataset_mismatches = validate_dataset_compatibility(
        provenance,
        generation_manifest=generation_manifest,
        training_manifest=training_manifest,
        mismatch_policy=args.dataset_mismatch_policy,
    )
    if training_manifest_path is not None:
        print(f"Training manifest: {training_manifest_path}", flush=True)
    generated_graphs = _load_graph_list(generated_path)
    sampled_degree_path = _resolve_sampled_degree_sequences(
        generated_dir, generation_manifest
    )
    sampled_degree_sequences = (
        _load_degree_sequence_list(sampled_degree_path)
        if sampled_degree_path is not None
        else []
    )
    base_graphs = (
        _load_graph_list(base_path)
        if base_path is not None and base_path.is_file()
        else []
    )
    additional_managed_stages = [
        (stage, _load_graph_list(path), path)
        for path, stage in additional_managed_stage_paths
    ]
    enriched_base_path = generated_dir / "enriched_base_graphs.pkl"
    enriched_base_graphs = (
        _load_graph_list(enriched_base_path)
        if enriched_base_path.is_file()
        else []
    )
    if not reference_graphs:
        raise ValueError(f"The configured dataset has no {reference_split} graphs.")

    protocol_cfg = config.get("protocol", {}) or {}
    configured_reference_cap = protocol_cfg.get("max_reference_graphs")
    reference_count, generated_count = resolve_evaluation_counts(
        num_reference=len(reference_graphs),
        num_generated=len(generated_graphs),
        configured_reference_cap=(
            int(configured_reference_cap)
            if configured_reference_cap is not None
            else None
        ),
        max_generated_graphs=args.max_graphs,
    )
    if reference_count <= 0:
        raise ValueError("No held-out reference graphs are available.")
    if generated_count <= 0:
        raise ValueError("No generated graphs are available.")
    reference = reference_graphs[:reference_count]
    generated = generated_graphs[:generated_count]
    degree_sequence_prior_metrics: dict[str, Any] | None = None
    if sampled_degree_sequences:
        degree_candidate_count = len(sampled_degree_sequences)
        if args.max_graphs is not None:
            degree_candidate_count = min(degree_candidate_count, int(args.max_graphs))
        degree_candidates = sampled_degree_sequences[:degree_candidate_count]
        degree_sequence_prior_metrics = evaluate_degree_sequence_sets(
            _graph_degree_sequences(reference),
            degree_candidates,
            train=_graph_degree_sequences(train_graphs) if train_graphs else None,
            degree_mmd_sigma=None,
        )
        degree_sequence_prior_metrics = {
            **degree_sequence_prior_metrics,
            "artifact_path": str(sampled_degree_path),
            "comparison": f"dhvae_degree_sequences_to_{reference_split}",
            "used_for_graph_generation": False,
        }
    molecular = is_molecular_evaluation(dataset_cfg, generated_graphs)
    graphlet_options = {
        "k_min": int(evaluation_cfg.get("graphlet_k_min", 3)),
        "k_max": int(evaluation_cfg.get("graphlet_k_max", 5)),
        "connected_only": bool(evaluation_cfg.get("graphlet_connected_only", True)),
        "topology_filter": str(evaluation_cfg.get("graphlet_topology_filter", "all")),
        "num_samples": evaluation_cfg.get("graphlet_num_samples"),
        "backend": graphlet_backend,
    }
    if not molecular and exact_orca_backend and not compute_orbit:
        orca_exec = configure_orca_executable(
            evaluation_cfg.get("orca_exec"), required=True,
        )
    final_stage = resolve_generated_stage(
        config,
        generated_path,
        generation_manifest=generation_manifest,
        explicit_stage=args.generated_stage,
    )
    print(f"Generated graph input: {generated_path}", flush=True)
    if generation_manifest_path is not None:
        print(
            f"Generation manifest: {generation_manifest_path}",
            flush=True,
        )
    print(f"Generated stage label: {final_stage}", flush=True)
    print(
        f"Evaluation counts: reference_{reference_split}={len(reference)} "
        f"generated={len(generated)} train_reference={len(train_graphs)}",
        flush=True,
    )

    metric_options = {
        "compute_orbit": compute_orbit,
        "metric_protocol": generic_mmd_protocol,
        "clustering_bins": generic_clustering_bins,
        "include_generic_metrics": not molecular,
        "graphlet_options": graphlet_options,
        "train": train_graphs,
    }
    rows: list[dict[str, Any]] = []
    if train_graphs:
        rows.append(
            _evaluate_comparison(
                reference,
                train_graphs,
                f"train_to_{reference_split}",
                novelty_candidate=reference,
                **metric_options,
            )
        )
    for stage, stage_graphs, _stage_path in additional_managed_stages:
        stage_count = len(stage_graphs)
        if args.max_graphs is not None:
            stage_count = min(stage_count, int(args.max_graphs))
        rows.append(
            _evaluate_comparison(
                reference,
                stage_graphs[:stage_count],
                f"{stage}_to_{reference_split}",
                **metric_options,
            )
        )
    if base_graphs:
        base_count = len(base_graphs)
        if args.max_graphs is not None:
            base_count = min(base_count, int(args.max_graphs))
        rows.append(
            _evaluate_comparison(
                reference, base_graphs[:base_count],
                f"{base_stage}_to_{reference_split}", **metric_options,
            )
        )
    if enriched_base_graphs and not _same_artifact(enriched_base_path, generated_path):
        enriched_count = len(enriched_base_graphs)
        if args.max_graphs is not None:
            enriched_count = min(enriched_count, int(args.max_graphs))
        rows.append(
            _evaluate_comparison(
                reference, enriched_base_graphs[:enriched_count],
                f"enriched_base_to_{reference_split}", **metric_options,
            )
        )
    rows.append(
        _evaluate_comparison(
            reference, generated, f"{final_stage}_to_{reference_split}", **metric_options,
        )
    )

    isomorphism_diagnostics: dict[str, Any] | None = None
    graded_diversity: dict[str, Any] | None = None
    merged_diversity_rows: list[dict[str, Any]] = []
    isomorphism_support_csv_path = output_dir / "isomorphism_support_metrics.csv"
    isomorphism_class_csv_path = output_dir / "isomorphism_class_frequencies.csv"
    generic_diversity_csv_path = output_dir / "generic_diversity_metrics.csv"
    generic_diversity_curve_csv_path = output_dir / "generic_diversity_curve.csv"

    generic_populations: dict[str, Sequence[nx.Graph]] | None = None
    reference_population_name = f"heldout_{reference_split}"
    if not molecular and train_graphs:
        generic_populations = {
            "train": train_graphs,
            reference_population_name: reference,
        }
        for stage, stage_graphs, _stage_path in additional_managed_stages:
            stage_count = len(stage_graphs)
            if args.max_graphs is not None:
                stage_count = min(stage_count, int(args.max_graphs))
            generic_populations[str(stage)] = stage_graphs[:stage_count]
        if base_graphs:
            base_count = len(base_graphs)
            if args.max_graphs is not None:
                base_count = min(base_count, int(args.max_graphs))
            generic_populations[str(base_stage)] = base_graphs[:base_count]
        if enriched_base_graphs and not _same_artifact(enriched_base_path, generated_path):
            enriched_count = len(enriched_base_graphs)
            if args.max_graphs is not None:
                enriched_count = min(enriched_count, int(args.max_graphs))
            generic_populations["enriched_base"] = enriched_base_graphs[:enriched_count]
        generic_populations[final_stage] = generated

    if generic_populations is not None and not args.skip_exact_isomorphism_diagnostics:
        print(
            "Computing exact-isomorphism support diagnostics with the complete "
            f"training split ({len(train_graphs)} graphs)...",
            flush=True,
        )
        isomorphism_diagnostics = exact_isomorphism_support_diagnostics(
            generic_populations,
            train_name="train",
            reference_name=reference_population_name,
        )
        _write_isomorphism_support_csv(
            isomorphism_diagnostics["population_metrics"],
            isomorphism_support_csv_path,
        )
        _write_isomorphism_support_csv(
            isomorphism_diagnostics["class_frequencies"],
            isomorphism_class_csv_path,
        )

    if generic_populations is not None and not args.skip_graded_diversity_diagnostics:
        print(
            "Computing graded topology diversity diagnostics "
            f"(WL iterations={diversity_wl_iterations}, "
            f"epsilons={sorted(set(diversity_epsilons))})...",
            flush=True,
        )
        graded_diversity = graded_diversity_diagnostics(
            generic_populations,
            train_name="train",
            reference_name=reference_population_name,
            epsilons=diversity_epsilons,
            wl_iterations=diversity_wl_iterations,
        )
        _write_generic_diversity_csv(
            graded_diversity["epsilon_curve"],
            generic_diversity_curve_csv_path,
        )

    if isomorphism_diagnostics is not None or graded_diversity is not None:
        merged_diversity_rows = _merge_generic_diversity_rows(
            (isomorphism_diagnostics or {}).get("population_metrics", []),
            (graded_diversity or {}).get("population_metrics", []),
        )
        _write_generic_diversity_csv(
            merged_diversity_rows,
            generic_diversity_csv_path,
        )

    csv_path = output_dir / "graph_mmd_metrics.csv"
    json_path = output_dir / "graph_evaluation_report.json"
    degree_sequence_csv_path = output_dir / "degree_sequence_mmd.csv"
    molecular_csv_path = output_dir / "molecular_quality_metrics.csv"
    valid_smiles_path = output_dir / "valid_generated.smi"
    _write_csv(rows, csv_path)
    if degree_sequence_prior_metrics is not None:
        _write_degree_sequence_csv(
            degree_sequence_prior_metrics,
            degree_sequence_csv_path,
            reference_split=reference_split,
        )
    molecular_metrics: dict[str, Any] | None = None
    molecular_stage_rows: list[dict[str, Any]] = []
    molecular_stage_errors: dict[str, dict[str, int]] = {}
    invalid_molecule_indices: list[int] = []
    conversion_error_counts: dict[str, int] = {}
    if molecular:
        stage_graphs: list[tuple[str, Sequence[nx.Graph]]] = [
            (f"real_{reference_split}", reference),
        ]
        if base_graphs:
            base_count = len(base_graphs)
            if args.max_graphs is not None:
                base_count = min(base_count, int(args.max_graphs))
            stage_graphs.append((str(base_stage), base_graphs[:base_count]))
        stage_graphs.append((final_stage, generated))
        valid_smiles: list[str] = []
        for stage, graphs in stage_graphs:
            (
                stage_metrics,
                stage_smiles,
                stage_invalid_indices,
                stage_errors,
            ) = molecular_quality_metrics(graphs, train_graphs)
            molecular_stage_rows.append({"stage": stage, **stage_metrics})
            molecular_stage_errors[stage] = stage_errors
            if stage == final_stage:
                molecular_metrics = stage_metrics
                valid_smiles = stage_smiles
                invalid_molecule_indices = stage_invalid_indices
                conversion_error_counts = stage_errors
        _write_molecular_csv(molecular_stage_rows, molecular_csv_path)
        with valid_smiles_path.open("w", encoding="utf-8") as handle:
            for smiles in valid_smiles:
                handle.write(smiles + "\n")

    save_json(
        {
            "format": "graph_generation_evaluation_report_v10",
            "reference_split": reference_split,
            "reference_split_sha256": provenance["split_sha256"][reference_split],
            "reference_graph_indices_zero_based": list(range(reference_count)),
            "orca_exec": orca_exec,
            "compute_orbit": compute_orbit,
            "generic_mmd_protocol": generic_mmd_protocol,
            "generic_clustering_bins": generic_clustering_bins,
            "graphlet_backend": graphlet_backend,
            "generic_graphlet_options": graphlet_options if not molecular else None,
            "generic_spectral_bins": 20 if not molecular else None,
            "generic_novelty_protocol": (
                {
                    "definition": "evaluated graphs with WL topology hashes absent from the complete training split / evaluated graphs",
                    "row_semantics": {
                        "train_to_<split>": "held-out split evaluated against complete training support",
                        "model_to_<split>": "generated model population evaluated against complete training support",
                    },
                    "reference_split": "train",
                    "num_training_graphs": len(train_graphs),
                    "node_and_edge_attributes": False,
                    "isomorphism_test": "approximate Weisfeiler-Lehman hash comparison",
                }
                if not molecular else None
            ),
            "generic_exact_isomorphism_diagnostics": (
                {
                    key: value
                    for key, value in (isomorphism_diagnostics or {}).items()
                    if key not in {"assignments", "class_frequencies"}
                }
                if not molecular else None
            ),
            "generic_exact_isomorphism_support_csv": (
                str(isomorphism_support_csv_path)
                if isomorphism_diagnostics is not None else None
            ),
            "generic_exact_isomorphism_class_frequency_csv": (
                str(isomorphism_class_csv_path)
                if isomorphism_diagnostics is not None else None
            ),
            "generic_diversity_diagnostics": (
                {
                    "protocol": (graded_diversity or {}).get("protocol"),
                    "population_metrics": merged_diversity_rows,
                    "epsilon_curve": (graded_diversity or {}).get("epsilon_curve", []),
                    "four_quantities": {
                        "exact_novelty": (
                            "fraction of graphs not exactly isomorphic to any graph "
                            "in the complete training split"
                        ),
                        "isomorphism_uniqueness": (
                            "number of exact topology isomorphism classes divided by "
                            "the number of graphs in the population"
                        ),
                        "nearest_training_distance": (
                            "mean nearest-training wl_component_structural_"
                            "pseudodistance_v1; lower means closer to training support"
                        ),
                        "epsilon_novelty_and_coverage": (
                            "epsilon novelty is the fraction farther than epsilon from "
                            "every training graph; held-out coverage is the fraction of "
                            "held-out graphs within epsilon of at least one population graph"
                        ),
                    },
                }
                if merged_diversity_rows else None
            ),
            "generic_diversity_metrics_csv": (
                str(generic_diversity_csv_path) if merged_diversity_rows else None
            ),
            "generic_diversity_curve_csv": (
                str(generic_diversity_curve_csv_path)
                if graded_diversity is not None else None
            ),
            "generic_spectral_descriptor": (
                "normalized_laplacian_isolates_zero_stable_boundaries_v2"
                if not molecular else None
            ),
            "dataset_provenance": provenance,
            "dataset_mismatch_policy": args.dataset_mismatch_policy,
            "dataset_provenance_mismatches": dataset_mismatches,
            "training_manifest": (
                str(training_manifest_path)
                if training_manifest_path is not None
                else None
            ),
            "generated_graphs": str(generated_path),
            "generated_stage": final_stage,
            "generation_manifest": (
                str(generation_manifest_path)
                if generation_manifest_path is not None
                else None
            ),
            "generation_manifest_format": (
                generation_manifest.get("format")
                if generation_manifest is not None
                else None
            ),
            "generation_model_id": (
                generation_manifest.get("model_id")
                if generation_manifest is not None
                else None
            ),
            "base_graphs": str(base_path) if base_path is not None else None,
            "base_stage": base_stage,
            "additional_managed_stages": [
                {"stage": stage, "path": str(path)}
                for path, stage in additional_managed_stage_paths
            ],
            "num_graphs_evaluated": len(generated),
            "num_reference_graphs": len(reference),
            "num_generated_graphs_evaluated": len(generated),
            "metrics": rows,
            "sampled_degree_sequences": (
                str(sampled_degree_path) if sampled_degree_path is not None else None
            ),
            "degree_sequence_prior": degree_sequence_prior_metrics,
            "molecular_evaluation": molecular,
            "molecular_quality": molecular_metrics,
            "molecular_quality_by_stage": {
                str(row["stage"]): {
                    key: value for key, value in row.items() if key != "stage"
                }
                for row in molecular_stage_rows
            },
            "molecular_protocol": (
                {
                    "validity_without_correction": (
                        "RDKit molecule construction followed by "
                        "Chem.SanitizeMol, without valency correction or "
                        "edge resampling."
                    ),
                    "uniqueness": (
                        "unique valid canonical SMILES / valid generated molecules"
                    ),
                    "novelty": (
                        "unique valid generated canonical SMILES absent from "
                        "the complete training split / unique valid generated "
                        "canonical SMILES"
                    ),
                    "canonical_smiles": True,
                    "isomeric_smiles": False,
                }
                if molecular
                else None
            ),
            "invalid_molecule_indices_zero_based": invalid_molecule_indices,
            "molecular_conversion_error_counts": conversion_error_counts,
            "molecular_conversion_error_counts_by_stage": (
                molecular_stage_errors if molecular else None
            ),
            "sample_selection": None,
            "sample_indices_zero_based": [],
            "sample_figure_png": None,
            "sample_figure_pdf": None,
        },
        json_path,
    )
    _print_table(rows, reference_split=reference_split)
    if isomorphism_diagnostics is not None:
        _print_isomorphism_support_diagnostics(
            isomorphism_diagnostics["population_metrics"]
        )
    if isomorphism_diagnostics is not None and graded_diversity is not None:
        _print_generic_diversity_diagnostics(
            merged_diversity_rows, graded_diversity["epsilon_curve"]
        )
    if degree_sequence_prior_metrics is not None:
        _print_degree_sequence_prior(
            degree_sequence_prior_metrics, reference_split=reference_split
        )
    if molecular_metrics is not None:
        _print_molecular_metrics(molecular_stage_rows)
    print(f"Saved metrics: {csv_path}")
    if isomorphism_diagnostics is not None:
        print(f"Saved support metrics: {isomorphism_support_csv_path}")
        print(f"Saved class frequencies: {isomorphism_class_csv_path}")
    if merged_diversity_rows:
        print(f"Saved diversity metrics: {generic_diversity_csv_path}")
    if graded_diversity is not None:
        print(f"Saved diversity curve: {generic_diversity_curve_csv_path}")
    if degree_sequence_prior_metrics is not None:
        print(f"Saved metrics: {degree_sequence_csv_path}")
    if molecular_metrics is not None:
        print(f"Saved metrics: {molecular_csv_path}")
        print(f"Saved SMILES:  {valid_smiles_path}")
    print(f"Saved report:  {json_path}")


if __name__ == "__main__":
    main()
