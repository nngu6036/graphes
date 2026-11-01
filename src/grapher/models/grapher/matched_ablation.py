"""Export matched controls from trusted, hash-anchored saved generic endpoints.

No reverse diffusion or neural inference is performed. The caller must verify
the managed training artifacts and original generation intent before calling.
Parent artifacts are never rewritten; positional identities are explicit in
the exported manifests when older graph pickles have no embedded sample IDs.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import networkx as nx
import numpy as np
import torch

from grapher.models.grapher import degree_constrained_realization as realization
from grapher.models.grapher.graphlet_stage3 import (
    refine_generated_graphs_with_fixed_predictions,
)
from grapher.utils.networkx_pickle import load_trusted_networkx_pickle

FORMAT = "graphes_matched_ablation_export_v1"
IDENTITY = "hash_anchored_parent_list_order"
COMMON = "matched_ablation_manifest.json"
INDEX = "ablation_manifest.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _subset(expected: Any, actual: Any, label: str) -> None:
    """Verify declared settings while allowing producer-resolved defaults."""
    if isinstance(expected, Mapping):
        _require(isinstance(actual, Mapping), f"Missing saved mapping: {label}")
        for key, value in expected.items():
            _require(key in actual, f"Unrecorded saved setting: {label}.{key}")
            _subset(value, actual[key], f"{label}.{key}")
    else:
        _require(expected == actual, f"Saved setting differs: {label}")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(value), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _record(directory: Path, record: Any, label: str) -> Path:
    _require(isinstance(record, dict), f"Missing artifact record: {label}")
    _require(bool(record.get("path")) and bool(record.get("sha256")), f"Missing path/hash: {label}")
    path = directory / str(record["path"])
    _require(path.resolve().is_relative_to(directory.resolve()), f"Artifact escapes directory: {label}")
    _require(path.is_file(), f"Missing artifact: {path}")
    _require(_sha(path) == record["sha256"], f"Artifact hash mismatch: {path}")
    return path


def _population(value: Any, ids: list[str], label: str, *, labelled: bool = False) -> list[nx.Graph]:
    _require(isinstance(value, list) and len(value) == len(ids), f"Population count differs: {label}")
    for graph, sample_id in zip(value, ids):
        _require(isinstance(graph, nx.Graph), f"Non-graph object: {label}:{sample_id}")
        _require(not graph.is_directed() and not graph.is_multigraph(), f"Graph is not simple/undirected: {label}:{sample_id}")
        _require(nx.number_of_selfloops(graph) == 0, f"Graph has self-loops: {label}:{sample_id}")
        _require(set(graph) == set(range(len(graph))), f"Indexed nodes differ: {label}:{sample_id}")
        actual = graph.graph.get("sample_id")
        _require((actual is None and not labelled) or str(actual) == sample_id, f"Saved graph ID/order differs: {label}:{sample_id}")
    return value


def _degrees(graph: nx.Graph) -> list[int]:
    return [int(graph.degree(node)) for node in range(len(graph))]


def _connected(graph: nx.Graph) -> bool:
    return len(graph) <= 1 or nx.is_connected(graph)


def _same_invariants(source: nx.Graph, graph: nx.Graph, *, connectivity: bool, label: str) -> None:
    _require(set(source) == set(graph) and _degrees(source) == _degrees(graph), f"Indexed degrees changed: {label}")
    _require(not connectivity or not _connected(source) or _connected(graph), f"Connectivity changed: {label}")


def _seed(seed: int, index: int, offset: int) -> int:
    return int(np.random.SeedSequence([seed, offset, index]).generate_state(1)[0])


def _refine_one(graph, prediction, payload, cfg, seed, index, budget):
    options = copy.deepcopy(cfg["graphlet_refinement"])
    options["random_steps"] = {"enabled": False}
    options.setdefault("refiner", {})["steps"] = budget
    graphs, _, diagnostics, _ = refine_generated_graphs_with_fixed_predictions(
        [graph.copy()], [prediction], graphlet_summary_payload=payload,
        config=options, seed=seed,
    )
    _require(len(graphs) == 1 and len(diagnostics) == 1, f"Refinement count differs at {index}")
    diagnostic = dict(diagnostics[0], graph_index=index, requested_steps=budget)
    return graphs[0], diagnostic


def _decode_zero(scores, degree, config, seed):
    matrix = np.asarray(scores, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    u, v = realization._edge_index(len(degree))
    normalized = realization._normalize_scores(
        matrix[u, v], str(config.get("score_transform", "zscore")),
        float(config.get("score_clip", 8.0)),
    )
    options = copy.deepcopy(config)
    options["gumbel_temperature"] = 0.0
    options["failure_policy"] = "error"
    return realization._solve_exact_degree_realization(
        np.asarray(degree, dtype=np.int64), normalized,
        rng=np.random.default_rng(seed), config=options,
    )


def _report_snapshot(root: Path) -> dict[str, str]:
    """Permit evaluation-only remnants from a prior failed seal, preserving bytes."""
    result = {}
    if not root.exists():
        return result
    _require(root.is_dir() and not root.is_symlink(), f"Invalid ablation directory: {root}")
    for path in root.rglob("*"):
        _require(not path.is_symlink(), f"Symlink in existing ablations: {path}")
        if not path.is_file():
            continue
        parts = path.relative_to(root).parts
        _require(
            len(parts) >= 3 and parts[0] in {"T", "D", "TR", "DR", "D0", "R1", "R12", "S0"}
            and parts[1].startswith("evaluation_"),
            f"Incomplete/conflicting ablation artifacts already exist: {path}",
        )
        result[path.relative_to(root).as_posix()] = _sha(path)
    return result


def _existing(directory: Path, provenance: dict, ids: list[str], variants: list[str], parents: dict, budgets: list[int], cfg: dict) -> Path:
    common_path = directory / COMMON
    common = _json(common_path)
    _require(common == _json(directory / "ablations" / INDEX), "Matched common indexes differ")
    _require(common.get("format") == FORMAT, "Unknown matched export format")
    _require(common.get("provenance") == provenance, "Matched export provenance differs")
    _require(common.get("sample_ids") == ids, "Matched sample ID order differs")
    _require(common.get("generation_seed") == provenance["seed"] and common.get("num_requested") == len(ids), "Matched seed/count differs")
    _require(common.get("saved_dr_requested_steps") == budgets, "Matched DR budgets differ from saved parent")
    _require(set(common.get("branches", {})) == set(variants), "Matched branch set differs")
    for variant in variants:
        branch = directory / "ablations" / variant
        child = _json(branch / "manifest.json")
        _require(child == common["branches"][variant], f"Branch/index mismatch: {variant}")
        _require(child.get("provenance") == provenance, f"Branch provenance differs: {variant}")
        _require(child.get("requested_sample_ids") == ids, f"Requested branch IDs differ: {variant}")
        _require(child.get("generation_seed") == provenance["seed"], f"Branch seed differs: {variant}")
        selected = child.get("sample_ids")
        _require(isinstance(selected, list) and len(set(selected)) == len(selected), f"Missing/duplicate IDs: {variant}")
        chosen = set(selected)
        _require(selected == [item for item in ids if item in chosen], f"Reordered/unknown branch IDs: {variant}")
        missing = [item for item in ids if item not in chosen]
        _require(not missing or variant == "D0", f"Incomplete primary branch: {variant}")
        _require(child.get("requested_count") == len(ids), f"Requested count differs: {variant}")
        _require(child.get("completed_count") == len(selected), f"Completed count differs: {variant}")
        _require(child.get("failed_count") == len(missing), f"Failed count differs: {variant}")
        _require([row.get("sample_id") for row in child.get("failures", [])] == missing, f"Failure records differ: {variant}")
        status = "complete" if not missing else ("partial" if selected else "failed")
        _require(child.get("status") == status, f"Branch status differs: {variant}")
        path = _record(branch, child.get("base_graphs"), f"{variant}.base_graphs")
        copied = {"T": "pre_realization_graphs", "D": "pre_refinement_graphs", "DR": "base_graphs"}.get(variant)
        if copied:
            source = provenance["source_artifacts"][copied]
            _require(child.get("identity_source") == IDENTITY and child.get("source_artifact") == source, f"Missing positional identity source: {variant}")
            _require(child["base_graphs"]["sha256"] == source["sha256"], f"Copied parent population changed: {variant}")
        graphs = _population(load_trusted_networkx_pickle(path), selected, variant, labelled=not bool(copied))
        diagnostic = _json(_record(branch, child.get("diagnostics"), f"{variant}.diagnostics"))
        if variant in {"TR", "R1", "R12", "D0"}:
            rows = diagnostic.get("per_graph", [])
            _require(len(rows) == len(selected), f"Diagnostic count differs: {variant}")
            for graph, sample_id, row in zip(graphs, selected, rows):
                index = ids.index(sample_id)
                _require(row.get("sample_id") == sample_id and row.get("graph_index") == index, f"Diagnostic ID/order differs: {variant}")
                source = parents["T" if variant == "TR" else "D"][index]
                connected = cfg.get("graphlet_refinement", {}).get("refiner", {}).get("preserve_connectivity", True)
                if variant == "D0":
                    connected = cfg.get("sample", {}).get("degree_constrained_realization", {}).get("require_connected", True)
                _same_invariants(source, graph, connectivity=bool(connected), label=f"{variant}:{sample_id}")
                if variant == "TR":
                    _require(row.get("requested_steps") == budgets[index], "TR and DR requested budgets differ")
                elif variant == "R1":
                    _require(row.get("requested_steps") == 1, "R1 budget differs")
                elif variant == "R12":
                    _require(row.get("requested_steps") in {1, 2}, "R12 budget differs")
    return common_path


def export_matched(directory: Path, cfg: dict, context: dict, checkpoint_path: Path) -> Path:
    """Export/reuse all configured matched views without resampling saved endpoints.

    The caller holds its generation lock and has validated the original begin
    intent. Existing evaluation-only directories survive installation unchanged.
    Incomplete or conflicting exported model artifacts are never overwritten.
    """
    directory, checkpoint_path = Path(directory), Path(checkpoint_path)
    _require(directory.is_dir(), f"Missing generation directory: {directory}")
    manifest_path = directory / "manifest.json"
    manifest_sha = _sha(manifest_path)
    generation = _json(manifest_path)
    seed, count = context.get("seed"), context.get("num_samples")
    _require(type(seed) is int and seed >= 0, "Invalid generation seed context")
    _require(type(count) is int and count > 0, "Invalid requested population context")
    ids = [f"{seed}:{index}" for index in range(count)]
    for key, expected in (
        ("model_id", "grapher"), ("run_id", context.get("run_id")),
        ("generation_id", directory.name), ("generation_seed", seed),
        ("num_requested", count), ("num_generated", count), ("variant", cfg.get("variant")),
    ):
        _require(generation.get(key) == expected, f"Generation {key} differs")
    _require(context.get("generation_id", directory.name) == directory.name, "Generation context ID differs")
    _require(checkpoint_path.is_file() and _sha(checkpoint_path) == context.get("checkpoint_sha256"), "Checkpoint hash differs from verified context")
    _require(generation.get("checkpoint", {}).get("sha256") == context["checkpoint_sha256"], "Generation checkpoint differs")
    sample = cfg.get("sample", {})
    matched = sample.get("matched_ablation", {})
    _require(matched.get("enabled") is True, "Matched ablation is not enabled")
    realization_cfg = realization.resolve_config(sample)
    _require(realization_cfg.get("enabled") is True, "Matched export needs exact-degree parent realizations")
    _require(not realization_cfg.get("candidate_selection", {}).get("enabled"), "Matched decoder export does not support summary-selected candidates")
    refinement = cfg.get("graphlet_refinement", {})
    _require(refinement.get("enabled") is True, "Matched export needs a saved rewired parent")
    sampling = generation.get("sampling", {})
    if "threshold" in sampling:
        _require(sampling["threshold"] == sample.get("threshold", 0.5), "Saved threshold differs")
    if "stochastic_diversification" in sampling:
        recorded = sampling["stochastic_diversification"] or {}
        for key in ("random_steps", "refiner", "target_source"):
            _subset(refinement.get(key), recorded.get(key), f"refinement.{key}")
    consumed, paths = {}, {}
    keys = [
        "pre_realization_graphs", "pre_refinement_graphs", "base_graphs",
        "projected_realization_degree_sequences", "predicted_structure_summaries",
        "sampled_basis_indices", "graphlet_refinement_diagnostics",
    ]
    if matched.get("include_d0", True):
        keys.append("continuous_adjacencies")
    for key in keys:
        paths[key] = _record(directory, generation.get(key), key)
        consumed[key] = copy.deepcopy(generation[key])
    populations = {
        variant: _population(load_trusted_networkx_pickle(paths[key]), ids, key)
        for variant, key in (("T", "pre_realization_graphs"), ("D", "pre_refinement_graphs"), ("DR", "base_graphs"))
    }
    degrees = load_trusted_networkx_pickle(paths["projected_realization_degree_sequences"])
    predictions = load_trusted_networkx_pickle(paths["predicted_structure_summaries"])
    diagnostics = load_trusted_networkx_pickle(paths["graphlet_refinement_diagnostics"])
    basis = load_trusted_networkx_pickle(paths["sampled_basis_indices"])
    for label, value in (("degrees", degrees), ("predictions", predictions), ("diagnostics", diagnostics), ("basis indices", basis)):
        _require(isinstance(value, list) and len(value) == count, f"Saved {label} count/order differs")
    budgets = []
    for index in range(count):
        _require(isinstance(predictions[index], Mapping) and predictions[index].get("graph_index") == index, f"Prediction graph_index differs: {index}")
        row = diagnostics[index]
        _require(isinstance(row, Mapping) and row.get("graph_index") == index, f"DR diagnostic graph_index differs: {index}")
        budget = row.get("requested_steps")
        _require(type(budget) is int and budget >= 0, f"Missing/invalid saved DR budget: {index}")
        random_cfg = refinement.get("random_steps", {})
        if random_cfg.get("enabled", False):
            _require(int(random_cfg.get("min_steps", 0)) <= budget <= int(random_cfg["max_steps"]), f"Saved DR budget outside configured range: {index}")
        else:
            _require(budget == int(refinement.get("refiner", {}).get("steps", 0)), f"Saved DR fixed budget differs: {index}")
        budgets.append(budget)
        target = degrees[index]
        _require(isinstance(target, (list, np.ndarray)), f"Invalid projected degree target: {index}")
        _require(all(isinstance(value, (int, np.integer)) and not isinstance(value, bool) for value in target), f"Noninteger degree target: {index}")
        target = [int(value) for value in target]
        _require(_degrees(populations["D"][index]) == target, f"Saved D degree target differs: {index}")
        _require(_degrees(populations["DR"][index]) == target, f"Saved DR degree target differs: {index}")
        _require(len(populations["T"][index]) == len(target), f"Threshold node count differs: {index}")
        _require(not realization_cfg.get("require_connected", True) or _connected(populations["D"][index]), f"Saved D is disconnected: {index}")
        _same_invariants(populations["D"][index], populations["DR"][index], connectivity=bool(refinement.get("refiner", {}).get("preserve_connectivity", True)), label=f"DR:{index}")
    provenance = {
        **copy.deepcopy(context), "sample_config": copy.deepcopy(sample),
        "sde": copy.deepcopy(cfg.get("sde", {})),
        "generation_batch_size": cfg.get("generation_batch_size"),
        "source_generation_manifest_sha256": manifest_sha,
        "source_artifacts": consumed, "identity_scheme": IDENTITY,
    }
    variants = ["T", "D", "TR", "DR"]
    if matched.get("include_d0", True):
        variants.append("D0")
    if matched.get("include_budget_sweep", True):
        variants.extend(("R1", "R12"))
    if (directory / COMMON).exists() or (directory / "ablations" / INDEX).exists():
        return _existing(directory, provenance, ids, variants, populations, budgets, cfg)
    report_root = directory / "ablations"
    had_reports = report_root.exists()
    report_snapshot = _report_snapshot(report_root)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload = state.get("structure_summary", {}) or {}
    if not payload.get("enabled", False):
        payload = state.get("graphlet_summary", {}) or {}
    _require(payload.get("enabled") is True, "Checkpoint lacks joint summary metadata")
    _require(isinstance(payload.get("graphlet_basis"), dict) and isinstance(payload.get("summary_config"), dict), "Checkpoint lacks graphlet basis/summary metadata")
    scores = load_trusted_networkx_pickle(paths["continuous_adjacencies"]) if "D0" in variants else None
    if scores is not None:
        _require(isinstance(scores, list) and len(scores) == count, "Saved continuous score count differs")
        for index, matrix in enumerate(scores):
            array = np.asarray(matrix)
            n = len(populations["D"][index])
            _require(array.shape == (n, n) and np.isfinite(array).all(), f"Invalid saved continuous scores: {index}")
    started = time.monotonic()
    staging = Path(tempfile.mkdtemp(prefix=".matched_export_", dir=directory))
    _require(staging.resolve().is_relative_to(directory.resolve()), "Export staging escapes generation")
    staged_root = staging / "ablations"
    backup = staging / "previous_ablations"
    installed = False
    published = False
    try:
        if had_reports:
            shutil.copytree(report_root, staged_root)
        else:
            staged_root.mkdir()
        children = {}

        def save_branch(variant, graphs, branch_ids, rows, *, failures=None, source_key=None):
            failures = failures or []
            child_dir = staged_root / variant
            child_dir.mkdir(exist_ok=True)
            graph_path = child_dir / "base_graphs.pkl"
            if source_key:
                shutil.copyfile(paths[source_key], graph_path)
            else:
                _population(graphs, branch_ids, variant, labelled=True)
                with graph_path.open("xb") as stream:
                    pickle.dump(graphs, stream, protocol=pickle.HIGHEST_PROTOCOL)
            diagnostic_path = child_dir / "diagnostics.json"
            _write(diagnostic_path, {"per_graph": rows, "failures": failures})
            child = {
                "format": FORMAT, "variant": variant, "label": variant,
                "generation_seed": seed, "requested_count": count,
                "completed_count": len(branch_ids), "failed_count": len(failures),
                "status": "complete" if not failures else ("partial" if branch_ids else "failed"),
                "sample_ids": branch_ids, "requested_sample_ids": ids,
                "failures": failures, "base_graphs": {"path": "base_graphs.pkl", "sha256": _sha(graph_path)},
                "diagnostics": {"path": "diagnostics.json", "sha256": _sha(diagnostic_path)},
                "provenance": provenance, "shared_context": provenance,
                "degree_source": "saved_projected_indexed_degree_targets" if variant != "T" and variant != "TR" else "saved_threshold_population",
                "posthoc_repair": False,
                "timing_scope": "saved parent population; export/derived control work only; no reverse diffusion rerun",
            }
            if source_key:
                child.update(identity_source=IDENTITY, source_artifact=consumed[source_key])
            _write(child_dir / "manifest.json", child)
            children[variant] = child

        save_branch("T", populations["T"], ids, [], source_key="pre_realization_graphs")
        save_branch("D", populations["D"], ids, [], source_key="pre_refinement_graphs")
        save_branch("DR", populations["DR"], ids, [dict(row) for row in diagnostics], source_key="base_graphs")
        rewire_offset = int(matched.get("rewire_seed_offset", 620001))
        budget_offset = int(matched.get("budget_seed_offset", 610001))
        for variant in (name for name in ("TR", "R1", "R12") if name in variants):
            print(f"Matched export deriving {variant} from saved endpoints: 0/{count}", flush=True)
            graphs, rows = [], []
            source_variant = "T" if variant == "TR" else "D"
            for index, source in enumerate(populations[source_variant]):
                budget = budgets[index] if variant == "TR" else (
                    1 if variant == "R1" else int(np.random.default_rng(_seed(seed, index, budget_offset)).integers(1, 3))
                )
                local_seed = _seed(seed, index, rewire_offset)
                graph, row = _refine_one(source, predictions[index], payload, cfg, local_seed, index, budget)
                _same_invariants(source, graph, connectivity=bool(refinement.get("refiner", {}).get("preserve_connectivity", True)), label=f"{variant}:{index}")
                graph.graph["sample_id"] = ids[index]
                graphs.append(graph)
                rows.append({**_plain(row), "sample_id": ids[index], "rewiring_seed": local_seed, "requested_steps": budget})
                if (index + 1) % 128 == 0 or index + 1 == count:
                    print(f"Matched export {variant}: {index + 1}/{count}", flush=True)
            save_branch(variant, graphs, ids, rows)
        if "D0" in variants:
            print(f"Matched export deriving D0 from saved degree targets: 0/{count}", flush=True)
            graphs, selected_ids, rows, failures = [], [], [], []
            for index, degree in enumerate(degrees):
                try:
                    graph, row = _decode_zero(scores[index], degree, realization_cfg, _seed(seed, index, int(realization_cfg.get("seed_offset", 810001))))
                    _require(_degrees(graph) == list(degree), f"D0 indexed degrees changed: {index}")
                    _require(not realization_cfg.get("require_connected", True) or _connected(graph), f"D0 is disconnected: {index}")
                    graph.graph["sample_id"] = ids[index]
                    _population([graph], [ids[index]], "D0", labelled=True)
                    graphs.append(graph)
                    selected_ids.append(ids[index])
                    rows.append({"sample_id": ids[index], "graph_index": index, "gumbel_temperature": 0.0, "solver": _plain(row)})
                except Exception as error:
                    failures.append({"sample_id": ids[index], "graph_index": index, "error": f"{type(error).__name__}: {error}", "replacement_generated": False})
                if (index + 1) % 128 == 0 or index + 1 == count:
                    print(f"Matched export D0: {index + 1}/{count}; {len(failures)} explicit failures", flush=True)
            save_branch("D0", graphs, selected_ids, rows, failures=failures)
        common = {
            "format": FORMAT, "generation_seed": seed, "num_requested": count,
            "sample_ids": ids, "branches": {name: children[name] for name in variants},
            "provenance": provenance, "shared_context": provenance,
            "saved_dr_requested_steps": budgets,
            "configured_dr_budget": copy.deepcopy(refinement.get("random_steps", {})),
            "derived_control_rng": "isolated SeedSequence(seed, configured offset, parent list index); saved DR RNG unchanged",
            "time_accounting": {
                "parent_generation_seconds": generation.get("duration_seconds"),
                "export_and_derived_controls_seconds": time.monotonic() - started,
                "reverse_diffusion_rerun": False,
                "derived_control_time_includes": "fixed-prediction rewiring and optional zero-temperature exact-degree MILP only",
            },
        }
        _write(staged_root / INDEX, common)
        _write(staging / COMMON, common)
        _require(_sha(manifest_path) == manifest_sha, "Parent generation manifest changed during export")
        _require(_sha(checkpoint_path) == context["checkpoint_sha256"], "Checkpoint changed during export")
        for key, record in consumed.items():
            _record(directory, record, key)
        _require(report_root.exists() == had_reports and _report_snapshot(report_root) == report_snapshot, "Existing evaluation reports changed during export")
        _require(not (directory / COMMON).exists(), "Matched index appeared during export")
        if had_reports:
            report_root.rename(backup)
        try:
            staged_root.rename(report_root)
            installed = True
            # Atomic no-overwrite publication of the already-complete root index.
            os.link(staging / COMMON, directory / COMMON)
            published = True
        except BaseException:
            if installed:
                report_root.rename(staging / "unpublished_ablations")
            if backup.exists():
                backup.rename(report_root)
            raise
        return directory / COMMON
    finally:
        if backup.exists() and not published:
            # If rollback itself failed, retain the original reports for manual
            # restoration rather than deleting the only remaining backup.
            print(f"Original evaluation backup retained at {backup}", file=sys.stderr)
        else:
            shutil.rmtree(staging)
