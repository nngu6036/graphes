"""Export paired molecular endpoints without changing any saved populations.

The caller verifies the managed training run, original generation intent and
effective sampler. This module checks the saved population hashes and strict
same-bond invariants, then publishes byte-identical T/TR views with explicit
positional identities. No model/checkpoint is deserialized or evaluated.
"""
from __future__ import annotations

import copy
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

from grapher.models.grapher.attributed_diversification import (
    _bond_valence_vectors_equal,
    _neighbor_atom_bond_signature,
    _weighted_bond_valence_vector,
)
from grapher.models.grapher.matched_ablation import (
    IDENTITY,
    INDEX,
    _connected,
    _degrees,
    _json,
    _plain,
    _population,
    _record,
    _report_snapshot,
    _require,
    _sha,
    _write,
)
from grapher.rewiring_mlp.molecular.typed_invariants import extract_typed_invariant
from grapher.utils.networkx_pickle import load_trusted_networkx_pickle

FORMAT = "graphes_molecular_ablation_export_v1"
COMMON = "molecular_ablation_manifest.json"
SOURCES = {"T": "pre_refinement_graphs", "TR": "base_graphs"}


def _verify_pairs(before, after, ids, cfg):
    attributed = cfg.get("attributed", {})
    node_attribute = str(attributed.get("node_attribute", "atomic_num"))
    edge_attribute = str(attributed.get("edge_attribute", "bond_type"))
    edge_types = tuple(attributed.get("edge_categories", (1, 2, 3)))
    _require(len(set(edge_types)) == len(edge_types), "Duplicate configured bond categories")
    node_types = attributed.get("node_categories")
    diversification = cfg.get("attributed_diversification", {})
    _require(diversification.get("enabled") is True, "Molecular T/TR export requires enabled diversification")
    _require(diversification.get("strict_same_bond", True) is True, "Molecular export requires strict same-bond rewiring")
    _require(diversification.get("preserve_neighbor_atom_type_by_bond_type", True) is True, "Molecular export requires preserved atom/bond environments")
    _require(diversification.get("verify_bond_valence_preservation", True) is True, "Molecular export requires preserved bond valences")
    connected_sources = 0
    for source, graph, sample_id in zip(before, after, ids):
        _require(set(source) == set(graph), f"Molecular indexed nodes differ: {sample_id}")
        _require(_degrees(source) == _degrees(graph), f"Molecular indexed degrees differ: {sample_id}")
        for node in source:
            _require(node_attribute in source.nodes[node] and node_attribute in graph.nodes[node], f"Missing atom attribute at {sample_id}:{node}")
            _require(_plain(dict(source.nodes[node])) == _plain(dict(graph.nodes[node])), f"Molecular indexed atom attributes differ: {sample_id}:{node}")
            if node_types is not None:
                _require(source.nodes[node][node_attribute] in node_types, f"Unsupported configured atom category: {sample_id}:{node}")
        for name, population in (("T", source), ("TR", graph)):
            for u, v, data in population.edges(data=True):
                _require(edge_attribute in data, f"Missing bond label in {name}:{sample_id}:{u}-{v}")
                _require(data[edge_attribute] in edge_types, f"Unsupported bond label in {name}:{sample_id}:{u}-{v}")
        source_counts = Counter(data[edge_attribute] for _, _, data in source.edges(data=True))
        actual_counts = Counter(data[edge_attribute] for _, _, data in graph.edges(data=True))
        _require(source_counts == actual_counts, f"Molecular bond-label counts differ: {sample_id}")
        invariant_args = {"edge_types": edge_types, "node_attribute": node_attribute, "edge_attribute": edge_attribute}
        _require(
            extract_typed_invariant(source, **invariant_args).signatures
            == extract_typed_invariant(graph, **invariant_args).signatures,
            f"Molecular indexed typed degrees differ: {sample_id}",
        )
        environment_args = {"node_attribute": node_attribute, "edge_attribute": edge_attribute}
        _require(
            _neighbor_atom_bond_signature(source, **environment_args)
            == _neighbor_atom_bond_signature(graph, **environment_args),
            f"Molecular neighbor atom/bond signatures differ: {sample_id}",
        )
        _require(
            _bond_valence_vectors_equal(
                _weighted_bond_valence_vector(source, edge_attribute=edge_attribute),
                _weighted_bond_valence_vector(graph, edge_attribute=edge_attribute),
            ),
            f"Molecular indexed bond valences differ: {sample_id}",
        )
        if _connected(source):
            connected_sources += 1
            _require(not diversification.get("preserve_connectivity", True) or _connected(graph), f"Molecular rewiring disconnected a connected source: {sample_id}")
        # The producer copies this endpoint provenance through diversification.
        # Its equality also detects accidental cross-index pairing when present.
        if "attribute_diffusion" in source.graph or "attribute_diffusion" in graph.graph:
            _require(_plain(source.graph.get("attribute_diffusion")) == _plain(graph.graph.get("attribute_diffusion")), f"Molecular endpoint provenance/order differs: {sample_id}")
    return {
        "num_checked": len(ids), "indexed_nodes_preserved": True,
        "indexed_atom_attributes_preserved": True, "indexed_degrees_preserved": True,
        "indexed_typed_degrees_preserved": True, "bond_type_counts_preserved": True,
        "neighbor_atom_bond_signatures_preserved": True, "indexed_bond_valences_preserved": True,
        "connected_sources_checked": connected_sources,
        "connectivity_preserved_for_connected_sources": bool(diversification.get("preserve_connectivity", True)),
        "chemical_validity_filter_applied": False,
        "decode_constraint_mode": attributed.get("decode", {}).get("constraint_mode"),
    }


def _branch_provenance(provenance, source):
    return {**provenance, "source_sha256": source["sha256"]}


def _existing(directory, provenance, ids):
    common = _json(directory / COMMON)
    _require(common == _json(directory / "ablations" / INDEX), "Molecular common indexes differ")
    _require(common.get("format") == FORMAT, "Unknown molecular export format")
    _require(common.get("provenance") == provenance, "Molecular export provenance differs")
    _require(common.get("sample_ids") == ids, "Molecular sample ID/order differs")
    _require(common.get("generation_seed") == provenance["seed"] and common.get("num_requested") == len(ids), "Molecular export seed/count differs")
    _require(set(common.get("branches", {})) == set(SOURCES), "Molecular branch set differs")
    for variant, key in SOURCES.items():
        child_dir = directory / "ablations" / variant
        child = _json(child_dir / "manifest.json")
        _require(common["branches"][variant] == child, f"Molecular branch/index mismatch: {variant}")
        source = provenance["source_artifacts"][key]
        _require(child.get("generation_seed") == provenance["seed"], f"Molecular branch seed differs: {variant}")
        for count in ("requested_count", "completed_count"):
            _require(child.get(count) == len(ids), f"Molecular branch {count} differs: {variant}")
        _require(child.get("failed_count") == 0 and child.get("failures") == [] and child.get("status") == "complete", f"Molecular branch is incomplete: {variant}")
        _require(child.get("sample_ids") == ids and child.get("requested_sample_ids") == ids, f"Molecular branch ID/order differs: {variant}")
        _require(child.get("identity_source") == IDENTITY and child.get("source_artifact") == source, f"Molecular branch source identity differs: {variant}")
        _require(child.get("provenance") == _branch_provenance(provenance, source), f"Molecular branch provenance differs: {variant}")
        _require(child.get("shared_context") == provenance, f"Molecular shared context differs: {variant}")
        path = _record(child_dir, child.get("base_graphs"), f"{variant}.base_graphs")
        _require(child["base_graphs"]["sha256"] == source["sha256"], f"Molecular branch differs from saved population: {variant}")
        _population(load_trusted_networkx_pickle(path), ids, variant)
    return directory / COMMON


def export_molecular(directory: Path, cfg: dict, context: dict) -> Path:
    """Publish/reuse T and TR views after the caller verifies original intent.

    The caller holds its generation lock. Parent graph lists and generation
    manifests remain byte-identical; incomplete existing exports are rejected.
    """
    directory = Path(directory)
    _require(directory.is_dir(), f"Missing molecular generation directory: {directory}")
    manifest_path = directory / "manifest.json"
    manifest_sha = _sha(manifest_path)
    generation = _json(manifest_path)
    seed, count = context.get("seed"), context.get("num_samples")
    _require(type(seed) is int and seed >= 0, "Invalid molecular generation seed context")
    _require(type(count) is int and count > 0, "Invalid molecular requested count context")
    for key, expected in (
        ("model_id", "grapher"), ("run_id", context.get("run_id")),
        ("generation_id", directory.name), ("generation_seed", seed),
        ("num_requested", count), ("num_generated", count), ("variant", cfg.get("variant")),
    ):
        _require(generation.get(key) == expected, f"Molecular generation {key} differs")
    _require(context.get("generation_id", directory.name) == directory.name, "Molecular generation context ID differs")
    _require(bool(context.get("checkpoint_sha256")) and generation.get("checkpoint", {}).get("sha256") == context["checkpoint_sha256"], "Molecular generation checkpoint differs from verified context")
    ids = [f"{seed}:{index}" for index in range(count)]
    consumed, paths = {}, {}
    for key in (*SOURCES.values(), "sampled_basis_indices"):
        paths[key] = _record(directory, generation.get(key), key)
        consumed[key] = copy.deepcopy(generation[key])
    populations = {
        variant: _population(load_trusted_networkx_pickle(paths[key]), ids, key)
        for variant, key in SOURCES.items()
    }
    indices = load_trusted_networkx_pickle(paths["sampled_basis_indices"])
    _require(isinstance(indices, list) and len(indices) == count, "Molecular sampled-basis count/order differs")
    _require(all(isinstance(index, (int, np.integer)) and not isinstance(index, bool) and index >= 0 for index in indices), "Invalid molecular sampled-basis index")
    invariant_diagnostics = _verify_pairs(populations["T"], populations["TR"], ids, cfg)
    provenance = {
        **copy.deepcopy(context), "sample_config": copy.deepcopy(cfg.get("sample", {})),
        "sde": copy.deepcopy(cfg.get("sde", {})),
        "generation_batch_size": cfg.get("generation_batch_size"),
        "source_generation_manifest_sha256": manifest_sha,
        "identity_scheme": IDENTITY, "source_artifacts": consumed,
    }
    if (directory / COMMON).exists() or (directory / "ablations" / INDEX).exists():
        return _existing(directory, provenance, ids)
    report_root = directory / "ablations"
    had_reports = report_root.exists()
    report_snapshot = _report_snapshot(report_root)
    started = time.monotonic()
    staging = Path(tempfile.mkdtemp(prefix=".molecular_export_", dir=directory))
    _require(staging.resolve().is_relative_to(directory.resolve()), "Molecular staging escapes generation")
    staged_root, backup = staging / "ablations", staging / "previous_ablations"
    installed = published = False
    try:
        if had_reports:
            shutil.copytree(report_root, staged_root)
        else:
            staged_root.mkdir()
        branches = {}
        for variant, key in SOURCES.items():
            child_dir = staged_root / variant
            child_dir.mkdir(exist_ok=True)
            graph_path = child_dir / "base_graphs.pkl"
            shutil.copyfile(paths[key], graph_path)
            _require(_sha(graph_path) == consumed[key]["sha256"], f"Molecular population changed while copying: {variant}")
            child = {
                "format": FORMAT, "variant": variant, "label": variant,
                "status": "complete", "generation_seed": seed,
                "requested_count": count, "completed_count": count, "failed_count": 0,
                "failures": [], "sample_ids": ids, "requested_sample_ids": ids,
                "base_graphs": {"path": "base_graphs.pkl", "sha256": _sha(graph_path)},
                "identity_source": IDENTITY, "source_artifact": consumed[key],
                "provenance": _branch_provenance(provenance, consumed[key]),
                "shared_context": provenance, "posthoc_repair": False,
                "degree_source": "saved_attributed_endpoint",
                "timing_scope": "saved population byte-copy and invariant verification only; no sampling or rewiring",
            }
            _write(child_dir / "manifest.json", child)
            branches[variant] = child
        common = {
            "format": FORMAT, "generation_seed": seed, "num_requested": count,
            "sample_ids": ids, "branches": branches, "provenance": provenance,
            "shared_context": provenance, "invariants": invariant_diagnostics,
            "time_accounting": {
                "parent_generation_seconds": generation.get("duration_seconds"),
                "export_seconds": time.monotonic() - started,
                "model_sampling_rerun": False, "rewiring_rerun": False,
            },
        }
        _write(staged_root / INDEX, common)
        _write(staging / COMMON, common)
        _require(_sha(manifest_path) == manifest_sha, "Molecular generation manifest changed during export")
        for key, record in consumed.items():
            _record(directory, record, key)
        _require(report_root.exists() == had_reports and _report_snapshot(report_root) == report_snapshot, "Existing molecular evaluation reports changed during export")
        _require(not (directory / COMMON).exists(), "Molecular index appeared during export")
        if had_reports:
            report_root.rename(backup)
        try:
            staged_root.rename(report_root)
            installed = True
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
            print(f"Original molecular evaluation backup retained at {backup}", file=sys.stderr)
        else:
            shutil.rmtree(staging)
