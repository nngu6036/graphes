#!/usr/bin/env python3
"""Audit and analyze saved GraphES paper experiments without fitting a model.

Only load project-owned, trusted graph pickles.  ``audit`` and ``summarize``
never invent missing measurements.  ``coverage`` uses the existing evaluator's
WL/component pseudodistance, with explicit, recorded thresholds.  ``plot``
requires hash-anchored populations and pairs by sample ID, including failures.
Run any subcommand with --help for its complete interface.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import summarize_grapher_ablation as saved

FORMAT = "graphes_paper_evidence_v1"
DEFAULT_VARIANTS = ("T", "D", "TR", "DR")
RUNTIME_FIELDS = ("basis_reverse_seconds", "degree_projection_seconds",
                  "topology_decode_seconds", "attributes_seconds", "rewiring_seconds",
                  "end_to_end_seconds", "peak_cpu_memory_bytes", "peak_gpu_memory_bytes")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")
    return value


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(saved._clean(value), indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True)
                             if isinstance(value, (list, dict)) else value
                             for key, value in saved._clean(row).items()})


def nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def resolve_runs(args) -> tuple[list[dict], list[int], list[str]]:
    """--run supports independently trained branches without moving artifacts."""
    entries: dict[tuple[int, str], Path] = {}
    variants = list(args.variants)
    for item in args.seed_dir:
        seed_text, separator, path_text = item.partition("=")
        if not separator or not path_text:
            raise ValueError("--seed-dir must be SEED=GEN/ablations")
        for variant in variants:
            key = (int(seed_text), variant)
            if key in entries:
                raise ValueError(f"Duplicate seed/variant: {key}")
            entries[key] = Path(path_text).resolve() / variant
    for item in args.run:
        label, separator, path_text = item.partition("=")
        seed_text, comma, variant = label.partition(",")
        if not separator or not comma or not path_text or not variant:
            raise ValueError("--run must be SEED,VARIANT=BRANCH_DIR")
        key = (int(seed_text), variant)
        if key in entries:
            raise ValueError(f"Duplicate seed/variant: {key}")
        entries[key] = Path(path_text).resolve()
        if variant not in variants:
            variants.append(variant)
    if not entries:
        raise ValueError("At least one --seed-dir or --run is required")
    records = []
    for (seed, variant), branch in sorted(entries.items()):
        record = saved.load_run(seed, branch.parent, branch.name, args.evaluation_subdir)
        record["variant"] = variant
        records.append(record)
    return records, sorted({seed for seed, _ in entries}), variants


def branch_metadata(branch: Path) -> tuple[dict, dict, Path | None]:
    common_path = next((path for path in (
        branch.parent / "ablation_manifest.json",
        branch.parent.parent / "matched_ablation_manifest.json",
        branch.parent.parent / "molecular_ablation_manifest.json",
    ) if path.is_file()), None)
    common = read_json(common_path) if common_path else {}
    metadata = dict(common.get("branches", {}).get(branch.name, {}))
    if (branch / "manifest.json").is_file():
        metadata.update(read_json(branch / "manifest.json"))
    return metadata, common, common_path


def declared_artifact(directory: Path, record: Any) -> Path:
    if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
        raise ValueError(f"Missing path/hash for an artifact under {directory}")
    path = (directory / record["path"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f"Artifact escapes directory: {path}")
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError(f"Missing or hash-mismatched artifact: {path}")
    return path


def load_graphs(path: Path) -> list:
    import networkx as nx
    from grapher.utils.networkx_pickle import load_trusted_networkx_pickle
    graphs = load_trusted_networkx_pickle(path)
    if not isinstance(graphs, (list, tuple)) or not all(isinstance(g, nx.Graph) for g in graphs):
        raise ValueError(f"Expected a list of NetworkX graphs: {path}")
    return list(graphs)


def checked_population(branch: Path) -> tuple[dict[str, Any], dict, dict, Path]:
    """No positional pairing unless the supplied exporter anchored source bytes."""
    metadata, common, _ = branch_metadata(branch)
    shared_branch = common.get("branches", {}).get(branch.name)
    if isinstance(shared_branch, dict):
        for key, value in shared_branch.items():
            if key != "manifest_path" and metadata.get(key) != value:
                raise ValueError(f"Branch/common-index disagreement: {branch}:{key}")
    graph_path = declared_artifact(branch, metadata.get("base_graphs"))
    graphs = load_graphs(graph_path)
    ids = metadata.get("sample_ids")
    if not isinstance(ids, list) or len(ids) != len(graphs) or len(set(ids)) != len(ids):
        raise ValueError(f"Invalid successful sample IDs/count: {branch}")
    requested = metadata.get("requested_sample_ids", common.get("sample_ids"))
    if not isinstance(requested, list) or len(set(requested)) != len(requested):
        raise ValueError(f"Missing/duplicate requested sample IDs: {branch}")
    if any(not isinstance(i, str) for i in requested + ids):
        raise ValueError(f"Sample IDs must be strings: {branch}")
    if ids != [i for i in requested if i in set(ids)]:
        raise ValueError(f"Unknown/reordered successful sample IDs: {branch}")
    if metadata.get("completed_count") != len(ids) or metadata.get("requested_count") != len(requested):
        raise ValueError(f"Manifest counts disagree with sample IDs: {branch}")
    missing = [i for i in requested if i not in set(ids)]
    failures = metadata.get("failures", [])
    if (metadata.get("failed_count") != len(missing)
            or [r.get("sample_id") for r in failures] != missing):
        raise ValueError(f"Missing explicit failure records: {branch}")
    expected_status = "complete" if not missing else ("partial" if ids else "failed")
    if metadata.get("status") != expected_status:
        raise ValueError(f"Branch status disagrees with retained/failed IDs: {branch}")
    positional = metadata.get("identity_source") == "hash_anchored_parent_list_order"
    source = metadata.get("source_artifact", {})
    if positional and source.get("sha256") != metadata["base_graphs"]["sha256"]:
        raise ValueError(f"Positional IDs are not anchored to identical source bytes: {branch}")
    for sample_id, graph in zip(ids, graphs):
        actual = graph.graph.get("sample_id")
        if actual != sample_id and not (actual is None and positional):
            raise ValueError(f"Embedded graph ID differs: {branch}:{sample_id}: {actual!r}")
    return dict(zip(ids, graphs)), metadata, common, graph_path


def _flatten(value: Any, prefix="") -> dict:
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(item, name))
    elif isinstance(value, (str, bool, int, float)) or value is None:
        result[prefix] = value
    return result


def aggregate_diagnostic_rows(rows: Sequence[Mapping]) -> dict:
    """Denominators count measured entries, never absent booleans as False."""
    flat = [_flatten(row) for row in rows]
    result = {}
    for field in sorted({key for row in flat for key in row}):
        if field.endswith(("graph_index", "sample_id", "seed")):
            continue
        values = [row[field] for row in flat if field in row and row[field] is not None]
        if not values:
            continue
        if all(isinstance(value, bool) for value in values):
            result[field] = {"n_measured": len(values), "n_missing": len(rows)-len(values),
                             "true_count": sum(values), "rate": sum(values)/len(values)}
        elif all(saved._finite(value) is not None for value in values):
            result[field] = {"n_measured": len(values), "n_missing": len(rows)-len(values),
                             "mean": statistics.fmean(values), "sum": sum(values),
                             "std": statistics.stdev(values) if len(values)>1 else None,
                             "min": min(values), "max": max(values)}
        elif all(isinstance(value, str) for value in values):
            result[field] = {"n_measured": len(values), "n_missing": len(rows)-len(values),
                             "counts": dict(Counter(values))}
    return result


def _generation_manifest(branch: Path) -> tuple[dict, Path | None]:
    for directory in list(branch.parents)[:4]:
        path = directory / "manifest.json"
        if path.is_file():
            value = read_json(path)
            if "duration_seconds" in value or "sampling" in value or "effective_sample_config" in value:
                return value, path
    return {}, None


def diagnostics_for(branch: Path, metadata: dict, generation: dict, generation_path: Path | None,
                    variant: str) -> tuple[list[dict], dict]:
    sources = []
    candidates = []
    if metadata.get("diagnostics"):
        candidates.append((branch, metadata["diagnostics"], "branch"))
    if generation_path:
        keys = []
        if variant in {"D", "DR"}:
            keys.append("degree_constrained_realization_diagnostics")
        if variant == "TR" and generation.get("attributed_diversification_diagnostics"):
            keys.append("attributed_diversification_diagnostics")
        for key in keys:
            candidates.append((generation_path.parent, generation[key], key)) if key in generation else None
    summaries = {}
    for directory, record, role in candidates:
        path = declared_artifact(directory, record)
        if path.suffix != ".json":
            continue
        payload = read_json(path)
        rows = payload.get("per_graph", payload.get("per_sample", []))
        if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
            summaries[role] = {"per_sample_count": len(rows), "fields": aggregate_diagnostic_rows(rows),
                               "failures": payload.get("failures", []),
                               "producer_aggregate": payload.get("aggregate")}
            sources.append({"role": role, "path": str(path), "sha256": sha256(path)})
    return sources, summaries


def invariant_pair(before: dict, after: dict, *, node_attribute: str, edge_attribute: str) -> dict:
    import networkx as nx
    common_ids = [i for i in before if i in after]
    rows = []
    for sample_id in common_ids:
        a, b = before[sample_id], after[sample_id]
        if a.is_directed() or b.is_directed() or a.is_multigraph() or b.is_multigraph():
            raise ValueError("Invariant checks require simple undirected graphs")
        same_nodes = set(a) == set(b)
        connected_a = len(a)>0 and nx.is_connected(a)
        connected_b = len(b)>0 and nx.is_connected(b)
        edges_a = {frozenset(edge) for edge in a.edges()}
        edges_b = {frozenset(edge) for edge in b.edges()}
        row = {"sample_id": sample_id, "indexed_nodes_preserved": same_nodes,
               "indexed_degrees_preserved": same_nodes and dict(a.degree()) == dict(b.degree()),
               "connected_source": connected_a, "connected_output": connected_b,
               "connectivity_preserved_for_connected_sources": connected_b if connected_a else None,
               "unchanged_indexed_topology": same_nodes and edges_a == edges_b,
               "nonisomorphic_topology": not nx.is_isomorphic(a, b),
               "normalized_edge_symmetric_difference": len(edges_a ^ edges_b)/max(len(edges_a),1)}
        typed = (same_nodes and all(node_attribute in g.nodes[n] for g in (a,b) for n in g)
                 and all(edge_attribute in data for g in (a,b) for _,_,data in g.edges(data=True)))
        row.update(typed_degrees_preserved=None, neighbor_atom_bond_signatures_preserved=None,
                   bond_valences_preserved=None, node_types_preserved=None)
        if typed:
            def td(g,n):
                return Counter(g.edges[n,v][edge_attribute] for v in g.neighbors(n))
            def signature(g,n):
                return Counter((g.nodes[v][node_attribute],g.edges[n,v][edge_attribute]) for v in g.neighbors(n))
            row["node_types_preserved"] = all(a.nodes[n][node_attribute]==b.nodes[n][node_attribute] for n in a)
            row["typed_degrees_preserved"] = all(td(a,n)==td(b,n) for n in a)
            row["neighbor_atom_bond_signatures_preserved"] = row["node_types_preserved"] and all(signature(a,n)==signature(b,n) for n in a)
            bonds = [data[edge_attribute] for g in (a,b) for _,_,data in g.edges(data=True)]
            if all(saved._finite(value) is not None for value in bonds):
                from grapher.rewiring_mlp.molecular.constraints import bond_order
                def valence(g,n):
                    return sum(bond_order(int(g.edges[n,v][edge_attribute])) for v in g.neighbors(n))
                row["bond_valences_preserved"] = all(math.isclose(valence(a,n), valence(b,n), abs_tol=1e-9) for n in a)
        rows.append(row)
    return {"num_pairs": len(rows), "before_count": len(before), "after_count": len(after),
            "missing_before_ids": [i for i in after if i not in before],
            "missing_after_ids": [i for i in before if i not in after],
            "definition": "Indexed invariants on matched successful IDs; connectivity denominator is connected source graphs; edge difference divided by max(source edge count,1); isomorphism ignores attributes; molecular bond valences use project bond_order (aromatic=1.5)",
            "fields": aggregate_diagnostic_rows(rows), "per_sample": rows}


def audit_runs(args) -> int:
    records, seeds, variants = resolve_runs(args)
    populations = {}
    audits = []
    for record in records:
        branch = Path(record["branch_dir"])
        issues = []
        sources = []
        out = {"seed": record["seed"], "variant": record["variant"], "branch_dir": str(branch),
               "evaluation_status": record["status"], "evaluation_reason": record.get("reason"),
               "report_path": record.get("report_path"), "report_sha256": record.get("report_sha256"),
               "protocol": record.get("protocol"), "protocol_fingerprint": record.get("protocol_fingerprint"),
               "counts": {"N_req": record["counts"].get("requested_count"), "N_att": None,
                          "N_out": record["counts"].get("completed_count"),
                          "N_evaluated": record["counts"].get("evaluated_count"),
                          "V": record["counts"].get("valid_count"), "U": record["counts"].get("unique_count"),
                          "W": record["counts"].get("novel_count")},
               "runtime": {key: None for key in RUNTIME_FIELDS},
               "metrics": record["metrics"], "artifact_integrity": "unverified"}
        try:
            metadata, common, common_path = branch_metadata(branch)
            generation, generation_path = _generation_manifest(branch)
            direct_generation = ("num_generated" in metadata or "num_graphs" in metadata) and "sample_ids" not in metadata
            if direct_generation:
                generation, generation_path = metadata, branch / "manifest.json"
            for path in (branch / "manifest.json", common_path, generation_path):
                if path and path.is_file():
                    sources.append({"path": str(path), "sha256": sha256(path)})
            out["sample_ids"] = metadata.get("sample_ids")
            out["requested_sample_ids"] = metadata.get("requested_sample_ids",common.get("sample_ids"))
            out["failures"] = metadata.get("failures")
            out["shared_context"] = metadata.get("shared_context", common.get("shared_context"))
            out["producer_invariants"] = common.get("invariants")
            out["training_provenance"] = generation.get("provenance", generation.get("training"))
            counts = out["counts"]
            if counts["N_req"] is None:
                counts["N_req"] = nonnegative_int(metadata.get("num_requested"))
            attempt_candidates = []
            for name, source in (("branch",metadata),("generation",generation),
                                 ("generation.sampling",generation.get("sampling",{}))):
                for key in ("attempted_count", "num_attempts", "N_att"):
                    value = nonnegative_int(source.get(key)) if isinstance(source,dict) else None
                    if value is not None:
                        attempt_candidates.append({"source": f"{name}.{key}", "value":value})
            out["attempt_evidence"] = attempt_candidates
            if len({row["value"] for row in attempt_candidates}) == 1:
                counts["N_att"] = attempt_candidates[0]["value"]
            elif attempt_candidates:
                issues.append("Conflicting recorded attempt counts; N_att remains null")
            if counts["N_att"] is None:
                issues.append("N_att was not recorded; requested/successful outputs do not establish total sampling attempts")
            for key in RUNTIME_FIELDS:
                values = [saved._finite(container.get(key)) for container in
                          (metadata, metadata.get("runtime",{}), generation.get("runtime",{}))
                          if isinstance(container,dict)]
                values = [value for value in values if value is not None]
                if values and len(set(values)) == 1:
                    out["runtime"][key] = values[0]
            out["runtime_evidence"] = {
                "branch_reported_stage_seconds": metadata.get("time_seconds"),
                "parent_generation_seconds": generation.get("duration_seconds"),
                "parent_scope": "complete parent generation pipeline, not any one derived branch",
                "time_accounting": metadata.get("time_accounting", common.get("time_accounting")),
                "timing_scope": metadata.get("timing_scope", common.get("timing_scope")),
                "clock_and_sync": generation.get("clock_and_sync", generation.get("runtime",{}).get("clock_and_sync"))
                    if isinstance(generation.get("runtime",{}),dict) else None,
            }
            if args.metadata_only:
                issues.append("Graph hashes, embedded sample IDs, and graph invariants not checked (--metadata-only)")
            else:
                if direct_generation:
                    graph_path = declared_artifact(branch, metadata.get("base_graphs"))
                    graphs = load_graphs(graph_path)
                    recorded_count = nonnegative_int(metadata.get("num_generated",metadata.get("num_graphs")))
                    if recorded_count != len(graphs):
                        raise ValueError("Direct generation count differs from hash-verified saved graphs")
                    out["artifact_integrity"] = "population_hash_verified_unmatched"
                    out["matching_note"] = "Direct generation population; no requested IDs inferred and no matched graph comparisons performed"
                else:
                    graphs, metadata, common, graph_path = checked_population(branch)
                    out["artifact_integrity"] = "branch_hash_and_ids_verified"
                    populations[record["seed"],record["variant"]] = graphs
                counts["N_out"] = len(graphs)
                sources.append({"path":str(graph_path),"sha256":sha256(graph_path)})
                if counts["N_evaluated"] is not None and counts["N_evaluated"] != len(graphs):
                    issues.append("Evaluated population differs from saved output count")
                if record.get("report_path"):
                    report = read_json(Path(record["report_path"]))
                    evaluated_path = report.get("generated_graphs",report.get("paths",{}).get("generated_source"))
                    out["evaluated_graph_path"] = evaluated_path
                    out["evaluated_graph_path_matches"] = bool(evaluated_path and Path(evaluated_path).resolve()==graph_path.resolve())
                    if not out["evaluated_graph_path_matches"]:
                        issues.append("Evaluation source path cannot be reconciled with this branch")
                    eval_hash = report.get("generated_graphs_sha256",report.get("source_files",{}).get("generated_graphs",{}).get("sha256"))
                    out["evaluated_graph_hash_matches"] = eval_hash == sha256(graph_path) if eval_hash else None
                    if eval_hash and not out["evaluated_graph_hash_matches"]:
                        issues.append("Evaluation source hash conflicts with saved branch")
                    elif eval_hash is None:
                        issues.append("Legacy report has no evaluated-graph SHA-256; path/count agreement alone does not prove historical population identity")
                    if report.get("failure_diagnostics") is not None:
                        out["molecular_conversion_failures"] = report["failure_diagnostics"]
            diag_sources, diag = diagnostics_for(branch,metadata,generation,generation_path,record["variant"])
            sources.extend(diag_sources)
            out["diagnostics"] = diag
            if (record.get("protocol") or {}).get("kind") == "molecular":
                consistency = {}
                for metric,numerator,denominator in (("validity_rate","V","N_evaluated"),
                                                      ("uniqueness_rate","U","V"),("novelty_rate","W","U")):
                    a,b = nonnegative_int(counts[numerator]),nonnegative_int(counts[denominator])
                    derived = a/b if a is not None and b is not None and b>0 else None
                    reported = saved._finite(record["metrics"].get(metric))
                    agrees = math.isclose(derived,reported,abs_tol=args.rate_tolerance) if derived is not None and reported is not None else None
                    consistency[metric] = {"reported":reported,"count_ratio":derived,"agrees":agrees}
                    if agrees is False:
                        issues.append(f"{metric} conflicts with {numerator}/{denominator}")
                out["molecular_rate_reconciliation"] = consistency
        except (OSError,ValueError,TypeError,KeyError,ImportError) as error:
            issues.append(f"Audit error: {type(error).__name__}: {error}")
            out["artifact_integrity"] = "error"
        out["sources"] = sources
        out["issues"] = issues
        out["missing_runtime_fields"] = [key for key,value in out["runtime"].items() if value is None]
        out["publication_ready"] = record["status"]=="complete" and not issues and not out["missing_runtime_fields"]
        audits.append(out)
    invariants = []
    for seed in seeds:
        for before, after in (("T","TR"),("D","DR")):
            if (seed,before) in populations and (seed,after) in populations:
                a = next(r for r in audits if r["seed"]==seed and r["variant"]==before)
                b = next(r for r in audits if r["seed"]==seed and r["variant"]==after)
                if not a.get("shared_context") or a["shared_context"]!=b.get("shared_context"):
                    invariants.append({"seed":seed,"comparison":f"{after}_vs_{before}","status":"unverified_shared_context"})
                    continue
                value = invariant_pair(populations[seed,before],populations[seed,after],
                                       node_attribute=args.node_attribute,edge_attribute=args.edge_attribute)
                invariants.append({"seed":seed,"comparison":f"{after}_vs_{before}","status":"checked",**value})
    payload = {"format":FORMAT,"kind":"population_and_diagnostic_audit","seeds":seeds,"variants":variants,
               "runs":audits,"invariant_pairs":invariants,
               "count_semantics":{"N_req":"requested outputs","N_att":"explicitly recorded full generation attempts, never inferred from N_req",
                                  "N_out":"saved outputs including invalid graphs","V":"valid evaluated outputs","U":"unique valid outputs for molecules; exact topology classes for generic reports",
                                  "W":"novel unique valid molecules; generic exact-novel graph count has different semantics"},
               "required_followup":[
                   "Re-evaluate mismatched/legacy populations from hash-verified saved branch files; preserve original reports and record selected run/checkpoint, graph hash, split hashes, and evaluation options in the paper manifest.",
                   "For null N_att, rerun generation with a counter incremented on every full sampling attempt, including rejected/restarted draws; retain the same declared output/attempt policy.",
                   "For null stage times, run an instrumented generation with synchronized device timing, explicit stage boundaries and cache policy; parent/export timings cannot be split retrospectively.",
                   "Peak memory and preprocessing/cache timings require a fresh instrumented run; report hardware, batch size, repetitions and warm/cold cache separately.",
                   "Saved generated-summary evaluators measure prediction/decoded-graph consistency; held-out log-gap/spectrum errors require held-out targets and a declared corruption/timestep protocol.",
               ]}
    write_json(args.output_dir / "paper_evidence_audit.json",payload)
    write_csv(args.output_dir / "populations.csv",[{"seed":r["seed"],"variant":r["variant"],**r["counts"],
              "evaluation_status":r["evaluation_status"],"artifact_integrity":r["artifact_integrity"],"issues":r["issues"]} for r in audits])
    print(f"Audited {len(audits)} saved runs; {sum(bool(r['issues']) for r in audits)} have unresolved evidence issues.")
    return 1 if args.require_complete and any(not r["publication_ready"] for r in audits) else 0


def pairing_reason(records: Sequence[dict], mode: str) -> str | None:
    if any(record["status"] not in {"complete","partial"} for record in records):
        return "missing_or_failed_report"
    if len({record.get("protocol_fingerprint") for record in records}) != 1:
        return "incompatible_evaluation_protocol"
    if mode == "seed":
        return None
    requested = [record.get("requested_sample_ids") for record in records]
    if any(not isinstance(ids,list) for ids in requested) or any(ids!=requested[0] for ids in requested):
        return "unverified_or_mismatched_requested_ids"
    contexts = [record.get("shared_context") for record in records]
    if not all(contexts) or any(context!=contexts[0] for context in contexts):
        return "unverified_or_mismatched_shared_context"
    return None


def summarize_runs(args) -> int:
    records,seeds,variants = resolve_runs(args)
    index = {(r["seed"],r["variant"]):r for r in records}
    metrics = sorted({name for r in records for name in r["metrics"]})
    protocols = {r.get("protocol_fingerprint") for r in records if r["status"] in {"complete","partial"}}
    blocked = len(protocols)>1
    summaries = []
    for variant in variants:
        for metric in metrics:
            values = {seed: saved._finite(index.get((seed,variant),{}).get("metrics",{}).get(metric))
                      if index.get((seed,variant),{}).get("status") in {"complete","partial"} and not blocked else None for seed in seeds}
            summaries.append({"variant":variant,"metric":metric,**saved.summarize_values(values,seeds)})
    contrast_specs = [(name,left,right) for name,left,right,_ in saved.DEFAULT_CONTRASTS if left in variants and right in variants]
    for item in args.contrast:
        name,separator,pair = item.partition("=")
        arms=pair.split(",")
        if not separator or len(arms)!=2 or len(set(arms))!=2 or any(arm not in variants for arm in arms):
            raise ValueError("--contrast must be NAME=LEFT,RIGHT using supplied variants")
        contrast_specs = [r for r in contrast_specs if r[0]!=name]+[(name,*arms)]
    definitions = [(name,{left:1,right:-1}) for name,left,right in contrast_specs]
    if all(v in variants for v in DEFAULT_VARIANTS):
        definitions.append(("decoder_rewiring_interaction",{"DR":1,"D":-1,"TR":-1,"T":1}))
    deltas=[]
    contrast_summaries=[]
    for name,weights in definitions:
        for metric in metrics:
            per_seed={}
            for seed in seeds:
                selected=[index.get((seed,variant)) for variant in weights]
                reason="missing_run" if any(r is None for r in selected) else pairing_reason(selected,args.pairing)
                values = {v:saved._finite(index.get((seed,v),{}).get("metrics",{}).get(metric)) for v in weights}
                if reason is None and any(v is None for v in values.values()):
                    reason="missing_metric"
                if reason is None and metric in saved.TIMING_METRICS:
                    scopes=[r.get("timing_scope") or r.get("time_accounting") for r in selected]
                    if not all(scopes) or any(s!=scopes[0] for s in scopes):
                        reason="incompatible_or_undeclared_timing_scope"
                difference=sum(weights[v]*values[v] for v in weights) if reason is None and not blocked else None
                per_seed[seed]=difference
                deltas.append({"contrast":name,"seed":seed,"metric":metric,"difference":difference,
                               "status":reason or ("incompatible_evaluation_protocol" if blocked else "complete"),
                               "values":values,"weights":weights,"pairing":args.pairing,
                               "output_counts":{v:index.get((seed,v),{}).get("counts",{}).get("evaluated_count") for v in weights},
                               "successful_output_ids_equal": all(r.get("sample_ids")==selected[0].get("sample_ids") for r in selected)
                                   if selected and all(r is not None and r.get("sample_ids") is not None for r in selected) else None})
            contrast_summaries.append({"contrast":name,"metric":metric,"weights":weights,
                                       **saved.summarize_values(per_seed,seeds)})
    payload={"format":FORMAT,"kind":"run_level_statistics","seeds":seeds,"variants":variants,
             "aggregation":"Equal-weight training seeds; sample SD ddof=1; no graph-level pseudo-replication; no zero imputation",
             "pairing":args.pairing,"pairing_definition":"matched requires identical requested IDs and nonempty identical shared_context; seed compares independent retraining runs by training seed only",
             "interaction":"(DR-D)-(TR-T), formed within each seed before averaging",
             "aggregation_blocked":blocked,"per_seed":records,"variant_summary":summaries,
             "paired_differences":deltas,"contrast_summary":contrast_summaries}
    write_json(args.output_dir / "paper_experiment_summary.json",payload)
    write_csv(args.output_dir / "means_and_sd.csv",summaries)
    write_csv(args.output_dir / "paired_contrasts.csv",contrast_summaries)
    write_csv(args.output_dir / "per_seed_differences.csv",deltas)
    print(f"Summarized {len(records)} runs; incompatible protocol aggregation blocked={blocked}.")
    incomplete=blocked or any(r["status"]!="complete" for r in records) or any(r["missing_seeds"] for r in summaries+contrast_summaries)
    return 1 if args.require_complete and incomplete else 0


def structural_profile(graph, iterations: int) -> dict:
    """Same deterministic colors and features as evaluate_graph_generation_report."""
    import networkx as nx
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("Paper coverage accepts only simple undirected graphs")
    def norm(counter):
        total=sum(counter.values())
        return {str(k):v/total for k,v in counter.items()} if total else {}
    component_counts=Counter()
    for component in nx.connected_components(graph):
        component_counts[len(component)]+=len(component)
    labels={node:f"d:{int(graph.degree(node))}" for node in graph}
    layers=[norm(Counter(labels.values()))]
    for _ in range(iterations):
        labels={node:hashlib.sha256("\x1f".join(["self",labels[node],"nbr",*sorted(labels[n] for n in graph.neighbors(node))]).encode()).hexdigest()[:24] for node in graph}
        layers.append(norm(Counter(labels.values())))
    return {"num_nodes":len(graph),"num_edges":graph.number_of_edges(),
            "component_histogram":norm(component_counts),"wl_layers":layers}


def structural_distance(a: dict,b: dict) -> float:
    def tv(p,q):
        return 0.5*sum(abs(p.get(k,0)-q.get(k,0)) for k in p.keys()|q.keys())
    if len(a["wl_layers"])!=len(b["wl_layers"]):
        raise ValueError("Profiles use different WL depths")
    terms=[abs(a[k]-b[k])/max(a[k],b[k],1) for k in ("num_nodes","num_edges")]
    terms.append(tv(a["component_histogram"],b["component_histogram"]))
    terms.extend(tv(p,q) for p,q in zip(a["wl_layers"],b["wl_layers"]))
    return sum(terms)/len(terms)


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered=sorted(values)
    x=q*(len(values)-1)
    lo,hi=math.floor(x),math.ceil(x)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(x-lo)


def compute_coverage(populations: dict[str,list], train_name: str, heldout_name: str,
                     epsilons: list[float], iterations: int) -> dict:
    profiles={name:[structural_profile(g,iterations) for g in graphs] for name,graphs in populations.items()}
    train,heldout=profiles[train_name],profiles[heldout_name]
    if not train or not heldout:
        raise ValueError("Training and held-out populations must be nonempty")
    rows,curves,nearest={},{},{}
    for name,query in profiles.items():
        distances=[min(structural_distance(g,h) for h in train) for g in query]
        coverage=[min(structural_distance(g,h) for h in query) for g in heldout] if query else []
        nearest[name]=distances
        rows[name]={"population":name,"num_graphs":len(query),
                    "mean_nearest_train_structural_distance":statistics.fmean(distances) if distances else None,
                    "median_nearest_train_structural_distance":quantile(distances,.5),
                    "q10_nearest_train_structural_distance":quantile(distances,.1),
                    "q90_nearest_train_structural_distance":quantile(distances,.9),
                    "mean_heldout_to_population_distance":statistics.fmean(coverage) if coverage else None}
        curves[name]=[{"population":name,"epsilon":e,
                       "epsilon_novelty_rate_against_train":sum(d>e for d in distances)/len(distances) if distances else None,
                       "heldout_coverage_rate":sum(d<=e for d in coverage)/len(heldout) if query else 0.0}
                      for e in epsilons]
    return {"population_metrics":list(rows.values()),"epsilon_curve":[r for group in curves.values() for r in group],
            "nearest_train_distances":nearest}


def coverage_command(args) -> int:
    if args.wl_iterations<0:
        raise ValueError("--wl-iterations must be nonnegative")
    epsilons=sorted(set(args.epsilons))
    if any(not math.isfinite(e) or not 0<=e<=1 for e in epsilons):
        raise ValueError("--epsilons must be finite values in [0,1]")
    paths={"train":args.train_graphs.resolve(),f"heldout_{args.reference_split}":args.heldout_graphs.resolve()}
    for item in args.population:
        name,separator,path=item.partition("=")
        if not separator or not name or name in paths:
            raise ValueError("--population must be a unique NAME=GRAPH_PICKLE")
        paths[name]=Path(path).resolve()
    populations={name:load_graphs(path) for name,path in paths.items()}
    result=compute_coverage(populations,"train",f"heldout_{args.reference_split}",epsilons,args.wl_iterations)
    result.update(format=FORMAT,kind="graded_novelty_and_coverage",protocol={
        "distance_name":"wl_component_structural_pseudodistance_v1","topology_only":True,
        "wl_iterations":args.wl_iterations,"epsilons":epsilons,"distance_range":[0,1],
        "distance_formula":"d_str=(relative |V| difference + relative |E| difference + component-size node-mass TV + sum_{h=0}^L WL-color-histogram TV)/(L+4)",
        "novelty_formula":"mean over generated graphs of 1[min training d_str > epsilon]",
        "coverage_formula":"mean over held-out graphs of 1[min generated d_str <= epsilon]",
        "reference_split":args.reference_split,"uses_entire_supplied_populations":True,
        "selection":"No filtering, truncation, or validity-based selection performed",
        "caveat":"This is a pseudodistance, not exact graph edit distance or an exact novelty test; nonisomorphic 1-WL-equivalent graphs may have distance zero.",
        "threshold_policy":"Explicit command-line thresholds; freeze using validation data before any test comparison",
        "empty_generated_population":"novelty/nearest-distance are null; coverage is mathematically zero against nonempty held-out population",
    },source_files={name:{"path":str(path),"sha256":sha256(path),"count":len(populations[name])} for name,path in paths.items()})
    write_json(args.output_dir / "graded_coverage_report.json",result)
    write_csv(args.output_dir / "graded_coverage_curve.csv",result["epsilon_curve"])
    write_csv(args.output_dir / "nearest_train_summary.csv",result["population_metrics"])
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout="constrained")
        for name in paths:
            if name=="train":
                continue
            rows=[r for r in result["epsilon_curve"] if r["population"]==name]
            for ax,key,title in zip(axes,("epsilon_novelty_rate_against_train","heldout_coverage_rate"),("Novelty against training","Held-out coverage")):
                ax.plot(epsilons,[r[key] for r in rows],marker="o",label=name)
                ax.set(xlabel="Structural distance threshold ε",ylabel="Fraction",title=title,ylim=(-.02,1.02))
                ax.grid(alpha=.2)
        axes[1].legend()
        fig.savefig(args.output_dir / "graded_coverage.pdf",bbox_inches="tight")
        fig.savefig(args.output_dir / "graded_coverage.png",dpi=180,bbox_inches="tight")
        plt.close(fig)
    print(f"Measured coverage for {len(paths)-2} generated populations; complete hash/protocol manifest saved.")
    return 0


def plot_command(args) -> int:
    import networkx as nx
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    populations,metadatas,sources={},{},{}
    for variant in DEFAULT_VARIANTS:
        graphs,metadata,common,path=checked_population(args.ablation_dir.resolve()/variant)
        populations[variant]=graphs
        metadatas[variant]=metadata
        sources[variant]={"path":str(path),"sha256":sha256(path),"manifest_sha256":sha256(args.ablation_dir/variant/"manifest.json")}
    requested=[metadatas[v].get("requested_sample_ids") for v in DEFAULT_VARIANTS]
    contexts=[metadatas[v].get("shared_context") for v in DEFAULT_VARIANTS]
    if not all(requested) or any(ids!=requested[0] for ids in requested):
        raise ValueError("All four variants must share the same nonempty requested sample IDs")
    if not all(contexts) or any(context!=contexts[0] for context in contexts):
        raise ValueError("All four variants must share the same nonempty generation context")
    ids=args.sample_ids or requested[0][:args.num_samples]
    if len(set(ids))!=len(ids) or any(i not in requested[0] for i in ids):
        raise ValueError("Requested gallery IDs are duplicate or outside the matched requested population")
    fig,axes=plt.subplots(len(ids),4,figsize=(11,2.6*len(ids)),squeeze=False,layout="constrained")
    gallery=[]
    for row_index,sample_id in enumerate(ids):
        union=nx.Graph()
        expected_nodes=None
        for variant in DEFAULT_VARIANTS:
            graph=populations[variant].get(sample_id)
            if graph is None:
                continue
            if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
                raise ValueError("Generic gallery requires simple undirected loop-free graphs")
            if expected_nodes is not None and set(graph)!=expected_nodes:
                raise ValueError(f"Indexed nodes differ across matched variants: {sample_id}")
            expected_nodes=set(graph)
            union.add_nodes_from(graph.nodes())
            union.add_edges_from(graph.edges())
        positions=nx.spring_layout(union,seed=args.layout_seed)
        entry={"sample_id":sample_id,"coordinates":[{"node":str(n),"x":float(p[0]),"y":float(p[1])} for n,p in positions.items()],"variants":{}}
        for column,variant in enumerate(DEFAULT_VARIANTS):
            ax=axes[row_index,column]
            ax.axis("off")
            graph=populations[variant].get(sample_id)
            if graph is None:
                failure=next((r for r in metadatas[variant].get("failures",[]) if r.get("sample_id")==sample_id),{})
                ax.text(.5,.5,"FAILED / NO OUTPUT",ha="center",va="center",transform=ax.transAxes,color="#a22")
                entry["variants"][variant]={"status":"failed","failure":failure}
                ax.set_title(f"{variant} · ID {sample_id}")
                continue
            connected=len(graph)>0 and nx.is_connected(graph)
            nx.draw_networkx(graph,pos=positions,ax=ax,with_labels=args.node_labels,node_size=35,font_size=6,
                             width=.65,node_color="#2478a8",edge_color="#7b8792")
            ax.set(xlim=(-1.15,1.15),ylim=(-1.15,1.15))
            ax.set_title(f"{variant} · ID {sample_id}\nn={len(graph)}, m={graph.number_of_edges()}, connected={connected}",fontsize=9)
            entry["variants"][variant]={"status":"saved_output","n":len(graph),"m":graph.number_of_edges(),"connected":connected}
        gallery.append(entry)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output_dir / "matched_T_D_TR_DR.pdf",bbox_inches="tight")
    fig.savefig(args.output_dir / "matched_T_D_TR_DR.png",dpi=args.dpi,bbox_inches="tight")
    plt.close(fig)
    write_json(args.output_dir / "matched_gallery_manifest.json",{
        "format":FORMAT,"kind":"matched_generic_gallery","variants":DEFAULT_VARIANTS,"samples":gallery,
        "selection":"explicit IDs" if args.sample_ids else "first requested IDs in manifest order, retaining failures",
        "layout":"one spring layout on the union of four indexed graphs for each sample; fixed axis limits in every panel",
        "layout_seed":args.layout_seed,"source_files":sources,"shared_context":contexts[0],
        "trajectory_status":"Endpoint comparison only; no unrecorded intermediate trajectory is fabricated"})
    print(f"Saved matched four-variant gallery for {len(ids)} requested IDs.")
    return 0


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    for name,handler in (("audit",audit_runs),("summarize",summarize_runs)):
        sub=commands.add_parser(name)
        sub.add_argument("--seed-dir",action="append",default=[],metavar="SEED=GEN/ablations")
        sub.add_argument("--run",action="append",default=[],metavar="SEED,VARIANT=BRANCH_DIR")
        sub.add_argument("--variants",nargs="+",default=list(DEFAULT_VARIANTS))
        sub.add_argument("--evaluation-subdir",default="evaluation_test")
        sub.add_argument("--output-dir",type=Path,required=True)
        sub.add_argument("--require-complete",action="store_true")
        sub.set_defaults(handler=handler)
    audit=commands.choices["audit"]
    audit.add_argument("--metadata-only",action="store_true")
    audit.add_argument("--node-attribute",default="atomic_num")
    audit.add_argument("--edge-attribute",default="bond_type")
    audit.add_argument("--rate-tolerance",type=float,default=1e-6)
    summary=commands.choices["summarize"]
    summary.add_argument("--contrast",action="append",default=[],metavar="NAME=LEFT,RIGHT")
    summary.add_argument("--pairing",choices=("matched","seed"),default="matched",
                         help="seed: paired training seeds only; matched: additionally require identical requested IDs/shared context")
    coverage=commands.add_parser("coverage")
    coverage.add_argument("--train-graphs",type=Path,required=True)
    coverage.add_argument("--heldout-graphs",type=Path,required=True)
    coverage.add_argument("--reference-split",choices=("val","test"),required=True)
    coverage.add_argument("--population",action="append",required=True,metavar="NAME=GRAPH_PICKLE")
    coverage.add_argument("--epsilons",nargs="+",type=float,required=True)
    coverage.add_argument("--wl-iterations",type=int,required=True)
    coverage.add_argument("--plot",action="store_true")
    coverage.add_argument("--output-dir",type=Path,required=True)
    coverage.set_defaults(handler=coverage_command)
    plot=commands.add_parser("plot")
    plot.add_argument("--ablation-dir",type=Path,required=True)
    plot.add_argument("--sample-ids",nargs="+")
    plot.add_argument("--num-samples",type=int,default=4)
    plot.add_argument("--layout-seed",type=int,default=42)
    plot.add_argument("--node-labels",action="store_true")
    plot.add_argument("--dpi",type=int,default=180)
    plot.add_argument("--output-dir",type=Path,required=True)
    plot.set_defaults(handler=plot_command)
    args=parser.parse_args(argv)
    try:
        if getattr(args,"num_samples",1)<1:
            raise ValueError("--num-samples must be positive")
        if hasattr(args,"rate_tolerance") and (not math.isfinite(args.rate_tolerance) or args.rate_tolerance<0):
            raise ValueError("--rate-tolerance must be finite and nonnegative")
        return args.handler(args)
    except (OSError,ValueError,TypeError,ImportError,KeyError) as error:
        parser.exit(2,f"{type(error).__name__}: {error}\n")


if __name__=="__main__":
    raise SystemExit(main())
