#!/usr/bin/env python3
"""Summarize saved GraphES ablations with per-seed paired differences.

No graphs are loaded and no metric is recomputed here. Each --seed-dir is a
GEN/ablations directory containing branch/evaluation_test (or evaluation_val).
Missing/failed runs and unavailable measurements remain explicit nulls. Generic
novelty and uniqueness are read from exact-isomorphism diagnostics, never
silently substituted by the evaluator's legacy WL rates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

GENERIC_METRICS = (
    "degree_mmd", "clustering_mmd", "orbit_mmd", "spectral_mmd", "graphlet_mmd",
    "validity_rate", "novelty_exact_rate", "uniqueness_exact_rate", "joint_vun_rate",
    "mean_nearest_train_structural_distance",
)
MOLECULAR_METRICS = (
    "validity_rate", "uniqueness_rate", "novelty_rate", "nspdk_mmd", "fcd", "qed_mean",
)
TIMING_METRICS = {"reported_generation_time_seconds", "end_to_end_seconds", "seconds_per_requested_output"}
DEFAULT_CONTRASTS = (
    ("D_minus_T", "D", "T", "decoder"),
    ("TR_minus_T", "TR", "T", "rewiring"),
    ("DR_minus_D", "DR", "D", "rewiring"),
    ("DR_minus_TR", "DR", "TR", "decoder_after_rewiring"),
    ("D_minus_D0", "D", "D0", "gumbel"),
    ("R1_minus_D", "R1", "D", "budget"),
    ("R12_minus_D", "R12", "D", "budget"),
    ("S0_minus_D", "S0", "D", "training"),
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _clean(value: Any) -> Any:
    """Keep saved JSON standards-compliant even when evaluators emit NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _digest(value: Any) -> str:
    payload = json.dumps(_clean(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _population_row(container: Any, stage: str) -> dict[str, Any]:
    if not isinstance(container, list):
        return {}
    found = [row for row in container if isinstance(row, dict) and row.get("population") == stage]
    if len(found) > 1:
        raise ValueError(f"Duplicate population diagnostic rows for {stage!r}")
    return dict(found[0]) if found else {}


def _generic_metrics(report: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    stage = report.get("generated_stage")
    split = report.get("reference_split")
    if not isinstance(stage, str) or not isinstance(split, str):
        raise ValueError("Generic report lacks generated_stage/reference_split")
    target = f"{stage}_to_{split}"
    rows = report.get("metrics", [])
    matches = [row for row in rows if isinstance(row, dict) and row.get("comparison") == target]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one generated comparison row {target!r}")
    row = matches[0]
    metrics = {key: _finite(row.get(key)) for key in GENERIC_METRICS}
    metrics["graphlet_mmd"] = _finite(row.get("graphlet_history_mmd"))
    metrics["novelty_wl_rate"] = _finite(row.get("novelty_rate"))
    metrics["uniqueness_wl_rate"] = _finite(row.get("uniqueness_rate"))
    pct = _finite(row.get("joint_vun_percent"))
    metrics["joint_vun_rate"] = None if pct is None else pct / 100.0
    exact_container = _mapping(report.get("generic_exact_isomorphism_diagnostics"))
    diversity_container = _mapping(report.get("generic_diversity_diagnostics"))
    exact = _population_row(exact_container.get("population_metrics"), stage)
    diversity = _population_row(diversity_container.get("population_metrics"), stage)
    source = {**exact, **diversity}
    metrics["novelty_exact_rate"] = _finite(source.get("exact_isomorphism_novelty_rate_against_train"))
    metrics["uniqueness_exact_rate"] = _finite(source.get("exact_isomorphism_uniqueness_rate"))
    for name in (
        "mean_nearest_train_structural_distance", "median_nearest_train_structural_distance",
        "q10_nearest_train_structural_distance", "q90_nearest_train_structural_distance",
        "mean_heldout_to_population_distance", "train_isomorphism_class_coverage_rate",
    ):
        metrics[name] = _finite(source.get(name))
    for curve in diversity_container.get("epsilon_curve", []):
        if not isinstance(curve, dict) or curve.get("population") != stage:
            continue
        epsilon = _finite(curve.get("epsilon"))
        if epsilon is None:
            continue
        suffix = format(epsilon, ".12g")
        metrics[f"heldout_coverage@{suffix}"] = _finite(curve.get("heldout_coverage_rate"))
        metrics[f"epsilon_novelty@{suffix}"] = _finite(curve.get("epsilon_novelty_rate_against_train"))
    counts = {
        "evaluated_count": report.get("num_generated_graphs_evaluated", report.get("num_graphs_evaluated")),
        "reference_count": report.get("num_reference_graphs"),
        "valid_count": source.get("num_valid_graphs"),
        "unique_count": source.get("num_exact_isomorphism_classes"),
        "novel_count": source.get("num_exact_novel_graphs_against_train"),
    }
    provenance = _mapping(report.get("dataset_provenance"))
    protocol = {
        "kind": "generic", "reference_split": split,
        "reference_split_sha256": report.get("reference_split_sha256"),
        "reference_graph_indices_zero_based": report.get("reference_graph_indices_zero_based"),
        "split_sha256": provenance.get("split_sha256"),
        **{k: report.get(k) for k in (
            "generic_mmd_protocol", "generic_clustering_bins", "generic_graphlet_options",
            "generic_spectral_descriptor", "generic_spectral_bins", "compute_orbit", "graphlet_backend",
            "generic_validity_options",
        )},
        "exact_protocol": exact_container.get("protocol"),
        "diversity_protocol": diversity_container.get("protocol"),
    }
    return metrics, counts, protocol


def _molecular_metrics(report: Mapping[str, Any], evaluation: Path) -> tuple[dict, dict, dict]:
    values = _mapping(report.get("metrics"))
    metrics = {key: _finite(values.get(key)) for key in MOLECULAR_METRICS}
    metrics["validity_rate"] = _finite(values.get("validity_without_correction", values.get("validity")))
    for key in ("graphlet_histogram_mmd", "graphlet_selected_mass_mmd"):
        metrics[key] = _finite(values.get(key))
    qed_path = evaluation / "qed" / "qed_metrics.json"
    qed_protocol = None
    if qed_path.is_file():
        qed = _read_json(qed_path)
        metrics["qed_mean"] = _finite(_mapping(qed.get("metrics")).get("qed_mean"))
        qed_protocol = {
            **{key: qed.get(key) for key in (
                "metric_molecule_source", "qed_definition", "rdkit_version", "weights",
                "stereochemistry", "primary_weighting",
            )},
            "reference_sha256": _mapping(
                _mapping(qed.get("source_files")).get("reference_graphs")
            ).get("sha256"),
        }
    counts = {
        "evaluated_count": values.get("num_generated_graphs"),
        "valid_count": values.get("num_valid_generated_molecules"),
        "invalid_count": values.get("num_invalid_generated_molecules"),
        "unique_count": values.get("unique_valid_count"),
        "novel_count": values.get("novel_unique_valid_count"),
        "metric_population_count": _mapping(report.get("protocol")).get("metric_population_count"),
        "fcd_population_count": values.get("fcd_num_valid_generated_molecules"),
    }
    p = _mapping(report.get("protocol"))
    nspdk = _mapping(p.get("nspdk"))
    nspdk.pop("metric_cache_dir", None)
    fcd = _mapping(p.get("fcd"))
    paths = _mapping(report.get("paths"))
    protocol = {
        "kind": "molecular",
        "reference_source": paths.get("reference_source"), "train_source": paths.get("train_source"),
        **{k: p.get(k) for k in (
            "strict_raw_metrics", "validity_denominator_complete", "canonical_smiles", "isomeric_smiles",
            "metric_molecule_source", "hogdiff_compatible_distribution_metrics",
            "uniqueness_definition", "novelty_definition",
        )},
        "nspdk": nspdk, "graphlet": p.get("graphlet"),
        "fcd": {key: fcd.get(key) for key in ("backend", "generated_smiles_source")},
        "qed": qed_protocol,
    }
    return metrics, counts, protocol


def load_run(seed: int, root: Path, variant: str, evaluation_subdir: str) -> dict[str, Any]:
    branch = root / variant
    record: dict[str, Any] = {
        "seed": seed, "variant": variant, "branch_dir": str(branch),
        "status": "missing", "reason": None, "metrics": {}, "counts": {}, "protocol": None,
        "report_path": None, "sample_ids": None, "requested_sample_ids": None,
        "matching_manifest": None, "generation_status": None,
    }
    try:
        common = {}
        for path in (root / "ablation_manifest.json", root.parent / "matched_ablation_manifest.json"):
            if path.is_file():
                common = _read_json(path)
                record["matching_manifest"] = str(path)
                break
        metadata = _mapping(_mapping(common.get("branches")).get(variant))
        if (branch / "manifest.json").is_file():
            metadata = {**metadata, **_read_json(branch / "manifest.json")}
        # A fail-fast primary decoder can fail before the atomic generation
        # directory is committed. The sampler preserves its error next to that
        # directory; retain it rather than calling every branch merely missing.
        generation = root.parent
        failure_path = next((path for path in (
            generation.parent / f"{generation.name}.matched_ablation_failure.json",
            generation.parent / f"{generation.name}.ablation_command_failure.json",
        ) if path.is_file()), None)
        if not common and not metadata and not (generation / "manifest.json").is_file() and failure_path is not None:
            failure = _read_json(failure_path)
            requested = failure.get("num_requested")
            generation_seed = failure.get("generation_seed", seed)
            record.update(
                status="failed", generation_status="failed",
                reason=f"Atomic generation failed: {failure.get('error', failure.get('stage', 'unspecified error'))}",
                generation_failure=failure, generation_failure_path=str(failure_path),
                sample_ids=[],
                requested_sample_ids=[f"{generation_seed}:{i}" for i in range(requested)]
                if isinstance(requested, int) and requested >= 0 else None,
            )
            record["counts"].update(requested_count=requested, completed_count=0, failed_count=None)
            return _clean(record)
        record["generation_status"] = metadata.get("status")
        record["sample_ids"] = metadata.get("sample_ids")
        record["requested_sample_ids"] = metadata.get("requested_sample_ids", common.get("sample_ids"))
        record["shared_context"] = metadata.get(
            "shared_context", common.get("shared_context", common.get("provenance"))
        )
        record["generation_metadata"] = {
            key: metadata[key] for key in (
                "label", "degree_source", "posthoc_repair", "base_graphs", "provenance",
                "checkpoint_sha256", "source_manifest", "shared_manifest", "decoder",
            ) if key in metadata
        }
        record["time_accounting"] = metadata.get("time_accounting", common.get("time_accounting"))
        record["timing_scope"] = metadata.get("timing_scope", common.get("timing_scope"))
        record["failures"] = metadata.get("failures", [])
        record["counts"].update({
            "requested_count": metadata.get("requested_count", common.get("num_requested")),
            "completed_count": metadata.get("completed_count"),
            "failed_count": metadata.get("failed_count"),
        })
        stage_time = _finite(metadata.get("time_seconds"))
        if stage_time is not None:
            record["metrics"]["reported_generation_time_seconds"] = stage_time
        explicit_total = _finite(metadata.get("end_to_end_seconds"))
        if explicit_total is not None:
            record["metrics"]["end_to_end_seconds"] = explicit_total
            n = record["counts"].get("requested_count")
            if isinstance(n, int) and n > 0:
                record["metrics"]["seconds_per_requested_output"] = explicit_total / n
        evaluation = branch / evaluation_subdir
        evaluation_status_path = evaluation / "evaluation_status.json"
        evaluation_status = None
        if evaluation_status_path.is_file():
            evaluation_status = _read_json(evaluation_status_path)
            record["evaluation_status"] = evaluation_status
            record["evaluation_status_path"] = str(evaluation_status_path)
            phase = evaluation_status.get("status")
            if phase not in {"running", "complete", "failed"}:
                raise ValueError(f"Unknown evaluation status {phase!r}")
            if phase != "complete":
                record["status"] = "pending" if phase == "running" else "failed"
                record["reason"] = (
                    f"Evaluation {phase}; any earlier saved report is ignored"
                    + (f" (exit code {evaluation_status['exit_code']})"
                       if evaluation_status.get("exit_code") is not None else "")
                )
                return _clean(record)
        reports = [p for p in (
            evaluation / "graph_evaluation_report.json", evaluation / "molecular_evaluation_metrics.json",
        ) if p.is_file()]
        if len(reports) > 1:
            raise ValueError("Both generic and molecular reports exist; select an unambiguous evaluation directory")
        if not reports:
            if evaluation_status is not None:
                raise ValueError("Evaluation is marked complete but no evaluator report exists")
            if metadata.get("status") == "failed":
                record["status"], record["reason"] = "failed", "Generation failed; no saved evaluator report"
            elif metadata:
                record["status"], record["reason"] = "pending", "Generation metadata exists; evaluation is pending"
            else:
                record["status"], record["reason"] = "missing", "No generation metadata or saved evaluator report"
            return record
        path = reports[0]
        report = _read_json(path)
        values, counts, protocol = (
            _generic_metrics(report) if path.name == "graph_evaluation_report.json"
            else _molecular_metrics(report, evaluation)
        )
        record["metrics"].update(values)
        record["counts"].update(counts)
        record["protocol"] = protocol
        record["protocol_fingerprint"] = _digest(protocol)
        record["report_path"] = str(path)
        record["report_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        record["status"] = "partial" if metadata.get("status") == "partial" else "complete"
        if metadata.get("status") == "failed":
            raise ValueError("Saved evaluator report conflicts with failed generation status (possibly stale output)")
        expected = record["counts"].get("completed_count")
        evaluated = record["counts"].get("evaluated_count")
        if expected is not None and evaluated is not None and expected != evaluated:
            raise ValueError(f"Evaluation population {evaluated} differs from branch completed_count {expected}")
        ids = record.get("sample_ids")
        if ids is not None:
            if not isinstance(ids, list) or len(set(map(str, ids))) != len(ids):
                raise ValueError("Successful sample IDs must be a unique list")
            if evaluated is not None and len(ids) != evaluated:
                raise ValueError("Successful sample IDs do not match evaluated population size")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        record["status"] = "error"
        record["reason"] = str(error)
        record["metrics"] = {}
    return _clean(record)


def summarize_values(values: Mapping[int, float | None], expected_seeds: Sequence[int]) -> dict[str, Any]:
    available = {int(seed): value for seed, value in values.items() if _finite(value) is not None}
    measured = list(available.values())
    return {
        "n": len(measured),
        "mean": statistics.fmean(measured) if measured else None,
        "std": statistics.stdev(measured) if len(measured) > 1 else None,
        "seeds": sorted(available),
        "missing_seeds": [seed for seed in expected_seeds if seed not in available],
    }


def _pairing(left: Mapping, right: Mapping, kind: str) -> str:
    left_ids, right_ids = left.get("requested_sample_ids"), right.get("requested_sample_ids")
    if not isinstance(left_ids, list) or not isinstance(right_ids, list):
        return "unverified_requested_ids"
    if left_ids != right_ids:
        return "mismatched_requested_ids"
    if kind != "training" and left.get("matching_manifest") != right.get("matching_manifest"):
        return "different_matching_manifests"
    return "verified_requested_ids"


def build_summary(seed_dirs: Mapping[int, Path], variants: Sequence[str], evaluation_subdir: str,
                  *, allow_unverified_pairing: bool = False) -> dict[str, Any]:
    seeds = sorted(seed_dirs)
    records = [load_run(seed, Path(seed_dirs[seed]), variant, evaluation_subdir)
               for seed in seeds for variant in variants]
    valid = [r for r in records if r["status"] in {"complete", "partial"}]
    protocol_groups: dict[str, list[dict]] = {}
    for record in valid:
        protocol_groups.setdefault(record["protocol_fingerprint"], []).append(
            {"seed": record["seed"], "variant": record["variant"]})
    # Mixed references, populations, kernels or validity definitions cannot share a mean.
    blocked = len(protocol_groups) > 1
    metric_names = sorted({name for record in records for name in record["metrics"]})
    if not metric_names:
        metric_names = list(GENERIC_METRICS)
    index = {(r["seed"], r["variant"]): r for r in records}
    by_variant = {}
    for variant in variants:
        by_variant[variant] = {}
        for metric in metric_names:
            values = {
                seed: (_finite(index[seed, variant]["metrics"].get(metric))
                       if index[seed, variant]["status"] in {"complete", "partial"} and not blocked else None)
                for seed in seeds
            }
            by_variant[variant][metric] = summarize_values(values, seeds)
    contrasts = {}
    differences = []
    for label, left, right, kind in DEFAULT_CONTRASTS:
        if left not in variants or right not in variants:
            continue
        per_metric = {metric: {} for metric in metric_names}
        for seed in seeds:
            a, b = index[seed, left], index[seed, right]
            matching = _pairing(a, b, kind)
            status = "complete"
            reason = None
            if a["status"] not in {"complete", "partial"} or b["status"] not in {"complete", "partial"}:
                status, reason = "missing_or_failed", "Both evaluator reports are required"
            elif blocked or a.get("protocol_fingerprint") != b.get("protocol_fingerprint"):
                status, reason = "incompatible_protocol", "Evaluation references/metric definitions differ"
            elif matching != "verified_requested_ids" and not (
                allow_unverified_pairing and matching == "unverified_requested_ids"
            ):
                status, reason = "unverified_pairing", matching
            for metric in metric_names:
                av, bv = _finite(a["metrics"].get(metric)), _finite(b["metrics"].get(metric))
                row_status, row_reason = status, reason
                if row_status == "complete" and metric in TIMING_METRICS and av is not None and bv is not None:
                    scope_a = a.get("timing_scope") or a.get("time_accounting")
                    scope_b = b.get("timing_scope") or b.get("time_accounting")
                    if not scope_a or scope_a != scope_b:
                        row_status, row_reason = "incompatible_timing_scope", "Generation timing scopes differ or are undeclared"
                delta = av - bv if row_status == "complete" and av is not None and bv is not None else None
                row_status = row_status if row_status != "complete" or delta is not None else "missing_metric"
                differences.append({
                    "seed": seed, "contrast": label, "kind": kind, "left": left, "right": right,
                    "metric": metric, "left_value": av, "right_value": bv, "difference": delta,
                    "status": row_status, "reason": row_reason, "matching_status": matching,
                    "output_ids_equal": a.get("sample_ids") == b.get("sample_ids")
                    if a.get("sample_ids") is not None and b.get("sample_ids") is not None else None,
                })
                per_metric[metric][seed] = delta
        contrasts[label] = {"left": left, "right": right, "kind": kind,
                            "metrics": {metric: summarize_values(vals, seeds) for metric, vals in per_metric.items()}}
    return _clean({
        "format": "graphes_ablation_summary_v1", "seeds": seeds, "variants": list(variants),
        "evaluation_subdir": evaluation_subdir,
        "aggregation": "Equal-weight training-seed metrics; sample SD (ddof=1); never pooled graph batches",
        "paired_difference": "Within each training seed compute left minus right, then mean and sample SD",
        "units": {"rates": "fractions", "joint_vun_rate": "source percent divided by 100", "std": "null when fewer than 2 seeds"},
        "missing_data": "null, never imputed as zero; contributing and missing seeds accompany every statistic",
        "timing": "Use end_to_end_seconds only when explicitly declared; each run retains generation time_accounting. Shared sampling may be attributed once to each hypothetical branch, so branch costs must not be summed to obtain suite wall time.",
        "generic_diversity": "Exact-isomorphism metrics have explicit names; legacy WL rates are separate",
        "aggregation_blocked": blocked,
        "aggregation_block_reason": "Different evaluation protocol fingerprints" if blocked else None,
        "protocol_groups": protocol_groups,
        "protocols": {r["protocol_fingerprint"]: r["protocol"] for r in valid},
        "per_seed": records, "variant_summary": by_variant,
        "paired_differences": differences, "contrast_summary": contrasts,
    })


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def write_summary(summary: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_clean(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    tmp = output_dir / "ablation_summary.json.tmp"
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(output_dir / "ablation_summary.json")
    rows = []
    status_rows = []
    for record in summary["per_seed"]:
        common = {key: record.get(key) for key in ("seed", "variant", "status", "reason", "report_path", "generation_status")}
        common.update(record["counts"])
        status_rows.append(common)
        rows.extend({**common, "metric": name, "value": value} for name, value in record["metrics"].items())
    count_fields = ("requested_count", "completed_count", "failed_count", "evaluated_count", "valid_count", "unique_count", "novel_count")
    common_fields = ("seed", "variant", "status", "reason", "generation_status", *count_fields, "report_path")
    _write_csv(output_dir / "ablation_run_status.csv", status_rows, common_fields)
    _write_csv(output_dir / "ablation_per_seed.csv", rows, (*common_fields, "metric", "value"))
    variants = [dict(variant=variant, metric=metric, **stats)
                for variant, metrics in summary["variant_summary"].items() for metric, stats in metrics.items()]
    _write_csv(output_dir / "ablation_variant_summary.csv", variants, ("variant", "metric", "n", "mean", "std", "seeds", "missing_seeds"))
    _write_csv(output_dir / "ablation_paired_differences.csv", summary["paired_differences"],
               ("seed", "contrast", "kind", "left", "right", "metric", "left_value", "right_value", "difference", "status", "reason", "matching_status", "output_ids_equal"))
    contrasts = [dict(contrast=name, left=data["left"], right=data["right"], kind=data["kind"], metric=metric, **stats)
                 for name, data in summary["contrast_summary"].items() for metric, stats in data["metrics"].items()]
    _write_csv(output_dir / "ablation_contrast_summary.csv", contrasts,
               ("contrast", "left", "right", "kind", "metric", "n", "mean", "std", "seeds", "missing_seeds"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir", action="append", required=True, metavar="SEED=GEN/ablations")
    parser.add_argument("--variants", nargs="+", default=["T", "D", "TR", "DR"])
    parser.add_argument("--evaluation-subdir", default="evaluation_test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-unverified-pairing", action="store_true",
                        help="Compute seed differences when requested-ID manifests are absent; explicit ID/manifest mismatches are still rejected.")
    parser.add_argument("--require-complete", action="store_true",
                        help="Save the audit reports but exit nonzero if any requested branch or primary metric is incomplete.")
    args = parser.parse_args(argv)
    seed_dirs = {}
    for value in args.seed_dir:
        try:
            seed_text, path = value.split("=", 1)
            seed = int(seed_text)
        except ValueError:
            parser.error(f"Invalid --seed-dir {value!r}; use SEED=PATH")
        if seed in seed_dirs or not path:
            parser.error(f"Duplicate seed or empty path: {value!r}")
        seed_dirs[seed] = Path(path)
    if len(set(args.variants)) != len(args.variants):
        parser.error("--variants must not contain duplicates")
    for variant in args.variants:
        if not variant or Path(variant).name != variant or variant in {".", ".."}:
            parser.error("Variant names must be single directory names")
    if Path(args.evaluation_subdir).is_absolute() or ".." in Path(args.evaluation_subdir).parts:
        parser.error("--evaluation-subdir must remain inside each branch directory")
    summary = build_summary(seed_dirs, args.variants, args.evaluation_subdir,
                            allow_unverified_pairing=args.allow_unverified_pairing)
    write_summary(summary, args.output_dir)
    missing = [r for r in summary["per_seed"] if r["status"] != "complete"]
    incomplete_metrics = []
    for r in summary["per_seed"]:
        required = MOLECULAR_METRICS if _mapping(r.get("protocol")).get("kind") == "molecular" else GENERIC_METRICS
        incomplete_metrics.extend((r["seed"], r["variant"], key) for key in required if r["metrics"].get(key) is None)
    unavailable_primary_pairs = [
        row for row in summary["paired_differences"]
        if row["status"] != "complete" and row["metric"] in set(GENERIC_METRICS + MOLECULAR_METRICS)
        and any(r["seed"] == row["seed"] and r["variant"] == row["left"]
                and row["metric"] in (MOLECULAR_METRICS if _mapping(r.get("protocol")).get("kind") == "molecular" else GENERIC_METRICS)
                for r in summary["per_seed"])
    ]
    print(f"Saved {args.output_dir / 'ablation_summary.json'}")
    print(f"Requested runs: {len(summary['per_seed'])}; non-complete runs: {len(missing)}; unavailable primary measurements: {len(incomplete_metrics)}")
    if summary["aggregation_blocked"]:
        print("Aggregation blocked: evaluator protocol/reference fingerprints differ. Inspect the saved audit.")
        return 2
    if args.require_complete and (missing or incomplete_metrics or unavailable_primary_pairs):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
