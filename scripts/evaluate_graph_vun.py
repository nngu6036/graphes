#!/usr/bin/env python
"""Fast, standalone exact joint-V.U.N. evaluation for saved generic graph batches.

Examples (run from the repository root with PYTHONPATH=src):

    python scripts/evaluate_graph_vun.py \
        --config configs/experiments/baselines/community_small_evaluation.yaml \
        --generated-dir outputs/baselines/gdsm/community_small/seed_42/generations/seed_42_n_1024 \
        --generated-dir outputs/baselines/gdsm/community_small/seed_43/generations/seed_43_n_1024 \
        --generated-dir outputs/baselines/gdsm/community_small/seed_44/generations/seed_44_n_1024 \
        --output-dir outputs/vun/community_small/gdsm

Use --include-stages for GraphER pre-rewire vs post-rewire comparisons.
This script *does not* recompute MMD, graphlets, or ORCA.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import evaluate_graph_generation_report as report
from grapher.data.io import load_dataset_splits, validate_prepared_dataset_contract
from grapher.utils.io import ensure_dir, load_yaml


_PER_RUN_COLUMNS = (
    "benchmark", "model", "experiment", "stage", "seed", "generated_dir",
    "graph_file", "num_training_graphs", "num_graphs", "num_valid_graphs",
    "num_distinct_valid_isomorphism_classes", "num_novel_valid_graphs",
    "num_distinct_valid_novel_isomorphism_classes",
    "joint_vun_rate", "joint_vun_percent",
)
_SUMMARY_COLUMNS = (
    "benchmark", "model", "experiment", "stage", "num_runs", "seeds",
    "num_generated_per_run", "joint_vun_percent_mean", "joint_vun_percent_std",
    "joint_vun_rate_mean", "joint_vun_rate_std",
    "num_distinct_valid_novel_classes_mean",
)


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _seed_from_path(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = re.search(r"(?<!\w)seed_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def _experiment_id(generated_dir: Path) -> str:
    """Normalize seed-specific run IDs but retain distinct parameter sweeps."""

    if generated_dir.parent.name == "generations":
        run_id = generated_dir.parent.parent.name
        batch_id = generated_dir.name
    else:
        run_id = generated_dir.name
        batch_id = ""
    normalize = lambda value: re.sub(r"seed_\d+", "seed_*", value)
    run_id = normalize(run_id)
    batch_id = normalize(batch_id)
    # The ordinary artifact name only records seed and graph count; it must
    # not create a separate experiment identity for each seed.
    if re.fullmatch(r"seed_\*_n_\d+", batch_id):
        batch_id = ""
    return "/".join(part for part in (run_id, batch_id) if part)


def _resolve_stage_paths(
    directory: Path,
    *,
    generated_graphs: Path | None,
    generated_stage: str | None,
    include_stages: bool,
    config: Mapping[str, Any],
) -> list[tuple[str, Path, dict[str, Any] | None]]:
    final_path, manifest, _ = report.resolve_generated_graph_path(
        directory, generated_graphs=generated_graphs,
    )
    final_stage = report.resolve_generated_stage(
        dict(config), final_path,
        generation_manifest=manifest,
        explicit_stage=generated_stage,
    )
    paths: list[tuple[str, Path]] = [(final_stage, final_path)]
    if include_stages:
        base, base_stage = report.resolve_base_graph_path(
            directory, generated_path=final_path, generation_manifest=manifest,
        )
        if base is not None and base_stage is not None:
            paths.append((base_stage, base))
        paths.extend(report.resolve_additional_managed_stage_paths(
            directory, generated_path=final_path, generation_manifest=manifest,
            primary_base_path=base,
        ))
        enriched = directory / "enriched_base_graphs.pkl"
        if enriched.is_file():
            paths.append(("enriched_base", enriched))

    unique: list[tuple[str, Path, dict[str, Any] | None]] = []
    seen: set[Path] = set()
    for stage, graph_file in paths:
        key = graph_file.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append((stage, graph_file, manifest))
    return unique


def _summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["benchmark"]), str(row["model"]),
            str(row["experiment"]), str(row["stage"]),
        )
        grouped[key].append(row)

    summary: list[dict[str, Any]] = []
    for (benchmark, model, experiment, stage), group in sorted(grouped.items()):
        seeds = [row["seed"] for row in group]
        if len(set(seeds)) != len(seeds):
            raise ValueError(
                "More than one batch was supplied for the same seed and "
                f"experiment/stage: {benchmark}/{model}/{experiment}/{stage}. "
                "Evaluate different decoder sweeps separately."
            )
        counts = {int(row["num_graphs"]) for row in group}
        if len(counts) != 1:
            raise ValueError(
                f"Different generation sizes in {model}/{stage}: {sorted(counts)}. "
                "Match sample counts before comparing diversity."
            )
        percentages = [float(row["joint_vun_percent"]) for row in group]
        rates = [float(row["joint_vun_rate"]) for row in group]
        novel_counts = [
            int(row["num_distinct_valid_novel_isomorphism_classes"])
            for row in group
        ]
        summary.append({
            "benchmark": benchmark,
            "model": model,
            "experiment": experiment,
            "stage": stage,
            "num_runs": len(group),
            "seeds": ",".join(str(seed) for seed in sorted(seeds)),
            "num_generated_per_run": counts.pop(),
            "joint_vun_percent_mean": statistics.mean(percentages),
            "joint_vun_percent_std": (
                statistics.stdev(percentages) if len(percentages) > 1 else 0.0
            ),
            "joint_vun_rate_mean": statistics.mean(rates),
            "joint_vun_rate_std": (
                statistics.stdev(rates) if len(rates) > 1 else 0.0
            ),
            "num_distinct_valid_novel_classes_mean": statistics.mean(novel_counts),
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Existing generic evaluation YAML.")
    parser.add_argument(
        "--generated-dir", required=True, action="append", type=Path,
        help="One generation directory; repeat for seeds 42/43/44.",
    )
    parser.add_argument(
        "--generated-graphs", type=Path, default=None,
        help="Override graph pickle (only when one --generated-dir is given).",
    )
    parser.add_argument("--generated-stage", default=None)
    parser.add_argument(
        "--include-stages", action="store_true",
        help="Evaluate source/pre-rewire and final graph batches separately.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-samples", type=int, default=None,
        help="Default: protocol.generated_graphs_per_seed from the config (1024).",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Allow a batch with fewer samples than the configured experiment count.",
    )
    parser.add_argument(
        "--max-graphs", type=int, default=None,
        help="Debug only: evaluate the first N generated graphs; disables expected-count check.",
    )
    parser.add_argument(
        "--dataset-mismatch-policy", choices=("error", "warn", "ignore"),
        default="error",
    )
    args = parser.parse_args()

    if args.generated_graphs is not None and len(args.generated_dir) != 1:
        parser.error("--generated-graphs requires exactly one --generated-dir.")
    if args.expected_samples is not None and args.expected_samples < 1:
        parser.error("--expected-samples must be positive.")
    if args.max_graphs is not None and args.max_graphs < 1:
        parser.error("--max-graphs must be positive.")

    config = load_yaml(args.config)
    dataset = config.get("dataset", {}) or {}
    root = dataset.get("root", "outputs/datasets")
    dataset_name = str(dataset.get("name", "sbm"))
    validate_prepared_dataset_contract(
        dataset_name, root=root, config_path=dataset.get("config_path"),
        mismatch_policy=args.dataset_mismatch_policy,
    )
    splits = load_dataset_splits(
        dataset_name, root=root, build_if_missing=False,
        config_path=dataset.get("config_path"),
    )
    train = list(splits.get("train", []))
    provenance = report.dataset_provenance(dataset, splits)
    if not train:
        raise ValueError("The complete training split is missing or empty; cannot evaluate novelty.")
    benchmark = str(config.get("benchmark") or dataset.get("benchmark") or dataset_name)
    if report.is_molecular_evaluation(dataset, train):
        raise ValueError("This command evaluates generic graph topology VUN, not molecular SMILES VUN.")
    expected = (
        args.expected_samples if args.expected_samples is not None
        else (config.get("protocol", {}) or {}).get("generated_graphs_per_seed")
    )
    expected = int(expected) if expected is not None else None

    per_run: list[dict[str, Any]] = []
    paths_seen: set[Path] = set()
    for directory in args.generated_dir:
        for stage, graph_file, manifest in _resolve_stage_paths(
            directory,
            generated_graphs=args.generated_graphs,
            generated_stage=args.generated_stage,
            include_stages=args.include_stages,
            config=config,
        ):
            if graph_file.resolve() in paths_seen:
                raise ValueError(f"Duplicate input graph batch: {graph_file}")
            paths_seen.add(graph_file.resolve())
            model = report._baseline_model_id(graph_file, manifest) or "unknown"
            # Prefer the training manifest's dataset fingerprint, as in the
            # full evaluator; generation manifests may only include a name.
            training_manifest, _ = report._find_training_manifest(graph_file.parent)
            report.validate_dataset_compatibility(
                provenance,
                generation_manifest=manifest,
                training_manifest=training_manifest,
                mismatch_policy=args.dataset_mismatch_policy,
            )
            graphs = report._load_graph_list(graph_file)
            raw_count = len(graphs)
            if (expected is not None and args.max_graphs is None
                    and not args.allow_partial and raw_count != expected):
                raise ValueError(
                    f"Incomplete generation batch {graph_file}: "
                    f"found {raw_count}, expected {expected}. "
                    "Use --allow-partial only for exploratory diagnostics."
                )
            if args.max_graphs is not None:
                graphs = graphs[:args.max_graphs]
            result = report.exact_joint_vun_metrics(graphs, train)
            seed = _seed_from_path(graph_file)
            identity = _experiment_id(directory)
            row = {
                "benchmark": benchmark,
                "model": model,
                "experiment": identity,
                "stage": stage,
                "seed": seed if seed is not None else f"run_{len(per_run) + 1}",
                "generated_dir": str(directory),
                "graph_file": str(graph_file),
                **result,
            }
            per_run.append(row)
            print(
                f"{model}/{benchmark}/{stage} seed={row['seed']} "
                f"joint VUN={float(result['joint_vun_percent']):.2f}% "
                f"({result['num_distinct_valid_novel_isomorphism_classes']}/"
                f"{result['num_graphs']}; valid={result['num_valid_graphs']})",
                flush=True,
            )

    summary = _summarize(per_run)
    output_dir = ensure_dir(args.output_dir)
    _write_csv(output_dir / "vun_per_run.csv", _PER_RUN_COLUMNS, per_run)
    _write_csv(output_dir / "vun_summary.csv", _SUMMARY_COLUMNS, summary)
    (output_dir / "vun_report.json").write_text(
        json.dumps({
            "format": "graphes_exact_joint_vun_v1",
            "benchmark": benchmark,
            "dataset": dataset_name,
            "num_training_graphs": len(train),
            "train_sha256": provenance["split_sha256"]["train"],
            "dataset_fingerprint": provenance["fingerprint"],
            "config": str(args.config),
            "definition": (
                "100 * distinct valid generated topology isomorphism classes "
                "not isomorphic to any graph in the complete training split / "
                "all evaluated generated graphs"
            ),
            "validity": "undirected, non-multigraph, no self-loops",
            "equivalence": "attribute-free NetworkX exact isomorphism (WL prefilter)",
            "expected_num_generated_per_seed": expected,
            "allow_partial": bool(args.allow_partial or args.max_graphs is not None),
            "per_run": per_run,
            "summary": summary,
        }, indent=2),
        encoding="utf-8",
    )
    print("\nJoint V.U.N. mean ± sample standard deviation (across independent runs):")
    for item in summary:
        print(
            f"  {item['model']}/{item['stage']}: "
            f"{item['joint_vun_percent_mean']:.2f} ± "
            f"{item['joint_vun_percent_std']:.2f}% "
            f"(runs={item['num_runs']}, seeds={item['seeds']})"
        )
    print(f"Saved CSV/JSON: {output_dir}")


if __name__ == "__main__":
    main()
