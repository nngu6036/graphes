#!/usr/bin/env python
"""Compare saved joint-checkpoint evaluations under a common generation protocol."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from grapher.utils.io import save_json

SELECTIONS = ("best_joint", "best_histogram", "best_orbit", "last")


def summarize(root: str | Path, *, reference_split: str = "val",
              selections=SELECTIONS, require_same_sources: bool = False) -> dict:
    rows = []
    reference_signature = None
    generation_signature = None
    sources = []
    for kind in selections:
        folder = Path(root) / kind
        with (folder / "report.json").open(encoding="utf-8") as handle:
            generation = json.load(handle)
        with (folder / f"evaluation_{reference_split}" / "graph_evaluation_report.json").open(encoding="utf-8") as handle:
            evaluation = json.load(handle)
        selected = generation.get("checkpoint_selection") or {}
        if selected.get("kind") != kind:
            raise ValueError(f"{folder}: selected checkpoint kind does not match folder name.")
        if evaluation.get("reference_split") != reference_split:
            raise ValueError(f"{folder}: wrong evaluation reference split.")
        if not evaluation.get("reference_split_sha256"):
            raise ValueError(f"{folder}: reference split fingerprint is missing.")
        ref = {key: evaluation.get(key) for key in (
            "reference_split", "reference_split_sha256", "reference_graph_indices_zero_based",
            "num_reference_graphs", "num_graphs_evaluated", "generic_mmd_protocol",
            "generic_clustering_bins", "compute_orbit", "orca_exec",
        )}
        settings = {"seed": generation.get("seed"), "degree_source": generation.get("degree_source"),
                    "num_generated": generation.get("num_generated"),
                    "topology_refiner": generation.get("config", {}).get("topology_refiner", {}),
                    "source_enrichment": generation.get("config", {}).get("source_enrichment", {})}
        if reference_signature is not None and ref != reference_signature:
            raise ValueError("Cannot compare different references, counts, or MMD protocols.")
        if generation_signature is not None and settings != generation_signature:
            raise ValueError("Cannot compare different generation seeds, degree-source modes, counts, or refiner settings.")
        reference_signature, generation_signature = ref, settings
        source_hash = generation.get("source_graph_fingerprint")
        sources.append(source_hash)
        stage = evaluation["generated_stage"]
        comparison = f"{stage}_to_{reference_split}"
        matches = [row for row in evaluation["metrics"] if row["comparison"] == comparison]
        if len(matches) != 1:
            raise ValueError(f"{folder}: missing or ambiguous final MMD row {comparison}.")
        metrics = matches[0]
        for metric in ("degree_mmd", "clustering_mmd", "orbit_mmd"):
            if not math.isfinite(float(metrics[metric])):
                raise ValueError(f"{folder}: nonfinite {metric}; enable the common evaluator before comparison.")
        rows.append({"selection": kind, "epoch": selected["epoch"],
                     "checkpoint_sha256": generation.get("checkpoint_sha256"),
                     "source_graph_fingerprint": source_hash,
                     **{metric: float(metrics[metric]) for metric in ("degree_mmd", "clustering_mmd", "orbit_mmd")}})
    if not rows:
        raise ValueError("At least one selection is required.")
    matched = all(sources) and len(set(sources)) == 1
    if require_same_sources and not matched:
        raise ValueError("Source graphs are not exactly matched. Use train_empirical with a fixed seed, or supply a fixed bank; same learned-prior seed is insufficient.")
    return {"reference": reference_signature, "generation_settings": generation_signature,
            "identical_source_graphs": bool(matched), "results": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", required=True)
    parser.add_argument("--reference-split", choices=("val", "test"), default="val")
    parser.add_argument("--selections", nargs="+", choices=(*SELECTIONS, "best_graphlet"), default=list(SELECTIONS))
    parser.add_argument("--require-same-sources", action="store_true")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    result = summarize(args.generation_root, reference_split=args.reference_split,
                       selections=args.selections, require_same_sources=args.require_same_sources)
    print(f"Checkpoint generation comparison against {args.reference_split}; source graphs identical: {result['identical_source_graphs']}")
    print(f"{'Selection':18s} {'Epoch':>6s} {'Degree MMD':>13s} {'Clustering MMD':>16s} {'Orbit MMD':>13s}")
    for row in result['results']:
        print(f"{row['selection']:18s} {row['epoch']:6d} {row['degree_mmd']:13.6f} {row['clustering_mmd']:16.6f} {row['orbit_mmd']:13.6f}")
    if not result['identical_source_graphs']:
        print("This is an end-to-end comparison with different sources, not a paired conditional-refinement comparison.")
    if args.json_out:
        save_json(result, args.json_out)


if __name__ == '__main__':
    main()
