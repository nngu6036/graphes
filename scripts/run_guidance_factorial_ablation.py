#!/usr/bin/env python3
"""Run the controlled HH / spectral / graphlet guidance factorial ablation.

The script intentionally reuses one trained checkpoint. It changes only the
candidate-ranking weights at generation time and verifies that every refined
mode starts from exactly the same ordered HH source batch.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from grapher.rewiring_mlp.generic.joint_degree_training import graph_fingerprint
from grapher.utils.io import load_pickle, load_yaml


DEFAULT_PROFILE = (
    "configs/experiments/grapher/ablations/"
    "community_small_guidance_factorial.yaml"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--degree-source", choices=("learned", "edge_relocation"), default="learned")
    parser.add_argument("--num-generate", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--reference-split", choices=("val", "test"), default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def _mode_overrides(profile: dict[str, Any], mode: dict[str, Any]) -> list[str]:
    protocol = dict(profile.get("protocol", {}) or {})
    weights = dict(mode.get("weights", {}) or {})
    expected = {"edge", "spectral", "clustering", "orbit", "graphlet"}
    if set(weights) != expected:
        raise ValueError(f"Each ablation mode must define exactly {sorted(expected)}; got {sorted(weights)}")
    if sum(float(weights[name]) for name in expected) <= 0:
        raise ValueError("Each refined ablation mode must have at least one positive guidance weight.")
    overrides = [
        f"topology_refiner.steps={int(protocol.get('steps', 32))}",
        f"topology_refiner.proposal_budget={int(protocol.get('proposal_budget', 1024))}",
        f"topology_refiner.valid_candidate_budget={int(protocol.get('valid_candidate_budget', 256))}",
        f"topology_refiner.component_normalization={protocol.get('component_normalization', 'initial')}",
    ]
    overrides.extend(f"topology_refiner.weights.{name}={float(weights[name])}" for name in sorted(expected))
    return overrides


def _report_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    return list(report.get("metrics", []))


def _row_with_prefix(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    matches = [row for row in rows if str(row.get("comparison", "")).startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one report row beginning with {prefix!r}; found {len(matches)}")
    return matches[0]


def main() -> int:
    args = _parse_args()
    if args.generation_only and args.evaluation_only:
        raise ValueError("--generation-only and --evaluation-only are mutually exclusive.")

    profile_path = Path(args.profile)
    profile = load_yaml(profile_path)
    base_config = Path(profile["base_config"])
    checkpoint = Path(profile["checkpoint"])
    seed = int(profile.get("seed", 42) if args.seed is None else args.seed)
    num_generate = int(profile.get("num_generate", 1024) if args.num_generate is None else args.num_generate)
    device = str(profile.get("device", "gpu") if args.device is None else args.device)
    reference_split = str(profile.get("reference_split", "test") if args.reference_split is None else args.reference_split)
    root = Path(args.output_root or profile["output_root"]) / args.degree_source
    modes = dict(profile.get("modes", {}) or {})
    degree_sources = dict(profile.get("degree_sources", {}) or {})
    if args.degree_source not in degree_sources:
        raise ValueError(f"Profile has no degree-source definition for {args.degree_source!r}")
    degree_overrides = [str(value) for value in degree_sources[args.degree_source].get("overrides", [])]

    if not args.dry_run:
        if not base_config.is_file():
            raise FileNotFoundError(base_config)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        root.mkdir(parents=True, exist_ok=True)

    for mode_name, mode in modes.items():
        mode_dir = root / mode_name
        overrides = degree_overrides + _mode_overrides(profile, mode)
        if not args.evaluation_only:
            cmd = [
                sys.executable,
                "scripts/run_topology_grapher.py",
                "--config", str(base_config),
                "--checkpoint", str(checkpoint),
                "--output-dir", str(mode_dir),
                "--num-generate", str(num_generate),
                "--seed", str(seed),
                "--device", device,
            ]
            for override in overrides:
                cmd += ["--set", override]
            _run(cmd, dry_run=args.dry_run)

        if not args.generation_only:
            eval_dir = mode_dir / f"evaluation_{reference_split}"
            cmd = [
                sys.executable,
                "scripts/evaluate_graph_generation_report.py",
                "--config", str(base_config),
                "--generated-dir", str(mode_dir),
                "--reference-split", reference_split,
                "--output-dir", str(eval_dir),
            ]
            _run(cmd, dry_run=args.dry_run)

    if args.dry_run or args.generation_only:
        return 0

    fingerprints: dict[str, str] = {}
    summaries: list[dict[str, Any]] = []
    hh_row: dict[str, Any] | None = None
    for mode_name, mode in modes.items():
        mode_dir = root / mode_name
        coarse = load_pickle(mode_dir / "coarse_graphs.pkl")
        final = load_pickle(mode_dir / "topology_refined_graphs.pkl")
        fingerprints[mode_name] = graph_fingerprint(coarse)
        if profile.get("protocol", {}).get("require_degree_preservation", True):
            if any(sorted(dict(a.degree()).values()) != sorted(dict(b.degree()).values()) for a, b in zip(coarse, final)):
                raise AssertionError(f"Degree preservation failed for ablation mode {mode_name}")
        report_path = mode_dir / f"evaluation_{reference_split}" / "graph_evaluation_report.json"
        rows = _report_rows(report_path)
        source_row = _row_with_prefix(rows, "hh_source_to_")
        final_row = _row_with_prefix(rows, "topology_final_to_")
        if hh_row is None:
            hh_row = source_row
        summaries.append({
            "case": str(mode.get("label", mode_name)),
            "mode": mode_name,
            "degree_mmd": float(final_row["degree_mmd"]),
            "clustering_mmd": float(final_row["clustering_mmd"]),
            "orbit_mmd": float(final_row["orbit_mmd"]),
        })

    if profile.get("protocol", {}).get("require_identical_hh_sources", True):
        unique = set(fingerprints.values())
        if len(unique) != 1:
            raise AssertionError(f"HH source batches differ across modes: {fingerprints}")

    assert hh_row is not None
    rows_out = [{
        "case": "HH",
        "mode": "hh_source",
        "degree_mmd": float(hh_row["degree_mmd"]),
        "clustering_mmd": float(hh_row["clustering_mmd"]),
        "orbit_mmd": float(hh_row["orbit_mmd"]),
    }] + summaries

    summary_csv = root / f"guidance_factorial_{reference_split}.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "mode", "degree_mmd", "clustering_mmd", "orbit_mmd"])
        writer.writeheader()
        writer.writerows(rows_out)

    summary_json = root / f"guidance_factorial_{reference_split}.json"
    summary_json.write_text(json.dumps({
        "format": "grapher_guidance_factorial_v1",
        "profile": str(profile_path),
        "base_config": str(base_config),
        "checkpoint": str(checkpoint),
        "degree_source": args.degree_source,
        "reference_split": reference_split,
        "seed": seed,
        "num_generate": num_generate,
        "hh_source_fingerprints": fingerprints,
        "rows": rows_out,
    }, indent=2) + "\n", encoding="utf-8")

    print("\nControlled guidance factorial (lower is better)")
    print(f"{'Case':30s} {'Degree MMD':>12s} {'Clustering MMD':>15s} {'Orbit MMD':>12s}")
    for row in rows_out:
        print(f"{row['case']:30s} {row['degree_mmd']:12.6f} {row['clustering_mmd']:15.6f} {row['orbit_mmd']:12.6f}")
    print(f"\nHH source fingerprint: {next(iter(fingerprints.values()))}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
