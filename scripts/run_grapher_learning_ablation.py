#!/usr/bin/env python3
"""Prepare/run/evaluate missing GraphES learning controls on T decoding (B=0).

Examples:
  PYTHONPATH=src python scripts/run_grapher_learning_ablation.py --dataset community_small --variants S1 S2 S3 S4 K3 K34 --stage all
  PYTHONPATH=src python scripts/run_grapher_learning_ablation.py --dataset qm9 --variants A1 A2 --stage all

The second command requires the canonical reference checkpoint AND saved paired
molecular generation. Use --dry-run or --stage plan to inspect without torch/data.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from grapher.experiments.learning_ablation import (
    CONTROLS, DATASETS, DEFINITIONS, MOLECULAR, SERIALIZED, applicable_controls,
    build_control_config, json_digest, reference_paths, require_attribute_reference,
    sha256, verify_attribute_population, verify_manifest_files,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--variants", nargs="+", choices=(*CONTROLS, "all"), default=["all"])
    p.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    p.add_argument("--stage", choices=("plan", "train", "generate", "evaluate", "all"), default="plan")
    p.add_argument("--dry-run", action="store_true", help="Print exact commands/definitions without creating artifacts or importing torch")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dataset-root", type=Path, default=Path("outputs/datasets"))
    p.add_argument("--output-root", type=Path, default=Path("outputs/baselines"))
    p.add_argument("--report-root", type=Path, default=Path("outputs/ablations/grapher_paper/learning"))
    p.add_argument("--num-samples", type=int, default=None, help="Default 1024 generic, 10000 molecular")
    p.add_argument("--reference-split", choices=("val", "test"), default="test")
    p.add_argument("--orca-executable", default=os.environ.get("ORCA_EXEC", "/home/quang/orca/orca.out"))
    p.add_argument("--reference-run-template", default=None, help="Canonical final g345_v2 run by default; fields {seed}, {dataset}, {profile}, {n}")
    p.add_argument("--reference-generation-template", default=None, help="Default paper_ablation_v1_seed_{seed}_n_{n}")
    p.add_argument("--run-tag", default="paper_learning_v1", help="New provenance-compatible run namespace; never overwrite a previous experiment")
    return p


def _absolute(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def _run(command: list[str], seed: int, args) -> None:
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONHASHSEED=str(seed), GRAPHER_EIGH_BACKEND="cpu", ORCA_EXEC=str(args.orca_executable))
    print(shlex.join(["env", f"PYTHONHASHSEED={seed}", "GRAPHER_EIGH_BACKEND=cpu", "PYTHONPATH=src", *command]), flush=True)
    if not args.dry_run and args.stage != "plan":
        subprocess.run(command, cwd=REPO, env=env, check=True)


def _commands(args, *, seed: int, variant: str, config_path: Path, run_id: str, generation_id: str, generation_dir: Path) -> dict[str, list[str]]:
    common = [sys.executable, "scripts/run_grapher_baseline.py", "--dataset", args.dataset,
              "--no-common-config", "--wrapper-config", str(config_path), "--dataset-root", str(args.dataset_root),
              "--serialized-dataset", SERIALIZED[args.dataset], "--output-root", str(args.output_root),
              "--seed-id", str(seed), "--run-id", run_id, "--device", args.device]
    result = {"train": [*common, "--stage", "train"], "generate": [*common, "--stage", "generate", "--generation-seed", str(seed), "--generation-id", generation_id, "--num-samples", str(args.num_samples)]}
    evaluation_dir = generation_dir / f"evaluation_{args.reference_split}"
    if args.dataset not in MOLECULAR:
        result["evaluate"] = [sys.executable, "scripts/evaluate_graph_generation_report.py", "--config", f"configs/evaluation/{args.dataset}.yaml",
                              "--dataset-root", str(args.dataset_root), "--reference-split", args.reference_split,
                              "--generic-mmd-protocol", "graphrnn", "--generic-validity-require-connected", "--generic-validity-min-nodes", "1",
                              "--generic-validity-max-nodes", "20" if args.dataset == "community_small" else "18",
                              "--generated-dir", str(generation_dir), "--generated-graphs", str(generation_dir / "base_graphs.pkl"),
                              "--generated-stage", variant, "--output-dir", str(evaluation_dir)]
    else:
        result["evaluate"] = [sys.executable, "scripts/evaluate_generated_molecules.py", "--generated-dir", str(generation_dir),
                              "--generated-graphs", str(generation_dir / "base_graphs.pkl"), "--dataset-root", str(args.dataset_root),
                              "--dataset", SERIALIZED[args.dataset], "--reference-split", args.reference_split, "--train-split", "train",
                              "--output-dir", str(evaluation_dir), "--strict-raw-metrics", "--metric-molecule-source", "raw_valid",
                              "--nspdk-backend", "eden", "--nspdk-complexity", "4", "--nspdk-bond-label-mode", "hogdiff",
                              "--require-fcd", "--fcd-device", args.device, "--graphlet-mmd", "--graphlet-k-min", "3", "--graphlet-k-max", "5",
                              "--graphlet-topology-filter", "all", "--graphlet-node-attribute", "atomic_num", "--graphlet-edge-attribute", "bond_type",
                              "--graphlet-attributed-backend", "python"]
        if args.dataset == "zinc":
            result["evaluate"].extend(["--graphlet-num-samples", "4096"])
        result["evaluate_qed"] = [sys.executable, "scripts/evaluate_qed.py",
                                  "--generated-graphs", str(generation_dir / "base_graphs.pkl"),
                                  "--reference-graphs", str(args.dataset_root / SERIALIZED[args.dataset] / f"{args.reference_split}.pkl"),
                                  "--molecular-report", str(evaluation_dir / "molecular_evaluation_metrics.json"),
                                  "--expected-num-generated", str(args.num_samples),
                                  "--output-dir", str(evaluation_dir / "qed")]
    return result


def _verify_training(run_dir: Path, config: dict, seed: int) -> dict:
    checkpoint = run_dir / "train/checkpoints/grapher.pt"
    manifest_path = run_dir / "train/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if sha256(checkpoint) != manifest.get("checkpoint", {}).get("sha256"):
        raise ValueError("Control checkpoint differs from training manifest")
    selection = manifest.get("checkpoint_selection", {})
    # All active implementations save the configured final epoch, never best-on-test.
    if int(selection.get("epoch", -1)) != 200:
        raise ValueError(f"Control checkpoint is not the configured final 200-epoch checkpoint: {selection}")
    if int(manifest.get("train_seed", -1)) != seed:
        raise ValueError("Control training seed differs")
    def check_subset(expected, actual, prefix="grapher"):
        for key, value in expected.items():
            if prefix == "grapher" and key == "runtime":
                continue  # The recorded command's --device supplies this override.
            location = f"{prefix}.{key}"
            if key not in actual:
                raise ValueError(f"Training manifest is missing expected option: {location}")
            if isinstance(value, dict):
                if not isinstance(actual[key], dict):
                    raise ValueError(f"Training option is not a mapping: {location}")
                check_subset(value, actual[key], location)
            elif value != actual[key]:
                raise ValueError(f"Training option differs from the sealed control: {location}")
    check_subset(config["grapher"], manifest.get("options", {}))
    return manifest


def _generation_contract(args, generation_dir: Path, run_dir: Path, seed: int, intent: dict) -> dict:
    manifest_path = generation_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checkpoint_hash = sha256(run_dir / "train/checkpoints/grapher.pt")
    if manifest.get("checkpoint", {}).get("sha256") != checkpoint_hash:
        raise ValueError("Control population was generated with another checkpoint")
    if int(manifest.get("generation_seed", -1)) != seed or int(manifest.get("num_generated", -1)) != args.num_samples or int(manifest.get("num_requested", -1)) != args.num_samples:
        raise ValueError("Control population seed or size differs")
    hashes = verify_manifest_files(generation_dir, manifest)
    if "base_graphs" not in hashes:
        raise ValueError("Generated graph hashes are missing")
    pairing = None
    if intent.get("attribute_reference"):
        pairing = verify_attribute_population(generation_dir, intent["attribute_reference"])
    return {"format": "grapher_learning_generation_v1", "intent_sha256": json_digest(intent), "generation_manifest_sha256": sha256(manifest_path), "checkpoint_sha256": checkpoint_hash, "artifact_sha256": hashes, "pairing": pairing}


def _one(args, seed: int, variant: str) -> dict:
    import yaml

    suffix = "paired" if args.dataset in MOLECULAR else "matched"
    base_path = REPO / f"configs/experiments/grapher_paper_ablation/{args.dataset}_seed_{seed}_{suffix}.yaml"
    base = yaml.safe_load(base_path.read_text())
    config = build_control_config(base, args.dataset, variant, seed)
    run_id = f"seed_{seed}_{args.dataset}_{args.run_tag}_{variant}"
    generation_id = f"{args.run_tag}_seed_{seed}_n_{args.num_samples}"
    run_dir = args.output_root / "grapher" / args.dataset / run_id
    generation_dir = run_dir / "generations" / generation_id
    report_dir = args.report_root / args.dataset / args.run_tag / f"seed_{seed}" / variant
    config_path = report_dir / "config.yaml"
    reference_run, reference_gen = reference_paths(args.output_root, args.dataset, seed, args.num_samples, args.reference_run_template, args.reference_generation_template)
    reference = None
    preview = args.dry_run or args.stage == "plan"
    if variant.startswith("A"):
        if not preview:
            reference = require_attribute_reference(reference_run, reference_gen, dataset=args.dataset, seed=seed, n=args.num_samples)
            config["grapher"]["train"]["attribute_only"] = True
            config["grapher"]["train"]["frozen_topology_reference"] = reference
            config["grapher"]["generation_batch_size"] = reference["batch_size"]
            config["grapher"]["sample"] = reference["effective_sample_config"]
        else:
            print(f"REQUIRES: {reference_run / 'train/checkpoints/grapher.pt'} and {reference_gen} (verified at execution)")
    commands = _commands(args, seed=seed, variant=variant, config_path=config_path, run_id=run_id, generation_id=generation_id, generation_dir=generation_dir)
    source_files = [Path(__file__).resolve(), REPO / "src/grapher/experiments/learning_ablation.py", *[REPO / f"src/grapher/models/grapher/{name}.py" for name in ("vanilla_gsdm", "hierarchical_attributed_loggap", "hybrid_attributed_loggap")]]
    intent = {"format": "grapher_learning_intent_v1", "dataset": args.dataset, "seed": seed, "variant": variant,
              "definition": DEFINITIONS[variant], "comparison_reference": "A0" if variant in {"A1", "A2"} else "T",
              "decoder": "T", "rewiring_budget": 0, "epochs": 200,
              "config": config, "base_config_sha256": sha256(base_path), "implementation_sha256": {str(path.relative_to(REPO)): sha256(path) for path in source_files},
              "attribute_reference": reference, "dataset_root": str(args.dataset_root), "device": args.device,
              "reference_run": str(reference_run), "reference_generation": str(reference_gen)}
    selected = ["train", "generate", "evaluate"] if args.stage in {"plan", "all"} else [args.stage]
    print(f"\n{args.dataset} seed={seed} {variant}: {DEFINITIONS[variant]} T, B=0, 200 epochs.", flush=True)
    if preview:
        for stage in selected:
            _run(commands[stage], seed, args)
            if stage == "evaluate" and "evaluate_qed" in commands:
                _run(commands["evaluate_qed"], seed, args)
        return {"dataset": args.dataset, "seed": seed, "variant": variant, "status": "planned", "commands": commands, "generation_dir": str(generation_dir)}
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".learning.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if "evaluate" in selected and generation_dir.exists():
            _write_json(generation_dir / f"evaluation_{args.reference_split}" / "evaluation_status.json",
                        {"status": "running", "exit_code": 0})
        intent_path = run_dir / "learning_intent.json"
        if intent_path.exists() and json.loads(intent_path.read_text()) != intent:
            raise ValueError(f"Existing learning run has different config/code/provenance; choose a new --run-tag: {run_dir}")
        if not intent_path.exists() and (run_dir / "train/manifest.json").exists():
            raise ValueError("Unsealed existing training cannot be adopted as a learning control")
        _write_json(intent_path, intent)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_text = yaml.safe_dump(config, sort_keys=False)
        if config_path.exists() and config_path.read_text() != config_text:
            raise ValueError("Learning control config changed; choose a new --run-tag")
        config_path.write_text(config_text)
        if "train" in selected:
            _run(commands["train"], seed, args)
        if any(stage in selected for stage in ("train", "generate", "evaluate")):
            _verify_training(run_dir, config, seed)
        seal_path = generation_dir / "learning_generation_seal.json"
        if "generate" in selected:
            if generation_dir.exists():
                if not seal_path.is_file():
                    raise ValueError("Existing control generation is unsealed; use a new --run-tag rather than adopting an unverified population")
                current = _generation_contract(args, generation_dir, run_dir, seed, intent)
                if json.loads(seal_path.read_text()) != current:
                    raise ValueError("Saved control population changed since it was sealed")
                print(f"Reusing verified learning population: {generation_dir}")
            else:
                _run(commands["generate"], seed, args)
                _write_json(seal_path, _generation_contract(args, generation_dir, run_dir, seed, intent))
        if "evaluate" in selected:
            _write_json(generation_dir / f"evaluation_{args.reference_split}" / "evaluation_status.json",
                        {"status": "running", "exit_code": 0})
            current = _generation_contract(args, generation_dir, run_dir, seed, intent)
            if not seal_path.is_file() or json.loads(seal_path.read_text()) != current:
                raise ValueError("Evaluate requires the original verified learning generation seal")
            _run(commands["evaluate"], seed, args)
            if "evaluate_qed" in commands:
                _run(commands["evaluate_qed"], seed, args)
            _write_json(generation_dir / f"evaluation_{args.reference_split}" / "evaluation_status.json",
                        {"status": "complete", "exit_code": 0})
        result = {"dataset": args.dataset, "seed": seed, "variant": variant, "status": "completed_" + args.stage,
                  "config": str(config_path), "definition": DEFINITIONS[variant], "generation_dir": str(generation_dir),
                  "checkpoint": str(run_dir / "train/checkpoints/grapher.pt"), "evaluation_dir": str(generation_dir / f"evaluation_{args.reference_split}"),
                  "intent_sha256": json_digest(intent), "commands": commands}
        _write_json(report_dir / "run_record.json", result)
        return result


def main(argv=None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if any(seed not in (41, 42, 43, 44, 45) for seed in args.seeds) or len(args.seeds) != len(set(args.seeds)):
        p.error("Paper learning controls use distinct seeds from 41, 42, 43, 44, 45")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_tag):
        p.error("--run-tag must be a path-free identifier")
    if "all" in args.variants and len(args.variants) != 1:
        p.error("Use --variants all alone, or list specific controls")
    if args.num_samples is None:
        args.num_samples = 10000 if args.dataset in MOLECULAR else 1024
    if args.num_samples < 1:
        p.error("--num-samples must be positive")
    for key in ("dataset_root", "output_root", "report_root"):
        setattr(args, key, _absolute(getattr(args, key)))
    variants = applicable_controls(args.dataset) if args.variants == ["all"] else tuple(args.variants)
    if any(variant in {"A1", "A2"} for variant in variants) and "A0" not in variants:
        variants = ("A0", *variants)
        print("Adding required A0: A1/A2 compare with matched frozen-topology attribute-only retraining; joint-training T uses a different gradient-clipping norm.")
    if len(variants) != len(set(variants)):
        p.error("--variants must be distinct")
    for variant in variants:
        if variant not in applicable_controls(args.dataset):
            p.error("QM9 S3 is N/A: ordinary degree supervision is already off" if args.dataset == "qm9" and variant == "S3" else f"{variant} is not applicable to {args.dataset}")
    if args.dataset == "qm9" and args.variants == ["all"]:
        print("N/A: QM9 S3 has no active ordinary degree objective and is deliberately excluded.")
    records = []
    try:
        for seed in args.seeds:
            for variant in variants:
                records.append(_one(args, seed, variant))
    except (OSError, ValueError, AssertionError, subprocess.CalledProcessError, KeyboardInterrupt) as exc:
        if not args.dry_run and args.stage in {"all", "evaluate"}:
            failed_run = f"seed_{seed}_{args.dataset}_{args.run_tag}_{variant}"
            failed_gen = f"{args.run_tag}_seed_{seed}_n_{args.num_samples}"
            failed_root = args.output_root / "grapher" / args.dataset / failed_run
            failed_generation = failed_root / "generations" / failed_gen
            status_path = (failed_generation / f"evaluation_{args.reference_split}" / "evaluation_status.json") if failed_generation.exists() else (failed_root / "learning_evaluation_status.json")
            _write_json(status_path, {"status": "failed", "exit_code": int(getattr(exc, "returncode", 130 if isinstance(exc, KeyboardInterrupt) else 2)), "error": str(exc)})
        print(f"Learning ablation failed: {exc}", file=sys.stderr)
        return 2
    if not args.dry_run and args.stage != "plan":
        index = args.report_root / args.dataset / args.run_tag / "latest_invocation.json"
        _write_json(index, {"records": records, "not_applicable": ["QM9 S3: ordinary degree supervision already disabled"] if args.dataset == "qm9" else []})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
