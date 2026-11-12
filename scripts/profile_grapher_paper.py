#!/usr/bin/env python3
"""Run an existing GraphES entry point with opt-in stage/cost measurements.

Example: PYTHONPATH=src python scripts/profile_grapher_paper.py
  --output outputs/profiles/qm9_41_TR.json --label qm9_TR --requested-count 10000
  -- --stage generate --dataset qm9 ... (native run_grapher_baseline.py flags)

Use a fresh generation ID. Reusing an existing generation measures cache reuse,
not generation. A failed operation still writes its measured costs and counts.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import runpy
import subprocess
import sys
import time
import traceback


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment():
    packages = {}
    for name in ("torch", "numpy", "scipy", "networkx", "rdkit", "PyYAML", "pynauty", "eden-kernel", "fcd-torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(), "logical_cpu_count": os.cpu_count(), "packages": packages, "git_commit": commit}


def install_wrappers(recorder, phase):
    specs = {
        "vanilla_gsdm": {"sample_batch": "basis_reverse", "_laplacian_eigh_padded": "basis_reverse" if phase == "generate" else "spectral_preprocessing", "_padded_dataset": "spectral_preprocessing", "_fit_laplacian_log_gap_stats": "spectral_normalizer_fit", "_joint_structure_targets": "summary_targets", "train": "training_pipeline", "generate": "generation_pipeline"},
        "hybrid_attributed_loggap": {"_attribute_diffusion_decode": "attributes", "_fit_attribute_diffusion_schema": "attribute_schema", "train": "training_pipeline", "generate": "generation_pipeline"},
        "hierarchical_attributed_loggap": {"_hierarchical_structure_targets": "summary_targets_and_cache", "_fit_typed_graphlet_basis": "typed_vocabulary_fit", "_fit_typed_degree_basis": "typed_vocabulary_fit"},
        "eigenbasis_perturbation": {"perturb_laplacian_basis": "basis_reverse"},
        "degree_constrained_realization": {"realize_graph": "topology_decode", "project_degree_sequence": "degree_projection"},
        "graphlet_stage3": {"refine_generated_graphs_with_fixed_predictions": "rewiring", "refine_generated_graphs": "rewiring"},
        "attributed_diversification": {"diversify_same_bond_graph": "rewiring"},
    }
    originals = {}
    for module_name, functions in specs.items():
        module = importlib.import_module("grapher.models.grapher." + module_name)
        for name, label in functions.items():
            original = getattr(module, name, None)
            if original is None:
                continue
            @functools.wraps(original)
            def wrapped(*args, _original=original, _label=label, **kwargs):
                with recorder.stage(_label):
                    return _original(*args, **kwargs)
            originals[id(original)] = (original, wrapped)
    # Update imported aliases too: hybrid generation imports sample_batch by name.
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith("grapher."):
            continue
        for name, value in list(vars(module).items()):
            item = originals.get(id(value))
            if item is not None and value is item[0]:
                setattr(module, name, item[1])
    def restore():
        replacements = {id(wrapped): (wrapped, original) for original, wrapped in originals.values()}
        # Include aliases imported by the target after instrumentation started.
        for module_name, module in list(sys.modules.items()):
            if module is None or not module_name.startswith("grapher."):
                continue
            for name, value in list(vars(module).items()):
                item = replacements.get(id(value))
                if item is not None and value is item[0]:
                    setattr(module, name, item[1])
    return restore


def native_arg(command, key, default=None):
    if key in command and command.index(key) + 1 < len(command):
        return command[command.index(key) + 1]
    return default


def cache_snapshot(run_dir):
    root = run_dir.parent / "_preprocess_cache"
    return [{"path": str(path.resolve()), "size_bytes": path.stat().st_size,
             "mtime_ns": path.stat().st_mtime_ns} for path in sorted(root.glob("*.pt"))]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--requested-count", type=int, required=True)
    p.add_argument("--target-script", type=Path, default=Path("scripts/run_grapher_baseline.py"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command or args.requested_count < 1:
        p.error("A positive requested count and native arguments after -- are required")
    if args.dry_run:
        print(json.dumps({"output": str(args.output), "label": args.label, "requested_count": args.requested_count, "target_script": str(args.target_script), "native_arguments": command}, indent=2))
        return 0
    if args.output.exists():
        p.error("Profile output already exists; choose a new path to preserve measured evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    phase = native_arg(command, "--stage", "all")
    if phase not in ("train", "generate"):
        p.error("Profile --stage train or generate separately, to avoid ambiguous costs")
    if "--overwrite" in command:
        p.error("Use a fresh run/generation ID for profiling; --overwrite is not allowed")
    run_id = native_arg(command, "--run-id")
    dataset = native_arg(command, "--dataset")
    gen_id = native_arg(command, "--generation-id")
    if not run_id or not dataset or (phase == "generate" and not gen_id):
        p.error("Explicit --run-id, --dataset and (for generation) --generation-id are required")
    run_dir = Path(native_arg(command, "--output-root", "outputs/baselines")) / "grapher" / dataset / run_id
    artifact_dir = run_dir / "train" if phase == "train" else run_dir / "generations" / gen_id
    if artifact_dir.exists():
        p.error("Target artifacts already exist; cache reuse cannot measure a new run")
    if phase == "generate":
        if int(native_arg(command, "--num-samples", "1024")) != args.requested_count:
            p.error("--requested-count must equal the native --num-samples")
        # This verifier retains checkpoint epoch, configuration, and split checks.
        from types import SimpleNamespace
        import check_grapher_ablation_artifacts as check
        config_path = native_arg(command, "--wrapper-config")
        if not config_path:
            p.error("An explicit --wrapper-config is required")
        verification_args = SimpleNamespace(run_dir=run_dir, seed=int(native_arg(command, "--seed-id", native_arg(command, "--seed", "42"))), dataset=dataset, serialized_dataset=native_arg(command, "--serialized-dataset", dataset), dataset_root=Path(native_arg(command, "--dataset-root", "outputs/datasets")))
        check.training(verification_args, check.config_options(Path(config_path)))
    report = {"format": "graphes_paper_runtime_v1", "label": args.label, "phase": phase, "requested_count": args.requested_count, "target_script": str(args.target_script.resolve()), "target_script_sha256": sha256(args.target_script), "arguments": command, "environment": environment(), "artifact_dir": str(artifact_dir.resolve()), "status": "running", "normalization": "seconds_per_requested_sample" if phase == "generate" else "total_training_job_seconds", "measurement": "CUDA-synchronized wall clock; exclusive nested stages; full operation includes failed work, startup, IO and unclassified overhead", "preprocessing_cache_before": cache_snapshot(run_dir)}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    start = time.perf_counter()
    exit_code = 0
    recorder = None
    torch = None
    device = None
    restore_wrappers = None
    try:
        import torch
        from grapher.models.grapher.paper_measurement import Recorder
        raw_device = native_arg(command, "--device", "auto")
        device = torch.device("cuda:0" if raw_device == "auto" and torch.cuda.is_available() else ("cpu" if raw_device == "auto" else raw_device))
        def sync():
            if device.type == "cuda" and torch.cuda.is_initialized():
                torch.cuda.synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            report["environment"]["gpu"] = torch.cuda.get_device_name(device)
            report["environment"]["cuda_version"] = torch.version.cuda
        recorder = Recorder(synchronize=sync)
        restore_wrappers = install_wrappers(recorder, phase)
        old_argv = sys.argv
        sys.argv = [str(args.target_script), *command]
        try:
            with recorder.activate():
                runpy.run_path(str(args.target_script), run_name="__main__")
        finally:
            sys.argv = old_argv
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        if exit_code:
            report["error"] = str(exc)
    except BaseException as exc:
        exit_code = 1
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        if restore_wrappers is not None:
            restore_wrappers()
        report["wall_seconds"] = time.perf_counter() - start
        report["exit_code"] = exit_code
        report["status"] = "complete" if exit_code == 0 else "failed"
        report["preprocessing_cache_after"] = cache_snapshot(run_dir)
        report["peak_host_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["child_peak_host_rss_kib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        if recorder is not None:
            report["stages"] = dict(recorder.stages)
            report["attempted_samples"] = recorder.attempted_samples if phase == "generate" else None
            report["attempt_count_definition"] = "sample slots whose reverse-diffusion batch began; failed batches retained; no replacement draws" if phase == "generate" else None
            report["completed_endpoints_before_rewiring"] = recorder.completed_endpoints if phase == "generate" else None
            report["attempt_batch_sizes"] = recorder.attempt_batches
            assigned = sum(s["exclusive_seconds"] for s in recorder.stages.values())
            report["unclassified_seconds"] = max(0.0, report["wall_seconds"] - assigned)
        if torch is not None and device is not None and device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_initialized():
            report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            report["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        else:
            report["peak_gpu_allocated_bytes"] = None
            report["peak_gpu_reserved_bytes"] = None
        if phase == "generate":
            report["seconds_per_requested_sample"] = report["wall_seconds"] / args.requested_count
            report["stage_seconds_per_requested_sample"] = {k: v["exclusive_seconds"] / args.requested_count for k, v in report.get("stages", {}).items()}
        manifest_path = artifact_dir / "manifest.json"
        if manifest_path.is_file():
            report["artifact_manifest_sha256"] = sha256(manifest_path)
            manifest = json.loads(manifest_path.read_text())
            report["persisted_output_count"] = manifest.get("num_generated", manifest.get("num_graphs"))
        else:
            report["persisted_output_count"] = 0 if phase == "generate" else None
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(args.output)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
