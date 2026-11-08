#!/usr/bin/env python3
"""Export the separately trained S0 population for the paper ablation.

Consumes trusted project artifacts only. No training, sampling, or evaluation
is performed. S0 has its own checkpoint; matching requested IDs does not imply
that it shares the D branch's reverse-diffusion trajectories. The sealed parent
indexes and reuse receipt are deliberately left byte-for-byte unchanged.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

# Also support direct invocation without PYTHONPATH=src.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FORMAT = "graphes_s0_ablation_export_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def artifact(root: Path, record: dict, label: str) -> Path:
    require(isinstance(record, dict) and bool(record.get("path")) and bool(record.get("sha256")),
            f"Missing artifact path/hash: {label}")
    path = root / record["path"]
    require(path.resolve().is_relative_to(root.resolve()), f"Artifact escapes directory: {label}")
    require(path.is_file() and sha256(path) == record["sha256"], f"Missing or changed artifact: {path}")
    return path


def generation(root: Path) -> tuple[dict, dict, Path]:
    """Bind a complete generation to its managed run and training checkpoint."""
    require(root.parent.name == "generations", f"Not a managed generation directory: {root}")
    data = read_json(root / "manifest.json")
    trained = read_json(root.parent.parent / "train" / "manifest.json")
    for name, value in (("model_id", "grapher"), ("run_id", root.parent.parent.name)):
        require(data.get(name) == value and trained.get(name) == value, f"Generation/training {name} differs: {root}")
    require(data.get("generation_id") == root.name, f"Generation ID differs: {root}")
    seed, count = data.get("generation_seed"), data.get("num_requested")
    require(type(seed) is int and seed >= 0 and trained.get("train_seed") == seed, "Training/generation seed differs")
    require(type(count) is int and count > 0 and data.get("num_generated") == count, "Generation is incomplete")
    require(data.get("variant") == trained.get("variant"), "Training/generation model variant differs")
    checkpoint = artifact(root.parent.parent / "train", trained.get("checkpoint"), "training checkpoint")
    require(data.get("checkpoint", {}).get("sha256") == sha256(checkpoint), "Generation checkpoint differs from training")
    return data, trained, artifact(root, data.get("base_graphs"), "base_graphs")


def population(path: Path, ids: list[str], *, labelled: bool = False) -> list:
    import networkx as nx
    from grapher.utils.networkx_pickle import load_trusted_networkx_pickle

    graphs = load_trusted_networkx_pickle(path)
    require(isinstance(graphs, list) and len(graphs) == len(ids), "Saved population count differs")
    for graph, sample_id in zip(graphs, ids):
        require(isinstance(graph, nx.Graph) and not graph.is_directed() and not graph.is_multigraph(),
                f"Expected a simple undirected graph: {sample_id}")
        require(nx.number_of_selfloops(graph) == 0, f"Self-loop in {sample_id}")
        actual = graph.graph.get("sample_id")
        require((actual is None and not labelled) or str(actual) == sample_id,
                f"Saved graph sample ID/order differs: expected {sample_id}, got {actual}")
    return graphs


def export_s0(source: Path, output_root: Path, reference: Path, *, overwrite: bool = False) -> Path:
    source, output_root, reference = source.resolve(), output_root.resolve(), reference.resolve()
    require(source != reference, "S0 must use a separate generation")
    require(output_root == reference / "ablations", "Output root must be the reference generation's ablations directory")
    require(output_root.is_dir(), "Reference ablation population must be exported first")
    # Serialize S0 exports without acquiring the shell's already-held generation lock.
    with (reference / ".s0_export.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _export_s0(source, output_root, reference, overwrite=overwrite)


def _export_s0(source: Path, output_root: Path, reference: Path, *, overwrite: bool) -> Path:
    source_data, trained, graph_path = generation(source)
    ref_data, ref_trained, _ = generation(reference)
    for key in ("generation_seed", "num_requested", "variant"):
        require(source_data.get(key) == ref_data.get(key), f"S0/reference {key} differs")
    dataset = trained.get("dataset", {})
    for key in ("benchmark_id", "serialized_id", "fingerprint", "split_sha256"):
        require(bool(dataset.get(key)) and dataset[key] == ref_trained.get("dataset", {}).get(key),
                f"S0/reference dataset {key} differs")
    require(dataset["benchmark_id"] in {"community_small", "ego_small"}, "S0 is a generic graph study")
    options = trained.get("options", {})
    summary = options.get("structure_summary", {})
    require(summary.get("enabled") is True and summary.get("loss_weight") == 0.0,
            "S0 training must retain the summary heads with loss_weight=0")
    require(ref_trained.get("options", {}).get("structure_summary", {}).get("loss_weight", 0) > 0,
            "Reference training must use summary supervision")
    for training in (trained, ref_trained):
        selection = training.get("checkpoint_selection", {})
        require(selection.get("kind") == "final_configured_epoch"
                and selection.get("epoch") == training.get("options", {}).get("train", {}).get("epochs"),
                "Expected a final configured-epoch checkpoint")
    require(trained["checkpoint_selection"] == ref_trained["checkpoint_selection"], "Training budgets differ")
    sample = source_data.get("effective_sample_config", {})
    decoder = sample.get("degree_constrained_realization", {})
    require(decoder.get("enabled") is True, "S0 must use exact-degree realization")
    require(decoder == ref_data.get("effective_sample_config", {}).get("degree_constrained_realization"),
            "S0/reference exact-degree decoder settings differ")
    require("rewiring" in source_data.get("sampling", {}) and not source_data["sampling"]["rewiring"],
            "S0 generation must disable rewiring")
    require(not options.get("graphlet_refinement", {}).get("enabled"), "S0 config enables refinement")

    seed, count = source_data["generation_seed"], source_data["num_requested"]
    ids = [f"{seed}:{index}" for index in range(count)]
    index_path = output_root / "ablation_manifest.json"
    common = read_json(index_path)
    require(common.get("sample_ids") == ids, "Reference requested sample IDs/order differs")
    # Track input bytes so provenance cannot silently become stale during export.
    tracked = [source / "manifest.json", reference / "manifest.json", index_path,
               source.parent.parent / "train" / "manifest.json",
               reference.parent.parent / "train" / "manifest.json", graph_path]
    for root, training in ((source, trained), (reference, ref_trained)):
        tracked.append(root.parent.parent / "train" / training["checkpoint"]["path"])
        receipt_path = root / "ablation_reuse_receipt.json"
        if receipt_path.is_file():
            receipt = read_json(receipt_path)
            require(receipt.get("generation_manifest_sha256") == sha256(root / "manifest.json"),
                    f"Generation changed after sealing: {root}")
            for key, relative in (("ablation_manifest_sha256", "ablations/ablation_manifest.json"),
                                  ("matched_ablation_manifest_sha256", "matched_ablation_manifest.json")):
                if key in receipt:
                    require(receipt[key] == sha256(root / relative), f"Sealed index changed: {root / relative}")
                    tracked.append(root / relative)
            tracked.append(receipt_path)
    hashes = {path: sha256(path) for path in tracked}
    graphs = population(graph_path, ids)
    provenance = {
        "source_generation_dir": str(source), "source_generation_manifest_sha256": hashes[source / "manifest.json"],
        "source_artifact": source_data["base_graphs"],
        "training_manifest_sha256": hashes[source.parent.parent / "train" / "manifest.json"],
        "checkpoint_sha256": source_data["checkpoint"]["sha256"],
        "reference_generation_dir": str(reference),
        "reference_generation_manifest_sha256": hashes[reference / "manifest.json"],
        "reference_ablation_manifest_sha256": hashes[index_path],
        "dataset": dataset, "summary_loss_weight": 0.0,
        "identity_source": "hash_anchored_parent_list_order",
        "pairing_scope": "requested sample IDs and training seed; separately trained checkpoint",
        "shared_reverse_trajectory": False,
    }
    metadata = {
        "format": FORMAT, "variant": "S0", "label": "No topology-summary supervision",
        "generation_seed": seed, "requested_count": count, "completed_count": count,
        "failed_count": 0, "status": "complete", "sample_ids": ids,
        "requested_sample_ids": ids, "failures": [], "provenance": provenance,
        "shared_context": provenance, "checkpoint_sha256": provenance["checkpoint_sha256"],
        "decoder": decoder, "degree_source": decoder.get("degree_source"),
        "posthoc_repair": False,
        "timing_scope": "separate S0 generation; export does not rerun sampling",
    }
    destination = output_root / "S0"
    require(not destination.is_symlink(), "S0 destination must not be a symlink")
    if destination.exists():
        require(destination.is_dir(), "S0 destination is not a directory")
        try:
            existing = read_json(destination / "manifest.json")
            require(all(existing.get(key) == value for key, value in metadata.items()), "Existing S0 provenance/metadata differs")
            saved = artifact(destination, existing.get("base_graphs"), "S0.base_graphs")
            population(saved, ids, labelled=True)
            require(all(sha256(path) == value for path, value in hashes.items()), "Inputs changed during export")
            return destination / "manifest.json"
        except (OSError, ValueError):
            if not overwrite:
                # Failed evaluations can create a directory before S0 is exported.
                files = [p for p in destination.rglob("*") if p.is_file() or p.is_symlink()]
                evaluation_only = all(not p.is_symlink() and len(p.relative_to(destination).parts) >= 2
                                      and p.relative_to(destination).parts[0].startswith("evaluation_") for p in files)
                require(evaluation_only, "Existing S0 export is incompatible or incomplete; use --overwrite only to replace this branch")
    staging = Path(tempfile.mkdtemp(prefix=".s0_export_", dir=output_root))
    backup = staging / "previous_S0"
    branch = staging / "S0"
    installed = False
    try:
        # A fresh/replaced population invalidates stale evaluation results.
        branch.mkdir()
        exported = []
        for graph, sample_id in zip(graphs, ids):
            graph = graph.copy()
            graph.graph["sample_id"] = sample_id
            exported.append(graph)
        with (branch / "base_graphs.pkl").open("wb") as stream:
            pickle.dump(exported, stream, protocol=pickle.HIGHEST_PROTOCOL)
        metadata["base_graphs"] = {"path": "base_graphs.pkl", "sha256": sha256(branch / "base_graphs.pkl")}
        write_json(branch / "manifest.json", metadata)
        require(all(sha256(path) == value for path, value in hashes.items()), "Inputs changed during export")
        if destination.exists():
            destination.rename(backup)
        try:
            branch.rename(destination)
            installed = True
        except BaseException:
            if backup.exists():
                backup.rename(destination)
            raise
        return destination / "manifest.json"
    finally:
        if backup.exists() and not installed:
            print(f"Original S0 backup retained at {backup}", file=sys.stderr)
        else:
            shutil.rmtree(staging)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s0 = sub.add_parser("s0", help="Export an existing no-summary-supervision generation")
    s0.add_argument("--generation-dir", required=True, type=Path)
    s0.add_argument("--output-root", required=True, type=Path)
    s0.add_argument("--reference-generation-dir", required=True, type=Path)
    s0.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        path = export_s0(args.generation_dir, args.output_root, args.reference_generation_dir, overwrite=args.overwrite)
    except (OSError, ValueError, pickle.UnpicklingError, EOFError) as error:
        print(f"S0 export failed: {error}", file=sys.stderr)
        return 2
    print(f"S0 export ready: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
