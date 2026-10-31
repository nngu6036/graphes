#!/usr/bin/env python3
"""Validate trusted saved ablation artifacts before reusing them.

Exit codes: 0 verified; 3 absent (creation may be allowed by the caller);
2 incompatible, corrupt or incomplete (never silently regenerate). ``begin``
and ``seal`` bracket a new successful generation and record configuration
provenance that older molecular manifests did not retain. No model is loaded,
trained or sampled by this helper. Graph pickles must be trusted project files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

RECEIPT = "ablation_reuse_receipt.json"
FORMAT = "graphes_ablation_reuse_receipt_v1"
GENERATION_ONLY = {"sample", "graphlet_refinement", "attributed_diversification", "generation_batch_size", "runtime"}


class Absent(Exception):
    pass


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")
    return value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def subset(expected: Any, actual: Any, label: str) -> None:
    """Compare every explicitly configured field; resolved defaults may add keys."""
    if isinstance(expected, Mapping):
        require(isinstance(actual, Mapping), f"Missing configuration mapping: {label}")
        for key, value in expected.items():
            require(key in actual, f"Unrecorded configuration field: {label}.{key}")
            subset(value, actual[key], f"{label}.{key}")
    else:
        require(expected == actual, f"Configuration mismatch at {label}: expected {expected!r}, recorded {actual!r}")


def config_options(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict) and isinstance(value.get("grapher"), dict), f"Not an explicit GraphER YAML: {path}")
    return value["grapher"]


def artifact(directory: Path, record: Any, label: str) -> Path:
    require(isinstance(record, dict) and bool(record.get("path")) and bool(record.get("sha256")), f"Missing hash/path for {label}")
    path = directory / str(record["path"])
    require(path.resolve().is_relative_to(directory.resolve()), f"Artifact escapes its directory: {label}")
    require(path.is_file(), f"Missing artifact: {path}")
    require(sha256(path) == record["sha256"], f"Artifact hash mismatch: {path}")
    return path


def graph_population(path: Path, count: int, ids: list[str], *, allow_unlabelled: bool = False) -> None:
    import networkx as nx
    from grapher.utils.networkx_pickle import load_trusted_networkx_pickle
    graphs = load_trusted_networkx_pickle(path)
    require(isinstance(graphs, list) and len(graphs) == count, f"Saved population count differs: {path}")
    require(len(ids) == count, f"Metadata sample count differs: {path}")
    for graph, sample_id in zip(graphs, ids):
        require(isinstance(graph, nx.Graph), f"Non-graph object in {path}")
        actual = graph.graph.get("sample_id")
        require((actual is None and allow_unlabelled) or str(actual) == sample_id, f"Saved graph ID/order differs: {path}: {sample_id}")


def training(args, cfg: dict, *, missing_ok: bool = False) -> tuple[dict, dict]:
    directory = args.run_dir / "train"
    if not directory.exists():
        if missing_ok:
            raise Absent(f"Training artifacts are absent: {directory}")
        raise ValueError(f"Cannot verify generation without its training artifacts: {directory}")
    manifest = read_json(directory / "manifest.json")
    require(manifest.get("model_id") == "grapher", "Training model_id is not grapher")
    require(manifest.get("run_id") == args.run_dir.name, "Training run ID differs from requested managed run")
    require(manifest.get("train_seed") == args.seed, "Training seed differs")
    checkpoint = artifact(directory, manifest.get("checkpoint"), "training checkpoint")
    recorded = manifest.get("options", {})
    for key, value in cfg.items():
        if key in GENERATION_ONLY:
            continue
        if key == "attributed":
            value = {k: v for k, v in value.items() if k != "decode"}
        subset(value, recorded.get(key), f"training.{key}")
    selection = manifest.get("checkpoint_selection", {})
    require(selection.get("kind") == "final_configured_epoch", "Checkpoint is not the final configured epoch")
    require(selection.get("epoch") == cfg.get("train", {}).get("epochs"), "Training epoch budget differs")
    data = manifest.get("dataset", {})
    require(data.get("benchmark_id") == args.dataset and data.get("serialized_id") == args.serialized_dataset, "Training dataset identity differs")
    split_hashes = {split: sha256(args.dataset_root / args.serialized_dataset / f"{split}.pkl") for split in ("train", "val", "test")}
    require(data.get("split_sha256") == split_hashes, "Prepared dataset split hashes differ from training")
    fingerprint = digest({split: {"path": f"{split}.pkl", "sha256": split_hashes[split]} for split in sorted(split_hashes)})
    require(data.get("fingerprint") == fingerprint, "Training dataset fingerprint differs")
    context = {
        "format": FORMAT, "dataset": args.dataset, "serialized_dataset": args.serialized_dataset,
        "seed": args.seed, "run_id": args.run_dir.name,
        "checkpoint_sha256": sha256(checkpoint), "training_manifest_sha256": sha256(directory / "manifest.json"),
        "dataset_split_sha256": split_hashes, "config_semantic_sha256": digest(cfg),
    }
    return manifest, context


def expected_ids(args) -> list[str]:
    return [f"{args.seed}:{i}" for i in range(args.num_samples)]


def invocation_context(args, cfg: dict) -> tuple[dict, dict]:
    trained, context = training(args, cfg)
    require(args.generation_dir.parent.parent.resolve() == args.run_dir.resolve(), "Generation directory is outside the requested managed run")
    context.update(kind=args.kind, num_samples=args.num_samples, generation_id=args.generation_dir.name)
    return trained, context


def verify_generation(args, cfg: dict, *, allow_unsealed: bool = False) -> tuple[dict, dict, dict]:
    directory = args.generation_dir
    if not directory.exists():
        raise Absent(f"Generation is absent: {directory}")
    trained, context = invocation_context(args, cfg)
    generation = read_json(directory / "manifest.json")
    require(generation.get("model_id") == "grapher", "Generation model_id is not grapher")
    for key, value in (("run_id", args.run_dir.name), ("generation_id", directory.name), ("generation_seed", args.seed), ("num_requested", args.num_samples), ("num_generated", args.num_samples)):
        require(generation.get(key) == value, f"Generation {key} differs: expected {value!r}")
    require(generation.get("checkpoint", {}).get("sha256") == context["checkpoint_sha256"], "Generation uses a different checkpoint from the current managed run")
    require(generation.get("variant") == cfg.get("variant"), "Generation variant differs")
    base = artifact(directory, generation.get("base_graphs"), "generation base_graphs")
    graph_population(base, args.num_samples, expected_ids(args), allow_unlabelled=args.kind == "s0")
    artifact(directory, generation.get("sampled_basis_indices"), "sampled basis indices")
    sampling = generation.get("sampling", {})
    if args.kind in {"matched", "s0"}:
        subset(cfg["sample"], generation.get("effective_sample_config"), "effective_sample_config")
        subset(cfg["sde"], generation.get("sde"), "generation.sde")
        require(generation.get("generation_batch_size") == cfg.get("generation_batch_size"), "Generation batch size differs")
        if args.kind == "matched":
            require(generation.get("matched_ablation", {}).get("enabled") is True, "Generation has no matched decoder population")
            common_path = artifact(directory, generation.get("matched_ablation"), "matched ablation index")
            common = read_json(common_path)
            require(common.get("sample_ids") == expected_ids(args), "Shared requested sample IDs differ")
            require(common.get("generation_seed") == args.seed and common.get("num_requested") == args.num_samples, "Shared generation seed/count differs")
            provenance = common.get("provenance", {})
            require(provenance.get("checkpoint_sha256") == context["checkpoint_sha256"], "Shared-index checkpoint differs")
            subset(cfg["sample"], provenance.get("sample_config"), "shared.sample_config")
            subset(cfg["sde"], provenance.get("sde"), "shared.sde")
            require(provenance.get("generation_batch_size") == cfg.get("generation_batch_size"), "Shared-index batch size differs")
            refinement = cfg.get("graphlet_refinement", {})
            actual = sampling.get("stochastic_diversification", {})
            for key in ("random_steps", "refiner", "target_source"):
                subset(refinement.get(key), actual.get(key), f"generation.refinement.{key}")
            for name in ("pre_realization_graphs", "pre_refinement_graphs"):
                artifact(directory, generation.get(name), name)
        else:
            require(not sampling.get("rewiring"), "S0 must not rewire its decoded graph")
            require(cfg.get("structure_summary", {}).get("loss_weight") == 0.0, "S0 must use zero topology-summary supervision")
            require(not cfg.get("graphlet_refinement", {}).get("enabled"), "S0 configuration enables rewiring")
    else:
        require(not cfg.get("sample", {}).get("degree_constrained_realization", {}).get("enabled"), "Exact-degree molecular generation is unsupported")
        require(sampling.get("generation_batch_size") == cfg.get("generation_batch_size"), "Molecular generation batch size differs")
        require(sampling.get("separate_attribute_rng") == cfg.get("sample", {}).get("separate_attribute_rng"), "Attribute RNG policy differs")
        subset(cfg.get("attributed_diversification"), sampling.get("attributed_diversification"), "generation.attributed_diversification")
        for key, value in cfg.get("attributed", {}).get("decode", {}).items():
            mapped = {"node_mode": "node_decode_mode", "edge_mode": "edge_decode_mode"}.get(key, key)
            subset(value, sampling.get(mapped), f"generation.decode.{key}")
        for key, target in (("steps", "attribute_diffusion_steps"), ("sample_steps", "attribute_sampling_steps")):
            subset(cfg.get("attribute_diffusion", {}).get(key), sampling.get(target), f"generation.{target}")
        artifact(directory, generation.get("pre_refinement_graphs"), "molecular pre_refinement_graphs")
    receipt_path = directory / RECEIPT
    receipt = read_json(receipt_path) if receipt_path.is_file() else None
    if receipt is not None:
        require(receipt.get("context") == context, "Reuse receipt configuration/checkpoint/training provenance differs")
        require(receipt.get("generation_manifest_sha256") == sha256(directory / "manifest.json"), "Generation manifest changed after the reuse receipt")
    elif args.kind == "molecular" and not allow_unsealed:
        raise ValueError("Legacy molecular manifest does not record the full continuous sampler and has no fresh-generation receipt. Existing output is retained; explicitly choose a new GENERATION_TAG to generate a verifiable population.")
    return generation, context, receipt or {}


def verify_branches(args, cfg: dict, generation: dict, receipt: dict, variants: list[str]) -> dict[str, str]:
    root = args.generation_dir / "ablations"
    common = read_json(root / "ablation_manifest.json")
    require(common.get("sample_ids") == expected_ids(args), "Branch index requested IDs differ")
    snapshots = {}
    for variant in variants:
        directory = root / variant
        child_path = directory / "manifest.json"
        if not child_path.is_file():
            raise Absent(f"Required saved branch is absent: {directory}")
        child = read_json(child_path)
        shared = common.get("branches", {}).get(variant)
        if shared is not None:
            for key, value in shared.items():
                if key != "manifest_path":
                    require(child.get(key) == value, f"Branch/index mismatch for {variant}.{key}")
        require(child.get("generation_seed") == args.seed, f"Branch generation seed differs: {variant}")
        require(child.get("requested_count") == args.num_samples, f"Branch requested count differs: {variant}")
        ids = child.get("sample_ids")
        require(isinstance(ids, list) and len(ids) == len(set(ids)), f"Branch IDs are missing or duplicated: {variant}")
        successful_ids = set(ids)
        requested = child.get("requested_sample_ids", common.get("sample_ids"))
        require(requested == expected_ids(args), f"Requested branch ID order differs: {variant}")
        require(ids == [sample_id for sample_id in requested if sample_id in successful_ids], f"Successful branch IDs are unknown/reordered: {variant}")
        count = child.get("completed_count")
        require(count == len(ids), f"Branch completed count differs: {variant}")
        failures = child.get("failures", [])
        missing = [sample_id for sample_id in requested if sample_id not in successful_ids]
        require(child.get("failed_count") == len(missing), f"Branch failure count differs: {variant}")
        require([row.get("sample_id") for row in failures] == missing, f"Missing explicit branch failure records: {variant}")
        expected_status = "complete" if not missing else ("partial" if ids else "failed")
        require(child.get("status") == expected_status, f"Inconsistent branch status: {variant}")
        require(not missing or variant == "D0", f"Primary branch is incomplete: {variant}")
        graph_path = artifact(directory, child.get("base_graphs"), f"{variant}.base_graphs")
        graph_population(graph_path, count, ids)
        if args.kind == "matched" and variant in {"T", "D", "DR"}:
            source = {"T": "pre_realization_graphs", "D": "pre_refinement_graphs", "DR": "base_graphs"}[variant]
            artifact(args.generation_dir, generation.get(source), source)
            require(child["base_graphs"]["sha256"] == generation[source]["sha256"],
                    f"Branch {variant} differs from its hash-anchored parent population")
        if variant in receipt.get("branch_manifest_sha256", {}):
            require(sha256(child_path) == receipt["branch_manifest_sha256"][variant], f"Branch manifest changed after receipt: {variant}")
        if args.kind == "molecular":
            provenance = child.get("provenance", {})
            source = "pre_refinement_graphs" if variant == "T" else "base_graphs"
            require(provenance.get("source_sha256") == generation.get(source, {}).get("sha256"), f"Molecular branch source differs: {variant}")
            require(provenance.get("source_generation_manifest_sha256") == sha256(args.generation_dir / "manifest.json"), f"Molecular branch generation provenance differs: {variant}")
        snapshots[variant] = sha256(child_path)
        if missing:
            print(f"Retaining {variant}: {count}/{args.num_samples} outputs; {len(missing)} recorded failures; no replacements.")
    return snapshots


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("training", "generation", "branches", "begin", "seal"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--serialized-dataset", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--generation-dir", type=Path)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--kind", choices=("matched", "molecular", "s0"))
    parser.add_argument("--variants", nargs="+", choices=("T", "D", "TR", "DR", "D0", "R1", "R12", "S0"), default=[])
    args = parser.parse_args(argv)
    if args.mode != "training" and (args.generation_dir is None or args.num_samples is None or args.num_samples <= 0 or args.kind is None):
        parser.error("Generation operations require --generation-dir, positive --num-samples, and --kind")
    try:
        cfg = config_options(args.config)
        if args.mode == "training":
            training(args, cfg, missing_ok=True)
        elif args.mode == "begin":
            require(not args.generation_dir.exists(), "Generation already exists; begin cannot authorize regeneration")
            _, context = invocation_context(args, cfg)
            write_json(args.generation_dir.parent / f"{args.generation_dir.name}.ablation_generation_intent.json", {"context": context, "config_file_sha256": sha256(args.config)})
        elif args.mode == "seal":
            intent_path = args.generation_dir.parent / f"{args.generation_dir.name}.ablation_generation_intent.json"
            intent = read_json(intent_path)
            generation, context, _ = verify_generation(args, cfg, allow_unsealed=True)
            require(intent.get("context") == context, "Configuration/checkpoint changed during generation; refusing a reuse receipt")
            branches = {}
            if args.kind == "matched":
                variants = list(read_json(args.generation_dir / "ablations" / "ablation_manifest.json").get("branches", {}))
                branches = verify_branches(args, cfg, generation, {}, variants)
            write_json(args.generation_dir / RECEIPT, {
                "format": FORMAT, "context": context,
                "config_file_sha256": intent.get("config_file_sha256"),
                "generation_manifest_sha256": sha256(args.generation_dir / "manifest.json"),
                "branch_manifest_sha256": branches,
                "provenance": "Written only after a successfully completed, explicitly bracketed generation invocation",
            })
            intent_path.unlink()
        else:
            generation, _, receipt = verify_generation(args, cfg)
            if args.mode == "branches":
                require(bool(args.variants), "branches mode requires --variants")
                verify_branches(args, cfg, generation, receipt, args.variants)
        print(f"Verified {args.mode}: {args.generation_dir or args.run_dir}")
        return 0
    except Absent as error:
        print(str(error), file=sys.stderr)
        return 3
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        print(f"Reuse rejected: {error}\nExisting artifacts were not modified. Choose a new run/generation tag for a different experiment.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
