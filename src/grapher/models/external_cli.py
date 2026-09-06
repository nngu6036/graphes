"""Uniform separate train/generate entry points for source-backed baselines."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import yaml

from grapher.models import DatasetReference, GenerateRequest, RunSpec, TrainRequest, create_baseline, normalize_baseline_id
from grapher.models.external_codec import PROFILES


def positive(raw: str) -> int:
    n = int(raw)
    if n <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return n


def build_parser(model: str) -> argparse.ArgumentParser:
    wrapper = create_baseline(model)
    parser = argparse.ArgumentParser(description=f"Train/generate {wrapper.display_name} on frozen GraphER splits.")
    parser.add_argument("--stage", choices=("train", "generate", "all"), default="all")
    parser.add_argument("--dataset", required=True, choices=sorted(wrapper.supported_datasets))
    parser.add_argument("--seed-id", "--seed", dest="seed_id", type=int, default=42)
    parser.add_argument("--generation-seed", type=int)
    parser.add_argument("--num-samples", type=positive, default=1024)
    parser.add_argument("--run-id")
    parser.add_argument("--generation-id")
    parser.add_argument("--dataset-root", type=Path, default=Path("outputs/datasets"))
    parser.add_argument("--serialized-dataset")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--wrapper-config", type=Path)
    parser.add_argument("--source-root", f"--{model}-root", dest="source_root", type=Path)
    parser.add_argument("--python", f"--{model}-python", dest="python", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", "--num-epochs", "--n-epochs", dest="epochs", type=positive)
    parser.add_argument("--batch-size", type=positive)
    parser.add_argument("--generation-batch-size", type=positive)
    parser.add_argument("--epoch-progress-interval", type=positive)
    parser.add_argument("--device", default=None, help="auto, cpu, gpu, cuda or cuda:N")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--max-nodes", type=positive)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-training-estimates", action="store_true", help="Compatibility flag: training-estimate export is disabled by default.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(model_id: str, argv: Sequence[str] | None = None) -> int:
    model = normalize_baseline_id(model_id)
    args = build_parser(model).parse_args(argv)
    if args.seed_id < 0 or (args.generation_seed is not None and args.generation_seed < 0):
        raise ValueError("Seeds must be nonnegative.")
    if args.stage == "generate" and any(v is not None for v in (args.epochs, args.batch_size, args.max_nodes)):
        raise ValueError("Generation uses the trained architecture; training overrides require --stage train/all.")
    for name, value in ((model.upper(), args.source_root), (model.upper() + "_PYTHON", args.python)):
        if value is not None:
            os.environ[name] = str(value.expanduser().absolute())
    profile = PROFILES[args.dataset]
    run = RunSpec.for_seed(model_id=model, dataset_id=args.dataset, seed=args.seed_id,
                           output_root=args.output_root, run_id=args.run_id)
    options = {"runtime": {"progress": {"enabled": not args.quiet, "stream_output": not args.quiet}}}
    if args.source_root is not None:
        options["source_root"] = str(args.source_root.expanduser().absolute())
    if args.python is not None:
        options["python"] = str(args.python.expanduser().absolute())
    if args.checkpoint is not None and args.stage != "generate":
        raise ValueError("--checkpoint is for --stage generate; training writes a managed checkpoint.")
    for key in ("device", "cuda_visible_devices", "timeout_seconds"):
        value = getattr(args, key)
        if value is not None: options["runtime"][key] = value
    if args.generation_batch_size is not None: options["generation_batch_size"] = args.generation_batch_size
    config = args.wrapper_config
    if config is None:
        config = Path(__file__).resolve().parents[3] / "configs" / "baselines" / f"{model}_{args.dataset}.yaml"
        if not config.is_file():
            raise FileNotFoundError(f"Default wrapper config not found: {config}")
    wrapper = create_baseline(model)
    summary = {"model": model, "dataset": args.dataset, "run_dir": str(run.layout.run_dir), "stage": args.stage}
    if args.stage in ("train", "all"):
        training_options = dict(options)
        train = {}
        for key, value in (("epochs", args.epochs), ("batch_size", args.batch_size), ("log_every", args.epoch_progress_interval)):
            if value is not None: train[key] = value
        if train: training_options["train"] = train
        if args.max_nodes is not None: training_options["max_nodes"] = args.max_nodes
        result = wrapper.train(TrainRequest(run=run,
                dataset=DatasetReference(args.dataset, root=args.dataset_root, serialized_id=args.serialized_dataset or profile.serialized_id),
                config_path=config, options=training_options, overwrite=args.overwrite))
        checkpoint = result.checkpoint_path
        summary["training_manifest"] = str(result.manifest_path)
    else:
        checkpoint = args.checkpoint or run.layout.checkpoints_dir / (model + ".pt")
        # Runtime/sample options may be overridden at generation, but never
        # silently replace model hyperparameters stored with the checkpoint.
        raw = yaml.safe_load(config.read_text()) or {}
        section = raw.get(model, raw.get("gsdm", raw) if model == "gdsm" else raw)
        from grapher.models.external_wrapper import merge
        generated = {k: v for k, v in section.items() if k in {"runtime", "sample", "generation_batch_size"}}
        options = merge(generated, options)
    summary["checkpoint"] = str(checkpoint)
    if args.stage in ("generate", "all"):
        result = wrapper.generate(GenerateRequest(run=run, checkpoint_path=checkpoint, num_graphs=args.num_samples,
                     generation_seed=args.generation_seed if args.generation_seed is not None else args.seed_id,
                     generation_id=args.generation_id, options=options, overwrite=args.overwrite))
        summary.update(generated_graphs=str(result.graphs_path), generation_manifest=str(result.manifest_path),
                       num_generated=result.num_generated, graphs_sha256=result.graphs_sha256)
    print(json.dumps(summary, indent=2))
    return 0
