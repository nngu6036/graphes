#!/usr/bin/env python
"""Internal worker that runs DeFoG sampling and emits a neutral graph batch.

The worker intentionally imports no GraphER modules. DeFoG's source tree uses
both ``src.*`` and unqualified imports, so the parent process supplies only the
DeFoG source roots on ``PYTHONPATH``. The resulting NPZ contains primitive
numeric arrays and is safe to load with ``allow_pickle=False``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import random
import subprocess
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

EXPORT_FORMAT = "defog_graph_batch_v2"
GENERIC_DATASETS = frozenset({"comm20", "planar", "sbm", "tree"})
MOLECULAR_ATOMIC_NUMBERS = {
    "qm9": (6, 7, 8, 9),
    "zinc": (6, 7, 8, 9, 15, 16, 17, 35, 53),
}
MOLECULAR_BOND_TYPES = {
    "qm9": (1, 2, 3, 4),
    # The attached DeFoG ZINC pipeline kekulizes aromatic systems.
    "zinc": (1, 2, 3),
}
SUPPORTED_DATASETS = frozenset(GENERIC_DATASETS | MOLECULAR_ATOMIC_NUMBERS.keys())


def _progress_enabled() -> bool:
    return os.environ.get("GRAPHER_DEFOG_PROGRESS_ENABLED", "0") == "1"


def _progress(message: str) -> None:
    if _progress_enabled():
        print(f"[GraphER/DeFoG] {message}", flush=True)


def _generation_progress_interval() -> int:
    raw = os.environ.get("GRAPHER_DEFOG_GENERATION_PROGRESS_INTERVAL", "1")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "GRAPHER_DEFOG_GENERATION_PROGRESS_INTERVAL must be a positive "
            f"integer; received {raw!r}."
        ) from exc
    if value <= 0:
        raise ValueError(
            "GRAPHER_DEFOG_GENERATION_PROGRESS_INTERVAL must be positive; "
            f"received {value}."
        )
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=5,
            shell=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else None


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _integer_array(value: Any, *, name: str) -> np.ndarray:
    array = _to_numpy(value)
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite numeric values.")
    rounded = np.rint(array)
    if not np.array_equal(array, rounded):
        raise ValueError(f"{name} must contain integral class labels.")
    return rounded.astype(np.int64, copy=False)


def _schema(dataset: str) -> dict[str, Any]:
    if dataset in GENERIC_DATASETS:
        return {
            "domain": "generic",
            "node_classes": [0],
            "edge_classes": [0, 1],
            "representation": "simple_undirected_topology",
        }
    return {
        "domain": "molecular",
        "atomic_numbers_by_node_class": list(MOLECULAR_ATOMIC_NUMBERS[dataset]),
        "bond_types": list(MOLECULAR_BOND_TYPES[dataset]),
        "representation": (
            "explicit_aromatic" if dataset == "qm9" else "kekule_no_aromatic_class"
        ),
    }


def encode_samples(samples: Sequence[Any], *, dataset: str) -> dict[str, np.ndarray]:
    """Validate raw DeFoG ``[X, E]`` samples and encode a ragged graph batch."""

    dataset = str(dataset).lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported DeFoG dataset {dataset!r}.")
    if not samples:
        raise ValueError("DeFoG returned no samples.")

    node_ptr = [0]
    edge_ptr = [0]
    all_node_labels: list[np.ndarray] = []
    all_endpoints: list[np.ndarray] = []
    all_edge_labels: list[np.ndarray] = []
    max_node_class = (
        0
        if dataset in GENERIC_DATASETS
        else len(MOLECULAR_ATOMIC_NUMBERS[dataset]) - 1
    )
    allowed_edge_labels = (
        {0, 1}
        if dataset in GENERIC_DATASETS
        else {0, *MOLECULAR_BOND_TYPES[dataset]}
    )

    for index, sample in enumerate(samples):
        if not isinstance(sample, (list, tuple)) or len(sample) != 2:
            raise TypeError(f"DeFoG sample {index} must be a two-item [X, E] pair.")
        node_labels = _integer_array(sample[0], name=f"sample[{index}].X")
        edge_labels = _integer_array(sample[1], name=f"sample[{index}].E")
        if node_labels.ndim != 1 or node_labels.size <= 0:
            raise ValueError(f"DeFoG sample {index} X must have shape [N], N > 0.")
        n = int(node_labels.size)
        if edge_labels.shape != (n, n):
            raise ValueError(
                f"DeFoG sample {index} E has shape {edge_labels.shape}, "
                f"expected {(n, n)}."
            )
        if np.any(node_labels < 0) or np.any(node_labels > max_node_class):
            if dataset in GENERIC_DATASETS:
                raise ValueError(
                    f"DeFoG sample {index} contains multiple node classes; "
                    "generic topology exports require class zero only."
                )
            raise ValueError(
                f"DeFoG sample {index} contains a node class outside "
                f"[0, {max_node_class}]."
            )
        if not np.all(np.isin(edge_labels, tuple(sorted(allowed_edge_labels)))):
            if dataset in GENERIC_DATASETS:
                raise ValueError(
                    f"DeFoG sample {index} contains an edge label outside 0/1."
                )
            raise ValueError(
                f"DeFoG sample {index} contains an unsupported edge class for "
                f"{dataset}."
            )
        if not np.array_equal(edge_labels, edge_labels.T):
            raise ValueError(f"DeFoG sample {index} edge matrix is not symmetric.")
        if np.any(np.diag(edge_labels) != 0):
            raise ValueError(f"DeFoG sample {index} contains a self-loop class.")

        endpoints = np.argwhere(np.triu(edge_labels > 0, k=1)).astype(
            np.int64, copy=False
        )
        present_labels = (
            edge_labels[endpoints[:, 0], endpoints[:, 1]].astype(np.int64, copy=False)
            if endpoints.size
            else np.empty((0,), dtype=np.int64)
        )
        all_node_labels.append(node_labels)
        all_endpoints.append(endpoints.reshape(-1, 2))
        all_edge_labels.append(present_labels)
        node_ptr.append(node_ptr[-1] + n)
        edge_ptr.append(edge_ptr[-1] + int(endpoints.shape[0]))

    packed_endpoints = (
            np.concatenate(all_endpoints, axis=0).astype(np.int64, copy=False)
            if any(array.size for array in all_endpoints)
            else np.empty((0, 2), dtype=np.int64)
    )
    packed_edges = (
            np.concatenate(all_edge_labels).astype(np.int64, copy=False)
            if any(array.size for array in all_edge_labels)
            else np.empty((0,), dtype=np.int64)
    )
    packed_nodes = np.concatenate(all_node_labels).astype(np.int64, copy=False)
    common = {
        "node_ptr": np.asarray(node_ptr, dtype=np.int64),
        "edge_ptr": np.asarray(edge_ptr, dtype=np.int64),
        "edge_endpoints": packed_endpoints,
        "raw_indices": np.arange(len(samples), dtype=np.int64),
    }
    if dataset in MOLECULAR_ATOMIC_NUMBERS:
        atom_lookup = np.asarray(MOLECULAR_ATOMIC_NUMBERS[dataset], dtype=np.int64)
        return {
            "format": np.asarray("grapher_defog_molecular_v1"),
            "dataset": np.asarray(dataset),
            "representation": np.asarray("model"),
            **common,
            "node_atomic_numbers": atom_lookup[packed_nodes],
            "edge_bond_types": packed_edges,
        }
    return {
        "format_version": np.asarray([2], dtype=np.int64),
        "dataset": np.asarray([dataset], dtype="<U32"),
        **common,
        "node_labels": packed_nodes,
        "edge_labels": packed_edges,
    }


def encode_generic_samples(samples: Sequence[Any]) -> dict[str, np.ndarray]:
    """Backward-compatible generic encoder used by existing tests."""

    return encode_samples(samples, dataset="comm20")


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _load_raw_pickle(path: Path) -> list[Any]:
    # This is an explicit, trusted upstream artifact. Never expose this option
    # to arbitrary user uploads without an isolation boundary.
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, (list, tuple)):
        raise TypeError("The DeFoG pickle must contain a sample list.")
    return list(value)


def _compose_config(args: argparse.Namespace) -> Any:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    if args.resolved_config is not None:
        cfg = OmegaConf.load(args.resolved_config)
    else:
        with initialize_config_dir(
            version_base="1.3", config_dir=str(args.defog_root / "configs")
        ):
            cfg = compose(
                config_name="config",
                overrides=[
                    f"+experiment={args.experiment}",
                    f"dataset={args.dataset}",
                ],
            )
    with open_dict(cfg):
        if args.dataset_datadir is not None:
            cfg.dataset.datadir = str(args.dataset_datadir)
        cfg.train.seed = int(args.seed)
        cfg.train.batch_size = int(args.batch_size)
        cfg.train.num_workers = 0
        cfg.general.wandb = "disabled"
        cfg.general.gpus = 1 if str(args.device).startswith("cuda") else 0
        cfg.general.conditional = False
        if args.sample_steps is not None:
            cfg.sample.sample_steps = int(args.sample_steps)
        if args.eta is not None:
            cfg.sample.eta = float(args.eta)
        if args.omega is not None:
            cfg.sample.omega = float(args.omega)
        if args.time_distortion is not None:
            cfg.sample.time_distortion = str(args.time_distortion)
        if args.rdb is not None:
            cfg.sample.rdb = str(args.rdb)
        if args.rdb_crit is not None:
            cfg.sample.rdb_crit = str(args.rdb_crit)
        cfg.sample.search = False
    return cfg


def _resolve_device(requested: str, torch: Any) -> Any:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"DeFoG device {requested!r} requested without CUDA.")
    return torch.device(requested)


def _model_parts(
    cfg: Any,
    dataset_name: str,
    *,
    molecular_statistics: Path | None,
) -> tuple[Any, Any, Any]:
    from models.extra_features import DummyExtraFeatures, ExtraFeatures

    if dataset_name in GENERIC_DATASETS:
        from datasets.spectre_dataset import SpectreDatasetInfos, SpectreGraphDataModule

        datamodule = SpectreGraphDataModule(cfg)
        dataset_infos = SpectreDatasetInfos(datamodule, cfg.dataset)
        domain_features = DummyExtraFeatures()
    elif dataset_name == "qm9":
        from datasets import qm9_dataset
        from models.extra_features_molecular import ExtraMolecularFeatures

        datamodule = qm9_dataset.QM9DataModule(cfg)
        dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    elif dataset_name == "zinc":
        from datasets import zinc_dataset
        from models.extra_features_molecular import ExtraMolecularFeatures

        datamodule = zinc_dataset.ZINCDataModule(cfg)
        dataset_infos = zinc_dataset.ZINCinfos(datamodule=datamodule, cfg=cfg)
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    else:  # pragma: no cover - parser and caller both validate this
        raise ValueError(f"Unsupported DeFoG dataset {dataset_name!r}.")

    if dataset_name in MOLECULAR_ATOMIC_NUMBERS:
        if molecular_statistics is not None:
            if __package__:
                from .molecular_runtime import apply_cached_statistics
            else:
                from molecular_runtime import apply_cached_statistics

            apply_cached_statistics(
                dataset_infos,
                molecular_statistics,
                dataset=dataset_name,
            )
        else:
            if __package__:
                from .molecular_runtime import apply_empirical_statistics
            else:
                from molecular_runtime import apply_empirical_statistics

            apply_empirical_statistics(
                dataset_infos,
                datamodule,
                dataset=dataset_name,
            )

    extra_features = ExtraFeatures(
        cfg.model.extra_features,
        cfg.model.rrwp_steps,
        dataset_info=dataset_infos,
    )
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features,
    )
    return dataset_infos, extra_features, domain_features


def _generate_samples(
    args: argparse.Namespace,
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    import pytorch_lightning as pl
    import torch
    from graph_discrete_flow_model import GraphDiscreteFlowModel
    from omegaconf import OmegaConf

    _progress(
        "generation worker initializing: "
        f"dataset={args.dataset}, requested={args.num_samples}, "
        f"batch_size={args.batch_size}, seed={args.seed}"
    )
    cfg = _compose_config(args)
    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_name = str(cfg.dataset.name).lower()
    if dataset_name != args.dataset or dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Resolved DeFoG dataset {dataset_name!r} does not match "
            f"the requested dataset {args.dataset!r}."
        )

    _progress("loading DeFoG dataset metadata and model components")
    dataset_infos, extra_features, domain_features = _model_parts(
        cfg,
        dataset_name,
        molecular_statistics=args.molecular_statistics,
    )
    model = GraphDiscreteFlowModel(
        cfg=cfg,
        dataset_infos=dataset_infos,
        train_metrics=None,
        sampling_metrics=None,
        visualization_tools=None,
        extra_features=extra_features,
        domain_features=domain_features,
        test_labels=None,
    )
    _progress(f"loading checkpoint from {args.checkpoint}")
    try:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(args.checkpoint, map_location="cpu")
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise TypeError("The DeFoG checkpoint has no state_dict mapping.")
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    allowed_auxiliary_prefixes = ("sampling_metrics.", "train_metrics.")
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith(allowed_auxiliary_prefixes)
    ]
    if missing or unexpected:
        raise RuntimeError(
            "DeFoG checkpoint/model mismatch: "
            f"missing={missing}, unexpected={unexpected}. Only saved metric "
            "buffers may be absent from the sampling-only model."
        )
    device = _resolve_device(args.device, torch)
    model.to(device)
    model.limit_dist.to_device(device)
    model.eval()
    _progress(f"checkpoint loaded; sampling device={device}")

    samples: list[Any] = []
    offset = 0
    batch_index = 0
    total_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    report_every = _generation_progress_interval()
    # Sampling is an inference-only operation.  Disabling autograd is
    # especially important for long QM9/ZINC runs, where retaining graphs for
    # hundreds of reverse-flow steps can otherwise consume unnecessary memory.
    with torch.inference_mode():
        while offset < args.num_samples:
            current_batch = min(args.batch_size, args.num_samples - offset)
            batch_index += 1
            batch_samples, _ = model.sample_batch(
                batch_id=offset,
                batch_size=current_batch,
                keep_chain=0,
                number_chain_steps=1,
                save_final=0,
                num_nodes=None,
                save_visualization=False,
            )
            if len(batch_samples) != current_batch:
                raise RuntimeError(
                    f"DeFoG returned {len(batch_samples)} samples for a "
                    f"batch of {current_batch}."
                )
            samples.extend(batch_samples)
            offset += current_batch
            if (
                batch_index == 1
                or batch_index % report_every == 0
                or offset == args.num_samples
            ):
                _progress(
                    "generation progress: "
                    f"batch={batch_index}/{total_batches}, "
                    f"generated={offset}/{args.num_samples}"
                )
    _progress("all requested DeFoG samples have been produced")
    runtime: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "pytorch_lightning": str(pl.__version__),
        "cuda": str(torch.version.cuda),
    }
    empirical_statistics = getattr(
        dataset_infos, "grapher_empirical_statistics", None
    )
    if empirical_statistics is not None:
        runtime["molecular_statistics"] = empirical_statistics
    effective_sampling = {
        key: OmegaConf.select(cfg, f"sample.{key}", default=None)
        for key in (
            "sample_steps",
            "time_distortion",
            "eta",
            "omega",
            "rdb",
            "rdb_crit",
            "search",
        )
    }
    effective_sampling["device"] = str(device)
    return samples, runtime, effective_sampling


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and export DeFoG graph samples for GraphER."
    )
    parser.add_argument("--defog-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(SUPPORTED_DATASETS), required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--input-pickle", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--resolved-config", type=Path, default=None)
    parser.add_argument("--dataset-datadir", type=Path, default=None)
    parser.add_argument("--molecular-statistics", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--omega", type=float, default=None)
    parser.add_argument("--time-distortion", default=None)
    parser.add_argument("--rdb", default=None)
    parser.add_argument("--rdb-crit", default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.defog_root = args.defog_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    if args.resolved_config is not None:
        args.resolved_config = args.resolved_config.expanduser().resolve()
        if not args.resolved_config.is_file():
            raise FileNotFoundError(f"Missing DeFoG resolved config: {args.resolved_config}")
    if args.dataset_datadir is not None:
        args.dataset_datadir = args.dataset_datadir.expanduser().resolve()
        if not args.dataset_datadir.is_dir():
            raise FileNotFoundError(f"Missing DeFoG dataset directory: {args.dataset_datadir}")
    if args.molecular_statistics is not None:
        args.molecular_statistics = args.molecular_statistics.expanduser().resolve()
        if not args.molecular_statistics.is_file():
            raise FileNotFoundError(
                "Missing DeFoG molecular-statistics cache: "
                f"{args.molecular_statistics}"
            )
        if args.dataset not in MOLECULAR_ATOMIC_NUMBERS:
            raise ValueError(
                "--molecular-statistics is valid only for QM9/ZINC generation."
            )
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("num-samples and batch-size must be positive.")
    if args.sample_steps is not None and args.sample_steps <= 0:
        raise ValueError("sample-steps must be positive when specified.")
    if args.input_pickle is not None and args.checkpoint is not None:
        raise ValueError("Use either --input-pickle or --checkpoint, not both.")
    if args.input_pickle is None and args.checkpoint is None:
        raise ValueError("--checkpoint is required for DeFoG generation.")
    if not (args.defog_root / "src" / "main.py").is_file():
        raise FileNotFoundError(f"Invalid DeFoG root: {args.defog_root}")

    if args.input_pickle is not None:
        args.input_pickle = args.input_pickle.expanduser().resolve()
        samples = _load_raw_pickle(args.input_pickle)
        versions = {"python": platform.python_version(), "mode": "trusted_pickle_conversion"}
        source = {"kind": "trusted_pickle", "path": str(args.input_pickle)}
        checkpoint_info = None
        sampling_info: dict[str, Any] = {"status": "unknown_from_pickle", "device": None}
    else:
        args.checkpoint = args.checkpoint.expanduser().resolve()
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"Missing DeFoG checkpoint: {args.checkpoint}")
        samples, versions, effective_sampling = _generate_samples(args)
        source = {"kind": "checkpoint_generation"}
        checkpoint_info = {"path": str(args.checkpoint), "sha256": _sha256(args.checkpoint)}
        sampling_info = {
            "batch_size": int(args.batch_size),
            "sample_steps_override": args.sample_steps,
            "eta_override": args.eta,
            "omega_override": args.omega,
            "time_distortion_override": args.time_distortion,
            "rdb_override": args.rdb,
            "rdb_crit_override": args.rdb_crit,
            "defaults_source": "resolved_training_config_or_experiment",
            "device": str(args.device),
            "effective": effective_sampling,
        }
    if len(samples) != args.num_samples:
        raise ValueError(f"DeFoG supplied {len(samples)} samples, expected {args.num_samples}.")

    _progress("validating and encoding generated graphs into the neutral NPZ")
    arrays = encode_samples(samples, dataset=args.dataset)
    _atomic_npz(args.output, arrays)
    manifest = {
        "format": EXPORT_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "schema": _schema(args.dataset),
        "dataset_datadir": str(args.dataset_datadir) if args.dataset_datadir is not None else None,
        "resolved_config": str(args.resolved_config) if args.resolved_config is not None else None,
        "molecular_statistics_cache": (
            {
                "path": str(args.molecular_statistics),
                "sha256": _sha256(args.molecular_statistics),
            }
            if args.molecular_statistics is not None
            else None
        ),
        "experiment": args.experiment,
        "seed": int(args.seed),
        "requested_samples": int(args.num_samples),
        "exported_samples": len(samples),
        "filtered_or_dropped": 0,
        "sampling": sampling_info,
        "source": source,
        "checkpoint": checkpoint_info,
        "defog": {"root": str(args.defog_root), "revision": _source_revision(args.defog_root)},
        "runtime": versions,
        "export": {"path": str(args.output), "sha256": _sha256(args.output), "allow_pickle": False},
    }
    _atomic_json(args.manifest, manifest)
    _progress(f"generation export and manifest written under {args.output.parent}")
    print(f"Exported {len(samples)} {args.dataset} DeFoG graphs to {args.output}")


if __name__ == "__main__":
    main()
