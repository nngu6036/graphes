#!/usr/bin/env python
"""Isolated exact-count HOG-Diff two-stage generation worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

try:
    from ._compat import install_torch_functional_alias
except ImportError:  # Direct worker execution uses this file's directory.
    from _compat import install_torch_functional_alias


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _numerical_retry_seed(seed: int, round_index: int, attempt: int) -> int:
    """Return a reproducible, non-overlapping seed for one batch attempt."""
    return int(seed) + int(round_index) + int(attempt) * 1_000_003


def _singleton_retry_seed(
    seed: int,
    round_index: int,
    sample_index: int,
    attempt: int,
) -> int:
    """Return a deterministic seed for an isolated fallback trajectory."""
    return (
        int(seed)
        + 1_000_000_007
        + int(round_index) * 1_000_003
        + int(sample_index) * 10_007
        + int(attempt) * 97
    )


def _is_predictor_nan_error(exc: BaseException) -> bool:
    return isinstance(exc, ValueError) and "NaNs in predictor output" in str(exc)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hogdiff-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--domain", choices=("generic", "attributed"), required=True)
    parser.add_argument("--num-graphs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--progress-every-batches", type=int, default=1)
    parser.add_argument("--max-numerical-retries", type=int, default=8)
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def _load_mode_config(path: Path, *, mode: str):
    import yaml
    from easydict import EasyDict as edict

    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    if not isinstance(raw, dict):
        raise TypeError(f"HOG-Diff resolved config must be a mapping: {path}")
    if mode == "higher-order":
        selected = {key: value for key, value in raw.items() if not str(key).startswith("OU")}
    else:
        selected = dict(raw)
        selected.update(
            {
                str(key)[2:]: value
                for key, value in raw.items()
                if str(key).startswith("OU")
            }
        )
    config = edict(selected)
    edge_th = config.model.get("edge_th")
    if isinstance(edge_th, str):
        config.model.edge_th = float(Fraction(edge_th))
    if int(config.sampling.n_steps) <= 0:
        config.sampling.corrector = "None"
    return config


def _pack_generic(adjacency, *, max_nodes: int):
    import numpy as np

    raw = adjacency.detach().cpu().numpy()
    binary = (raw >= 0.5).astype(np.int8)
    outputs = np.zeros((binary.shape[0], max_nodes, max_nodes), dtype=np.int8)
    sizes = np.ones((binary.shape[0],), dtype=np.int64)
    for index, matrix in enumerate(binary):
        matrix = np.maximum(matrix, matrix.T)
        np.fill_diagonal(matrix, 0)
        active = np.flatnonzero(matrix.sum(axis=0) + matrix.sum(axis=1))
        if active.size == 0:
            sizes[index] = 1
            continue
        packed = matrix[np.ix_(active, active)]
        n = int(packed.shape[0])
        outputs[index, :n, :n] = packed
        sizes[index] = n
    return outputs, sizes


def _pack_molecular(atoms, bonds, sample_nodes, *, max_nodes: int):
    import numpy as np

    atom_channel = atoms.detach().cpu().numpy().argmax(axis=2).astype(np.int16)
    raw_bonds = bonds.detach().cpu().numpy() * 3.0
    quantized = np.zeros_like(raw_bonds, dtype=np.int8)
    quantized[(raw_bonds >= 0.5) & (raw_bonds < 1.5)] = 1
    quantized[(raw_bonds >= 1.5) & (raw_bonds < 2.5)] = 2
    quantized[raw_bonds >= 2.5] = 3
    sizes = sample_nodes.detach().cpu().numpy().astype(np.int64)
    out_bonds = np.zeros((quantized.shape[0], max_nodes, max_nodes), dtype=np.int8)
    out_atoms = np.full((quantized.shape[0], max_nodes), -1, dtype=np.int16)
    for index, n_value in enumerate(sizes):
        n = int(n_value)
        if n < 1 or n > max_nodes:
            raise ValueError(f"HOG-Diff returned invalid molecular node count {n}.")
        out_atoms[index, :n] = atom_channel[index, :n]
        # HOG-Diff's construct_mol reads the lower triangle.  Mirror exactly
        # that triangle into a symmetric GraphER adjacency representation.
        for u in range(n):
            for v in range(u + 1, n):
                category = int(quantized[index, v, u])
                out_bonds[index, u, v] = out_bonds[index, v, u] = category
    return out_atoms, out_bonds, sizes


def main() -> int:
    args = _parser().parse_args()
    if args.num_graphs <= 0 or args.batch_size <= 0:
        raise ValueError("--num-graphs and --batch-size must be positive.")
    if args.progress_every_batches <= 0:
        raise ValueError("--progress-every-batches must be positive.")
    if args.max_numerical_retries < 0:
        raise ValueError("--max-numerical-retries must be non-negative.")
    root = args.hogdiff_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    for required in (root / "sampler.py", config_path, checkpoint):
        if not required.is_file():
            raise FileNotFoundError(required)

    sys.path.insert(0, str(root))
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_SILENT", "true")

    import numpy as np
    import torch
    import torch.nn.functional as torch_functional
    from models import layers as hog_layers

    install_torch_functional_alias(hog_layers, torch_functional)
    from sampler import Sampler
    from utils import loader
    from utils.logger import Logger

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("HOG-Diff worker was configured for GPU but CUDA is unavailable.")

    hconfig = _load_mode_config(config_path, mode="higher-order")
    ouconfig = _load_mode_config(config_path, mode="OU")
    batch_size = min(int(args.batch_size), int(args.num_graphs))
    for config in (hconfig, ouconfig):
        config.ckpt = str(checkpoint)
        config.eval.batch_size = batch_size
        config.eval.num_samples = batch_size
        config.eval.seed = int(args.seed)
        config.exp.plot = False
        config.data.num_workers = 0
    loader.init_exp(ouconfig)
    device = loader.load_device(force=True)

    # Load both trained score networks and the OU training tensor exactly once.
    sample_logger = Logger(ouconfig, is_train=False, show_exc=True)
    hsampler = Sampler(hconfig, sample_logger, mode="higher-order", device=device)
    ousampler = Sampler(ouconfig, sample_logger, mode="OU", device=device)

    generic_batches: list[np.ndarray] = []
    generic_sizes: list[np.ndarray] = []
    molecule_atoms: list[np.ndarray] = []
    molecule_bonds: list[np.ndarray] = []
    molecule_sizes: list[np.ndarray] = []
    numerical_retries: list[dict[str, int]] = []
    singleton_fallbacks: list[dict[str, int]] = []
    rounds = int(math.ceil(float(args.num_graphs) / float(batch_size)))

    def capture_once(active_hsampler, active_ousampler, *, sample_seed: int):
        active_hsampler.config.eval.seed = sample_seed
        active_ousampler.config.eval.seed = sample_seed
        captured: dict[str, Any] = {}
        if args.domain == "generic":
            def capture_generic(all_bonds):
                captured["bonds"] = torch.concat(all_bonds, dim=0).detach().cpu()
                return {}
            active_ousampler._generic_eval_fn = capture_generic
        else:
            def capture_molecular(all_atoms, all_bonds, all_sample_nodes):
                captured["atoms"] = torch.concat(all_atoms, dim=0).detach().cpu()
                captured["bonds"] = torch.concat(all_bonds, dim=0).detach().cpu()
                captured["sample_nodes"] = torch.concat(all_sample_nodes, dim=0).detach().cpu()
                return {}
            active_ousampler._mol_eval_fn = capture_molecular

        mu, _ = active_hsampler.sample()
        if len(mu) != 1:
            raise RuntimeError(
                f"Expected one HOG-Diff higher-order batch, received {len(mu)}."
            )
        active_ousampler.sample(mu_list=mu)
        if "bonds" not in captured:
            raise RuntimeError("HOG-Diff phase-2 sampler did not expose generated tensors.")
        return captured

    singleton_samplers = None

    def get_singleton_samplers():
        nonlocal singleton_samplers
        if singleton_samplers is not None:
            return singleton_samplers
        singleton_hconfig = _load_mode_config(config_path, mode="higher-order")
        singleton_ouconfig = _load_mode_config(config_path, mode="OU")
        for config in (singleton_hconfig, singleton_ouconfig):
            config.ckpt = str(checkpoint)
            config.eval.batch_size = 1
            config.eval.num_samples = 1
            config.eval.seed = int(args.seed)
            config.exp.plot = False
            config.data.num_workers = 0
        singleton_samplers = (
            Sampler(singleton_hconfig, sample_logger, mode="higher-order", device=device),
            Sampler(singleton_ouconfig, sample_logger, mode="OU", device=device),
        )
        return singleton_samplers

    for round_index in range(rounds):
        captured: dict[str, Any] = {}
        use_singleton_fallback = False
        for attempt in range(args.max_numerical_retries + 1):
            round_seed = _numerical_retry_seed(args.seed, round_index, attempt)
            try:
                captured = capture_once(hsampler, ousampler, sample_seed=round_seed)
            except ValueError as exc:
                numerical_failure = _is_predictor_nan_error(exc)
                if not numerical_failure:
                    raise
                if attempt >= args.max_numerical_retries:
                    if batch_size == 1:
                        raise RuntimeError(
                            "HOG-Diff produced NaNs for isolated generation batch "
                            f"{round_index + 1} after {attempt + 1} attempts. "
                            "The finite checkpoint is numerically unstable "
                            "under the configured OU sampler."
                        ) from exc
                    use_singleton_fallback = True
                    singleton_fallbacks.append(
                        {"batch": round_index + 1, "failed_batch_attempts": attempt + 1}
                    )
                    print(
                        "[GraphER/HOG-Diff] vectorized generation batch "
                        f"{round_index + 1} remained non-finite after {attempt + 1} "
                        "attempts; isolating its trajectories at batch size 1",
                        flush=True,
                    )
                    break
                retry_seed = _numerical_retry_seed(args.seed, round_index, attempt + 1)
                numerical_retries.append(
                    {
                        "batch": round_index + 1,
                        "failed_attempt": attempt + 1,
                        "failed_seed": round_seed,
                        "retry_seed": retry_seed,
                    }
                )
                print(
                    "[GraphER/HOG-Diff] rejected non-finite trajectory for "
                    f"generation batch {round_index + 1}; retrying with "
                    f"deterministic seed {retry_seed} "
                    f"({attempt + 1}/{args.max_numerical_retries})",
                    flush=True,
                )
                continue
            break

        if use_singleton_fallback:
            singleton_hsampler, singleton_ousampler = get_singleton_samplers()
            singleton_outputs: list[dict[str, Any]] = []
            for sample_index in range(batch_size):
                for attempt in range(args.max_numerical_retries + 1):
                    sample_seed = _singleton_retry_seed(
                        args.seed, round_index, sample_index, attempt
                    )
                    try:
                        sample = capture_once(
                            singleton_hsampler,
                            singleton_ousampler,
                            sample_seed=sample_seed,
                        )
                    except ValueError as exc:
                        if not _is_predictor_nan_error(exc):
                            raise
                        if attempt >= args.max_numerical_retries:
                            raise RuntimeError(
                                "HOG-Diff produced NaNs for isolated trajectory "
                                f"{sample_index + 1} in generation batch "
                                f"{round_index + 1} after {attempt + 1} attempts. "
                                "The checkpoint requires retraining or a modified sampler."
                            ) from exc
                        retry_seed = _singleton_retry_seed(
                            args.seed, round_index, sample_index, attempt + 1
                        )
                        numerical_retries.append(
                            {
                                "batch": round_index + 1,
                                "fallback_sample": sample_index + 1,
                                "failed_attempt": attempt + 1,
                                "failed_seed": sample_seed,
                                "retry_seed": retry_seed,
                            }
                        )
                        print(
                            "[GraphER/HOG-Diff] rejected isolated non-finite "
                            f"trajectory {sample_index + 1}/{batch_size} for generation "
                            f"batch {round_index + 1}; retrying with deterministic seed "
                            f"{retry_seed} ({attempt + 1}/{args.max_numerical_retries})",
                            flush=True,
                        )
                        continue
                    singleton_outputs.append(sample)
                    break
            captured = {
                key: torch.concat([sample[key] for sample in singleton_outputs], dim=0)
                for key in singleton_outputs[0]
            }

        max_nodes = int(ouconfig.data.max_node)
        if args.domain == "generic":
            adjacency, sizes = _pack_generic(captured["bonds"], max_nodes=max_nodes)
            generic_batches.append(adjacency)
            generic_sizes.append(sizes)
        else:
            atoms, adjacency, sizes = _pack_molecular(
                captured["atoms"],
                captured["bonds"],
                captured["sample_nodes"],
                max_nodes=max_nodes,
            )
            molecule_atoms.append(atoms)
            molecule_bonds.append(adjacency)
            molecule_sizes.append(sizes)

        if (round_index + 1) % int(args.progress_every_batches) == 0 or round_index + 1 == rounds:
            print(
                f"[GraphER/HOG-Diff] generation batch {round_index + 1}/{rounds}",
                flush=True,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    if args.domain == "generic":
        adjacency = np.concatenate(generic_batches, axis=0)[: args.num_graphs]
        sizes = np.concatenate(generic_sizes, axis=0)[: args.num_graphs]
        np.savez_compressed(
            output,
            adjacency=adjacency,
            num_nodes=sizes,
            sample_index=np.arange(args.num_graphs, dtype=np.int64),
        )
    else:
        atoms = np.concatenate(molecule_atoms, axis=0)[: args.num_graphs]
        adjacency = np.concatenate(molecule_bonds, axis=0)[: args.num_graphs]
        sizes = np.concatenate(molecule_sizes, axis=0)[: args.num_graphs]
        np.savez_compressed(
            output,
            adjacency=adjacency,
            node_types=atoms,
            num_nodes=sizes,
            sample_index=np.arange(args.num_graphs, dtype=np.int64),
        )
    if not output.is_file():
        raise RuntimeError(f"HOG-Diff worker did not write {output}.")
    _atomic_json(
        manifest,
        {
            "format": "grapher_hogdiff_export_v1",
            "native_dataset": str(ouconfig.data.name),
            "domain": args.domain,
            "num_requested": int(args.num_graphs),
            "num_generated": int(args.num_graphs),
            "batch_size": batch_size,
            "sampling_rounds": rounds,
            "seed": int(args.seed),
            "device": str(device),
            "max_numerical_retries_per_batch": int(args.max_numerical_retries),
            "numerical_retry_count": len(numerical_retries),
            "numerical_retries": numerical_retries,
            "singleton_fallback_count": len(singleton_fallbacks),
            "singleton_fallbacks": singleton_fallbacks,
            "phase_1": "higher-order VPSDE score sampler",
            "phase_2": "OU-bridge conditional score sampler",
            "postprocessing": (
                "upstream generic 0.5 threshold + isolate removal"
                if args.domain == "generic"
                else "raw atom argmax + upstream bond quantization; no MoFlow/RDKit correction"
            ),
            "output": {"path": str(output), "sha256": _sha256(output)},
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
