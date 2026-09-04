#!/usr/bin/env python3
"""Isolated exact-count raw GDSS generation worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_config(path: Path):
    import yaml
    from types import SimpleNamespace

    def convert(value):
        if isinstance(value, dict):
            return SimpleNamespace(**{str(k): convert(v) for k, v in value.items()})
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value

    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError(f"GDSS resolved config must be a mapping: {path}")
    return convert(value)


def _torch_load(path: Path, map_location):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _sample_flags(dataset_dir: Path, *, batch_size: int, max_nodes: int, device: str):
    import numpy as np
    import torch

    with np.load(dataset_dir / "train.npz", allow_pickle=False) as payload:
        adjacency = np.asarray(payload["adjacency"])
    active = (np.abs(adjacency).sum(axis=-1) > 1.0e-5).astype(np.float32)
    if len(active) == 0:
        raise RuntimeError("GDSS training projection is empty; cannot sample node masks.")
    indices = np.random.randint(0, len(active), size=int(batch_size))
    return torch.tensor(active[indices, :max_nodes], dtype=torch.float32, device=device)


def _sampling_fn(config, *, batch_size: int, device: str):
    from solver import S4_solver, get_pc_sampler
    from utils.loader import load_sde

    sde_x = load_sde(config.sde.x)
    sde_adj = load_sde(config.sde.adj)
    shape_x = (batch_size, int(config.data.max_node_num), int(config.data.max_feat_num))
    shape_adj = (batch_size, int(config.data.max_node_num), int(config.data.max_node_num))
    get_sampler = S4_solver if str(config.sampler.predictor) == "S4" else get_pc_sampler
    return get_sampler(
        sde_x=sde_x,
        sde_adj=sde_adj,
        shape_x=shape_x,
        shape_adj=shape_adj,
        predictor=config.sampler.predictor,
        corrector=config.sampler.corrector,
        snr=float(config.sampler.snr),
        scale_eps=float(config.sampler.scale_eps),
        n_steps=int(config.sampler.n_steps),
        probability_flow=bool(config.sample.probability_flow),
        continuous=True,
        denoise=bool(config.sample.noise_removal),
        eps=float(config.sample.eps),
        device=device,
    )


def _generic_postprocess(adj, *, max_nodes: int):
    import numpy as np

    binary = (adj < 0.5).astype(np.int8)
    binary = 1 - binary
    result_adj = np.zeros((len(binary), max_nodes, max_nodes), dtype=np.int8)
    result_n = np.zeros((len(binary),), dtype=np.int64)
    for i, raw in enumerate(binary):
        raw = np.asarray(raw, dtype=np.int8).copy()
        np.fill_diagonal(raw, 0)
        active = np.where(raw.sum(axis=0) + raw.sum(axis=1) > 0)[0]
        if active.size == 0:
            # Exact upstream adjs_to_graphs behavior: an empty thresholded graph
            # is represented by one isolated node.
            result_n[i] = 1
            continue
        compact = raw[np.ix_(active, active)]
        n = int(active.size)
        result_adj[i, :n, :n] = compact
        result_n[i] = n
    return result_adj, result_n


def _molecular_postprocess(x, adj, *, max_nodes: int, atom_channels: int):
    import numpy as np

    # Match GDSS Sampler_mol up to, but not including, gen_mol/correct_mol:
    # threshold atom channels, append the virtual-node score, quantize bonds.
    x_binary = (x > 0.5).astype(np.int8)
    virtual = 1 - x_binary.sum(axis=-1, keepdims=True)
    atom_choice = np.concatenate([x_binary, virtual], axis=-1).argmax(axis=-1)
    exists = atom_choice < atom_channels

    bond = np.zeros_like(adj, dtype=np.int8)
    bond[adj >= 2.5] = 3
    bond[(adj >= 1.5) & (adj < 2.5)] = 2
    bond[(adj >= 0.5) & (adj < 1.5)] = 1

    result_adj = np.zeros((len(bond), max_nodes, max_nodes), dtype=np.int8)
    result_types = np.full((len(bond), max_nodes), -1, dtype=np.int16)
    result_n = np.zeros((len(bond),), dtype=np.int64)
    for i in range(len(bond)):
        positions = np.where(exists[i])[0]
        n = int(positions.size)
        result_n[i] = n
        if n == 0:
            continue
        compact = bond[i][np.ix_(positions, positions)].copy()
        np.fill_diagonal(compact, 0)
        # The score process is symmetric by construction. Fail rather than
        # silently repair an upstream numerical/schema violation.
        if not np.array_equal(compact, compact.T):
            raise RuntimeError(f"GDSS molecular sample {i} quantized to a non-symmetric adjacency.")
        result_adj[i, :n, :n] = compact
        result_types[i, :n] = atom_choice[i, positions].astype(np.int16)
    return result_adj, result_types, result_n



def _install_evaluation_stub() -> None:
    """Bypass GDSS native metric imports during pure train/sample workers.

    ``utils.loader`` imports ``evaluation.mmd`` only so its separate evaluation
    helper can expose Gaussian kernels. GraphER never calls that evaluator from
    these workers, and the attached release otherwise drags in pyemd/dill.
    """
    import types

    module = types.ModuleType("evaluation.mmd")

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("GDSS native MMD evaluation is not invoked by the GraphER wrapper.")

    module.gaussian = unavailable
    module.gaussian_emd = unavailable
    sys.modules["evaluation.mmd"] = module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdss-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--domain", choices=("generic", "attributed"), required=True)
    parser.add_argument("--num-graphs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--atom-channels", type=int, default=0)
    parser.add_argument("--progress-every-batches", type=int, default=1)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    if args.num_graphs <= 0 or args.batch_size <= 0:
        raise ValueError("num-graphs and batch-size must be positive.")
    root = args.gdss_root.expanduser().resolve()
    sys.path.insert(0, str(root))

    _install_evaluation_stub()
    import numpy as np
    import torch
    from utils.loader import load_ema_from_ckpt, load_model_from_ckpt, load_seed

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("GDSS worker was configured for GPU but CUDA is unavailable.")
    device = "cuda:0" if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
    config = _load_config(args.config.expanduser().resolve())
    load_seed(int(args.seed))

    checkpoint = _torch_load(args.checkpoint.expanduser().resolve(), map_location=device)
    params_x = checkpoint["params_x"]
    params_adj = checkpoint["params_adj"]
    model_x = load_model_from_ckpt(params_x, checkpoint["x_state_dict"], [0] if device.startswith("cuda") else "cpu")
    model_adj = load_model_from_ckpt(params_adj, checkpoint["adj_state_dict"], [0] if device.startswith("cuda") else "cpu")
    model_x.eval()
    model_adj.eval()
    if bool(config.sample.use_ema):
        if "ema_x" not in checkpoint or "ema_adj" not in checkpoint:
            raise RuntimeError("GDSS resolved sampling requests EMA but checkpoint has no EMA state.")
        ema_x = load_ema_from_ckpt(model_x, checkpoint["ema_x"], float(config.train.ema))
        ema_adj = load_ema_from_ckpt(model_adj, checkpoint["ema_adj"], float(config.train.ema))
        ema_x.copy_to(model_x.parameters())
        ema_adj.copy_to(model_adj.parameters())

    max_nodes = int(config.data.max_node_num)
    rounds = int(math.ceil(args.num_graphs / args.batch_size))
    adjacency_batches = []
    node_type_batches = []
    size_batches = []
    generated = 0
    for round_index in range(rounds):
        current = min(args.batch_size, args.num_graphs - generated)
        sampling_fn = _sampling_fn(config, batch_size=current, device=device)
        flags = _sample_flags(args.dataset_dir.expanduser().resolve(), batch_size=current, max_nodes=max_nodes, device=device)
        x, adj, _ = sampling_fn(model_x, model_adj, flags)
        x_np = x.detach().cpu().numpy()
        adj_np = adj.detach().cpu().numpy()
        if args.domain == "generic":
            batch_adj, batch_n = _generic_postprocess(adj_np, max_nodes=max_nodes)
            adjacency_batches.append(batch_adj)
            size_batches.append(batch_n)
        else:
            batch_adj, batch_types, batch_n = _molecular_postprocess(
                x_np, adj_np, max_nodes=max_nodes, atom_channels=int(args.atom_channels)
            )
            adjacency_batches.append(batch_adj)
            node_type_batches.append(batch_types)
            size_batches.append(batch_n)
        generated += current
        if (round_index + 1) % max(1, int(args.progress_every_batches)) == 0:
            print(f"[GraphER/GDSS] generation batch {round_index + 1}/{rounds}", flush=True)

    adjacency = np.concatenate(adjacency_batches, axis=0)[: args.num_graphs]
    sizes = np.concatenate(size_batches, axis=0)[: args.num_graphs]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "adjacency": adjacency,
        "num_nodes": sizes,
        "sample_index": np.arange(args.num_graphs, dtype=np.int64),
    }
    if args.domain == "attributed":
        payload["node_types"] = np.concatenate(node_type_batches, axis=0)[: args.num_graphs]
    np.savez_compressed(output, **payload)
    if not output.is_file():
        raise RuntimeError(f"GDSS worker did not write {output}.")
    _atomic_json(
        args.manifest.expanduser().resolve(),
        {
            "format": "grapher_gdss_export_v1",
            "native_dataset": str(config.data.data),
            "domain": args.domain,
            "num_requested": int(args.num_graphs),
            "num_generated": int(args.num_graphs),
            "batch_size": int(args.batch_size),
            "sampling_rounds": rounds,
            "seed": int(args.seed),
            "device": device,
            "postprocessing": (
                "upstream 0.5 adjacency threshold + isolate removal"
                if args.domain == "generic"
                else "raw GDSS atom threshold/virtual argmax + bond quantization; no correct_mol, no largest-component rewrite"
            ),
            "output": {"path": str(output), "sha256": _sha256(output)},
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
