"""Atomic artifacts, explicit devices, and reproducible standalone run state."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import torch


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_write(path: str | Path, writer) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    os.close(fd)
    try:
        writer(Path(temporary))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value) -> None:
    atomic_write(path, lambda tmp: tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"))


def save_torch(path, state) -> None:
    atomic_write(path, lambda tmp: torch.save(state, tmp))


def load_checkpoint(path: str | Path) -> dict:
    from . import CHECKPOINT_FORMAT
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Expected a new Option-C checkpoint; old GDSM/categorical checkpoints are incompatible")
    return state


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    return value


def seed_everything(seed: int) -> None:
    if not 0 <= int(seed) < 2**32:
        raise ValueError("seed must be in [0,2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "gpu":
        value = "cuda"
    device = torch.device(value)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Option C supports cpu, gpu/cuda[:index], or auto")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but CUDA is unavailable; use --device cpu explicitly")
        device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
    return device


def versions() -> dict:
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "networkx": nx.__version__, "cuda_runtime": torch.version.cuda,
            "torch_cpu_threads": torch.get_num_threads()}


def validate_output_dir(path: Path, *, format_name: str, overwrite: bool, resume: bool = False) -> None:
    """Never remove files belonging to another experiment or legacy runner."""
    if path.exists() and any(path.iterdir()):
        marker = path / "manifest.json"
        if not marker.is_file() or json.loads(marker.read_text()).get("format") != format_name:
            raise FileExistsError(f"Refusing to use non-Option-C output directory: {path}")
        if not resume and not overwrite:
            raise FileExistsError(f"Output exists: {path}; choose a new directory, --resume, or --overwrite")
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")


def source_fingerprint() -> dict:
    """Record the exact new implementation and reused categorical/graphlet helpers."""
    package = Path(__file__).resolve().parent
    root = package.parents[1]  # grapher/
    paths = list(package.glob("*.py"))
    shared = root / "models/gdsm_simple/categorical"
    paths.extend(shared / name for name in ("noise.py", "data.py", "multiscale.py", "refiner.py"))
    paths.append(root / "rewiring_mlp/attributed/data.py")
    hashes = {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}
    return {"sha256": hashes, "fingerprint": object_hash(hashes)}
