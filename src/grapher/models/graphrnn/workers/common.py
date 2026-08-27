"""Modern compatibility runtime for the attached GraphRNN implementation.

Only the upstream neural modules in ``model.py`` are imported. The original
training entry point is intentionally not executed because it hard-codes CUDA,
NetworkX 1.11 APIs, and PyTorch 0.2 scalar semantics. This worker retains the
GraphRNN BFS adjacency sequence, GRU/MLP architectures, autoregressive sampler,
and optimizer schedule while using current PyTorch APIs and neutral NPZ I/O.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

CHECKPOINT_FORMAT = "grapher_graphrnn_checkpoint_v1"
SUPPORTED_VARIANTS = frozenset({"GraphRNN_RNN", "GraphRNN_MLP"})


@dataclass(frozen=True)
class DatasetArrays:
    train_adjacency: np.ndarray
    train_num_nodes: np.ndarray
    val_adjacency: np.ndarray
    val_num_nodes: np.ndarray
    test_adjacency: np.ndarray
    test_num_nodes: np.ndarray
    max_num_node: int


@dataclass(frozen=True)
class BuiltModels:
    rnn: torch.nn.Module
    output: torch.nn.Module
    architecture: dict[str, Any]


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def resolve_device(raw: str) -> torch.device:
    requested = str(raw).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "gpu":
        requested = "cuda"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"GraphRNN device {raw!r} requested CUDA, but CUDA is unavailable."
        )
    return device


def import_upstream_model(graphrnn_root: str | Path) -> ModuleType:
    root = Path(graphrnn_root).expanduser().resolve()
    model_path = root / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing upstream GraphRNN model.py: {model_path}")
    name = "graphrnn_upstream_model_adapter"
    spec = importlib.util.spec_from_file_location(name, model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load upstream GraphRNN model module: {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for required in ("GRU_plain", "MLP_plain"):
        if not hasattr(module, required):
            raise RuntimeError(f"Upstream GraphRNN model.py has no {required}.")
    return module


def load_dataset_arrays(path: str | Path) -> DatasetArrays:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        values: dict[str, np.ndarray] = {}
        for split in ("train", "val", "test"):
            for suffix in ("adjacency", "num_nodes"):
                key = f"{split}_{suffix}"
                if key not in payload.files:
                    raise RuntimeError(f"GraphRNN dataset export is missing {key!r}.")
                values[key] = np.asarray(payload[key])

    maxima: set[int] = set()
    for split in ("train", "val", "test"):
        adjacency = values[f"{split}_adjacency"]
        sizes = values[f"{split}_num_nodes"]
        if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2]:
            raise ValueError(f"Invalid {split} adjacency tensor shape {adjacency.shape}.")
        if sizes.shape != (adjacency.shape[0],):
            raise ValueError(f"Invalid {split} num_nodes shape {sizes.shape}.")
        if adjacency.shape[0] <= 0:
            raise ValueError(f"GraphRNN {split} split is empty.")
        max_nodes = int(adjacency.shape[1])
        maxima.add(max_nodes)
        if np.any(sizes <= 0) or np.any(sizes > max_nodes):
            raise ValueError(f"GraphRNN {split} contains an invalid graph size.")
        if not np.all((adjacency == 0) | (adjacency == 1)):
            raise ValueError(f"GraphRNN {split} adjacency is not binary.")
        if not np.array_equal(adjacency, np.swapaxes(adjacency, 1, 2)):
            raise ValueError(f"GraphRNN {split} adjacency is not symmetric.")
        diagonal = np.diagonal(adjacency, axis1=1, axis2=2)
        if np.any(diagonal != 0):
            raise ValueError(f"GraphRNN {split} adjacency contains self-loops.")
        for index, size in enumerate(sizes.tolist()):
            n = int(size)
            if n < max_nodes and (
                np.any(adjacency[index, n:, :])
                or np.any(adjacency[index, :, n:])
            ):
                raise ValueError(
                    f"GraphRNN {split}[{index}] has non-zero padding outside n={n}."
                )
    if len(maxima) != 1:
        raise ValueError("GraphRNN split tensors use inconsistent max_num_node values.")

    return DatasetArrays(
        train_adjacency=values["train_adjacency"].astype(np.uint8, copy=False),
        train_num_nodes=values["train_num_nodes"].astype(np.int64, copy=False),
        val_adjacency=values["val_adjacency"].astype(np.uint8, copy=False),
        val_num_nodes=values["val_num_nodes"].astype(np.int64, copy=False),
        test_adjacency=values["test_adjacency"].astype(np.uint8, copy=False),
        test_num_nodes=values["test_num_nodes"].astype(np.int64, copy=False),
        max_num_node=maxima.pop(),
    )


def normalize_config(raw: Mapping[str, Any], *, dataset_max_num_node: int) -> dict[str, Any]:
    config = dict(raw)
    variant = str(config.get("variant", "GraphRNN_RNN"))
    aliases = {
        "rnn": "GraphRNN_RNN",
        "graphrnn_rnn": "GraphRNN_RNN",
        "mlp": "GraphRNN_MLP",
        "graphrnn_mlp": "GraphRNN_MLP",
    }
    variant = aliases.get(variant.strip().lower(), variant)
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"variant must be one of {sorted(SUPPORTED_VARIANTS)}, got {variant!r}."
        )

    max_num_node = int(config.get("max_num_node", dataset_max_num_node))
    if max_num_node != int(dataset_max_num_node):
        raise ValueError(
            "The worker dataset tensor and resolved max_num_node disagree: "
            f"{dataset_max_num_node} != {max_num_node}."
        )
    raw_prev = config.get("max_prev_node")
    max_prev_node = max_num_node - 1 if raw_prev is None else int(raw_prev)
    max_prev_node_limit = max_num_node - 1
    if max_prev_node <= 0 or max_prev_node > max_prev_node_limit:
        raise ValueError(
            "max_prev_node must be in [1, max_num_node - 1]; got "
            f"{max_prev_node} for max_num_node={max_num_node}."
        )

    defaults: dict[str, Any] = {
        "variant": variant,
        "max_num_node": max_num_node,
        "max_prev_node": max_prev_node,
        "hidden_size_rnn": 128,
        "hidden_size_rnn_output": 16,
        "embedding_size_rnn": 64,
        "embedding_size_rnn_output": 8,
        "embedding_size_output": 64,
        "num_layers": 4,
        "batch_size": 32,
        "batch_ratio": 32,
        "epochs": 3000,
        "learning_rate": 0.003,
        "milestones": [400, 1000],
        "lr_rate": 0.3,
        "scheduler_step_unit": "batch",
        "num_workers": 0,
        "save_every_epochs": 100,
        "log_every_epochs": 10,
        "gradient_clip_norm": None,
        "sample_time": 1,
        "generation_batch_size": 32,
        "deterministic": False,
        "torch_num_threads": None,
    }
    defaults.update(config)
    defaults["variant"] = variant
    defaults["max_num_node"] = max_num_node
    defaults["max_prev_node"] = max_prev_node

    positive_ints = (
        "hidden_size_rnn",
        "hidden_size_rnn_output",
        "embedding_size_rnn",
        "embedding_size_rnn_output",
        "embedding_size_output",
        "num_layers",
        "batch_size",
        "batch_ratio",
        "epochs",
        "save_every_epochs",
        "log_every_epochs",
        "sample_time",
        "generation_batch_size",
    )
    for key in positive_ints:
        defaults[key] = int(defaults[key])
        if defaults[key] <= 0:
            raise ValueError(f"{key} must be positive.")
    defaults["num_workers"] = int(defaults["num_workers"])
    if defaults["num_workers"] < 0:
        raise ValueError("num_workers must be non-negative.")
    defaults["learning_rate"] = float(defaults["learning_rate"])
    defaults["lr_rate"] = float(defaults["lr_rate"])
    if defaults["learning_rate"] <= 0 or defaults["lr_rate"] <= 0:
        raise ValueError("learning_rate and lr_rate must be positive.")
    defaults["milestones"] = sorted({int(value) for value in defaults["milestones"]})
    if any(value <= 0 for value in defaults["milestones"]):
        raise ValueError("milestones must be positive integers.")
    unit = str(defaults["scheduler_step_unit"]).lower()
    if unit not in {"batch", "epoch"}:
        raise ValueError("scheduler_step_unit must be 'batch' or 'epoch'.")
    defaults["scheduler_step_unit"] = unit
    clip = defaults.get("gradient_clip_norm")
    defaults["gradient_clip_norm"] = None if clip is None else float(clip)
    if defaults["gradient_clip_norm"] is not None and defaults["gradient_clip_norm"] <= 0:
        raise ValueError("gradient_clip_norm must be positive when configured.")
    threads = defaults.get("torch_num_threads")
    defaults["torch_num_threads"] = None if threads is None else int(threads)
    if defaults["torch_num_threads"] is not None and defaults["torch_num_threads"] <= 0:
        raise ValueError("torch_num_threads must be positive when configured.")
    defaults["deterministic"] = bool(defaults["deterministic"])
    return defaults


def bfs_sequence(adjacency: np.ndarray, start: int) -> np.ndarray:
    """Return GraphRNN's BFS node order, extended to a BFS forest if needed."""

    n = int(adjacency.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    unvisited = set(range(n))
    roots = [int(start)] + [node for node in range(n) if node != int(start)]
    output: list[int] = []
    for root in roots:
        if root not in unvisited:
            continue
        queue: deque[int] = deque([root])
        unvisited.remove(root)
        while queue:
            current = queue.popleft()
            output.append(current)
            for neighbor in np.flatnonzero(adjacency[current]).tolist():
                neighbor = int(neighbor)
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    queue.append(neighbor)
    if len(output) != n:
        raise RuntimeError("BFS ordering did not cover all nodes.")
    return np.asarray(output, dtype=np.int64)


def encode_adj(adjacency: np.ndarray, max_prev_node: int) -> np.ndarray:
    """Encode the strict lower triangle using the official GraphRNN layout."""

    lower = np.tril(np.asarray(adjacency), k=-1)
    n = int(lower.shape[0])
    lower = lower[1:n, 0 : n - 1]
    output = np.zeros((lower.shape[0], int(max_prev_node)), dtype=np.float32)
    for row in range(lower.shape[0]):
        input_start = max(0, row - int(max_prev_node) + 1)
        input_end = row + 1
        output_start = int(max_prev_node) + input_start - input_end
        output[row, output_start:int(max_prev_node)] = lower[row, input_start:input_end]
        output[row, :] = output[row, :][::-1]
    return output


def decode_adj(encoded: np.ndarray) -> np.ndarray:
    """Decode the official GraphRNN adjacency sequence."""

    encoded = np.asarray(encoded)
    max_prev_node = int(encoded.shape[1])
    lower = np.zeros((encoded.shape[0], encoded.shape[0]), dtype=np.uint8)
    for row in range(encoded.shape[0]):
        input_start = max(0, row - max_prev_node + 1)
        input_end = row + 1
        output_start = max_prev_node + input_start - input_end
        lower[row, input_start:input_end] = encoded[row, ::-1][
            output_start:max_prev_node
        ]
    full = np.zeros((encoded.shape[0] + 1, encoded.shape[0] + 1), dtype=np.uint8)
    n = int(full.shape[0])
    full[1:n, 0 : n - 1] = np.tril(lower, 0)
    full = full + full.T
    return (full != 0).astype(np.uint8, copy=False)


class GraphSequenceDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        adjacency: np.ndarray,
        num_nodes: np.ndarray,
        *,
        max_num_node: int,
        max_prev_node: int,
    ) -> None:
        self.adjacency = np.asarray(adjacency, dtype=np.uint8)
        self.num_nodes = np.asarray(num_nodes, dtype=np.int64)
        self.max_num_node = int(max_num_node)
        self.max_prev_node = int(max_prev_node)

    def __len__(self) -> int:
        return int(self.adjacency.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        n = int(self.num_nodes[index])
        matrix = self.adjacency[index, :n, :n].copy()
        permutation = np.random.permutation(n)
        matrix = matrix[np.ix_(permutation, permutation)]
        start = int(np.random.randint(n))
        order = bfs_sequence(matrix, start)
        matrix = matrix[np.ix_(order, order)]
        encoded = encode_adj(matrix, self.max_prev_node)
        x = np.zeros((self.max_num_node, self.max_prev_node), dtype=np.float32)
        y = np.zeros((self.max_num_node, self.max_prev_node), dtype=np.float32)
        x[0, :] = 1.0
        y[: encoded.shape[0], :] = encoded
        x[1 : encoded.shape[0] + 1, :] = encoded
        return {"x": x, "y": y, "len": n}


def build_models(
    upstream: ModuleType,
    config: Mapping[str, Any],
    *,
    device: torch.device,
) -> BuiltModels:
    variant = str(config["variant"])
    if variant == "GraphRNN_RNN":
        rnn = upstream.GRU_plain(
            input_size=int(config["max_prev_node"]),
            embedding_size=int(config["embedding_size_rnn"]),
            hidden_size=int(config["hidden_size_rnn"]),
            num_layers=int(config["num_layers"]),
            has_input=True,
            has_output=True,
            output_size=int(config["hidden_size_rnn_output"]),
        )
        output = upstream.GRU_plain(
            input_size=1,
            embedding_size=int(config["embedding_size_rnn_output"]),
            hidden_size=int(config["hidden_size_rnn_output"]),
            num_layers=int(config["num_layers"]),
            has_input=True,
            has_output=True,
            output_size=1,
        )
    elif variant == "GraphRNN_MLP":
        rnn = upstream.GRU_plain(
            input_size=int(config["max_prev_node"]),
            embedding_size=int(config["embedding_size_rnn"]),
            hidden_size=int(config["hidden_size_rnn"]),
            num_layers=int(config["num_layers"]),
            has_input=True,
            has_output=False,
        )
        output = upstream.MLP_plain(
            h_size=int(config["hidden_size_rnn"]),
            embedding_size=int(config["embedding_size_output"]),
            y_size=int(config["max_prev_node"]),
        )
    else:  # guarded by normalize_config
        raise AssertionError(variant)
    rnn = rnn.to(device)
    output = output.to(device)
    architecture = {
        key: config[key]
        for key in (
            "variant",
            "max_num_node",
            "max_prev_node",
            "hidden_size_rnn",
            "hidden_size_rnn_output",
            "embedding_size_rnn",
            "embedding_size_rnn_output",
            "embedding_size_output",
            "num_layers",
        )
    }
    return BuiltModels(rnn=rnn, output=output, architecture=architecture)


def _hidden(module: torch.nn.Module, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(
        int(module.num_layers),
        int(batch_size),
        int(module.hidden_size),
        device=device,
    )


def _packed_clean(values: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    packed = pack_padded_sequence(values, lengths, batch_first=True)
    return pad_packed_sequence(packed, batch_first=True)[0]


def _train_mlp_batch(
    batch: Mapping[str, torch.Tensor],
    models: BuiltModels,
    config: Mapping[str, Any],
    device: torch.device,
) -> torch.Tensor:
    x_unsorted = batch["x"].float()
    y_unsorted = batch["y"].float()
    lengths_unsorted = batch["len"].long()
    max_length = int(lengths_unsorted.max().item())
    x_unsorted = x_unsorted[:, :max_length, :]
    y_unsorted = y_unsorted[:, :max_length, :]
    lengths_sorted, order = torch.sort(lengths_unsorted, descending=True)
    lengths = [int(value) for value in lengths_sorted.tolist()]
    x = x_unsorted.index_select(0, order).to(device)
    y = y_unsorted.index_select(0, order).to(device)
    models.rnn.hidden = _hidden(models.rnn, x.size(0), device)
    hidden = models.rnn(x, pack=True, input_len=lengths)
    probabilities = torch.sigmoid(models.output(hidden))
    probabilities = _packed_clean(probabilities, lengths)
    return F.binary_cross_entropy(probabilities, y)


def _train_rnn_batch(
    batch: Mapping[str, torch.Tensor],
    models: BuiltModels,
    config: Mapping[str, Any],
    device: torch.device,
) -> torch.Tensor:
    x_unsorted = batch["x"].float()
    y_unsorted = batch["y"].float()
    lengths_unsorted = batch["len"].long()
    max_length = int(lengths_unsorted.max().item())
    x_unsorted = x_unsorted[:, :max_length, :]
    y_unsorted = y_unsorted[:, :max_length, :]
    lengths_sorted, order = torch.sort(lengths_unsorted, descending=True)
    lengths = [int(value) for value in lengths_sorted.tolist()]
    x_cpu = x_unsorted.index_select(0, order)
    y_cpu = y_unsorted.index_select(0, order)

    packed_y = pack_padded_sequence(y_cpu, lengths, batch_first=True).data
    packed_y = torch.flip(packed_y, dims=(0,)).view(packed_y.size(0), packed_y.size(1), 1)
    output_x = torch.cat(
        (torch.ones(packed_y.size(0), 1, 1), packed_y[:, :-1, :]), dim=1
    )
    output_y = packed_y
    length_bins = np.bincount(np.asarray(lengths, dtype=np.int64))
    output_lengths: list[int] = []
    for position in range(len(length_bins) - 1, 0, -1):
        active = int(np.sum(length_bins[position:]))
        output_lengths.extend(
            [min(position, int(y_cpu.size(2)))] * active
        )
    if not output_lengths:
        raise RuntimeError("GraphRNN produced no output-RNN sequence lengths.")

    x = x_cpu.to(device)
    output_x = output_x.to(device)
    output_y = output_y.to(device)
    models.rnn.hidden = _hidden(models.rnn, x.size(0), device)
    hidden = models.rnn(x, pack=True, input_len=lengths)
    hidden = pack_padded_sequence(hidden, lengths, batch_first=True).data
    hidden = torch.flip(hidden, dims=(0,))
    hidden_null = torch.zeros(
        int(config["num_layers"]) - 1,
        hidden.size(0),
        hidden.size(1),
        device=device,
    )
    models.output.hidden = torch.cat(
        (hidden.view(1, hidden.size(0), hidden.size(1)), hidden_null), dim=0
    )
    probabilities = torch.sigmoid(
        models.output(output_x, pack=True, input_len=output_lengths)
    )
    probabilities = _packed_clean(probabilities, output_lengths)
    target = _packed_clean(output_y, output_lengths)
    return F.binary_cross_entropy(probabilities, target)


def _torch_load(path: Path, *, map_location: torch.device | str) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_checkpoint(
    path: Path,
    *,
    models: BuiltModels,
    optimizer_rnn: torch.optim.Optimizer,
    optimizer_output: torch.optim.Optimizer,
    scheduler_rnn: torch.optim.lr_scheduler.MultiStepLR,
    scheduler_output: torch.optim.lr_scheduler.MultiStepLR,
    epoch: int,
    config: Mapping[str, Any],
    seed: int,
    dataset_sha256: str,
) -> None:
    value = {
        "format": CHECKPOINT_FORMAT,
        "epoch": int(epoch),
        "seed": int(seed),
        "architecture": dict(models.architecture),
        "resolved_config": dict(config),
        "dataset_sha256": str(dataset_sha256),
        "rnn_state_dict": models.rnn.state_dict(),
        "output_state_dict": models.output.state_dict(),
        "optimizer_rnn_state_dict": optimizer_rnn.state_dict(),
        "optimizer_output_state_dict": optimizer_output.state_dict(),
        "scheduler_rnn_state_dict": scheduler_rnn.state_dict(),
        "scheduler_output_state_dict": scheduler_output.state_dict(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: str | Path,
    *,
    models: BuiltModels,
    device: torch.device,
    optimizer_rnn: torch.optim.Optimizer | None = None,
    optimizer_output: torch.optim.Optimizer | None = None,
    scheduler_rnn: torch.optim.lr_scheduler.MultiStepLR | None = None,
    scheduler_output: torch.optim.lr_scheduler.MultiStepLR | None = None,
) -> dict[str, Any]:
    value = _torch_load(Path(path), map_location=device)
    if not isinstance(value, dict) or value.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError(f"Unsupported GraphRNN checkpoint format in {path}.")
    if dict(value.get("architecture", {})) != dict(models.architecture):
        raise RuntimeError(
            "GraphRNN checkpoint architecture does not match the resolved config."
        )
    models.rnn.load_state_dict(value["rnn_state_dict"])
    models.output.load_state_dict(value["output_state_dict"])
    if optimizer_rnn is not None and "optimizer_rnn_state_dict" in value:
        optimizer_rnn.load_state_dict(value["optimizer_rnn_state_dict"])
    if optimizer_output is not None and "optimizer_output_state_dict" in value:
        optimizer_output.load_state_dict(value["optimizer_output_state_dict"])
    if scheduler_rnn is not None and "scheduler_rnn_state_dict" in value:
        scheduler_rnn.load_state_dict(value["scheduler_rnn_state_dict"])
    if scheduler_output is not None and "scheduler_output_state_dict" in value:
        scheduler_output.load_state_dict(value["scheduler_output_state_dict"])
    return value


def _worker_seed(worker_id: int) -> None:
    del worker_id
    seed = int(torch.initial_seed() % (2**32))
    np.random.seed(seed)
    random.seed(seed)


def train_model(
    *,
    graphrnn_root: str | Path,
    dataset: DatasetArrays,
    config: Mapping[str, Any],
    output_dir: str | Path,
    seed: int,
    device: torch.device,
    dataset_sha256: str,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    resolved = normalize_config(config, dataset_max_num_node=dataset.max_num_node)
    threads = resolved.get("torch_num_threads")
    if threads is not None:
        torch.set_num_threads(int(threads))
    seed_everything(seed, deterministic=bool(resolved["deterministic"]))
    upstream = import_upstream_model(graphrnn_root)
    models = build_models(upstream, resolved, device=device)
    training_dataset = GraphSequenceDataset(
        dataset.train_adjacency,
        dataset.train_num_nodes,
        max_num_node=int(resolved["max_num_node"]),
        max_prev_node=int(resolved["max_prev_node"]),
    )
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        weights=torch.ones(len(training_dataset), dtype=torch.double),
        num_samples=int(resolved["batch_size"]) * int(resolved["batch_ratio"]),
        replacement=True,
        generator=sampler_generator,
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(seed) + 1)
    loader = DataLoader(
        training_dataset,
        batch_size=int(resolved["batch_size"]),
        sampler=sampler,
        num_workers=int(resolved["num_workers"]),
        worker_init_fn=_worker_seed if int(resolved["num_workers"]) else None,
        generator=loader_generator,
        persistent_workers=bool(int(resolved["num_workers"])),
    )
    optimizer_rnn = torch.optim.Adam(
        models.rnn.parameters(), lr=float(resolved["learning_rate"])
    )
    optimizer_output = torch.optim.Adam(
        models.output.parameters(), lr=float(resolved["learning_rate"])
    )
    scheduler_rnn = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_rnn,
        milestones=list(resolved["milestones"]),
        gamma=float(resolved["lr_rate"]),
    )
    scheduler_output = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_output,
        milestones=list(resolved["milestones"]),
        gamma=float(resolved["lr_rate"]),
    )

    start_epoch = 1
    resumed_epoch = None
    if resume_from is not None:
        checkpoint = load_checkpoint(
            resume_from,
            models=models,
            device=device,
            optimizer_rnn=optimizer_rnn,
            optimizer_output=optimizer_output,
            scheduler_rnn=scheduler_rnn,
            scheduler_output=scheduler_output,
        )
        checkpoint_dataset = str(checkpoint.get("dataset_sha256", ""))
        if not checkpoint_dataset or checkpoint_dataset != str(dataset_sha256):
            raise RuntimeError(
                "GraphRNN resume checkpoint was trained on a different dataset "
                "export. Start a new run or resume with the original frozen "
                "GraphES splits."
            )
        resumed_epoch = int(checkpoint["epoch"])
        if int(resolved["epochs"]) < resumed_epoch:
            raise ValueError(
                "Configured GraphRNN epoch horizon is earlier than the resume "
                f"checkpoint: epochs={resolved['epochs']}, checkpoint_epoch="
                f"{resumed_epoch}."
            )
        start_epoch = resumed_epoch + 1

    destination = Path(output_dir)
    checkpoints_dir = destination / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    history_path = destination / "loss_history.jsonl"
    final_checkpoint = checkpoints_dir / "graphrnn.pt"
    started = time.monotonic()
    last_loss = math.nan
    completed_epoch = start_epoch - 1

    for epoch in range(start_epoch, int(resolved["epochs"]) + 1):
        models.rnn.train()
        models.output.train()
        losses: list[float] = []
        for batch in loader:
            optimizer_rnn.zero_grad(set_to_none=True)
            optimizer_output.zero_grad(set_to_none=True)
            if resolved["variant"] == "GraphRNN_RNN":
                loss = _train_rnn_batch(batch, models, resolved, device)
            else:
                loss = _train_mlp_batch(batch, models, resolved, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite GraphRNN loss at epoch {epoch}: {loss.item()}"
                )
            loss.backward()
            clip = resolved.get("gradient_clip_norm")
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(models.rnn.parameters(), float(clip))
                torch.nn.utils.clip_grad_norm_(models.output.parameters(), float(clip))
            optimizer_output.step()
            optimizer_rnn.step()
            if resolved["scheduler_step_unit"] == "batch":
                scheduler_output.step()
                scheduler_rnn.step()
            losses.append(float(loss.detach().cpu().item()))
        if resolved["scheduler_step_unit"] == "epoch":
            scheduler_output.step()
            scheduler_rnn.step()
        completed_epoch = epoch
        last_loss = float(np.mean(losses))
        record = {
            "epoch": epoch,
            "loss": last_loss,
            "learning_rate": float(optimizer_rnn.param_groups[0]["lr"]),
            "duration_seconds": time.monotonic() - started,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if epoch == 1 or epoch % int(resolved["log_every_epochs"]) == 0:
            print(
                "GRAPHRNN_PROGRESS "
                f"epoch={epoch}/{resolved['epochs']} "
                f"loss={last_loss:.8f} "
                f"lr={record['learning_rate']:.8g}",
                flush=True,
            )
        if epoch % int(resolved["save_every_epochs"]) == 0:
            numbered = checkpoints_dir / f"graphrnn_epoch_{epoch}.pt"
            save_checkpoint(
                numbered,
                models=models,
                optimizer_rnn=optimizer_rnn,
                optimizer_output=optimizer_output,
                scheduler_rnn=scheduler_rnn,
                scheduler_output=scheduler_output,
                epoch=epoch,
                config=resolved,
                seed=seed,
                dataset_sha256=dataset_sha256,
            )

    if completed_epoch <= 0:
        raise RuntimeError("GraphRNN training completed no epochs.")
    save_checkpoint(
        final_checkpoint,
        models=models,
        optimizer_rnn=optimizer_rnn,
        optimizer_output=optimizer_output,
        scheduler_rnn=scheduler_rnn,
        scheduler_output=scheduler_output,
        epoch=completed_epoch,
        config=resolved,
        seed=seed,
        dataset_sha256=dataset_sha256,
    )
    return {
        "format": "grapher_graphrnn_training_worker_v1",
        "checkpoint": str(final_checkpoint.resolve()),
        "checkpoint_epoch": completed_epoch,
        "configured_epochs": int(resolved["epochs"]),
        "resumed_from_epoch": resumed_epoch,
        "last_loss": last_loss,
        "history": str(history_path.resolve()),
        "resolved_config": resolved,
        "architecture": models.architecture,
        "device": str(device),
        "duration_seconds": time.monotonic() - started,
        "train_graph_count": int(dataset.train_num_nodes.shape[0]),
        "val_graph_count": int(dataset.val_num_nodes.shape[0]),
        "test_graph_count": int(dataset.test_num_nodes.shape[0]),
    }


def _bernoulli(logits: torch.Tensor, *, sample_time: int = 1) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    result = torch.bernoulli(probabilities)
    if int(sample_time) <= 1:
        return result
    # Match the original MLP sampler: retry an all-zero row a bounded number
    # of times, while retaining zero if all attempts are zero.
    for index in range(result.size(0)):
        for _ in range(int(sample_time) - 1):
            if bool(torch.sum(result[index]).item() > 0):
                break
            result[index] = torch.bernoulli(probabilities[index])
    return result


def _sample_batch(
    models: BuiltModels,
    config: Mapping[str, Any],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    max_num_node = int(config["max_num_node"])
    max_prev_node = int(config["max_prev_node"])
    encoded = torch.zeros(
        batch_size, max_num_node, max_prev_node, device=device
    )
    x_step = torch.ones(batch_size, 1, max_prev_node, device=device)
    models.rnn.hidden = _hidden(models.rnn, batch_size, device)
    for node_index in range(max_num_node):
        hidden = models.rnn(x_step)
        if config["variant"] == "GraphRNN_RNN":
            hidden_null = torch.zeros(
                int(config["num_layers"]) - 1,
                hidden.size(0),
                hidden.size(2),
                device=device,
            )
            models.output.hidden = torch.cat(
                (hidden.permute(1, 0, 2), hidden_null), dim=0
            )
            x_step = torch.zeros(batch_size, 1, max_prev_node, device=device)
            output_x_step = torch.ones(batch_size, 1, 1, device=device)
            for edge_index in range(min(max_prev_node, node_index + 1)):
                logits = models.output(output_x_step)
                output_x_step = _bernoulli(logits, sample_time=1)
                x_step[:, :, edge_index : edge_index + 1] = output_x_step
                models.output.hidden = models.output.hidden.detach()
        else:
            logits = models.output(hidden)
            x_step = _bernoulli(
                logits, sample_time=int(config.get("sample_time", 1))
            )
        encoded[:, node_index : node_index + 1, :] = x_step
        models.rnn.hidden = models.rnn.hidden.detach()
    return encoded.detach().cpu().to(torch.uint8)


def _trim_official(adjacency: np.ndarray) -> np.ndarray:
    """Mirror upstream ``utils.get_graph`` by dropping all-zero rows/columns."""

    keep = ~np.all(adjacency == 0, axis=1)
    trimmed = adjacency[keep, :]
    trimmed = trimmed[:, keep]
    return trimmed.astype(np.uint8, copy=False)


def generate_graphs(
    *,
    graphrnn_root: str | Path,
    checkpoint_path: str | Path,
    config: Mapping[str, Any],
    num_graphs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    progress_every_batches: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if int(num_graphs) <= 0 or int(batch_size) <= 0:
        raise ValueError("num_graphs and batch_size must be positive.")
    resolved = dict(config)
    seed_everything(seed, deterministic=bool(resolved.get("deterministic", False)))
    upstream = import_upstream_model(graphrnn_root)
    models = build_models(upstream, resolved, device=device)
    checkpoint = load_checkpoint(
        checkpoint_path,
        models=models,
        device=device,
    )
    models.rnn.eval()
    models.output.eval()
    max_output_nodes = int(resolved["max_num_node"]) + 1
    output = np.zeros(
        (int(num_graphs), max_output_nodes, max_output_nodes), dtype=np.uint8
    )
    sizes = np.zeros(int(num_graphs), dtype=np.int64)
    empty_count = 0
    disconnected_count = 0
    cursor = 0
    batch_index = 0
    started = time.monotonic()
    with torch.inference_mode():
        while cursor < int(num_graphs):
            current = min(int(batch_size), int(num_graphs) - cursor)
            encoded = _sample_batch(
                models,
                resolved,
                batch_size=current,
                device=device,
            ).numpy()
            for local in range(current):
                decoded = decode_adj(encoded[local])
                trimmed = _trim_official(decoded)
                n = int(trimmed.shape[0])
                output[cursor + local, :n, :n] = trimmed
                sizes[cursor + local] = n
                if n == 0:
                    empty_count += 1
                elif n > 1:
                    # Lightweight connectedness check without NetworkX.
                    seen = {0}
                    queue = deque([0])
                    while queue:
                        node = queue.popleft()
                        for neighbor in np.flatnonzero(trimmed[node]).tolist():
                            if int(neighbor) not in seen:
                                seen.add(int(neighbor))
                                queue.append(int(neighbor))
                    if len(seen) != n:
                        disconnected_count += 1
            cursor += current
            batch_index += 1
            if batch_index % int(progress_every_batches) == 0 or cursor == int(num_graphs):
                print(
                    "GRAPHRNN_GENERATION_PROGRESS "
                    f"generated={cursor}/{num_graphs} batches={batch_index}",
                    flush=True,
                )
    metadata = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "variant": str(resolved["variant"]),
        "empty_graph_count": empty_count,
        "disconnected_nonempty_graph_count": disconnected_count,
        "duration_seconds": time.monotonic() - started,
        "device": str(device),
    }
    return output, sizes, metadata
