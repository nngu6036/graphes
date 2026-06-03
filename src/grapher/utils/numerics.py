from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np


def is_finite_number(value: Any) -> bool:
    if isinstance(value, (bool, str, bytes)) or value is None:
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(float(value)))
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return True
    if arr.size == 0:
        return True
    return bool(np.all(np.isfinite(arr)))


def assert_finite_array(value: Any, *, context: str) -> None:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size and not np.all(np.isfinite(arr)):
        raise FloatingPointError(f"Non-finite values detected in {context}; shape={arr.shape}.")


def _find_nonfinite_attrs(data: Mapping[str, Any], *, prefix: str) -> list[str]:
    bad: list[str] = []
    for key, value in data.items():
        if not is_finite_number(value):
            bad.append(f"{prefix}.{key}")
    return bad


def graph_nonfinite_report(graph: nx.Graph, *, graph_index: int | None = None) -> list[str]:
    prefix = "graph" if graph_index is None else f"graph[{graph_index}]"
    bad = _find_nonfinite_attrs(graph.graph, prefix=f"{prefix}.graph")
    for node, data in graph.nodes(data=True):
        bad.extend(_find_nonfinite_attrs(data, prefix=f"{prefix}.node[{node!r}]"))
    for u, v, data in graph.edges(data=True):
        bad.extend(_find_nonfinite_attrs(data, prefix=f"{prefix}.edge[{u!r},{v!r}]"))
    return bad


def assert_finite_graphs(graphs: Sequence[nx.Graph], *, context: str, max_failures: int = 5) -> None:
    if not isinstance(graphs, Sequence):
        raise TypeError(f"{context}: expected a sequence of NetworkX graphs, got {type(graphs)}.")
    failures: list[str] = []
    for idx, graph in enumerate(graphs):
        if not isinstance(graph, nx.Graph):
            failures.append(f"graph[{idx}] is {type(graph)} instead of networkx.Graph")
        else:
            failures.extend(graph_nonfinite_report(graph, graph_index=idx))
        if len(failures) >= max_failures:
            break
    if failures:
        joined = "; ".join(failures[:max_failures])
        raise FloatingPointError(f"{context}: non-finite or invalid generated graph payload detected: {joined}")


def _torch_module_finite_report(module: Any, *, prefix: str) -> list[str]:
    try:
        import torch
    except Exception:
        return []
    if not isinstance(module, torch.nn.Module):
        return []
    bad: list[str] = []
    with torch.no_grad():
        for name, tensor in list(module.named_parameters(recurse=True)) + list(module.named_buffers(recurse=True)):
            if tensor is not None and tensor.numel() and not torch.isfinite(tensor).all().item():
                bad.append(f"{prefix}.{name}")
    return bad


def assert_model_tensors_finite(model_or_wrapper: Any, *, context: str, max_failures: int = 20) -> None:
    """Best-effort finite check for torch modules owned by a wrapper.

    External wrappers store their trained modules under different attribute names
    (`model`, `_model`, `score_model`, etc.).  This function scans direct wrapper
    attributes and validates any `torch.nn.Module` it finds.  It is intentionally
    conservative: non-module attributes are ignored.
    """

    failures: list[str] = []
    failures.extend(_torch_module_finite_report(model_or_wrapper, prefix=context))
    for attr_name, value in vars(model_or_wrapper).items() if hasattr(model_or_wrapper, "__dict__") else []:
        failures.extend(_torch_module_finite_report(value, prefix=f"{context}.{attr_name}"))
        if len(failures) >= max_failures:
            break
    if failures:
        raise FloatingPointError(f"{context}: non-finite parameters/buffers detected: {', '.join(failures[:max_failures])}")


def assert_torch_loss_finite(loss: Any, *, context: str) -> None:
    try:
        import torch
    except Exception:
        return
    if isinstance(loss, torch.Tensor) and (loss.numel() == 0 or not torch.isfinite(loss).all().item()):
        raise FloatingPointError(f"{context}: non-finite loss detected.")


def assert_torch_grads_finite(module: Any, *, context: str) -> None:
    try:
        import torch
    except Exception:
        return
    if not isinstance(module, torch.nn.Module):
        return
    bad: list[str] = []
    for name, param in module.named_parameters(recurse=True):
        if param.grad is not None and param.grad.numel() and not torch.isfinite(param.grad).all().item():
            bad.append(name)
            if len(bad) >= 20:
                break
    if bad:
        raise FloatingPointError(f"{context}: non-finite gradients detected: {', '.join(bad)}")


def sanitize_numpy_array(value: Any, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> np.ndarray:
    return np.nan_to_num(np.asarray(value), nan=nan, posinf=posinf, neginf=neginf)
