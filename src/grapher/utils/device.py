from __future__ import annotations

from typing import Any


def resolve_torch_device(device: Any = "auto"):
    import torch

    requested = "auto" if device is None else str(device).strip().lower()
    if requested in {"", "auto", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required, but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    resolved = torch.device(requested)
    if resolved.type != "cuda":
        raise RuntimeError(f"CUDA GPU is required; requested device {requested!r} is not supported.")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA GPU is required for device {requested!r}, but torch.cuda.is_available() is False.")
    return resolved
