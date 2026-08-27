from __future__ import annotations

from typing import Any


def resolve_torch_device(device: Any = "auto"):
    import torch

    requested = "auto" if device is None else str(device).strip().lower()
    if requested in {"", "auto", "gpu"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested for device {requested!r}, but "
            "torch.cuda.is_available() is False."
        )
    return resolved
