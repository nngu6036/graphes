from __future__ import annotations

from typing import Any

import torch


def torch_load_compat(*args: Any, **kwargs: Any) -> Any:
    """Load torch checkpoints across PyTorch versions.

    Newer PyTorch versions accept ``weights_only`` and default it to a safer
    value for untrusted files.  Older versions raise ``TypeError`` when that
    keyword is provided.  Benchmark wrapper checkpoints store Python objects in
    addition to tensors, so callers should default to normal unpickling for
    local, trusted checkpoint files.
    """
    kwargs.setdefault("weights_only", False)
    try:
        return torch.load(*args, **kwargs)
    except TypeError as exc:
        message = str(exc).lower()
        if "weights_only" not in kwargs or ("weights_only" not in message and "unexpected keyword" not in message and "invalid keyword" not in message):
            raise
        kwargs.pop("weights_only", None)
        return torch.load(*args, **kwargs)
