"""Small runtime repairs for defects in the attached HOG-Diff release."""

from __future__ import annotations

from typing import Any


def install_torch_functional_alias(
    layers_module: Any,
    functional_module: Any,
) -> bool:
    """Provide ``models.layers.F`` when upstream forgot to import it.

    HOG-Diff's attention layer calls ``F.softmax``. Some distributed source
    snapshots omit ``import torch.nn.functional as F``, so install the exact
    module alias in memory without changing the external checkout. The return
    value records whether a repair was needed.
    """

    if getattr(layers_module, "F", None) is not None:
        return False
    layers_module.F = functional_module
    return True
