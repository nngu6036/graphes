from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, *, include_torch: bool = False) -> None:
    """Set lightweight RNG seeds.

    Torch seeding is optional because importing torch can be slow and should not
    be required for dataset-only or metric-only commands.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_torch:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
