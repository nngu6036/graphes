from __future__ import annotations
from typing import Iterable
import numpy as np

def mean_std(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))
