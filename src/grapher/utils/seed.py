from __future__ import annotations

import random

import numpy as np


def make_rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
