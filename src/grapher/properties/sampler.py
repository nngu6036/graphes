from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from grapher.properties.summary import SummaryConfig, extract_summary


@dataclass
class EmpiricalSummarySampler:
    summaries: list[dict[str, Any]]
    seed: int = 0

    @classmethod
    def fit(cls, graphs: list[nx.Graph], config: SummaryConfig | dict[str, Any] | None = None, seed: int = 0) -> "EmpiricalSummarySampler":
        cfg = config if isinstance(config, SummaryConfig) else SummaryConfig.from_dict(config or {}, graphs)
        return cls([extract_summary(g, cfg) for g in graphs], seed=seed)

    def sample(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        if not self.summaries:
            raise ValueError("Cannot sample from an empty summary list.")
        generator = rng if rng is not None else np.random.default_rng(self.seed)
        idx = int(generator.integers(0, len(self.summaries)))
        return self.summaries[idx]
