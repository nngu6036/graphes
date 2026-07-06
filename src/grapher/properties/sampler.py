from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


class LearnedSummarySampler:
    """Sampler backed by a trained SummaryVAE checkpoint."""

    def __init__(self, checkpoint_path: str | Path, *, device: str = "auto", deterministic: bool = False, seed: int = 0):
        from grapher.utils.device import resolve_torch_device

        self.checkpoint_path = str(checkpoint_path)
        self.device = resolve_torch_device(device)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self._model = None
        self._vectorizer = None
        self._load()

    def _load(self) -> None:
        from grapher.generators.summary_vae import load_summary_vae_checkpoint

        model, vectorizer, _ = load_summary_vae_checkpoint(self.checkpoint_path, device=self.device)
        self._model = model
        self._vectorizer = vectorizer

    @classmethod
    def from_config(cls, data: dict[str, Any], *, seed: int = 0) -> "LearnedSummarySampler":
        checkpoint_path = data.get("checkpoint_path") or data.get("path")
        if not checkpoint_path:
            raise ValueError("Learned summary sampler requires summary_generator.checkpoint_path.")
        return cls(
            checkpoint_path,
            device=str(data.get("device", "auto")),
            deterministic=bool(data.get("deterministic", False)),
            seed=int(data.get("seed", seed)),
        )

    def sample(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        import torch

        if self._model is None or self._vectorizer is None:
            self._load()
        generator = rng if rng is not None else np.random.default_rng(self.seed)
        with torch.no_grad():
            outputs = self._model.sample_outputs(1, device=self.device)
        return self._vectorizer.outputs_to_summaries(outputs, rng=generator, deterministic=self.deterministic)[0]
