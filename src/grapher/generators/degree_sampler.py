from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from grapher.generators.degree_vae import load_degree_vae_checkpoint
from grapher.utils.device import resolve_torch_device


class DegreeVAESampler:
    """Sampler backed by a trained degree-histogram VAE."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        deterministic: bool = False,
        seed: int = 0,
        sample_num_nodes: str = "empirical",
        max_resample: int = 200,
        parity_conditioned: bool = True,
        max_parity_resample: int = 32,
        fallback: str = "empirical_nearest_n",
    ):
        self.checkpoint_path = str(checkpoint_path)
        self.device = resolve_torch_device(device)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.sample_num_nodes = str(sample_num_nodes)
        self.max_resample = int(max_resample)
        self.parity_conditioned = bool(parity_conditioned)
        self.max_parity_resample = int(max_parity_resample)
        self.fallback = str(fallback)
        self._model = None
        self._vectorizer = None
        self._load()

    def _load(self) -> None:
        model, vectorizer, _ = load_degree_vae_checkpoint(self.checkpoint_path, device=self.device)
        self._model = model
        self._vectorizer = vectorizer

    @classmethod
    def from_config(cls, data: dict[str, Any], *, seed: int = 0) -> "DegreeVAESampler":
        checkpoint_path = data.get("checkpoint_path") or data.get("checkpoint")
        if not checkpoint_path:
            raise ValueError("DegreeVAESampler requires degree_generator.checkpoint_path.")
        return cls(
            checkpoint_path,
            device=str(data.get("device", "auto")),
            deterministic=bool(data.get("deterministic", False)),
            seed=int(data.get("seed", seed)),
            sample_num_nodes=str(data.get("sample_num_nodes", "empirical")),
            max_resample=int(data.get("max_resample", 200)),
            parity_conditioned=bool(data.get("parity_conditioned", True)),
            max_parity_resample=int(data.get("max_parity_resample", 32)),
            fallback=str(data.get("fallback", "empirical_nearest_n")),
        )

    def sample(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        import torch

        if self._model is None or self._vectorizer is None:
            self._load()
        generator = rng if rng is not None else np.random.default_rng(self.seed)
        node_counts = None
        if self.sample_num_nodes.lower() == "empirical":
            node_counts = [
                self._vectorizer.sample_empirical_node_count(generator)
            ]
        with torch.no_grad():
            outputs = self._model.sample_outputs(
                1,
                node_counts=node_counts,
                deterministic_node_count=self.deterministic,
                device=self.device,
            )
        return self._vectorizer.outputs_to_summaries(
            outputs,
            rng=generator,
            deterministic=self.deterministic,
            sample_num_nodes=self.sample_num_nodes,
            max_resample=self.max_resample,
            parity_conditioned=self.parity_conditioned,
            max_parity_resample=self.max_parity_resample,
            fallback=self.fallback,
        )[0]


class EmpiricalDegreeSampler:
    def __init__(self, degree_sequences: list[list[int]], seed: int = 0):
        if not degree_sequences:
            raise ValueError("EmpiricalDegreeSampler requires at least one degree sequence.")
        self.degree_sequences = [[int(d) for d in seq] for seq in degree_sequences]
        self.seed = int(seed)

    @classmethod
    def fit_from_graphs(cls, graphs, seed: int = 0) -> "EmpiricalDegreeSampler":
        return cls([sorted([int(d) for _, d in g.degree()], reverse=True) for g in graphs], seed=seed)

    def sample(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        generator = rng if rng is not None else np.random.default_rng(self.seed)
        seq = list(self.degree_sequences[int(generator.integers(0, len(self.degree_sequences)))])
        n = len(seq)
        m = int(sum(seq) // 2)
        max_degree = max(max(seq) if seq else 0, 1)
        hist = np.bincount(seq, minlength=max_degree + 1).astype(np.float64)
        hist = hist / max(hist.sum(), 1.0)
        density = (2.0 * m / (n * (n - 1))) if n > 1 else 0.0
        return {
            "num_nodes": int(n),
            "num_edges": int(m),
            "degree_sequence": sorted([int(d) for d in seq], reverse=True),
            "degree_hist": hist,
            "density": float(density),
        }
