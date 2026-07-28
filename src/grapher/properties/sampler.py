from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from grapher.generators.degree_sampler import DegreeVAESampler, EmpiricalDegreeSampler
from grapher.properties.summary import SummaryConfig, extract_summary


@dataclass
class EmpiricalSummarySampler:
    summaries: list[dict[str, Any]]
    seed: int = 0

    @classmethod
    def fit(
        cls,
        graphs: list[nx.Graph],
        config: SummaryConfig | dict[str, Any] | None = None,
        seed: int = 0,
    ) -> EmpiricalSummarySampler:
        cfg = (
            config
            if isinstance(config, SummaryConfig)
            else SummaryConfig.from_dict(config or {}, graphs)
        )
        return cls([extract_summary(g, cfg) for g in graphs], seed=seed)

    def sample(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        if not self.summaries:
            raise ValueError("Cannot sample from an empty summary list.")
        generator = rng if rng is not None else np.random.default_rng(self.seed)
        idx = int(generator.integers(0, len(self.summaries)))
        # Return a shallow copy so downstream hybrid samplers can safely modify fields.
        return dict(self.summaries[idx])

    def sample_conditioned(
        self,
        degree_summary: dict[str, Any],
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        """Sample a real structural target with a nearby degree condition.

        This is the non-parametric conditional baseline.  Exact matches are
        preferred; otherwise it chooses randomly among the nearest summaries
        by node count and padded degree-histogram distance.
        """

        if not self.summaries:
            raise ValueError("Cannot sample from an empty summary list.")
        generator = rng if rng is not None else np.random.default_rng(self.seed)
        target_n = int(degree_summary.get("num_nodes", 0))
        target_hist = np.asarray(
            degree_summary.get("degree_hist", []), dtype=np.float64
        )

        scores: list[float] = []
        for summary in self.summaries:
            hist = np.asarray(summary.get("degree_hist", []), dtype=np.float64)
            width = max(target_hist.size, hist.size, 1)
            left = np.zeros(width, dtype=np.float64)
            right = np.zeros(width, dtype=np.float64)
            left[: target_hist.size] = target_hist
            right[: hist.size] = hist
            node_penalty = abs(int(summary.get("num_nodes", 0)) - target_n)
            scores.append(float(node_penalty + np.linalg.norm(left - right)))

        best = np.flatnonzero(np.isclose(scores, np.min(scores)))
        idx = int(generator.choice(best))
        return dict(self.summaries[idx])


class LearnedSummarySampler:
    """Sampler backed by a trained SummaryVAE checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        deterministic: bool = False,
        seed: int = 0,
    ):
        from grapher.utils.device import resolve_torch_device

        self.checkpoint_path = str(checkpoint_path)
        self.device = resolve_torch_device(device)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self._model = None
        self._vectorizer = None
        self._load()

    def _load(self) -> None:
        from grapher.generators.summary_vae import (
            ConditionalSummaryVAE,
            load_summary_vae_checkpoint,
        )

        model, vectorizer, _ = load_summary_vae_checkpoint(
            self.checkpoint_path, device=self.device
        )
        self._model = model
        self._vectorizer = vectorizer
        self._conditional = isinstance(model, ConditionalSummaryVAE)

    @classmethod
    def from_config(
        cls, data: dict[str, Any], *, seed: int = 0
    ) -> LearnedSummarySampler:
        checkpoint_path = (
            data.get("checkpoint_path") or data.get("path") or data.get("checkpoint")
        )
        if not checkpoint_path:
            raise ValueError(
                "Learned summary sampler requires summary_generator.checkpoint_path."
            )
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
        if getattr(self, "_conditional", False):
            raise ValueError(
                "This target-summary checkpoint is conditional on a degree "
                "sequence. Call sample_conditioned(degree_summary, rng), or "
                "enable degree_generator so HybridSummarySampler supplies it."
            )
        generator = rng if rng is not None else np.random.default_rng(self.seed)
        with torch.no_grad():
            outputs = self._model.sample_outputs(1, device=self.device)
        return self._vectorizer.outputs_to_summaries(
            outputs, rng=generator, deterministic=self.deterministic
        )[0]

    def sample_conditioned(
        self,
        degree_summary: dict[str, Any],
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        import torch

        if self._model is None or self._vectorizer is None:
            self._load()
        if not getattr(self, "_conditional", False):
            return self.sample(rng)
        generator = rng if rng is not None else np.random.default_rng(self.seed)
        condition_np = self._vectorizer.to_condition_vector(degree_summary)
        condition = torch.as_tensor(
            condition_np[None, :],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            outputs = self._model.sample_outputs(condition, device=self.device)
        return self._vectorizer.outputs_to_summaries(
            outputs,
            rng=generator,
            deterministic=self.deterministic,
            condition_summaries=[degree_summary],
        )[0]


class HybridSummarySampler:
    """Merge a structure-summary sampler with an explicit degree sampler.

    The structure sampler supplies clustering/spectral/motif/orbit targets.
    The degree sampler supplies num_nodes, num_edges, degree_sequence,
    degree_hist, and density. This avoids asking the generic SummaryVAE to
    learn the degree sequence distribution, which is a hard invariant for
    degree-preserving rewiring.
    """

    def __init__(self, structure_sampler: Any, degree_sampler: Any):
        self.structure_sampler = structure_sampler
        self.degree_sampler = degree_sampler

    def sample(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        generator = rng if rng is not None else np.random.default_rng(0)
        degree_summary = self.degree_sampler.sample(generator)
        if hasattr(self.structure_sampler, "sample_conditioned"):
            summary = dict(
                self.structure_sampler.sample_conditioned(
                    degree_summary,
                    generator,
                )
            )
        else:
            summary = dict(self.structure_sampler.sample(generator))
        n = int(degree_summary["num_nodes"])
        sequence = sorted(
            [int(d) for d in degree_summary["degree_sequence"]], reverse=True
        )
        m = int(sum(sequence) // 2)
        summary["num_nodes"] = n
        summary["num_edges"] = m
        summary["degree_sequence"] = sequence
        summary["degree_hist"] = np.asarray(
            degree_summary["degree_hist"], dtype=np.float64
        )
        summary["density"] = float((2.0 * m / (n * (n - 1))) if n > 1 else 0.0)
        return summary


def build_degree_sampler_from_config(
    data: dict[str, Any], train_graphs: list[nx.Graph], *, seed: int = 0
):
    data = data or {}
    if not bool(data.get("enabled", False)):
        return None
    degree_type = str(data.get("type", "degree_histogram_vae")).lower()
    if degree_type in {"degree_histogram_vae", "degree_vae", "vae", "learned"}:
        return DegreeVAESampler.from_config(data, seed=seed)
    if degree_type in {"empirical", "empirical_degree"}:
        return EmpiricalDegreeSampler.fit_from_graphs(train_graphs, seed=seed)
    raise ValueError(f"Unknown degree_generator.type: {degree_type!r}")


def maybe_wrap_with_degree_sampler(
    structure_sampler: Any,
    config: dict[str, Any],
    train_graphs: list[nx.Graph],
    *,
    seed: int = 0,
):
    degree_sampler = build_degree_sampler_from_config(
        config.get("degree_generator", {}) or {}, train_graphs, seed=seed
    )
    if degree_sampler is None:
        return structure_sampler
    return HybridSummarySampler(structure_sampler, degree_sampler)
