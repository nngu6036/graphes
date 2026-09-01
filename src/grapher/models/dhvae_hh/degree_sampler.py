from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from grapher.models.dhvae_hh.degree_vae import load_degree_vae_checkpoint
from grapher.models.dhvae_hh.typed_degree_vae import (
    load_typed_signature_checkpoint,
)
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
        sample_num_edges: str = "model",
        exact_degree_sum_conditioning: bool = True,
        max_resample: int = 200,
        model_resample_attempts: int = 32,
        parity_conditioned: bool = True,
        max_parity_resample: int = 32,
        fallback: str = "empirical_nearest_n",
        postprocess_policy: str = "repair",
    ):
        self.checkpoint_path = str(checkpoint_path)
        self.device = resolve_torch_device(device)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.sample_num_nodes = str(sample_num_nodes)
        self.sample_num_edges = str(sample_num_edges)
        self.exact_degree_sum_conditioning = bool(exact_degree_sum_conditioning)
        self.max_resample = int(max_resample)
        self.model_resample_attempts = max(int(model_resample_attempts), 1)
        self.parity_conditioned = bool(parity_conditioned)
        self.max_parity_resample = int(max_parity_resample)
        self.fallback = str(fallback)
        self.postprocess_policy = str(postprocess_policy)
        self._model = None
        self._vectorizer = None
        self._load()

    def _load(self) -> None:
        model, vectorizer, _ = load_degree_vae_checkpoint(
            self.checkpoint_path, device=self.device
        )
        self._model = model
        self._vectorizer = vectorizer

    @classmethod
    def from_config(cls, data: dict[str, Any], *, seed: int = 0) -> "DegreeVAESampler":
        checkpoint_path = data.get("checkpoint_path") or data.get("checkpoint")
        if not checkpoint_path:
            raise ValueError(
                "DegreeVAESampler requires degree_generator.checkpoint_path."
            )
        return cls(
            checkpoint_path,
            device=str(data.get("device", "auto")),
            deterministic=bool(data.get("deterministic", False)),
            seed=int(data.get("seed", seed)),
            sample_num_nodes=str(data.get("sample_num_nodes", "empirical")),
            sample_num_edges=str(data.get("sample_num_edges", "model")),
            exact_degree_sum_conditioning=bool(
                data.get("exact_degree_sum_conditioning", True)
            ),
            max_resample=int(data.get("max_resample", 200)),
            model_resample_attempts=int(data.get("model_resample_attempts", 32)),
            parity_conditioned=bool(data.get("parity_conditioned", True)),
            max_parity_resample=int(data.get("max_parity_resample", 32)),
            fallback=str(data.get("fallback", "empirical_nearest_n")),
            postprocess_policy=str(data.get("postprocess_policy", "repair")),
        )

    def sample(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        import torch

        if self._model is None or self._vectorizer is None:
            self._load()
        generator = rng if rng is not None else np.random.default_rng(self.seed)

        # ``max_resample`` controls degree-histogram rejection for a *fixed*
        # prior draw.  Sparse datasets such as Ego-small can occasionally
        # produce a latent/edge-count draw whose categorical degree law has
        # negligible mass on graphical connected sequences.  Redrawing the
        # model prior is still genuine rejection sampling from p(D); it is
        # preferable to silently replacing the sample with an empirical
        # sequence.
        last_error: RuntimeError | None = None
        full_attempt_limit = (
            self.model_resample_attempts if self.fallback.lower() == "error" else 1
        )
        for full_attempt in range(1, full_attempt_limit + 1):
            node_counts = None
            if self.sample_num_nodes.lower() == "empirical":
                node_counts = [self._vectorizer.sample_empirical_node_count(generator)]
            edge_counts = None
            if (
                node_counts is not None
                and self.sample_num_edges.lower() == "empirical"
            ):
                edge_counts = [
                    self._vectorizer.sample_empirical_edge_count(
                        node_counts[0], generator
                    )
                ]
            with torch.no_grad():
                outputs = self._model.sample_outputs(
                    1,
                    node_counts=node_counts,
                    edge_counts=edge_counts,
                    deterministic_node_count=self.deterministic,
                    deterministic_edge_count=self.deterministic,
                    device=self.device,
                )
            try:
                summary = self._vectorizer.outputs_to_summaries(
                    outputs,
                    rng=generator,
                    deterministic=self.deterministic,
                    sample_num_nodes=self.sample_num_nodes,
                    sample_num_edges=self.sample_num_edges,
                    exact_degree_sum_conditioning=self.exact_degree_sum_conditioning,
                    max_resample=self.max_resample,
                    parity_conditioned=self.parity_conditioned,
                    max_parity_resample=self.max_parity_resample,
                    fallback=self.fallback,
                    postprocess_policy=self.postprocess_policy,
                    include_diagnostics=True,
                )[0]
                diagnostic = summary.get("sampling_diagnostics")
                if isinstance(diagnostic, dict):
                    diagnostic["model_resample_attempts"] = int(full_attempt)
                    diagnostic["model_resample_redraws"] = int(full_attempt - 1)
                return summary
            except RuntimeError as exc:
                last_error = exc

        raise RuntimeError(
            "Degree generator exhausted full prior redraws without a valid "
            "graphical, connected-feasible degree sequence "
            f"({full_attempt_limit} model draws; {self.max_resample} "
            "degree-histogram attempts per draw)."
        ) from last_error


class EmpiricalDegreeSampler:
    def __init__(self, degree_sequences: list[list[int]], seed: int = 0):
        if not degree_sequences:
            raise ValueError(
                "EmpiricalDegreeSampler requires at least one degree sequence."
            )
        self.degree_sequences = [[int(d) for d in seq] for seq in degree_sequences]
        self.seed = int(seed)

    @classmethod
    def fit_from_graphs(cls, graphs, seed: int = 0) -> "EmpiricalDegreeSampler":
        return cls(
            [sorted([int(d) for _, d in g.degree()], reverse=True) for g in graphs],
            seed=seed,
        )

    def sample(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        generator = rng if rng is not None else np.random.default_rng(self.seed)
        seq = list(
            self.degree_sequences[
                int(generator.integers(0, len(self.degree_sequences)))
            ]
        )
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


class TypedDegreeVAESampler:
    """Sampler for the joint atom/edge-type-degree invariant prior."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        deterministic: bool = False,
        seed: int = 0,
        sample_num_nodes: str = "empirical",
        max_resample: int = 1000,
        fallback: str = "error",
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.device = resolve_torch_device(device)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.sample_num_nodes = str(sample_num_nodes)
        self.max_resample = int(max_resample)
        self.fallback = str(fallback)
        self._model, self._vectorizer, _ = load_typed_signature_checkpoint(
            self.checkpoint_path,
            device=self.device,
        )

    @classmethod
    def from_config(
        cls, data: dict[str, Any], *, seed: int = 0
    ) -> "TypedDegreeVAESampler":
        checkpoint_path = data.get("checkpoint_path") or data.get("checkpoint")
        if not checkpoint_path:
            raise ValueError("TypedDegreeVAESampler requires checkpoint_path.")
        return cls(
            checkpoint_path,
            device=str(data.get("device", "auto")),
            deterministic=bool(data.get("deterministic", False)),
            seed=int(data.get("seed", seed)),
            sample_num_nodes=str(data.get("sample_num_nodes", "empirical")),
            max_resample=int(data.get("max_resample", 1000)),
            fallback=str(data.get("fallback", "error")),
        )

    def sample(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        import torch

        generator = rng if rng is not None else np.random.default_rng(self.seed)
        node_counts = None
        if self.sample_num_nodes.lower() == "empirical":
            node_counts = [self._vectorizer.sample_empirical_node_count(generator)]
        with torch.no_grad():
            outputs = self._model.sample_outputs(
                1,
                node_counts=node_counts,
                device=self.device,
            )
        return self._vectorizer.outputs_to_summaries(
            outputs,
            rng=generator,
            deterministic=self.deterministic,
            max_resample=self.max_resample,
            fallback=self.fallback,
        )[0]


def build_degree_sampler(
    config: dict[str, Any],
    train_graphs: list[Any],
    *,
    seed: int = 0,
) -> DegreeVAESampler | TypedDegreeVAESampler | EmpiricalDegreeSampler | None:
    """Build the invariant sampler used by the DH-VAE+HH baseline."""

    if not config or not bool(config.get("enabled", False)):
        return None
    sampler_type = str(config.get("type", "degree_histogram_vae")).lower()
    if sampler_type in {
        "degree_histogram_vae",
        "degree_vae",
        "vae",
        "learned",
    }:
        return DegreeVAESampler.from_config(config, seed=seed)
    if sampler_type in {
        "typed_degree_histogram_vae",
        "typed_signature_histogram_vae",
        "typed_signature_vae",
    }:
        return TypedDegreeVAESampler.from_config(config, seed=seed)
    if sampler_type in {"empirical", "empirical_degree"}:
        return EmpiricalDegreeSampler.fit_from_graphs(train_graphs, seed=seed)
    raise ValueError(f"Unknown degree_generator.type: {sampler_type!r}")
